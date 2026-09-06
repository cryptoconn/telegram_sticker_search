"""Vision captioning backends.

Three wire protocols are supported:

  openai     — llama.cpp, Ollama, vLLM, OpenAI, OpenRouter, Groq, Mistral, xAI …
  anthropic  — api.anthropic.com/v1/messages
  google     — generativelanguage.googleapis.com

A CaptionRouter combines a local and a cloud backend so the bot can fall back
to the cloud when the local machine is asleep or the model is not loaded.
"""
from __future__ import annotations

import asyncio
import base64
import html
import logging
import time
from dataclasses import dataclass

import httpx

log = logging.getLogger(__name__)

PROMPT = """You are labelling a chat sticker so someone can find it later by \
describing it from memory.

Write 2-4 sentences in {language} covering, in this order:
- Exactly what is shown. Name the species, character, object or person as \
precisely as you can, and prefer the specific word over the general one: \
"wolf" rather than "animal", "pug" rather than "dog". Where it could be \
mistaken for something similar, say which one it is and name the detail that \
settles it.
- What it is doing: posture, gesture, facial expression and mood.
- Any text in the image, transcribed exactly, in quotes.
- Notable colours, art style, clothing and background.

Describe only what is visibly there, and commit to one reading instead of \
offering alternatives.

Finish with a line starting "Keywords:" listing 5-10 single words someone \
might search for. Use the precise term and its close synonyms only. Never \
list a word for something that is not in the picture: a dog is not to be \
tagged "wolf", and a pig is not to be tagged "boar". A wrong keyword makes \
this sticker turn up in searches for a different animal entirely.

No preamble, no markdown, no commentary."""

# The new prompt asks for a little more than the old one; leave headroom so a
# description is never cut off mid-sentence.
MAX_CAPTION_TOKENS = 400

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


@dataclass
class Caption:
    text: str
    backend: str


class CaptionError(RuntimeError):
    pass


async def _post(client: httpx.AsyncClient, url: str, *, json: dict,
                headers: dict | None = None, attempts: int = 4) -> httpx.Response:
    """POST with backoff on rate limits and transient server errors."""
    delay = 2.0
    last: Exception | None = None
    for i in range(attempts):
        try:
            resp = await client.post(url, json=json, headers=headers)
        except httpx.RequestError as exc:
            last = exc
            if i == attempts - 1:
                break
            await asyncio.sleep(delay)
            delay *= 2
            continue
        if resp.status_code in RETRY_STATUS and i < attempts - 1:
            wait = float(resp.headers.get("retry-after") or delay)
            log.warning("%s → %s, retrying in %.0fs", url, resp.status_code, wait)
            await asyncio.sleep(min(wait, 60))
            delay *= 2
            continue
        if resp.status_code >= 400:
            raise CaptionError(f"{resp.status_code}: {resp.text[:300]}")
        return resp
    raise CaptionError(f"request to {url} failed: {last}")


class Backend:
    """Common interface for every provider."""

    def __init__(self, label: str, base_url: str, api_key: str, model: str,
                 language: str, timeout: float):
        self.label = label
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.language = language
        self._client = httpx.AsyncClient(timeout=timeout)

    def __str__(self) -> str:
        return f"{self.label} ({self.model})"

    @property
    def prompt(self) -> str:
        return PROMPT.format(language=self.language)

    async def describe(self, png: bytes) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    async def healthy(self) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    async def close(self) -> None:
        await self._client.aclose()


class OpenAIBackend(Backend):
    async def describe(self, png: bytes) -> str:
        b64 = base64.b64encode(png).decode()
        resp = await _post(
            self._client,
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "max_tokens": MAX_CAPTION_TOKENS,
                "temperature": 0.2,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        {"type": "text", "text": self.prompt},
                    ],
                }],
            },
        )
        return resp.json()["choices"][0]["message"]["content"].strip()

    async def healthy(self) -> bool:
        try:
            r = await self._client.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10.0,
            )
            return r.status_code < 500
        except Exception as exc:  # noqa: BLE001
            log.info("%s unreachable: %s", self.label, exc)
            return False


class AnthropicBackend(Backend):
    def _headers(self) -> dict:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    async def describe(self, png: bytes) -> str:
        b64 = base64.b64encode(png).decode()
        resp = await _post(
            self._client,
            f"{self.base_url}/messages",
            headers=self._headers(),
            json={
                "model": self.model,
                "max_tokens": MAX_CAPTION_TOKENS,
                "temperature": 0.2,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image",
                         "source": {"type": "base64", "media_type": "image/png",
                                    "data": b64}},
                        {"type": "text", "text": self.prompt},
                    ],
                }],
            },
        )
        blocks = resp.json().get("content", [])
        return "\n".join(b["text"] for b in blocks if b.get("type") == "text").strip()

    async def healthy(self) -> bool:
        try:
            r = await self._client.get(f"{self.base_url}/models",
                                       headers=self._headers(), timeout=10.0)
            return r.status_code < 500
        except Exception as exc:  # noqa: BLE001
            log.info("%s unreachable: %s", self.label, exc)
            return False


class GoogleBackend(Backend):
    async def describe(self, png: bytes) -> str:
        b64 = base64.b64encode(png).decode()
        resp = await _post(
            self._client,
            f"{self.base_url}/models/{self.model}:generateContent",
            headers={"x-goog-api-key": self.api_key,
                     "content-type": "application/json"},
            json={
                "contents": [{
                    "parts": [
                        {"inline_data": {"mime_type": "image/png", "data": b64}},
                        {"text": self.prompt},
                    ],
                }],
                "generationConfig": {"maxOutputTokens": MAX_CAPTION_TOKENS, "temperature": 0.2},
            },
        )
        cands = resp.json().get("candidates", [])
        if not cands:
            raise CaptionError("no candidates returned (blocked by safety filter?)")
        parts = cands[0].get("content", {}).get("parts", [])
        return "\n".join(p["text"] for p in parts if "text" in p).strip()

    async def healthy(self) -> bool:
        try:
            r = await self._client.get(f"{self.base_url}/models",
                                       headers={"x-goog-api-key": self.api_key},
                                       timeout=10.0)
            return r.status_code < 500
        except Exception as exc:  # noqa: BLE001
            log.info("%s unreachable: %s", self.label, exc)
            return False


KINDS = {"openai": OpenAIBackend, "anthropic": AnthropicBackend,
         "google": GoogleBackend}


def build_backend(spec, language: str) -> Backend:
    """spec is a config.BackendSpec."""
    cls = KINDS[spec.kind]
    return cls(spec.label, spec.base_url, spec.api_key, spec.model, language,
               spec.timeout)


class CaptionRouter:
    """Picks a backend per request: local first, cloud as fallback.

    mode:
      local  — only the local endpoint
      cloud  — only the cloud API
      auto   — local first; on failure use cloud and put the local endpoint on a
               short cooldown so a sleeping machine does not stall every request
    """

    COOLDOWN = 300.0

    def __init__(self, local: Backend | None, cloud: Backend | None, mode: str):
        self.local = local
        self.cloud = cloud
        self.mode = mode
        self._local_down_until = 0.0

    @property
    def available(self) -> list[str]:
        return [b.label for b in (self.local, self.cloud) if b]

    def set_mode(self, mode: str) -> None:
        if mode not in {"local", "cloud", "auto"}:
            raise ValueError(mode)
        if mode in {"local", "auto"} and not self.local:
            raise ValueError("no local backend configured")
        if mode in {"cloud", "auto"} and not self.cloud:
            raise ValueError("no cloud backend configured")
        self.mode = mode
        self._local_down_until = 0.0

    def _order(self) -> list[Backend]:
        if self.mode == "local":
            return [b for b in (self.local,) if b]
        if self.mode == "cloud":
            return [b for b in (self.cloud,) if b]
        chain: list[Backend] = []
        if self.local and time.monotonic() >= self._local_down_until:
            chain.append(self.local)
        if self.cloud:
            chain.append(self.cloud)
        if not chain and self.local:
            chain.append(self.local)
        return chain

    async def describe(self, png: bytes) -> Caption:
        errors = []
        for backend in self._order():
            try:
                text = await backend.describe(png)
                if not text:
                    raise CaptionError("empty response")
                if backend is self.local:
                    self._local_down_until = 0.0
                return Caption(text, f"{backend.label}:{backend.model}")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{backend.label}: {exc}")
                if backend is self.local and self.mode == "auto" and self.cloud:
                    self._local_down_until = time.monotonic() + self.COOLDOWN
                    log.warning("local backend down, using %s for the next %.0fs",
                                self.cloud.label, self.COOLDOWN)
        raise CaptionError("; ".join(errors) or "no backend configured")

    async def status(self) -> str:
        lines = [f"mode: <b>{html.escape(self.mode)}</b>"]
        for role, b in (("local", self.local), ("cloud", self.cloud)):
            if not b:
                lines.append(f"{role}: not configured")
                continue
            ok = await b.healthy()
            mark = "reachable" if ok else "unreachable"
            lines.append(f"{role}: {html.escape(b.label)} · "
                         f"<code>{html.escape(b.model)}</code> · {mark}")
        if self._local_down_until > time.monotonic():
            left = self._local_down_until - time.monotonic()
            lines.append(f"local on cooldown for {left:.0f}s")
        return "\n".join(lines)

    async def close(self) -> None:
        for b in (self.local, self.cloud):
            if b:
                await b.close()
