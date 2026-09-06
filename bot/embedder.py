"""Text embeddings for semantic search, from three sources.

  local — sentence-transformers in-process. Best quality offline, but pulls in
          torch: ~2.5 GB RAM to build and ~700 MB resident. Too much for a
          small VPS.
  api   — any OpenAI-compatible /embeddings endpoint (OpenAI, llama.cpp,
          Ollama, Mistral, ...). Almost no local memory.
  none  — no embeddings at all. Search falls back to SQLite FTS5 keyword
          matching over the captions, which still works, just literally.
"""
from __future__ import annotations

import asyncio
import logging

import numpy as np

log = logging.getLogger(__name__)


class Embedder:
    label = "none"

    async def encode(self, texts: list[str]) -> list[np.ndarray | None]:
        return [None] * len(texts)

    async def encode_one(self, text: str) -> np.ndarray | None:
        return (await self.encode([text]))[0]

    async def close(self) -> None:
        return None


class NullEmbedder(Embedder):
    """Keyword-only search. Cheap, and fine for a few hundred stickers."""


class LocalEmbedder(Embedder):
    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise SystemExit(
                "EMBED_PROVIDER=local needs sentence-transformers. Rebuild with "
                "WITH_LOCAL_EMBEDDINGS=1, or use EMBED_PROVIDER=api / none."
            ) from exc

        log.info("loading embedding model %s", model_name)
        self.label = f"local:{model_name}"
        try:
            self._model = SentenceTransformer(model_name)
        except Exception as exc:  # noqa: BLE001
            # Almost always an API model name left in EMBED_MODEL, which
            # otherwise surfaces as an opaque huggingface traceback.
            raise SystemExit(
                f"EMBED_MODEL={model_name!r} is not a local "
                f"sentence-transformers model ({type(exc).__name__}: "
                f"{str(exc).splitlines()[0][:160]}).\n"
                f"Names like 'text-embedding-3-small' or 'mistral-embed' belong "
                f"to an API provider — set EMBED_PROVIDER=api together with "
                f"EMBED_BASE_URL and EMBED_API_KEY.\n"
                f"For in-process embeddings pick a sentence-transformers model, "
                f"e.g. sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2."
            ) from exc

    def _encode(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")

    async def encode(self, texts: list[str]) -> list[np.ndarray | None]:
        return list(await asyncio.to_thread(self._encode, texts))


class ApiEmbedder(Embedder):
    """OpenAI-compatible POST /embeddings."""

    def __init__(self, base_url: str, api_key: str, model: str,
                 timeout: float = 60.0):
        import httpx

        self.label = f"api:{model}"
        self.url = f"{base_url.rstrip('/')}/embeddings"
        self.model = model
        self._client = httpx.AsyncClient(
            timeout=timeout, headers={"Authorization": f"Bearer {api_key}"})

    async def encode(self, texts: list[str]) -> list[np.ndarray | None]:
        try:
            resp = await self._client.post(
                self.url, json={"model": self.model, "input": texts})
            resp.raise_for_status()
            rows = sorted(resp.json()["data"], key=lambda d: d.get("index", 0))
        except Exception as exc:  # noqa: BLE001
            log.warning("embedding request failed, falling back to keyword "
                        "search for this batch: %s", exc)
            return [None] * len(texts)

        out = []
        for row in rows:
            v = np.asarray(row["embedding"], dtype="float32")
            norm = np.linalg.norm(v)
            out.append(v / norm if norm else v)
        return out

    async def close(self) -> None:
        await self._client.aclose()


def build_embedder(provider: str, model: str, base_url: str, api_key: str) -> Embedder:
    if provider == "auto":
        try:
            import sentence_transformers  # noqa: F401
            provider = "local"
        except ImportError:
            provider = "api" if base_url else "none"
        log.info("EMBED_PROVIDER=auto resolved to %s", provider)

    if provider == "local":
        return LocalEmbedder(model)
    if provider == "api":
        if not base_url:
            raise SystemExit("EMBED_PROVIDER=api needs EMBED_BASE_URL")
        return ApiEmbedder(base_url, api_key, model)
    log.warning("no embedding model: search will be keyword-only")
    return NullEmbedder()
