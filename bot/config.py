"""Configuration, read once from the environment."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

# provider -> (wire protocol, default base url, default model)
PRESETS: dict[str, tuple[str, str, str]] = {
    # local / self-hosted
    "llamacpp":   ("openai", "http://host.docker.internal:8080/v1", "qwen2.5-vl"),
    "ollama":     ("openai", "http://host.docker.internal:11434/v1", "qwen2.5vl:7b"),
    "vllm":       ("openai", "http://host.docker.internal:8000/v1", "Qwen/Qwen2.5-VL-7B-Instruct"),
    # cloud
    "openai":     ("openai", "https://api.openai.com/v1", "gpt-4o-mini"),
    "openrouter": ("openai", "https://openrouter.ai/api/v1", "google/gemini-2.5-flash"),
    "groq":       ("openai", "https://api.groq.com/openai/v1", "meta-llama/llama-4-scout-17b-16e-instruct"),
    "mistral":    ("openai", "https://api.mistral.ai/v1", "pixtral-12b-2409"),
    "xai":        ("openai", "https://api.x.ai/v1", "grok-2-vision-1212"),
    "anthropic":  ("anthropic", "https://api.anthropic.com/v1", "claude-sonnet-5"),
    "google":     ("google", "https://generativelanguage.googleapis.com/v1beta",
                   "gemini-2.5-flash"),
    # anything else that speaks the OpenAI protocol
    "custom":     ("openai", "", ""),
}

CLOUD_PROVIDERS = {"openai", "openrouter", "groq", "mistral", "xai", "anthropic",
                   "google"}


def _ids(raw: str) -> set[int]:
    return {int(x) for x in raw.replace(",", " ").split() if x.strip()}


@dataclass(frozen=True)
class BackendSpec:
    label: str          # provider name, e.g. "llamacpp" or "anthropic"
    kind: str           # wire protocol: openai | anthropic | google
    base_url: str
    api_key: str
    model: str
    timeout: float


def _spec(prefix: str, provider: str, timeout: float) -> BackendSpec | None:
    """Build a backend spec from PREFIX_PROVIDER / _BASE_URL / _MODEL / _API_KEY."""
    provider = provider.strip().lower()
    if provider in ("", "none", "off", "disabled"):
        return None
    if provider not in PRESETS:
        raise SystemExit(
            f"{prefix}_PROVIDER={provider!r} is unknown. "
            f"Pick one of: {', '.join(sorted(PRESETS))}"
        )
    kind, base, model = PRESETS[provider]
    base = os.environ.get(f"{prefix}_BASE_URL", base).strip().rstrip("/")
    model = os.environ.get(f"{prefix}_MODEL", model).strip()
    key = os.environ.get(f"{prefix}_API_KEY", "").strip()

    if not base:
        raise SystemExit(f"{prefix}_BASE_URL must be set for provider {provider!r}")
    if not model:
        raise SystemExit(f"{prefix}_MODEL must be set for provider {provider!r}")
    if provider in CLOUD_PROVIDERS and not key:
        raise SystemExit(f"{prefix}_API_KEY is required for provider {provider!r}")

    return BackendSpec(provider, kind, base, key or "no-key", model, timeout)


@dataclass(frozen=True)
class Config:
    bot_token: str
    db_path: str = "/data/stickers.db"

    local: BackendSpec | None = None
    cloud: BackendSpec | None = None
    backend_mode: str = "auto"      # local | cloud | auto

    caption_language: str = "English"
    caption_concurrency: int = 2

    embed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    allowed_user_ids: set[int] = field(default_factory=set)
    index_user_ids: set[int] = field(default_factory=set)
    max_pack_size: int = 200
    daily_caption_limit: int = 0        # 0 = unlimited
    max_results: int = 3
    inline_results: int = 30

    @classmethod
    def from_env(cls) -> "Config":
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        if not token:
            raise SystemExit("TELEGRAM_BOT_TOKEN is not set")

        timeout = float(os.environ.get("VLM_TIMEOUT", "180"))
        local = _spec("VLM", os.environ.get("VLM_PROVIDER", "llamacpp"), timeout)
        cloud = _spec("CLOUD", os.environ.get("CLOUD_PROVIDER", ""),
                      float(os.environ.get("CLOUD_TIMEOUT", "120")))

        if not local and not cloud:
            raise SystemExit(
                "No captioning backend configured. Set VLM_PROVIDER (local) "
                "and/or CLOUD_PROVIDER (cloud API)."
            )

        mode = os.environ.get("CAPTION_BACKEND", "").strip().lower()
        if not mode:
            mode = "auto" if (local and cloud) else ("local" if local else "cloud")
        if mode not in {"local", "cloud", "auto"}:
            raise SystemExit("CAPTION_BACKEND must be local, cloud or auto")
        if mode == "local" and not local:
            raise SystemExit("CAPTION_BACKEND=local but no local backend configured")
        if mode == "cloud" and not cloud:
            raise SystemExit("CAPTION_BACKEND=cloud but no cloud backend configured")
        if mode == "auto" and not (local and cloud):
            mode = "local" if local else "cloud"

        # A caption concurrency of 2 suits one local GPU; cloud APIs take more.
        default_conc = "6" if mode == "cloud" else "2"

        allowed = _ids(os.environ.get("ALLOWED_USER_IDS", ""))
        # Captioning costs money and GPU time, so it needs its own, tighter list.
        # Unset: inherit ALLOWED_USER_IDS; if that is open too, nobody may index.
        raw_index = os.environ.get("INDEX_USER_IDS")
        indexers = _ids(raw_index) if raw_index is not None else set(allowed)

        return cls(
            bot_token=token,
            db_path=os.environ.get("DB_PATH", "/data/stickers.db"),
            local=local,
            cloud=cloud,
            backend_mode=mode,
            caption_language=os.environ.get("CAPTION_LANGUAGE", "English"),
            caption_concurrency=int(
                os.environ.get("CAPTION_CONCURRENCY", default_conc)),
            embed_model=os.environ.get(
                "EMBED_MODEL",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            ),
            allowed_user_ids=allowed,
            index_user_ids=indexers,
            max_pack_size=int(os.environ.get("MAX_PACK_SIZE", "200")),
            daily_caption_limit=int(os.environ.get("DAILY_CAPTION_LIMIT", "0")),
            max_results=int(os.environ.get("MAX_RESULTS", "3")),
            inline_results=int(os.environ.get("INLINE_RESULTS", "30")),
        )

    def is_allowed(self, user_id: int | None) -> bool:
        """May use the bot at all (searching is cheap and local)."""
        if not self.allowed_user_ids:
            return True
        return user_id in self.allowed_user_ids

    def may_index(self, user_id: int | None) -> bool:
        """May trigger captioning, i.e. spend GPU time or API credits.

        An empty list means nobody: an open bot must not be able to run up a
        bill just because someone forwards it stickers.
        """
        return bool(self.index_user_ids) and user_id in self.index_user_ids
