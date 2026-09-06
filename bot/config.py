"""Configuration, read once from the environment."""
from __future__ import annotations

import importlib.util
import logging
import os
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

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


def _have_sentence_transformers() -> bool:
    # find_spec locates the package without importing it, so resolving to a
    # different provider does not pay for loading torch.
    try:
        return importlib.util.find_spec("sentence_transformers") is not None
    except (ImportError, ValueError):
        return False


def _resolve_embed_provider(provider: str, base_url: str) -> str:
    """Turn 'auto' into a concrete provider.

    Local first: it is the one that costs nothing per query. Resolving here
    rather than in build_embedder means the default EMBED_MODEL below is chosen
    for the provider that will actually be used — picking it from the
    unresolved 'auto' used to hand a sentence-transformers model name to an API,
    which fails per request and silently leaves search keyword-only.
    """
    if provider != "auto":
        return provider
    if _have_sentence_transformers():
        resolved = "local"
    else:
        resolved = "api" if base_url else "none"
    log.info("EMBED_PROVIDER=auto resolved to %s", resolved)
    return resolved


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

    embed_provider: str = "auto"     # local | api | none | auto
    embed_model: str = ""
    embed_base_url: str = ""
    embed_api_key: str = ""

    allowed_user_ids: set[int] = field(default_factory=set)
    index_user_ids: set[int] = field(default_factory=set)
    max_pack_size: int = 200
    daily_caption_limit: int = 0        # 0 = unlimited
    max_results: int = 1        # a plain search answers with the best match
    max_results_cap: int = 10   # most a query may ask for by trailing number
    inline_results: int = 30
    # relevance floors: absolute cosine, and a fraction of the best hit
    search_min_score: float = 0.20
    search_relative_cutoff: float = 0.60

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

        embed_provider = os.environ.get("EMBED_PROVIDER", "auto").strip().lower()
        if embed_provider not in {"local", "api", "none", "auto"}:
            raise SystemExit("EMBED_PROVIDER must be local, api, none or auto")
        embed_base_url = os.environ.get("EMBED_BASE_URL", "").strip().rstrip("/")
        embed_api_key = os.environ.get("EMBED_API_KEY", "").strip()
        embed_provider = _resolve_embed_provider(embed_provider, embed_base_url)
        # chosen for the resolved provider, so the two can never disagree
        default_embed_model = (
            "text-embedding-3-small" if embed_provider == "api"
            else "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        if embed_provider == "local" and embed_api_key and embed_base_url:
            log.warning(
                "EMBED_PROVIDER resolved to local, but EMBED_BASE_URL and "
                "EMBED_API_KEY are set. If you meant to use the API, set "
                "EMBED_PROVIDER=api explicitly.")

        max_results_cap = int(os.environ.get("MAX_RESULTS_CAP", "10"))
        if max_results_cap < 1:
            raise SystemExit("MAX_RESULTS_CAP must be at least 1")

        min_score = float(os.environ.get("SEARCH_MIN_SCORE", "0.20"))
        rel_cutoff = float(os.environ.get("SEARCH_RELATIVE_CUTOFF", "0.60"))
        if not -1.0 <= min_score <= 1.0:
            raise SystemExit("SEARCH_MIN_SCORE must be between -1 and 1")
        if not 0.0 <= rel_cutoff <= 1.0:
            raise SystemExit("SEARCH_RELATIVE_CUTOFF must be between 0 and 1")

        return cls(
            bot_token=token,
            db_path=os.environ.get("DB_PATH", "/data/stickers.db"),
            local=local,
            cloud=cloud,
            backend_mode=mode,
            caption_language=os.environ.get("CAPTION_LANGUAGE", "English"),
            caption_concurrency=int(
                os.environ.get("CAPTION_CONCURRENCY", default_conc)),
            embed_provider=embed_provider,
            embed_model=os.environ.get("EMBED_MODEL", default_embed_model),
            embed_base_url=embed_base_url,
            embed_api_key=embed_api_key,
            allowed_user_ids=allowed,
            index_user_ids=indexers,
            max_pack_size=int(os.environ.get("MAX_PACK_SIZE", "200")),
            daily_caption_limit=int(os.environ.get("DAILY_CAPTION_LIMIT", "0")),
            max_results=int(os.environ.get("MAX_RESULTS", "1")),
            max_results_cap=max_results_cap,
            inline_results=int(os.environ.get("INLINE_RESULTS", "30")),
            search_min_score=min_score,
            search_relative_cutoff=rel_cutoff,
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
