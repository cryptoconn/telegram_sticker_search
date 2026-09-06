# Sticker Search Bot

A Telegram bot that finds stickers by description. It captions every sticker in
your packs with a vision model — local or cloud, whichever you configure —
embeds the captions, and searches them with a hybrid of semantic similarity and
keyword matching. It runs in one Docker container and keeps everything in a
local SQLite file. How much of the work also stays local is your choice: see
[What leaves your machine](#what-leaves-your-machine).

```
sticker  →  PNG frame  →  vision model caption  →  embedding  →  SQLite
query    →  embedding + FTS5  →  reciprocal rank fusion  →  sticker
```

## What you need

1. **A bot token** from [@BotFather](https://t.me/BotFather).
2. **A vision-capable model** — local, cloud, or both (see below). A text-only
   model will not work; the bot needs to actually see the images.
3. **Inline mode enabled**, if you want to use the bot in other chats:
   BotFather → `/setinline` → pick the bot → give it a placeholder like
   `describe a sticker`.

## Choosing a vision model

Two slots, either of which can be left empty: a **local** one (`VLM_*`) and a
**cloud** one (`CLOUD_*`). Set a provider and the base URL and model default
sensibly; override them if you need to.

| `VLM_PROVIDER` | endpoint | notes |
|---|---|---|
| `llamacpp` | `host.docker.internal:8080/v1` | start with `--host 0.0.0.0` |
| `ollama` | `host.docker.internal:11434/v1` | |
| `vllm` | `host.docker.internal:8000/v1` | |
| `custom` | set `VLM_BASE_URL` | anything OpenAI-compatible |
| `none` | — | cloud only |

| `CLOUD_PROVIDER` | default model | protocol |
|---|---|---|
| `openai` | `gpt-4o-mini` | OpenAI |
| `anthropic` | `claude-sonnet-5` | Anthropic Messages |
| `google` | `gemini-2.5-flash` | Gemini |
| `openrouter` | `google/gemini-2.5-flash` | OpenAI |
| `groq`, `mistral`, `xai` | see `bot/config.py` | OpenAI |

Model names change often — check the provider's docs and set `CLOUD_MODEL`
explicitly rather than trusting the default.

Local example:

```bash
llama-server -hf ggml-org/Qwen2.5-VL-7B-Instruct-GGUF --host 0.0.0.0 --port 8080
```

Cloud example (`.env`):

```ini
VLM_PROVIDER=none
CLOUD_PROVIDER=anthropic
CLOUD_API_KEY=sk-ant-...
CLOUD_MODEL=claude-sonnet-5
```

### `CAPTION_BACKEND`

- `local` — local endpoint only
- `cloud` — cloud API only
- `auto` — local first, cloud when the local one fails (the default when both
  are configured)

`auto` is the useful one if the machine running your model isn't always on:
when it's unreachable the bot switches to the cloud and puts the local endpoint
on a 5-minute cooldown, so one sleeping box doesn't make every sticker wait for
a timeout. `/backend` shows the current state and reachability, `/backend cloud`
switches on the fly — handy for indexing one important pack with a stronger
model while everything else runs locally. Both need `INDEX_USER_IDS`: reading
the setting exposes which providers you use, and changing it redirects what the
next `/index` spends.

Cloud calls are per image: a 120-sticker pack is 120 vision requests. Cheap on a
flash-tier model, less so on a frontier one — see *Who can do what* for the
limits that keep this bounded. `/packs` shows which model wrote
which captions, and rate limits and transient errors are retried with backoff.

## Run it

```bash
cp .env.example .env
# edit .env: TELEGRAM_BOT_TOKEN, ALLOWED_USER_IDS, VLM_MODEL
docker compose up -d --build
docker compose logs -f
```

The default image is slim: no torch, ~400 MB, comfortable in 300 MB of RAM.

## Embeddings

Search is a hybrid of vector similarity and SQLite FTS5 keyword matching over
the captions. Where the vectors come from is up to you:

| `EMBED_PROVIDER` | cost | notes |
|---|---|---|
| `api` | an API call per caption | any OpenAI-compatible `/embeddings`: OpenAI, Mistral, or llama.cpp / Ollama on your own box |
| `local` | ~700 MB RAM | sentence-transformers in-process, fully offline, best multilingual quality |
| `none` | nothing | keyword-only; works, but "sad cat" won't find "crying kitten" |
| `auto` | — | local if built in, else api, else none |

`local` needs the model bundled at build time:

```bash
WITH_LOCAL_EMBEDDINGS=1 docker compose build
```

That step installs torch and needs roughly **2.5 GB of RAM and 4 GB of disk**.
On a small VPS it will trip the OOM killer and can take the host down with it —
use `EMBED_PROVIDER=api` there, or build the image on a bigger machine and push
it to a registry. `mem_limit` in `docker-compose.yml` (1 GB by default) keeps a
runaway container from doing the same at runtime.

Embeddings and captions are independent: local model for captions with a cloud
embedding API is a perfectly normal combination, and vice versa.

Each vector records which embedding model wrote it. Changing `EMBED_PROVIDER` or
`EMBED_MODEL` puts new vectors in a different space — usually a different width
too — so the old ones stop being comparable and are left out of search rather
than mixed in. Nothing breaks and nothing is deleted: `/packs` reports how many
vectors are being skipped, and `/reindex` on those packs re-embeds them with the
new model. Keyword search covers everything the whole time. Vectors written
before the bot recorded this are kept if their width matches, so upgrading does
not blind an existing index.

Your Telegram user ID: message [@userinfobot](https://t.me/userinfobot).

## What leaves your machine

The database lookup itself is always local — SQLite FTS5 plus an in-process
vector comparison, no network, no external service. What the two model slots do
is up to how you set them:

| | stays on your machine | goes to a provider |
|---|---|---|
| Captioning (`/index`) | `CAPTION_BACKEND=local` | any `CLOUD_PROVIDER` use — one image per sticker |
| Search queries | `EMBED_PROVIDER=local` or `none` | `EMBED_PROVIDER=api` — the query text, on every search |
| Sending the sticker back | — | Telegram, always |

The row that surprises people is the middle one. Captioning is finished once a
pack is indexed and never runs again, but searching still has to embed the
query, so with `EMBED_PROVIDER=api` your search text is sent to the embedding
provider every time. Nothing is cached, and inline queries are sent as you type,
so a single inline search is several requests.

A query has to be embedded in the same vector space as the captions it is
compared against, which is why "cloud embeddings for indexing, local for
searching" is not a combination that exists. For search that touches nothing but
your own hardware, use `EMBED_PROVIDER=local` (or `none`) together with
`CAPTION_BACKEND=local`.

`CAPTION_BACKEND` is the startup default, not a lock: `/backend cloud` switches
it at runtime (for users in `INDEX_USER_IDS`), and `auto` falls back to the cloud
whenever the local endpoint is unreachable. Leave `CLOUD_PROVIDER` empty if captions must never go anywhere —
with no cloud backend configured there is nothing to fall back to or switch to.

Telegram sees the traffic in every configuration — the bot polls it for updates
and sends stickers back by `file_id` — as with any Telegram bot.

## Who can do what

Two separate lists, because searching is free and captioning is not:

| | env | empty means |
|---|---|---|
| Use the bot (search, inline) | `ALLOWED_USER_IDS` | everyone |
| Spend anything (`/index`, the button, `/reindex`, `/forget`, `/backend`) | `INDEX_USER_IDS` | **nobody** |

`/backend` is on that second list in both directions: it reports which providers
are configured and it chooses which one the next `/index` spends on. Someone who
may only search gets no answer from it at all, rather than a refusal.

`INDEX_USER_IDS` unset inherits `ALLOWED_USER_IDS`. Set to an empty value and
the bot becomes read-only: people can search what's already there but cannot
make it spend anything. That is also the default for a bot with no
`ALLOWED_USER_IDS` at all — an open bot can never be made to run up a bill by
someone forwarding it stickers.

Three more limits sit behind that list, so even an authorised user can't
accidentally burn a month of credits:

- `MAX_PACK_SIZE` (default 200) — most stickers one `/index` run may caption.
  A larger pack is captioned up to the cap; run `/index` again to continue.
- `DAILY_CAPTION_LIMIT` (default 0 = off) — captions per user per UTC day,
  counted in the database so a restart doesn't reset it. `/quota` shows where
  you stand.
- Only one pack is captioned at a time. Queued requests wait rather than
  fanning out into hundreds of parallel API calls.

Non-authorised users who send a sticker just get told whether that pack is
indexed; they never see the index button, and pressing someone else's button in
a group is rejected too.

## Using it

| Action | How |
|---|---|
| Index a pack | Send the bot any sticker from it, tap **Index this pack** |
| Index by link | `/index https://t.me/addstickers/PackName` |
| Search | Just write: `guy shrugging with a confused face` |
| Search anywhere | Type `@yourbot crying cat` in any chat |
| List packs | `/packs` |
| Re-caption | `/reindex PackName` (after changing model or language) |
| Remove | `/forget PackName` |
| Check / switch model | `/backend`, `/backend local\|cloud\|auto` (indexers only) |
| Check your budget | `/quota` |

Indexing a 50-sticker pack takes a few minutes and is the only slow part.
Progress is shown in the chat; already-captioned stickers are skipped on re-runs.

## Notes on sticker formats

- **Static** `.webp` — decoded directly, transparency flattened onto white.
- **Video** `.webm` — ffmpeg pulls a frame at 0.4 s.
- **Animated** `.tgs` — Lottie rendering needs `rlottie`, which is awkward to
  build, so the bot captions Telegram's own thumbnail instead. Lower resolution,
  but good enough for a description.

## Tuning

- `CAPTION_LANGUAGE=German` writes descriptions in German. The embedding model
  is multilingual, so you can search in German even with English captions —
  matching just gets a bit sharper if both sides use the same language.
- `CAPTION_CONCURRENCY` is how many images are sent to the model at once. With a
  single local GPU, 1–2 is right; for a cloud API 6–10 is fine (it defaults to 6
  in cloud-only mode). Too high and you'll just collect 429s.
- Search quality lives almost entirely in the captions. If results feel off,
  edit `PROMPT` in `bot/captioner.py` and `/reindex` a pack to compare.

## Data

Everything is in `./data/stickers.db` (SQLite, mounted into the container) —
captions, embeddings and the per-day caption counters.
Back it up and you keep your index; delete it and re-index from scratch.
Sticker `file_id`s are bot-specific — if you switch to a new bot token, re-index.

## Layout

```
bot/config.py      env config
bot/store.py       SQLite, FTS5, vectors, hybrid search
bot/render.py      webp/webm → PNG
bot/captioner.py   OpenAI / Anthropic / Google backends + fallback router
bot/embedder.py    sentence-transformers
bot/indexer.py     pack indexing pipeline
bot/main.py        Telegram handlers
```
