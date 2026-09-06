"""Download a sticker set, caption every sticker, store captions + embeddings."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable

from telegram import Bot, Sticker

from .captioner import CaptionRouter
from .embedder import Embedder  # noqa: F401
from .render import to_png
from .store import Store

log = logging.getLogger(__name__)

Progress = Callable[[int, int, int], Awaitable[None]]


@dataclass
class IndexResult:
    captioned: int = 0
    failed: int = 0
    skipped: int = 0      # already had a caption
    deferred: int = 0     # left out because the budget ran out

    @property
    def attempted(self) -> int:
        return self.captioned + self.failed


def kind_of(s: Sticker) -> str:
    if s.is_video:
        return "video"
    if s.is_animated:
        return "animated"
    return "static"


class Indexer:
    def __init__(self, store: Store, captioner: CaptionRouter, embedder: Embedder,
                 concurrency: int = 2):
        self.store = store
        self.captioner = captioner
        self.embedder = embedder
        self._sem = asyncio.Semaphore(concurrency)
        self._running: set[str] = set()
        # only one pack at a time, so a queue of packs cannot fan out into
        # hundreds of parallel API calls
        self._job_lock = asyncio.Lock()

    def is_running(self, set_name: str) -> bool:
        return set_name in self._running

    @property
    def busy(self) -> bool:
        return self._job_lock.locked()

    @property
    def current(self) -> str | None:
        return next(iter(self._running), None)

    async def _source_file_id(self, s: Sticker) -> str:
        """Animated .tgs cannot be rendered here, so use Telegram's thumbnail."""
        if kind_of(s) == "animated" and s.thumbnail:
            return s.thumbnail.file_id
        return s.file_id

    async def _one(self, bot: Bot, s: Sticker, set_name: str) -> bool:
        async with self._sem:
            try:
                k = kind_of(s)
                file_id = await self._source_file_id(s)
                tg_file = await bot.get_file(file_id)
                raw = bytes(await tg_file.download_as_bytearray())
                png = await to_png(raw, "video" if k == "video" else "image")
                result = await self.captioner.describe(png)
                caption, model = result.text, result.backend
            except Exception as exc:  # noqa: BLE001
                log.warning("caption failed for %s: %s", s.file_unique_id, exc)
                caption, model = None, None

            text = " ".join(filter(None, [caption, s.emoji]))
            vec = await self.embedder.encode_one(text) if text.strip() else None
            await self.store.upsert_sticker(
                {
                    "uid": s.file_unique_id,
                    "file_id": s.file_id,
                    "set_name": set_name,
                    "emoji": s.emoji,
                    "kind": kind_of(s),
                    "caption": caption,
                    "model": model,
                    "embed_model": self.embedder.label,
                },
                vec,
            )
            return caption is not None

    async def index_set(self, bot: Bot, set_name: str, force: bool = False,
                        progress: Progress | None = None,
                        budget: int | None = None) -> IndexResult:
        """Caption a pack. `budget` caps how many stickers may be captioned."""
        async with self._job_lock:
            self._running.add(set_name)
            try:
                sset = await bot.get_sticker_set(set_name)
                await self.store.upsert_set(sset.name, sset.title)
                known = set() if force else await self.store.known_uids(set_name)
                todo = [s for s in sset.stickers if s.file_unique_id not in known]

                res = IndexResult(skipped=len(sset.stickers) - len(todo))
                if budget is not None and len(todo) > max(budget, 0):
                    res.deferred = len(todo) - max(budget, 0)
                    todo = todo[: max(budget, 0)]
                if not todo:
                    return res

                done = 0
                tasks = [asyncio.create_task(self._one(bot, s, set_name))
                         for s in todo]
                for fut in asyncio.as_completed(tasks):
                    if await fut:
                        res.captioned += 1
                    else:
                        res.failed += 1
                    done += 1
                    if progress and (done % 5 == 0 or done == len(todo)):
                        await progress(done, len(todo), res.captioned)
                return res
            finally:
                self._running.discard(set_name)
