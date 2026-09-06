"""Telegram sticker search bot.

Send the bot a sticker to index its whole pack, then describe a sticker in
plain language to get it back. Works inline in any chat: @yourbot crying cat
"""
from __future__ import annotations

import asyncio
import html
import logging
import re
from uuid import uuid4

from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InlineQueryResultCachedSticker,
    Update,
)
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    InlineQueryHandler,
    MessageHandler,
    filters,
)

from .captioner import CaptionRouter, build_backend
from .config import Config
from .embedder import build_embedder
from .indexer import Indexer
from .store import Store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("stickerbot")

# HTML, not Markdown: Telegram's legacy Markdown has no backslash escaping, so
# an underscore in an interpolated value (a bot username, a pack title) silently
# opens an italic entity that never closes and the whole send fails.
HELP = (
    "<b>Sticker search</b>\n\n"
    "• Send me any sticker → I offer to index its whole pack.\n"
    "• <code>/index &lt;pack link or name&gt;</code> → index a pack directly.\n"
    "• Just write what you remember (\"cat crying in the rain\") → I send matches.\n"
    "• Type <code>@{me} crying cat</code> in <b>any</b> chat to insert a sticker "
    "inline.\n\n"
    "<code>/packs</code> indexed packs · <code>/forget &lt;name&gt;</code> remove a "
    "pack · <code>/reindex &lt;name&gt;</code> re-caption a pack\n"
    "<code>/backend</code> show the vision model · "
    "<code>/backend local|cloud|auto</code> switch it\n"
    "<code>/quota</code> your captioning budget\n\n"
    "<i>/index, /reindex, /forget and /backend are limited to the user IDs in "
    "INDEX_USER_IDS; everyone allowed can search.</i>"
)


def pack_name(text: str) -> str | None:
    text = text.strip()
    m = re.search(r"(?:t\.me|telegram\.me)/addstickers/([A-Za-z0-9_]+)", text)
    if m:
        return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9_]+", text):
        return text
    return None


class BotApp:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.store = Store(cfg.db_path, cfg.search_min_score,
                           cfg.search_relative_cutoff)
        self.captioner = CaptionRouter(
            build_backend(cfg.local, cfg.caption_language) if cfg.local else None,
            build_backend(cfg.cloud, cfg.caption_language) if cfg.cloud else None,
            cfg.backend_mode,
        )
        self.embedder = build_embedder(cfg.embed_provider, cfg.embed_model,
                                       cfg.embed_base_url, cfg.embed_api_key)
        self.indexer = Indexer(self.store, self.captioner, self.embedder,
                               cfg.caption_concurrency)

    # ---------- guards ----------

    def allowed(self, update: Update) -> bool:
        user = update.effective_user
        return bool(user) and self.cfg.is_allowed(user.id)

    def may_index(self, update: Update) -> bool:
        user = update.effective_user
        return bool(user) and self.cfg.may_index(user.id)

    async def deny_index(self, update: Update) -> None:
        text = ("Indexing is restricted — it spends GPU time or API credits. "
                "You can still search everything that is already indexed.")
        if update.callback_query:
            await update.callback_query.answer(text, show_alert=True)
        elif update.message:
            await update.message.reply_text(text)

    async def budget_for(self, user_id: int) -> tuple[int, int, int]:
        """(budget, used_today, limit) — how many captions this user may spend."""
        limit = self.cfg.daily_caption_limit
        used = await self.store.usage_today(user_id) if limit else 0
        remaining = max(limit - used, 0) if limit else self.cfg.max_pack_size
        return min(remaining, self.cfg.max_pack_size), used, limit

    # ---------- commands ----------

    async def start(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message:
            return
        if not self.allowed(update):
            await update.message.reply_text("Not authorised.")
            return
        me = (await ctx.bot.get_me()).username or ""
        await update.message.reply_html(HELP.format(me=html.escape(me)))

    async def packs(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self.allowed(update):
            return
        s = await self.store.stats()
        if not s["list"]:
            await update.message.reply_text("Nothing indexed yet. Send me a sticker.")
            return
        lines = [f"{html.escape(title)} — {n} stickers "
                 f"(<code>{html.escape(name)}</code>)"
                 for title, name, n in s["list"]]
        by_model = "\n".join(f"<code>{html.escape(m)}</code> — {n}"
                             for m, n in s.get("models", []))
        # vectors from another embedding model are not comparable with the
        # current one, so they are skipped when searching: don't count them
        # as searchable
        stale = 0
        if self.embedder.label != "none":
            stale = sum(n for m, n in s.get("embedders", [])
                        if m not in ("unknown", self.embedder.label))
        text = (f"<b>{s['sets']} packs, {s['stickers']} stickers, "
                f"{s['vectors'] - stale} searchable</b>\n\n" + "\n".join(lines))
        if by_model:
            text += "\n\ncaptioned by:\n" + by_model
        if self.embedder.label == "none":
            text += ("\n\n<i>No embedding model configured — search is "
                     "keyword-only.</i>")
        elif stale:
            was = "vector was" if stale == 1 else "vectors were"
            text += (f"\n\n⚠️ {stale} {was} written by a different embedding "
                     f"model and {'is' if stale == 1 else 'are'} skipped when "
                     f"searching. Re-run <code>/reindex</code> on those packs "
                     f"to restore them.")
        await update.message.reply_html(text)

    async def backend_cmd(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        # /backend reports which providers are configured and switches which one
        # spends GPU time or API credits, so it belongs to the indexing list
        # rather than the search list. Users who may only search get no reply at
        # all — not a refusal, which would just advertise that the command
        # exists.
        if not (self.allowed(update) and self.may_index(update)):
            return
        if ctx.args:
            try:
                self.captioner.set_mode(ctx.args[0].lower())
            except ValueError as exc:
                await update.message.reply_text(
                    f"Cannot switch: {exc}. Use local, cloud or auto.")
                return
        await update.message.reply_html(await self.captioner.status())

    async def quota_cmd(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self.allowed(update):
            return
        user = update.effective_user
        if not self.cfg.may_index(user.id):
            await update.message.reply_text(
                "You have search access. Indexing is restricted to the "
                "user IDs in INDEX_USER_IDS.")
            return
        budget, used, limit = await self.budget_for(user.id)
        cap = f"{used}/{limit} captions used today" if limit else \
            f"{used} captions today, no daily limit"
        busy = (f"\nbusy with <code>{html.escape(self.indexer.current or '')}</code>"
                if self.indexer.busy else "")
        await update.message.reply_html(
            f"{cap}\nmax {self.cfg.max_pack_size} per pack, "
            f"{budget} available right now{busy}")

    async def index_cmd(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self.allowed(update):
            return
        if not self.may_index(update):
            await self.deny_index(update)
            return
        arg = " ".join(ctx.args) if ctx.args else ""
        name = pack_name(arg)
        if not name:
            await update.message.reply_text(
                "Usage: /index https://t.me/addstickers/PackName")
            return
        await self.run_index(update, ctx, name, force=False)

    async def reindex_cmd(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self.allowed(update):
            return
        if not self.may_index(update):
            await self.deny_index(update)
            return
        name = pack_name(" ".join(ctx.args) if ctx.args else "")
        if not name:
            await update.message.reply_text("Usage: /reindex PackName")
            return
        await self.run_index(update, ctx, name, force=True)

    async def forget_cmd(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self.allowed(update):
            return
        if not self.may_index(update):
            await self.deny_index(update)
            return
        name = pack_name(" ".join(ctx.args) if ctx.args else "")
        if not name:
            await update.message.reply_text("Usage: /forget PackName")
            return
        n = await self.store.forget_set(name)
        await update.message.reply_text(f"Removed {n} stickers from {name}.")

    # ---------- indexing ----------

    async def run_index(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE,
                        name: str, force: bool) -> None:
        chat = update.effective_chat
        user = update.effective_user
        if not self.cfg.may_index(user.id):        # re-checked at the last gate
            await self.deny_index(update)
            return
        if self.indexer.is_running(name):
            await ctx.bot.send_message(chat.id, f"{name} is already being indexed.")
            return

        budget, used, limit = await self.budget_for(user.id)
        if budget <= 0:
            await ctx.bot.send_message(
                chat.id,
                f"Daily captioning limit reached ({used}/{limit}). "
                "It resets at midnight UTC.")
            return

        queued = self.indexer.busy
        note = " (queued behind another pack)" if queued else ""
        safe = html.escape(name)
        msg = await ctx.bot.send_message(
            chat.id,
            f"Indexing <code>{safe}</code> via <b>{self.captioner.mode}</b> "
            f"backend{note} …",
            parse_mode="HTML")

        last = [0.0]

        async def progress(done: int, total: int, ok: int) -> None:
            now = asyncio.get_running_loop().time()
            if now - last[0] < 3 and done != total:
                return
            last[0] = now
            try:
                await msg.edit_text(f"Indexing <code>{safe}</code> … {done}/{total}",
                                    parse_mode="HTML")
            except Exception:  # noqa: BLE001 - ignore "message not modified"
                pass

        res = None
        try:
            res = await self.indexer.index_set(
                ctx.bot, name, force=force, progress=progress, budget=budget)
        except Exception as exc:  # noqa: BLE001
            log.exception("indexing %s failed", name)
            await msg.edit_text(
                f"Could not index <code>{safe}</code>: {html.escape(str(exc))}",
                parse_mode="HTML")
            return
        finally:
            if res and res.attempted:
                await self.store.add_usage(user.id, res.attempted)

        parts = [f"✅ <code>{safe}</code>: {res.captioned} described"]
        if res.failed:
            parts.append(f"{res.failed} failed")
        if res.skipped:
            parts.append(f"{res.skipped} already known")
        text = ", ".join(parts) + "."
        if res.deferred:
            text += (f"\n\n{res.deferred} stickers left out — the pack is bigger "
                     f"than your remaining budget of {budget}. Run /index again "
                     "to continue.")
        if limit:
            spent = await self.store.usage_today(user.id)
            text += f"\n\nUsed today: {spent}/{limit} captions."
        await msg.edit_text(text, parse_mode="HTML")

    async def on_sticker(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self.allowed(update):
            return
        sticker = update.message.sticker
        if not sticker.set_name:
            await update.message.reply_text("That sticker isn't part of a pack.")
            return
        if not self.may_index(update):
            known = await self.store.known_uids(sticker.set_name)
            safe = html.escape(sticker.set_name)
            await update.message.reply_text(
                f"<code>{safe}</code> — {len(known)} stickers indexed."
                if known else
                f"<code>{safe}</code> isn't indexed, and indexing is restricted.",
                parse_mode="HTML")
            return
        kb = InlineKeyboardMarkup([[
            InlineKeyboardButton("Index this pack",
                                 callback_data=f"idx:{sticker.set_name}"),
            InlineKeyboardButton("Re-index",
                                 callback_data=f"rdx:{sticker.set_name}"),
        ]])
        await update.message.reply_text(
            f"Pack: <code>{html.escape(sticker.set_name)}</code>",
            parse_mode="HTML", reply_markup=kb)

    async def on_button(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        q = update.callback_query
        if not self.allowed(update):
            await q.answer()
            return
        if not self.may_index(update):
            # someone else in a group must not be able to press the button
            await self.deny_index(update)
            return
        await q.answer()
        action, _, name = q.data.partition(":")
        await self.run_index(update, ctx, name, force=(action == "rdx"))

    # ---------- search ----------

    async def on_text(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self.allowed(update):
            return
        query = update.message.text.strip()
        if pack_name(query) and query.startswith("http"):
            await self.run_index(update, ctx, pack_name(query), force=False)
            return

        await ctx.bot.send_chat_action(update.effective_chat.id, ChatAction.TYPING)
        vec = await self.embedder.encode_one(query)
        hits = await self.store.search(vec, query, self.cfg.max_results,
                                       self.embedder.label)
        if not hits:
            await update.message.reply_text(
                "Nothing matched. Index a few packs first, or try other words.")
            return
        for hit in hits:
            await update.message.reply_sticker(hit.file_id)

    async def on_inline(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        iq = update.inline_query
        if not self.cfg.is_allowed(iq.from_user.id):
            await iq.answer([], cache_time=5, is_personal=True)
            return
        query = iq.query.strip()
        if not query:
            await iq.answer([], cache_time=5, is_personal=True)
            return
        vec = await self.embedder.encode_one(query)
        hits = await self.store.search(vec, query, self.cfg.inline_results,
                                       self.embedder.label)
        results = [
            InlineQueryResultCachedSticker(id=str(uuid4()), sticker_file_id=h.file_id)
            for h in hits
        ]
        await iq.answer(results, cache_time=30, is_personal=True)

    # ---------- lifecycle ----------

    async def post_init(self, app: Application) -> None:
        me = await app.bot.get_me()
        log.info("embeddings: %s", self.embedder.label)
        log.info("running as @%s | backends: %s | mode: %s", me.username,
                 ", ".join(self.captioner.available) or "none",
                 self.captioner.mode)
        for line in (await self.captioner.status()).splitlines():
            log.info("  %s", re.sub(r"<[^>]+>", "", line))
        if not self.cfg.index_user_ids:
            log.warning("INDEX_USER_IDS is empty — nobody can trigger captioning. "
                        "Set it to your Telegram user ID.")
        else:
            log.info("indexing allowed for: %s | max %d per pack | daily limit %s",
                     sorted(self.cfg.index_user_ids), self.cfg.max_pack_size,
                     self.cfg.daily_caption_limit or "none")

    async def post_shutdown(self, app: Application) -> None:
        await self.captioner.close()
        await self.embedder.close()


async def on_error(update: object, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    log.error("unhandled error while processing an update", exc_info=ctx.error)


def main() -> None:
    cfg = Config.from_env()
    bot = BotApp(cfg)

    app = (
        Application.builder()
        .token(cfg.bot_token)
        .post_init(bot.post_init)
        .post_shutdown(bot.post_shutdown)
        .build()
    )
    app.add_handler(CommandHandler(["start", "help"], bot.start))
    app.add_handler(CommandHandler("index", bot.index_cmd))
    app.add_handler(CommandHandler("reindex", bot.reindex_cmd))
    app.add_handler(CommandHandler("forget", bot.forget_cmd))
    app.add_handler(CommandHandler(["packs", "stats"], bot.packs))
    app.add_handler(CommandHandler(["backend", "model"], bot.backend_cmd))
    app.add_handler(CommandHandler(["quota", "budget"], bot.quota_cmd))
    app.add_handler(MessageHandler(filters.Sticker.ALL, bot.on_sticker))
    app.add_handler(CallbackQueryHandler(bot.on_button))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.on_text))
    app.add_handler(InlineQueryHandler(bot.on_inline))
    app.add_error_handler(on_error)

    log.info("starting polling")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
