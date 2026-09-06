"""SQLite storage: sticker metadata, VLM captions, embeddings, FTS index."""
from __future__ import annotations

import asyncio
import datetime as dt
import logging
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# Bumped when the text put into stickers_fts changes, so the keyword index is
# rebuilt from the captions already stored rather than needing a re-caption.
FTS_VERSION = "3"

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS sticker_sets (
    name        TEXT PRIMARY KEY,
    title       TEXT,
    indexed_at  REAL
);

CREATE TABLE IF NOT EXISTS stickers (
    uid         TEXT PRIMARY KEY,           -- file_unique_id
    file_id     TEXT NOT NULL,
    set_name    TEXT,
    emoji       TEXT,
    kind        TEXT,                       -- static | video | animated
    caption     TEXT,
    model       TEXT,                       -- backend that wrote the caption
    embedding   BLOB,
    embed_model TEXT,                       -- embedder that wrote the vector
    indexed_at  REAL
);

CREATE INDEX IF NOT EXISTS idx_stickers_set ON stickers(set_name);

CREATE VIRTUAL TABLE IF NOT EXISTS stickers_fts
    USING fts5(uid UNINDEXED, text, tokenize='porter unicode61');

CREATE TABLE IF NOT EXISTS meta (
    key    TEXT PRIMARY KEY,
    value  TEXT
);

-- how many captions each user has spent, per UTC day
CREATE TABLE IF NOT EXISTS usage (
    user_id   INTEGER NOT NULL,
    day       TEXT NOT NULL,
    captions  INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (user_id, day)
);
"""


@dataclass
class StickerRow:
    uid: str
    file_id: str
    set_name: str | None
    emoji: str | None
    caption: str | None


def fts_terms(text: str) -> list[str]:
    """The words of a query, quoted so FTS5 reads them as literals."""
    return [f'"{t}"' for t in re.findall(r"\w+", text, flags=re.UNICODE)
            if len(t) > 1]


def fts_query(text: str) -> str:
    """Match any term. Used as the fallback when no caption has them all."""
    return " OR ".join(fts_terms(text))


class Store:
    def __init__(self, path: str, min_score: float = 0.20,
                 rel_cutoff: float = 0.60):
        self.min_score = min_score
        self.rel_cutoff = rel_cutoff
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(SCHEMA)
        self._migrate()
        self._db.commit()
        self._lock = asyncio.Lock()
        self._matrix: np.ndarray | None = None
        self._matrix_uids: list[str] = []
        self._matrix_key: tuple[str, int] | None = None

    def _migrate(self) -> None:
        cols = {r[1] for r in self._db.execute("PRAGMA table_info(stickers)")}
        if "model" not in cols:
            self._db.execute("ALTER TABLE stickers ADD COLUMN model TEXT")
        if "embed_model" not in cols:
            self._db.execute("ALTER TABLE stickers ADD COLUMN embed_model TEXT")

        row = self._db.execute(
            "SELECT value FROM meta WHERE key = 'fts_version'").fetchone()
        if (row["value"] if row else None) != FTS_VERSION:
            self._rebuild_fts()
            self._db.execute(
                """INSERT INTO meta (key, value) VALUES ('fts_version', ?)
                   ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
                (FTS_VERSION,),
            )

    def _fts_text(self, caption: str | None, emoji: str | None) -> str:
        """What a sticker is findable by.

        Deliberately not the pack name: it is the same for every sticker in the
        pack, so a query matching it matched the whole pack — and because the
        term then appears in every row, bm25's IDF collapses to zero and the
        ranking cannot tell the wolf from the pig beside it.
        """
        return " ".join(filter(None, [caption, emoji]))

    def _rebuild_fts(self) -> None:
        """Re-derive the keyword index from captions already in the database.

        Dropped and recreated rather than emptied: the tokenizer is fixed when
        the table is created, so changing it needs a new table. Captions are
        already stored, so this costs nothing but a moment at startup.
        """
        rows = self._db.execute(
            "SELECT uid, caption, emoji FROM stickers").fetchall()
        self._db.execute("DROP TABLE IF EXISTS stickers_fts")
        self._db.executescript(SCHEMA)
        self._db.executemany(
            "INSERT INTO stickers_fts (uid, text) VALUES (?, ?)",
            [(r["uid"], self._fts_text(r["caption"], r["emoji"])) for r in rows],
        )
        if rows:
            log.info("rebuilt keyword index for %d stickers", len(rows))

    # ---------- write ----------

    def _upsert(self, row: dict, vec: np.ndarray | None) -> None:
        blob = vec.astype("float32").tobytes() if vec is not None else None
        params = {"model": None, "embed_model": None, **row,
                  "emb": blob, "ts": time.time()}
        # no vector means no embedding model to record
        if blob is None:
            params["embed_model"] = None
        self._db.execute(
            """INSERT INTO stickers (uid, file_id, set_name, emoji, kind, caption,
                                     model, embedding, embed_model, indexed_at)
               VALUES (:uid, :file_id, :set_name, :emoji, :kind, :caption,
                       :model, :emb, :embed_model, :ts)
               ON CONFLICT(uid) DO UPDATE SET
                   file_id=excluded.file_id, set_name=excluded.set_name,
                   emoji=excluded.emoji, kind=excluded.kind,
                   caption=excluded.caption, model=excluded.model,
                   embedding=excluded.embedding,
                   embed_model=excluded.embed_model,
                   indexed_at=excluded.indexed_at""",
            params,
        )
        text = self._fts_text(row.get("caption"), row.get("emoji"))
        self._db.execute("DELETE FROM stickers_fts WHERE uid = ?", (row["uid"],))
        self._db.execute(
            "INSERT INTO stickers_fts (uid, text) VALUES (?, ?)", (row["uid"], text)
        )
        self._db.commit()

    async def upsert_sticker(self, row: dict, vec: np.ndarray | None) -> None:
        async with self._lock:
            await asyncio.to_thread(self._upsert, row, vec)
            self._matrix_key = None

    async def upsert_set(self, name: str, title: str) -> None:
        async with self._lock:
            await asyncio.to_thread(
                lambda: (
                    self._db.execute(
                        """INSERT INTO sticker_sets (name, title, indexed_at)
                           VALUES (?, ?, ?)
                           ON CONFLICT(name) DO UPDATE SET
                               title=excluded.title, indexed_at=excluded.indexed_at""",
                        (name, title, time.time()),
                    ),
                    self._db.commit(),
                )
            )

    async def forget_set(self, name: str) -> int:
        async with self._lock:
            def run() -> int:
                uids = [r["uid"] for r in self._db.execute(
                    "SELECT uid FROM stickers WHERE set_name = ?", (name,))]
                self._db.executemany(
                    "DELETE FROM stickers_fts WHERE uid = ?", [(u,) for u in uids])
                self._db.execute("DELETE FROM stickers WHERE set_name = ?", (name,))
                self._db.execute("DELETE FROM sticker_sets WHERE name = ?", (name,))
                self._db.commit()
                return len(uids)

            n = await asyncio.to_thread(run)
            self._matrix_key = None
            return n

    # ---------- usage quota ----------

    @staticmethod
    def _today() -> str:
        return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")

    async def usage_today(self, user_id: int) -> int:
        def run() -> int:
            row = self._db.execute(
                "SELECT captions FROM usage WHERE user_id = ? AND day = ?",
                (user_id, self._today()),
            ).fetchone()
            return row["captions"] if row else 0

        return await asyncio.to_thread(run)

    async def add_usage(self, user_id: int, n: int) -> int:
        if n <= 0:
            return await self.usage_today(user_id)

        def run() -> int:
            self._db.execute(
                """INSERT INTO usage (user_id, day, captions) VALUES (?, ?, ?)
                   ON CONFLICT(user_id, day) DO UPDATE SET
                       captions = captions + excluded.captions""",
                (user_id, self._today(), n),
            )
            self._db.commit()
            row = self._db.execute(
                "SELECT captions FROM usage WHERE user_id = ? AND day = ?",
                (user_id, self._today()),
            ).fetchone()
            return row["captions"]

        async with self._lock:
            return await asyncio.to_thread(run)

    # ---------- read ----------

    async def known_uids(self, set_name: str) -> set[str]:
        def run() -> set[str]:
            return {
                r["uid"] for r in self._db.execute(
                    "SELECT uid FROM stickers WHERE set_name = ? AND caption IS NOT NULL",
                    (set_name,),
                )
            }

        return await asyncio.to_thread(run)

    async def stats(self) -> dict:
        def run() -> dict:
            c = self._db.execute(
                "SELECT COUNT(*) n, COUNT(embedding) v FROM stickers").fetchone()
            s = self._db.execute("SELECT COUNT(*) n FROM sticker_sets").fetchone()
            sets = [
                (r["title"] or r["name"], r["name"], r["n"])
                for r in self._db.execute(
                    """SELECT s.name, s.title, COUNT(st.uid) n
                       FROM sticker_sets s LEFT JOIN stickers st ON st.set_name = s.name
                       GROUP BY s.name ORDER BY n DESC""")
            ]
            models = [
                (r["model"] or "unknown", r["n"])
                for r in self._db.execute(
                    """SELECT model, COUNT(*) n FROM stickers
                       WHERE caption IS NOT NULL GROUP BY model ORDER BY n DESC""")
            ]
            embedders = [
                (r["embed_model"] or "unknown", r["n"])
                for r in self._db.execute(
                    """SELECT embed_model, COUNT(*) n FROM stickers
                       WHERE embedding IS NOT NULL
                       GROUP BY embed_model ORDER BY n DESC""")
            ]
            return {"stickers": c["n"], "vectors": c["v"], "sets": s["n"],
                    "list": sets, "models": models, "embedders": embedders}

        return await asyncio.to_thread(run)

    def _load_matrix(self, embed_model: str, dim: int) -> tuple[np.ndarray, list[str]]:
        """Vectors written by one embedding model, at one width.

        Switching EMBED_PROVIDER or EMBED_MODEL changes the vector space, and
        usually the width too. Stacking widths together raises, which used to
        take out every search until the database was deleted; and even at equal
        width, comparing vectors from two models is meaningless. So the query
        model and width select the rows, and anything else is simply not
        searched until it is reindexed.

        Rows written before embed_model was recorded are matched on width alone,
        so upgrading an existing database does not blind the index.
        """
        key = (embed_model, dim)
        if self._matrix_key != key:
            rows = self._db.execute(
                """SELECT uid, embedding FROM stickers
                   WHERE embedding IS NOT NULL
                     AND length(embedding) = ?
                     AND (embed_model = ? OR embed_model IS NULL)""",
                (dim * 4, embed_model),      # float32: 4 bytes per component
            ).fetchall()
            if rows:
                self._matrix_uids = [r["uid"] for r in rows]
                self._matrix = np.vstack(
                    [np.frombuffer(r["embedding"], dtype="float32") for r in rows]
                )
            else:
                self._matrix_uids = []
                self._matrix = np.zeros((0, dim), dtype="float32")
            self._matrix_key = key
        return self._matrix, self._matrix_uids

    def _rows(self, uids: list[str]) -> dict[str, StickerRow]:
        if not uids:
            return {}
        q = ",".join("?" * len(uids))
        out = {}
        for r in self._db.execute(
            f"SELECT uid, file_id, set_name, emoji, caption FROM stickers WHERE uid IN ({q})",
            uids,
        ):
            out[r["uid"]] = StickerRow(r["uid"], r["file_id"], r["set_name"],
                                       r["emoji"], r["caption"])
        return out

    def _fts(self, match: str, limit: int) -> list:
        try:
            return self._db.execute(
                """SELECT uid, bm25(stickers_fts) AS score FROM stickers_fts
                   WHERE stickers_fts MATCH ?
                   ORDER BY score LIMIT ?""",
                (match, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []

    def _search(self, qvec: np.ndarray | None, text: str, limit: int,
                embed_model: str = "") -> list[StickerRow]:
        ranks: dict[str, float] = {}
        terms = fts_terms(text)

        # A caption carrying every term beats one carrying only some, so
        # "pig mountain" returns the pig on the mountain rather than every pig
        # in the pack. When no caption has them all, fall back to matching any
        # of them: the captions may just word it differently, and a partial
        # match beats nothing.
        rows: list = []
        only: set[str] | None = None
        if len(terms) > 1:
            rows = self._fts(" AND ".join(terms), limit * 5)
            if rows:
                only = {r["uid"] for r in rows}
        if not rows and terms:
            rows = self._fts(" OR ".join(terms), limit * 5)

        # dense / semantic
        if qvec is not None:
            qvec = np.asarray(qvec, dtype="float32")
            mat, uids = self._load_matrix(embed_model, int(qvec.shape[0]))
            if len(uids):
                sims = mat @ qvec
                order = np.argsort(-sims)[: limit * 5]
                # Nothing here used to be filtered, so a search always came
                # back with `limit` stickers however badly they matched — ask
                # for a wolf in a pack of three and you got the pig too. The
                # absolute floor rejects a query nothing matches; the relative
                # one drops the tail once something clearly does.
                floor = max(self.min_score,
                            float(sims[order[0]]) * self.rel_cutoff)
                for rank, i in enumerate(order):
                    if float(sims[i]) < floor:
                        break
                    # similarity ranks the all-term matches, it does not widen
                    # them: that is what kept the other pigs in the results
                    if only is not None and uids[i] not in only:
                        continue
                    ranks[uids[i]] = ranks.get(uids[i], 0.0) + 1.0 / (60 + rank)

        # sparse / keyword
        if rows:
            # bm25 is negative and lower is better. Same idea as above: keep the
            # rows that matched strongly and drop those that only caught a
            # common word. When every row scores 0 the term is in all of them
            # and they are equally right, so nothing is cut.
            cut = rows[0]["score"] * self.rel_cutoff
            for rank, r in enumerate(rows):
                if r["score"] > cut:
                    break
                ranks[r["uid"]] = ranks.get(r["uid"], 0.0) + 1.0 / (60 + rank)

        top = sorted(ranks.items(), key=lambda kv: -kv[1])[:limit]
        fetched = self._rows([uid for uid, _ in top])
        return [fetched[uid] for uid, _ in top if uid in fetched]

    async def search(self, qvec, text: str, limit: int,
                     embed_model: str = "") -> list[StickerRow]:
        return await asyncio.to_thread(
            self._search, qvec, text, limit, embed_model)
