"""Local multilingual sentence embeddings for semantic search."""
from __future__ import annotations

import asyncio
import logging

import numpy as np

log = logging.getLogger(__name__)


class Embedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        log.info("loading embedding model %s", model_name)
        self._model = SentenceTransformer(model_name)

    def _encode(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vecs.astype("float32")

    async def encode(self, texts: list[str]) -> np.ndarray:
        return await asyncio.to_thread(self._encode, texts)

    async def encode_one(self, text: str) -> np.ndarray:
        return (await self.encode([text]))[0]
