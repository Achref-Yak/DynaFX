"""Embedding model — sentence-transformers integration.

Provides a singleton EmbeddingModel that converts text to dense vectors
for semantic similarity, concept matching, and clustering.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_instance: Optional["EmbeddingModel"] = None


class EmbeddingModel:
    """Wraps sentence-transformers for text embedding.

    Usage:
        model = EmbeddingModel.get_instance()
        vector = model.encode("some text")
        sim = model.similarity("text A", "text B")
    """

    def __init__(self, model_name: str = _MODEL_NAME) -> None:
        import os
        if not os.environ.get("HF_TOKEN"):
            logger.warning(
                "HF_TOKEN not set — HuggingFace rate limits apply. "
                "Set HF_TOKEN=<your_token> in .env or export it."
            )
        from sentence_transformers import SentenceTransformer
        self._model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._dimension = self._model.get_embedding_dimension()
        logger.info("Loaded embedding model %s (dim=%d)", model_name, self._dimension)

    @classmethod
    def get_instance(cls) -> "EmbeddingModel":
        global _instance
        if _instance is None:
            _instance = cls()
        return _instance

    @classmethod
    def reset(cls) -> None:
        global _instance
        _instance = None

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, text: str) -> list[float]:
        """Encode a single text into a dense vector."""
        vec = self._model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """Encode multiple texts into dense vectors."""
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return vecs.tolist()

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def cosine_distance(a: list[float], b: list[float]) -> float:
        """Compute cosine distance (1 - similarity) between two vectors."""
        return 1.0 - EmbeddingModel.cosine_similarity(a, b)

    def similarity(self, text_a: str, text_b: str) -> float:
        """Compute cosine similarity between two texts."""
        vec_a = self.encode(text_a)
        vec_b = self.encode(text_b)
        return self.cosine_similarity(vec_a, vec_b)
