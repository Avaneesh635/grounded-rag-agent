"""Embeddings provider factory supporting FastEmbed (local ONNX), OpenAI, and Mock embeddings."""

import hashlib
import logging
from typing import List, Optional
import numpy as np
from langchain_core.embeddings import Embeddings

from src.config import settings

logger = logging.getLogger(__name__)


class FastEmbedEmbeddingsWrapper(Embeddings):
    """Local ONNX-accelerated embeddings using FastEmbed (zero API cost, fast on CPU)."""

    def __init__(self, model_name: str = settings.FASTEMBED_MODEL):
        from fastembed import TextEmbedding

        self.model_name = model_name
        self.client = TextEmbedding(model_name=self.model_name)
        self.total_tokens_embedded = 0

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of document strings."""
        if not texts:
            return []
        embeddings = list(self.client.embed(texts))
        return [e.tolist() for e in embeddings]

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        embedding = list(self.client.embed([text]))[0]
        return embedding.tolist()


class MockEmbeddings(Embeddings):
    """Deterministic hash-based embeddings for offline unit testing without external models."""

    def __init__(self, dimension: int = 64):
        self.dimension = dimension

    def _hash_to_vector(self, text: str) -> List[float]:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        # Generate pseudo-random vector seeded by hash
        np.random.seed(int.from_bytes(h[:4], "big"))
        vec = np.random.randn(self.dimension)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._hash_to_vector(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._hash_to_vector(text)


def get_embeddings(provider: Optional[str] = None) -> Embeddings:
    """Factory function returning the appropriate Embeddings instance."""
    prov = (provider or settings.EMBEDDING_PROVIDER).lower()

    if prov == "openai":
        api_key = settings.OPENAI_API_KEY
        if not api_key:
            logger.warning(
                "OPENAI_API_KEY is not set. Falling back to FastEmbed local embeddings."
            )
            return FastEmbedEmbeddingsWrapper()
        try:
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(
                model=settings.OPENAI_EMBEDDING_MODEL, api_key=api_key
            )
        except Exception as e:
            logger.error(
                "Failed to initialize OpenAI embeddings: %s. Falling back to FastEmbed.",
                e,
            )
            return FastEmbedEmbeddingsWrapper()

    elif prov == "mock":
        return MockEmbeddings()

    else:
        # Default: fastembed
        return FastEmbedEmbeddingsWrapper()
