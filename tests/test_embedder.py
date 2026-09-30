"""Unit tests for embedding providers."""

import pytest
from src.indexing.embedder import (
    FastEmbedEmbeddingsWrapper,
    MockEmbeddings,
    get_embeddings,
)


def test_mock_embeddings():
    embedder = MockEmbeddings(dimension=32)
    doc_vecs = embedder.embed_documents(["First document", "Second document"])
    assert len(doc_vecs) == 2
    assert len(doc_vecs[0]) == 32

    q_vec = embedder.embed_query("Search query")
    assert len(q_vec) == 32


def test_fastembed_embeddings():
    embedder = FastEmbedEmbeddingsWrapper()
    q_vec = embedder.embed_query("FastAPI routing")
    assert len(q_vec) == 384
