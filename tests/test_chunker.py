"""Unit tests for text chunker and token counter."""

import pytest
from src.crawler.cleaner import CleanedPage
from src.indexing.chunker import TextChunker


def test_chunker_basic():
    chunker = TextChunker(chunk_size=200, chunk_overlap=30)
    page = CleanedPage(
        url="https://fastapi.tiangolo.com/tutorial/first-steps/",
        title="First Steps",
        description="Getting started",
        text="FastAPI is a modern web framework. " * 30,
        char_count=1000,
        word_count=150,
    )

    docs, total_tokens = chunker.chunk_pages([page])
    assert len(docs) > 1
    assert total_tokens > 0

    first_doc = docs[0]
    assert first_doc.metadata["source"] == page.url
    assert first_doc.metadata["title"] == page.title
    assert "chunk_id" in first_doc.metadata
    assert first_doc.metadata["chunk_index"] == 0
    assert first_doc.metadata["token_count"] > 0


def test_token_counter():
    chunker = TextChunker()
    tokens = chunker.count_tokens("Hello world, this is a test.")
    assert tokens >= 6
