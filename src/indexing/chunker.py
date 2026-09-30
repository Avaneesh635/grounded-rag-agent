"""Text chunking and metadata preservation for crawled documentation."""

import hashlib
from typing import List, Tuple
import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings
from src.crawler.cleaner import CleanedPage


class TextChunker:
    """Chunks cleaned pages using recursive character splitting while preserving document metadata."""

    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " ", ""],
            keep_separator=True,
        )
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.tokenizer = None

    def count_tokens(self, text: str) -> int:
        """Count tokens accurately using tiktoken, fallback to word estimate."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text, disallowed_special=()))
        return max(1, len(text.split()) * 4 // 3)

    def chunk_page(self, page: CleanedPage) -> List[Document]:
        """Split a single cleaned page into LangChain Documents with full metadata."""
        if not page.text.strip():
            return []

        raw_chunks = self.splitter.split_text(page.text)
        documents: List[Document] = []

        url_hash = hashlib.md5(page.url.encode("utf-8")).hexdigest()[:8]

        for idx, chunk_text in enumerate(raw_chunks):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            token_count = self.count_tokens(chunk_text)
            chunk_id = f"{url_hash}-{idx}"

            doc = Document(
                page_content=chunk_text,
                metadata={
                    "source": page.url,
                    "title": page.title,
                    "description": page.description,
                    "chunk_id": chunk_id,
                    "chunk_index": idx,
                    "total_chunks_in_page": len(raw_chunks),
                    "token_count": token_count,
                    "char_count": len(chunk_text),
                },
            )
            documents.append(doc)

        return documents

    def chunk_pages(self, pages: List[CleanedPage]) -> Tuple[List[Document], int]:
        """Chunk a list of pages and return (documents, total_tokens)."""
        all_docs: List[Document] = []
        total_tokens = 0

        for page in pages:
            docs = self.chunk_page(page)
            all_docs.extend(docs)
            for d in docs:
                total_tokens += d.metadata.get("token_count", 0)

        return all_docs, total_tokens
