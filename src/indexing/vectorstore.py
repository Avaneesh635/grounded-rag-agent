"""Vector database manager using ChromaDB for persistent storage and retrieval."""

import logging
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from rich.console import Console

from src.config import settings
from src.indexing.embedder import get_embeddings

logger = logging.getLogger(__name__)
console = Console()


class VectorStoreManager:
    """Manages the persistent ChromaDB collection for the website knowledge base."""

    def __init__(
        self,
        persist_dir: Optional[Path] = None,
        collection_name: Optional[str] = None,
        embeddings: Optional[Embeddings] = None,
    ):
        self.persist_dir = Path(persist_dir or settings.CHROMA_PERSIST_DIR)
        self.collection_name = collection_name or settings.COLLECTION_NAME
        self.embeddings = embeddings or get_embeddings()
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self._vectorstore: Optional[Chroma] = None

    @property
    def vectorstore(self) -> Chroma:
        """Lazy load or return the active Chroma vectorstore instance."""
        if self._vectorstore is None:
            self._vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(self.persist_dir),
            )
        return self._vectorstore

    def index_documents(
        self,
        documents: List[Document],
        batch_size: int = 64,
        clear_existing: bool = True,
    ) -> int:
        """Index a list of Document objects into ChromaDB."""
        if clear_existing:
            self.clear_collection()

        if not documents:
            console.print("[yellow]No documents provided for indexing.[/yellow]")
            return 0

        console.print(
            f"[bold cyan]Indexing {len(documents)} chunks into ChromaDB...[/bold cyan]"
        )

        # Index in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            ids = [
                d.metadata.get("chunk_id", f"doc-{i + idx}")
                for idx, d in enumerate(batch)
            ]
            self.vectorstore.add_documents(documents=batch, ids=ids)
            console.print(
                f"  Indexed batch {i // batch_size + 1}/{(len(documents) - 1) // batch_size + 1} ({len(batch)} chunks)"
            )

        console.print(
            f"[bold green]Successfully indexed {len(documents)} chunks into ChromaDB![/bold green]"
        )
        return len(documents)

    def similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = settings.TOP_K,
        score_threshold: Optional[float] = None,
    ) -> List[Tuple[Document, float]]:
        """Perform similarity search with normalized relevance scores."""
        threshold = (
            score_threshold
            if score_threshold is not None
            else settings.SIMILARITY_SCORE_THRESHOLD
        )
        try:
            results = self.vectorstore.similarity_search_with_relevance_scores(
                query=query, k=k, score_threshold=threshold
            )
            return results
        except Exception as e:
            logger.warning(
                "Similarity search with score threshold failed: %s. Falling back to plain search.",
                e,
            )
            raw_docs = self.vectorstore.similarity_search(query=query, k=k)
            # Default placeholder score if relevance score is unavailable
            return [(doc, 0.75) for doc in raw_docs]

    def similarity_search(self, query: str, k: int = settings.TOP_K) -> List[Document]:
        """Simple top-k similarity search."""
        return self.vectorstore.similarity_search(query=query, k=k)

    def clear_collection(self) -> None:
        """Reset the vector database collection."""
        try:
            if self._vectorstore is not None:
                self._vectorstore.delete_collection()
                self._vectorstore = None
            if self.persist_dir.exists():
                shutil.rmtree(self.persist_dir)
                self.persist_dir.mkdir(parents=True, exist_ok=True)
            console.print("[dim]Vector store collection cleared.[/dim]")
        except Exception as e:
            logger.warning("Error clearing collection: %s", e)

    def get_stats(self) -> dict:
        """Retrieve collection statistics."""
        try:
            count = self.vectorstore._collection.count()
            return {
                "collection_name": self.collection_name,
                "persist_dir": str(self.persist_dir),
                "total_chunks": count,
            }
        except Exception:
            return {
                "collection_name": self.collection_name,
                "persist_dir": str(self.persist_dir),
                "total_chunks": 0,
            }
