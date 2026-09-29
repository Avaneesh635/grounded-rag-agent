"""Real-time token usage and cost accounting for ingestion and queries."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import tiktoken
from rich.console import Console
from rich.table import Table

from src.config import settings
from src.tracking.cost_model import CostBreakdown, CostModel

console = Console()


@dataclass
class IngestionStats:
    total_pages: int = 0
    total_chunks: int = 0
    total_tokens: int = 0
    embedding_model: str = settings.EMBEDDING_PROVIDER
    estimated_cost_usd: float = 0.0


@dataclass
class QueryStats:
    query: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    model_name: str
    retrieved_chunks_count: int
    cited_urls_count: int


class TokenTracker:
    """Singleton/Instance tracker for cumulative token metrics and costs."""

    def __init__(self):
        self.ingestion: IngestionStats = IngestionStats()
        self.queries: List[QueryStats] = []
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.tokenizer = None

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken or estimation."""
        if not text:
            return 0
        if self.tokenizer:
            return len(self.tokenizer.encode(text, disallowed_special=()))
        return max(1, len(text.split()) * 4 // 3)

    def record_ingestion(
        self,
        pages_count: int,
        chunks_count: int,
        total_tokens: int,
        model_name: str = settings.OPENAI_EMBEDDING_MODEL
    ) -> IngestionStats:
        """Record vector store ingestion statistics."""
        cost = CostModel.calculate_embedding_cost(total_tokens, model_name)
        self.ingestion = IngestionStats(
            total_pages=pages_count,
            total_chunks=chunks_count,
            total_tokens=total_tokens,
            embedding_model=model_name,
            estimated_cost_usd=cost
        )
        return self.ingestion

    def record_query(
        self,
        query: str,
        prompt_text: str,
        completion_text: str,
        model_name: str = settings.OPENAI_MODEL,
        retrieved_chunks_count: int = 0,
        cited_urls_count: int = 0,
        prompt_tokens_override: Optional[int] = None,
        completion_tokens_override: Optional[int] = None
    ) -> QueryStats:
        """Record a single user query execution with exact/estimated token counts."""
        prompt_tokens = (
            prompt_tokens_override
            if prompt_tokens_override is not None
            else self.count_tokens(prompt_text)
        )
        completion_tokens = (
            completion_tokens_override
            if completion_tokens_override is not None
            else self.count_tokens(completion_text)
        )

        cost_info = CostModel.calculate_llm_cost(prompt_tokens, completion_tokens, model_name)

        q_stats = QueryStats(
            query=query,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=cost_info.total_cost_usd,
            model_name=model_name,
            retrieved_chunks_count=retrieved_chunks_count,
            cited_urls_count=cited_urls_count
        )
        self.queries.append(q_stats)
        return q_stats

    def get_summary(self) -> Dict:
        """Return cumulative session statistics."""
        total_prompt_tokens = sum(q.prompt_tokens for q in self.queries)
        total_completion_tokens = sum(q.completion_tokens for q in self.queries)
        total_query_cost = sum(q.cost_usd for q in self.queries)
        total_queries = len(self.queries)

        avg_prompt = total_prompt_tokens // total_queries if total_queries else 0
        avg_completion = total_completion_tokens // total_queries if total_queries else 0

        return {
            "ingestion": {
                "pages": self.ingestion.total_pages,
                "chunks": self.ingestion.total_chunks,
                "tokens": self.ingestion.total_tokens,
                "model": self.ingestion.embedding_model,
                "cost_usd": self.ingestion.estimated_cost_usd,
            },
            "queries": {
                "total_queries": total_queries,
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
                "total_tokens": total_prompt_tokens + total_completion_tokens,
                "avg_prompt_tokens": avg_prompt,
                "avg_completion_tokens": avg_completion,
                "total_query_cost_usd": total_query_cost,
                "total_system_cost_usd": self.ingestion.estimated_cost_usd + total_query_cost,
            }
        }


# Global singleton instance
tracker = TokenTracker()
