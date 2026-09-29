"""Cost calculation and scaling projections for RAG ingestion and queries."""

from dataclasses import dataclass
from typing import Dict, List, Optional
from rich.table import Table

from src.config import settings


@dataclass
class CostBreakdown:
    """Detailed cost metrics for a single operation."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float
    model_name: str


class CostModel:
    """Calculates LLM and Embedding costs based on token consumption and model pricing."""

    PRICING = settings.PRICING

    @classmethod
    def get_pricing(cls, model_name: str) -> Dict[str, float]:
        """Fetch input/output price per 1M tokens for a given model."""
        # Normalize model name
        for key, p in cls.PRICING.items():
            if key in model_name:
                return p
        # Default fallback to gpt-4o-mini rates
        return cls.PRICING.get("gpt-4o-mini", {"input": 0.15, "output": 0.60})

    @classmethod
    def calculate_embedding_cost(cls, tokens: int, model_name: str = settings.OPENAI_EMBEDDING_MODEL) -> float:
        """Calculate embedding cost in USD."""
        pricing = cls.get_pricing(model_name)
        rate_per_1m = pricing.get("input", 0.0)
        return (tokens / 1_000_000.0) * rate_per_1m

    @classmethod
    def calculate_llm_cost(
        cls,
        prompt_tokens: int,
        completion_tokens: int,
        model_name: str = settings.OPENAI_MODEL
    ) -> CostBreakdown:
        """Calculate LLM generation cost breakdown in USD."""
        pricing = cls.get_pricing(model_name)
        input_rate = pricing.get("input", 0.15)
        output_rate = pricing.get("output", 0.60)

        input_cost = (prompt_tokens / 1_000_000.0) * input_rate
        output_cost = (completion_tokens / 1_000_000.0) * output_rate
        total_cost = input_cost + output_cost

        return CostBreakdown(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            input_cost_usd=input_cost,
            output_cost_usd=output_cost,
            total_cost_usd=total_cost,
            model_name=model_name
        )

    @classmethod
    def generate_projections(
        cls,
        avg_prompt_tokens: int = 1200,
        avg_completion_tokens: int = 250,
        ingestion_tokens: int = 45000,
        query_volumes: Optional[List[int]] = None,
        llm_model: str = settings.OPENAI_MODEL,
        embedding_model: str = settings.OPENAI_EMBEDDING_MODEL
    ) -> List[Dict]:
        """Project total costs across query volumes (1, 100, 1,000, 10,000)."""
        if query_volumes is None:
            query_volumes = [1, 100, 1_000, 10_000]

        ingestion_cost = cls.calculate_embedding_cost(ingestion_tokens, embedding_model)
        query_cost_info = cls.calculate_llm_cost(avg_prompt_tokens, avg_completion_tokens, llm_model)
        single_query_cost = query_cost_info.total_cost_usd

        projections = []
        for count in query_volumes:
            total_query_cost = count * single_query_cost
            total_cost = ingestion_cost + total_query_cost
            projections.append({
                "query_count": count,
                "ingestion_tokens": ingestion_tokens,
                "ingestion_cost_usd": ingestion_cost,
                "avg_query_tokens": avg_prompt_tokens + avg_completion_tokens,
                "query_cost_usd": total_query_cost,
                "total_cost_usd": total_cost,
                "llm_model": llm_model,
                "embedding_model": embedding_model,
            })
        return projections

    @classmethod
    def format_projections_table(cls, projections: List[Dict]) -> Table:
        """Build a Rich table summarizing cost projections."""
        table = Table(title="Website-Grounded RAG Cost Projections", show_header=True, header_style="bold magenta")
        table.add_column("Query Volume", justify="right", style="cyan")
        table.add_column("Ingestion Cost ($)", justify="right", style="dim")
        table.add_column("Query Cost ($)", justify="right", style="green")
        table.add_column("Total Cost ($)", justify="right", style="bold yellow")
        table.add_column("Cost / 1K Queries ($)", justify="right")

        for p in projections:
            count = p["query_count"]
            q_cost = p["query_cost_usd"]
            t_cost = p["total_cost_usd"]
            i_cost = p["ingestion_cost_usd"]
            rate_per_1k = (q_cost / count) * 1000 if count > 0 else 0.0

            table.add_row(
                f"{count:,}",
                f"${i_cost:.5f}",
                f"${q_cost:.4f}",
                f"${t_cost:.4f}",
                f"${rate_per_1k:.4f}"
            )
        return table
