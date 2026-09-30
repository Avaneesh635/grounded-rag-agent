"""Unit tests for cost model and projections."""

import pytest
from src.tracking.cost_model import CostModel


def test_cost_calculation():
    # Ingestion embedding cost
    cost_emb = CostModel.calculate_embedding_cost(50_000, "text-embedding-3-small")
    # 50,000 / 1M * 0.02 = 0.001
    assert abs(cost_emb - 0.001) < 1e-6

    # LLM query cost
    breakdown = CostModel.calculate_llm_cost(
        prompt_tokens=1000, completion_tokens=200, model_name="gpt-4o-mini"
    )
    # 1000 * 0.15/1M = 0.00015, 200 * 0.60/1M = 0.00012 -> total = 0.00027
    assert abs(breakdown.total_cost_usd - 0.00027) < 1e-6


def test_cost_projections():
    projections = CostModel.generate_projections(
        avg_prompt_tokens=1000,
        avg_completion_tokens=200,
        ingestion_tokens=50_000,
        query_volumes=[1, 100, 1000, 10000],
    )
    assert len(projections) == 4
    p_10k = [p for p in projections if p["query_count"] == 10000][0]
    assert p_10k["query_cost_usd"] > p_10k["ingestion_cost_usd"]
