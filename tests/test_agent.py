"""Integration and unit tests for RAG Agent and grounding."""

import pytest
from src.agent.graph import GroundedRAGAgent


def test_agent_grounded_answer():
    agent = GroundedRAGAgent()
    res = agent.ask("How do you declare path parameters with types in FastAPI?")
    assert "answer" in res
    assert len(res["sources"]) > 0
    assert "path" in res["answer"].lower() or "item_id" in res["answer"].lower()
    assert res["grounded"] is True


def test_agent_unanswerable_refusal():
    agent = GroundedRAGAgent()
    res = agent.ask("What is the authentic recipe for baking sourdough bread?")
    assert "cannot find sufficient information" in res["answer"].lower()
    assert len(res["sources"]) == 0


def test_agent_misleading_refutation():
    agent = GroundedRAGAgent()
    res = agent.ask("How do you enable FastAPI's built-in PHP compiler using the @app.php() decorator?")
    ans_lower = res["answer"].lower()
    assert "not support" in ans_lower or "does not mention" in ans_lower or "cannot find" in ans_lower or "not execute php" in ans_lower
