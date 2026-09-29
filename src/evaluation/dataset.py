"""Curated evaluation dataset containing 12 test questions across 5 core categories."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class EvalSample:
    id: int
    question: str
    category: str  # "straightforward", "paraphrased", "multi-page", "misleading", "unanswerable"
    expected_refusal: bool  # True if the agent MUST refuse or debunk
    expected_url_keywords: List[str]  # Partial URL slugs expected in citations
    expected_answer_keywords: List[str]  # Key technical terms expected
    description: str


EVAL_QUESTIONS: List[EvalSample] = [
    # 1. Straightforward Questions
    EvalSample(
        id=1,
        question="How do you create a basic FastAPI application and declare a GET root endpoint returning Hello World?",
        category="straightforward",
        expected_refusal=False,
        expected_url_keywords=["first-steps"],
        expected_answer_keywords=["FastAPI", "@app.get", "root", "async def"],
        description="Direct factual query matching 'First Steps' page."
    ),
    EvalSample(
        id=2,
        question="How do you declare path parameters with Python type annotations in FastAPI?",
        category="straightforward",
        expected_refusal=False,
        expected_url_keywords=["path-params"],
        expected_answer_keywords=["item_id", "int", "path"],
        description="Core basic functionality question on path parameters."
    ),

    # 2. Paraphrased Questions
    EvalSample(
        id=3,
        question="In what manner can an application accept non-path query parameters that are automatically converted to standard types?",
        category="paraphrased",
        expected_refusal=False,
        expected_url_keywords=["query-params"],
        expected_answer_keywords=["query", "default", "optional"],
        description="Paraphrased query testing semantic retrieval of query parameters without using standard phrasing."
    ),
    EvalSample(
        id=4,
        question="What mechanism does the framework provide to execute secondary background jobs after returning an HTTP response?",
        category="paraphrased",
        expected_refusal=False,
        expected_url_keywords=["background-tasks"],
        expected_answer_keywords=["BackgroundTasks", "add_task"],
        description="Paraphrased question about background tasks."
    ),

    # 3. Multi-page Synthesis Questions
    EvalSample(
        id=5,
        question="How do you share logic across endpoints using Dependency Injection and integrate it with Security OAuth2?",
        category="multi-page",
        expected_refusal=False,
        expected_url_keywords=["dependencies", "security"],
        expected_answer_keywords=["Depends", "OAuth2", "security"],
        description="Requires cross-referencing Dependency Injection tutorial with Security tutorials."
    ),
    EvalSample(
        id=6,
        question="How do you define SQL database sessions with yield dependencies to ensure connections are closed?",
        category="multi-page",
        expected_refusal=False,
        expected_url_keywords=["sql-databases", "dependencies"],
        expected_answer_keywords=["yield", "finally", "db", "close"],
        description="Requires synthesizing SQL database setup with yield dependencies."
    ),

    # 4. Misleading Questions (False Presuppositions)
    EvalSample(
        id=7,
        question="How do you enable FastAPI's built-in PHP compiler using the @app.php() decorator?",
        category="misleading",
        expected_refusal=True,
        expected_url_keywords=[],
        expected_answer_keywords=["not support", "does not mention", "cannot find", "php"],
        description="Misleading question assuming FastAPI executes PHP via a fake decorator."
    ),
    EvalSample(
        id=8,
        question="Why is Django ORM mandatory for connecting to SQLite databases in FastAPI?",
        category="misleading",
        expected_refusal=True,
        expected_url_keywords=[],
        expected_answer_keywords=["not mandatory", "does not require", "cannot find", "sqlalchemy"],
        description="Misleading question claiming Django ORM is required."
    ),
    EvalSample(
        id=9,
        question="Which configuration flag in FastAPI disables Python type hints to force dynamic JavaScript compilation?",
        category="misleading",
        expected_refusal=True,
        expected_url_keywords=[],
        expected_answer_keywords=["cannot find", "not support", "does not", "information"],
        description="Misleading question assuming fake JavaScript compilation flag."
    ),

    # 5. Unanswerable Questions (Completely Out of Scope)
    EvalSample(
        id=10,
        question="What is the authentic recipe for baking traditional San Francisco sourdough bread?",
        category="unanswerable",
        expected_refusal=True,
        expected_url_keywords=[],
        expected_answer_keywords=["cannot find sufficient information"],
        description="Unrelated out-of-domain cooking question."
    ),
    EvalSample(
        id=11,
        question="Who won the 2024 ICC Men's T20 Cricket World Cup?",
        category="unanswerable",
        expected_refusal=True,
        expected_url_keywords=[],
        expected_answer_keywords=["cannot find sufficient information"],
        description="Unrelated out-of-domain sports question."
    ),
    EvalSample(
        id=12,
        question="How do you configure Kubernetes pod autoscaling with Helm charts in FastAPI?",
        category="unanswerable",
        expected_refusal=True,
        expected_url_keywords=[],
        expected_answer_keywords=["cannot find sufficient information"],
        description="Out-of-scope infrastructure topic not covered in basic tutorial pages."
    ),
]
