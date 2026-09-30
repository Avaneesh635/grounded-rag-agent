"""LangGraph Agent State definition."""

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict
from langchain_core.documents import Document


class AgentState(TypedDict, total=False):
    """Represents the complete state of the RAG agent throughout graph execution."""

    query: str
    rewritten_query: str
    documents: List[Document]
    filtered_documents: List[Document]
    has_sufficient_info: bool
    generation: str
    sources: List[Dict[str, Any]]
    grounded: bool
    hallucination_grade: str
    token_usage: Dict[str, int]
    cost_usd: float
    error: Optional[str]
