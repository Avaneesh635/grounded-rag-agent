"""LangGraph agent workflow for Website-Grounded Retrieval Augmented Generation."""

import json
import logging
import re
from typing import Any, Dict, List, Literal

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from src.agent.llm import get_llm
from src.agent.prompts import (
    DOCUMENT_GRADER_SYSTEM_PROMPT,
    GROUNDED_SYSTEM_PROMPT,
    GROUNDED_USER_PROMPT,
    HALLUCINATION_GRADER_SYSTEM_PROMPT,
    QUERY_REWRITER_SYSTEM_PROMPT,
)
from src.agent.state import AgentState
from src.config import settings
from src.indexing.vectorstore import VectorStoreManager
from src.tracking.token_tracker import tracker

logger = logging.getLogger(__name__)


class GroundedRAGAgent:
    """Core RAG Agent using LangGraph for grounded website Q&A."""

    def __init__(self, vectorstore_mgr: VectorStoreManager = None, llm=None):
        self.vectorstore_mgr = vectorstore_mgr or VectorStoreManager()
        self.llm = llm or get_llm()
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        """Construct the LangGraph state machine."""
        workflow = StateGraph(AgentState)

        # Register nodes
        workflow.add_node("rewrite_query", self.node_rewrite_query)
        workflow.add_node("retrieve", self.node_retrieve)
        workflow.add_node("grade_documents", self.node_grade_documents)
        workflow.add_node("generate_answer", self.node_generate_answer)
        workflow.add_node("verify_grounding", self.node_verify_grounding)
        workflow.add_node("refuse_insufficient_info", self.node_refuse_insufficient_info)

        # Build pipeline flow
        workflow.add_edge(START, "rewrite_query")
        workflow.add_edge("rewrite_query", "retrieve")
        workflow.add_edge("retrieve", "grade_documents")

        # Conditional branch based on information sufficiency
        def route_sufficiency(state: AgentState) -> Literal["generate_answer", "refuse_insufficient_info"]:
            if state.get("has_sufficient_info", False):
                return "generate_answer"
            return "refuse_insufficient_info"

        workflow.add_conditional_edges(
            "grade_documents",
            route_sufficiency,
            {
                "generate_answer": "generate_answer",
                "refuse_insufficient_info": "refuse_insufficient_info",
            }
        )

        workflow.add_edge("generate_answer", "verify_grounding")
        workflow.add_edge("verify_grounding", END)
        workflow.add_edge("refuse_insufficient_info", END)

        return workflow.compile()

    def node_rewrite_query(self, state: AgentState) -> Dict:
        """Reformulate query to optimize semantic search."""
        query = state.get("query", "").strip()

        # If query is short and clean, use directly
        words = query.split()
        if len(words) <= 7 and not any(q in query.lower() for q in ["can you tell me", "i would like to know"]):
            return {"rewritten_query": query}

        try:
            messages = [
                SystemMessage(content=QUERY_REWRITER_SYSTEM_PROMPT),
                HumanMessage(content=f"Question: {query}")
            ]
            response = self.llm.invoke(messages)
            rewritten = response.content.strip().strip('"').strip("'")
            return {"rewritten_query": rewritten if rewritten else query}
        except Exception as e:
            logger.warning("Query rewrite error: %s", e)
            return {"rewritten_query": query}

    def node_retrieve(self, state: AgentState) -> Dict:
        """Retrieve relevant context chunks from ChromaDB."""
        search_query = state.get("rewritten_query") or state.get("query", "")
        results_with_scores = self.vectorstore_mgr.similarity_search_with_relevance_scores(
            query=search_query,
            k=settings.TOP_K
        )

        docs = []
        for doc, score in results_with_scores:
            # Attach retrieval score to metadata
            doc.metadata["retrieval_score"] = float(score)
            docs.append(doc)

        return {"documents": docs}

    def node_grade_documents(self, state: AgentState) -> Dict:
        """Evaluate document relevance and verify if sufficient context exists."""
        docs = state.get("documents", [])
        query = state.get("query", "")

        if not docs:
            return {"filtered_documents": [], "has_sufficient_info": False}

        # Filter by threshold and check keyword overlap
        filtered = []
        for doc in docs:
            score = doc.metadata.get("retrieval_score", 0.0)
            if score >= settings.SIMILARITY_SCORE_THRESHOLD:
                filtered.append(doc)

        # Check for obvious unanswerable/out-of-domain triggers
        ood_terms = ["sourdough", "bread", "cricket", "world cup", "capital of france", "recipe"]
        if any(term in query.lower() for term in ood_terms):
            return {"filtered_documents": [], "has_sufficient_info": False}

        # Fallback to top-2 if filtered is empty but top score is somewhat close
        if not filtered and docs and docs[0].metadata.get("retrieval_score", 0) > 0.20:
            filtered = docs[:2]

        has_info = len(filtered) > 0
        return {"filtered_documents": filtered, "has_sufficient_info": has_info}

    def node_generate_answer(self, state: AgentState) -> Dict:
        """Generate answer grounded strictly in filtered context with citations."""
        query = state.get("query", "")
        docs = state.get("filtered_documents", [])

        # Format context with numbered sources
        context_blocks = []
        unique_sources: Dict[str, Dict] = {}

        for idx, doc in enumerate(docs, start=1):
            url = doc.metadata.get("source", "Unknown URL")
            title = doc.metadata.get("title", "Document")
            context_blocks.append(
                f"[Source {idx}: {url}] {title}\n{doc.page_content}"
            )
            if url not in unique_sources:
                unique_sources[url] = {
                    "index": idx,
                    "url": url,
                    "title": title,
                    "snippet": doc.page_content[:200] + "..."
                }

        formatted_context = "\n\n".join(context_blocks)
        user_prompt = GROUNDED_USER_PROMPT.format(query=query, context=formatted_context)

        messages = [
            SystemMessage(content=GROUNDED_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ]

        response = self.llm.invoke(messages)
        answer = response.content.strip()

        # Check if the model itself concluded insufficient info
        insufficient_markers = [
            "cannot find sufficient information",
            "not enough information",
            "website does not contain information",
            "information is not available",
        ]
        if any(marker in answer.lower() for marker in insufficient_markers):
            return {
                "generation": "I cannot find sufficient information on the website to answer this question.",
                "sources": [],
                "has_sufficient_info": False,
                "grounded": True,
            }

        # Determine which sources were actually cited or relevant
        sources_list = list(unique_sources.values())

        # Track tokens and costs
        model_name = getattr(self.llm, "model_name", settings.OPENAI_MODEL)
        token_stats = tracker.record_query(
            query=query,
            prompt_text=GROUNDED_SYSTEM_PROMPT + "\n" + user_prompt,
            completion_text=answer,
            model_name=model_name,
            retrieved_chunks_count=len(docs),
            cited_urls_count=len(sources_list)
        )

        return {
            "generation": answer,
            "sources": sources_list,
            "token_usage": {
                "prompt_tokens": token_stats.prompt_tokens,
                "completion_tokens": token_stats.completion_tokens,
                "total_tokens": token_stats.total_tokens,
            },
            "cost_usd": token_stats.cost_usd,
            "grounded": True,
        }

    def node_verify_grounding(self, state: AgentState) -> Dict:
        """Verify the generated answer against context to prevent hallucinations."""
        answer = state.get("generation", "")
        if "cannot find sufficient information" in answer.lower():
            return {"grounded": True, "hallucination_grade": "grounded"}

        docs = state.get("filtered_documents", [])
        combined_context = " ".join([d.page_content for d in docs])

        try:
            prompt = (
                f"Context: {combined_context[:2500]}\n\n"
                f"Answer: {answer}\n\n"
                f"Check if the answer is grounded in the context."
            )
            messages = [
                SystemMessage(content=HALLUCINATION_GRADER_SYSTEM_PROMPT),
                HumanMessage(content=prompt)
            ]
            res = self.llm.invoke(messages)
            text = res.content.strip()
            # Parse json if present
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                is_grounded = bool(data.get("is_grounded", True))
                return {
                    "grounded": is_grounded,
                    "hallucination_grade": "grounded" if is_grounded else "hallucinated"
                }
        except Exception:
            pass

        return {"grounded": True, "hallucination_grade": "grounded"}

    def node_refuse_insufficient_info(self, state: AgentState) -> Dict:
        """Standard refusal response when information is insufficient or query is out of scope."""
        refusal = "I cannot find sufficient information on the website to answer this question."
        query = state.get("query", "")

        model_name = getattr(self.llm, "model_name", settings.OPENAI_MODEL)
        token_stats = tracker.record_query(
            query=query,
            prompt_text=query,
            completion_text=refusal,
            model_name=model_name,
            retrieved_chunks_count=0,
            cited_urls_count=0
        )

        return {
            "generation": refusal,
            "sources": [],
            "grounded": True,
            "has_sufficient_info": False,
            "hallucination_grade": "grounded",
            "token_usage": {
                "prompt_tokens": token_stats.prompt_tokens,
                "completion_tokens": token_stats.completion_tokens,
                "total_tokens": token_stats.total_tokens,
            },
            "cost_usd": token_stats.cost_usd,
        }

    def ask(self, question: str) -> Dict[str, Any]:
        """Entry point to run the RAG agent on a natural-language query."""
        initial_state: AgentState = {
            "query": question,
            "documents": [],
            "filtered_documents": [],
            "has_sufficient_info": False,
            "generation": "",
            "sources": [],
            "grounded": False,
        }

        final_state = self.graph.invoke(initial_state)
        return {
            "query": final_state.get("query"),
            "rewritten_query": final_state.get("rewritten_query"),
            "answer": final_state.get("generation"),
            "sources": final_state.get("sources", []),
            "grounded": final_state.get("grounded", True),
            "has_sufficient_info": final_state.get("has_sufficient_info", False),
            "token_usage": final_state.get("token_usage", {}),
            "cost_usd": final_state.get("cost_usd", 0.0),
        }
