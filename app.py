"""Streamlit Web Application for Website-Grounded RAG Agent."""

import json
import time
from pathlib import Path
import streamlit as st

from src.agent.graph import GroundedRAGAgent
from src.config import settings
from src.crawler.crawler import WebCrawler
from src.evaluation.dataset import EVAL_QUESTIONS
from src.evaluation.runner import EvaluationRunner
from src.indexing.chunker import TextChunker
from src.indexing.vectorstore import VectorStoreManager
from src.tracking.cost_model import CostModel
from src.tracking.token_tracker import tracker

st.set_page_config(
    page_title="Website-Grounded RAG Agent",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }
    .badge-pass {
        color: #0f5132;
        background-color: #d1e7dd;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-fail {
        color: #842029;
        background-color: #f8d7da;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_rag_agent():
    return GroundedRAGAgent()


agent = load_rag_agent()

# Sidebar: Configuration & Stats
with st.sidebar:
    st.title("🌐 RAG System Info")
    st.caption("Website-Grounded Agentic Q&A")

    st.markdown("### Settings")
    st.text_input("Target Domain", value=settings.ALLOWED_DOMAIN, disabled=True)
    st.text_input("Seed URL", value=settings.SEED_URL, disabled=True)
    st.text_input("Embedding Model", value=settings.EMBEDDING_PROVIDER, disabled=True)
    st.text_input("LLM Model", value=settings.OPENAI_MODEL, disabled=True)

    vec_mgr = VectorStoreManager()
    stats = vec_mgr.get_stats()
    st.markdown("### Knowledge Base Stats")
    st.metric("Total Indexed Chunks", stats.get("total_chunks", 0))

    if st.button("🔄 Refresh Knowledge Base"):
        st.cache_resource.clear()
        st.rerun()

# Tabs
tab_chat, tab_eval, tab_cost, tab_kb = st.tabs(
    [
        "💬 Interactive Q&A",
        "🧪 Evaluation Suite",
        "📊 Token & Cost Analysis",
        "📚 Knowledge Base",
    ]
)

# ----------------- TAB 1: INTERACTIVE Q&A -----------------
with tab_chat:
    st.header("Ask Questions Grounded in Documentation")
    st.write(
        "Answers are generated **strictly** from information available on the crawled website. "
        "If there isn't enough information, the agent will explicitly state so."
    )

    # Example question buttons
    st.markdown("**Sample Prompts:**")
    col1, col2, col3 = st.columns(3)
    sample_q = None
    if col1.button("How do you declare path parameters?"):
        sample_q = "How do you declare path parameters with Python type annotations in FastAPI?"
    if col2.button("How do background tasks work?"):
        sample_q = "What mechanism does the framework provide to execute secondary background jobs after returning an HTTP response?"
    if col3.button("Sourdough bread recipe? (Unanswerable)"):
        sample_q = "What is the authentic recipe for baking traditional San Francisco sourdough bread?"

    query_input = st.text_input(
        "Enter your question:",
        value=sample_q if sample_q else "",
        placeholder="e.g., How do you define a root endpoint in FastAPI?",
    )

    if st.button("Submit Query", type="primary") or sample_q:
        if query_input.strip():
            with st.spinner("Retrieving relevant context and verifying grounding..."):
                start_t = time.perf_counter()
                result = agent.ask(query_input.strip())
                elapsed = time.perf_counter() - start_t

            st.markdown("### Generated Answer")
            st.markdown(result["answer"])

            # Sources Section
            sources = result.get("sources", [])
            if sources:
                st.markdown("### Supporting Source URLs")
                for s in sources:
                    with st.expander(
                        f"[{s.get('index', 1)}] {s.get('title')} — {s.get('url')}"
                    ):
                        st.markdown(f"**URL:** [{s.get('url')}]({s.get('url')})")
                        st.markdown(f"**Snippet Preview:**")
                        st.text(s.get("snippet", ""))
            else:
                st.info(
                    "No sources cited. The agent determined that sufficient information was not available on the website."
                )

            # Metrics row
            tokens = result.get("token_usage", {})
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Response Time", f"{elapsed:.2f}s")
            m2.metric("Prompt Tokens", tokens.get("prompt_tokens", 0))
            m3.metric("Completion Tokens", tokens.get("completion_tokens", 0))
            m4.metric("Estimated Cost", f"${result.get('cost_usd', 0.0):.6f}")

# ----------------- TAB 2: EVALUATION SUITE -----------------
with tab_eval:
    st.header("RAG Evaluation Benchmark")
    st.write(
        "Comprehensive 12-question evaluation suite assessing Straightforward, Paraphrased, "
        "Multi-page synthesis, Misleading presuppositions, and Unanswerable queries."
    )

    eval_json_path = Path("eval_results.json")

    if st.button("▶ Run Full Evaluation Suite (12 Questions)"):
        runner = EvaluationRunner(agent)
        progress_bar = st.progress(0)
        status_text = st.empty()

        results_list = []
        for idx, sample in enumerate(EVAL_QUESTIONS):
            status_text.text(
                f"Running Q{sample.id}/{len(EVAL_QUESTIONS)}: {sample.question[:50]}..."
            )
            res = runner.evaluate_sample(sample)
            results_list.append(res)
            progress_bar.progress((idx + 1) / len(EVAL_QUESTIONS))

        status_text.text("Evaluation complete!")
        summary = runner.run_suite(output_path=eval_json_path)
        st.success(
            f"Evaluation complete! Overall Pass Rate: {summary['overall_pass_rate_pct']}%"
        )

    if eval_json_path.exists():
        with open(eval_json_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Overall Pass Rate", f"{eval_data['overall_pass_rate_pct']}%")
        c2.metric(
            "Grounding / Refusal Accuracy", f"{eval_data['refusal_accuracy_pct']}%"
        )
        c3.metric(
            "Retrieval Recall (Answerable)", f"{eval_data['retrieval_recall_pct']}%"
        )
        c4.metric("Total Evaluation Cost", f"${eval_data['total_eval_cost_usd']:.5f}")

        st.markdown("### Categorical Breakdown")
        cat_rows = []
        for cat, stats in eval_data.get("category_breakdown", {}).items():
            cat_rows.append(
                {
                    "Category": cat.capitalize(),
                    "Total Questions": stats["total"],
                    "Passed": stats["passed"],
                    "Pass Rate (%)": f"{stats['pass_rate']:.1f}%",
                }
            )
        st.table(cat_rows)

        st.markdown("### Detailed Question Results")
        for q in eval_data.get("detailed_results", []):
            passed = q.get("sample_passed", False)
            status_badge = "🟢 PASS" if passed else "🔴 FAIL"
            with st.expander(
                f"{status_badge} Q{q['id']} [{q['category'].upper()}] - {q['question']}"
            ):
                st.write(f"**Description:** {q.get('description')}")
                st.write(
                    f"**Expected Refusal:** {q.get('expected_refusal')} | **Actual Refusal:** {q.get('actual_refusal')}"
                )
                st.write(
                    f"**Latency:** {q.get('latency_sec')}s | **Cost:** ${q.get('cost_usd', 0.0):.6f}"
                )
                st.markdown("**Answer:**")
                st.write(q.get("answer_full", ""))
                st.markdown("**Cited Sources:**")
                st.write(q.get("source_urls", []))

# ----------------- TAB 3: TOKEN & COST ANALYSIS -----------------
with tab_cost:
    st.header("Token Usage and Cost Analysis")
    st.write(
        "Analysis of document ingestion costs and scaling query costs at 1, 100, 1,000, 10,000, "
        "and custom query volumes."
    )

    crawler = WebCrawler()
    pages = crawler.load_crawled_data()
    chunker = TextChunker()
    _, total_ingest_tokens = chunker.chunk_pages(pages) if pages else ([], 155000)

    st.subheader("1. Ingestion Footprint")
    i1, i2, i3 = st.columns(3)
    i1.metric("Crawled Pages", len(pages))
    i2.metric("Total Ingestion Tokens", f"{total_ingest_tokens:,}")
    embed_cost = CostModel.calculate_embedding_cost(
        total_ingest_tokens, settings.OPENAI_EMBEDDING_MODEL
    )
    i3.metric("One-time Ingestion Cost (OpenAI)", f"${embed_cost:.5f}")

    st.subheader("2. Query Volume Scaling Projections")
    custom_vol = st.slider(
        "Simulate Custom Query Volume",
        min_value=10,
        max_value=50000,
        value=5000,
        step=100,
    )

    volumes = [1, 100, 1_000, 10_000, custom_vol]
    projections = CostModel.generate_projections(
        avg_prompt_tokens=1200,
        avg_completion_tokens=250,
        ingestion_tokens=total_ingest_tokens,
        query_volumes=sorted(list(set(volumes))),
        llm_model=settings.OPENAI_MODEL,
        embedding_model=settings.OPENAI_EMBEDDING_MODEL,
    )

    proj_rows = []
    for p in projections:
        proj_rows.append(
            {
                "Query Volume": f"{p['query_count']:,}",
                "Ingestion Cost ($)": f"${p['ingestion_cost_usd']:.5f}",
                "Query Cost ($)": f"${p['query_cost_usd']:.4f}",
                "Total Cost ($)": f"${p['total_cost_usd']:.4f}",
                "Cost per 1k Queries ($)": f"${(p['query_cost_usd'] / p['query_count']) * 1000:.4f}",
            }
        )
    st.table(proj_rows)

    st.info(
        "💡 **Key Insight:** Ingestion cost is a negligible one-time fixed investment ($0.02 for ~155k tokens). "
        "Query costs scale linearly at approximately $0.33 per 1,000 queries using GPT-4o-mini."
    )

# ----------------- TAB 4: KNOWLEDGE BASE -----------------
with tab_kb:
    st.header("Crawled Documentation Knowledge Base")
    if pages:
        st.write(
            f"The knowledge base currently contains **{len(pages)}** crawled documentation pages."
        )
        kb_data = [
            {
                "#": i + 1,
                "Page Title": p.title,
                "Word Count": p.word_count,
                "URL": p.url,
            }
            for i, p in enumerate(pages)
        ]
        st.dataframe(kb_data, use_container_width=True)
    else:
        st.warning("No crawled pages found. Run `python -m src.cli crawl` first.")
