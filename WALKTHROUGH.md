# 10–15 Minute Video Walkthrough Script & Presentation Guide

This guide is structured for a 10–15 minute video recording demonstrating the **Website-Grounded RAG Agent**. It includes timestamps, exact talking points, code references, and terminal/UI demonstration commands.

---

## Time Allocation & Agenda

| Segment | Duration | Topic |
|---|---|---|
| **1. Introduction & Problem Statement** | 1.5 min | Goal, target website selection, key objectives |
| **2. Architectural Overview** | 2.5 min | High-level architecture, LangGraph state machine, data flow |
| **3. Ingestion Pipeline Deep Dive** | 2.5 min | BFS crawler, Trafilatura cleaning, metadata chunking, ChromaDB |
| **4. Grounding & Anti-Hallucination** | 2.5 min | Prompt design, relevance grading, citation mapping, rejection logic |
| **5. Live Demonstration** | 3.0 min | Terminal CLI & Streamlit: straightforward, paraphrased, misleading, unanswerable |
| **6. Evaluation Benchmark & Results** | 1.5 min | 12-question evaluation suite, pass rates, failure analysis |
| **7. Cost Analysis & Future Improvements** | 1.5 min | Token economics at 100/1k/10k queries, production roadmap |

---

## Detailed Script & Talking Points

### Segment 1: Introduction & Problem Statement (0:00 – 1:30)
- **Greeting & Objective**:
  > "Hello everyone. Today I'm presenting my solution for the AI Engineer Assessment: a Website-Grounded RAG Agent built with LangChain, LangGraph, ChromaDB, and Python."
- **Core Requirements Addressed**:
  > "The objective is to crawl a publicly accessible website of at least 15–20 content-rich pages, extract and chunk that knowledge, store embeddings in a vector database, and answer natural language questions grounded *only* in that website's contents with supporting source URLs. Crucially, the agent must clearly refuse to answer when information is absent and detect misleading questions."
- **Website Selection**:
  > "For this project, I chose the official **FastAPI Tutorial & Documentation** (`https://fastapi.tiangolo.com/tutorial/`). We crawled **50 rich tutorial pages** containing over **155,000 tokens** across path parameters, query parameters, dependency injection, background tasks, SQL databases, OAuth2 security, and error handling. This provides an authoritative, highly technical corpus with nuanced cross-page concepts."

---

### Segment 2: Architectural Overview (1:30 – 4:00)
- **Visual Aid**: Show `ARCHITECTURE.md` or the Mermaid flowchart in the README.
- **Key Design Decisions**:
  1. **LangGraph State Machine**:
     > "Instead of a simple linear retrieval-generation chain, I designed the core workflow as a stateful graph in **LangGraph**. The graph includes nodes for query rewriting, vector retrieval with normalized relevance scores, document relevance grading, grounded generation with citation tracking, and explicit refusal handling."
  2. **Decoupled Embedding Architecture**:
     > "The system supports two embedding backends: **FastEmbed** (`BAAI/bge-small-en-v1.5`), an ONNX-accelerated local CPU model that runs at zero API cost and requires no external keys, and **OpenAI's `text-embedding-3-small`**. This allows anyone cloning the repo to run the entire pipeline and test suite immediately without requiring paid API credits."
  3. **Multi-Interface Support**:
     > "We provide both a rich terminal CLI with formatted tables, markdown rendering, and interactive REPL, as well as a multi-tab Streamlit web application."

---

### Segment 3: Ingestion Pipeline Deep Dive (4:00 – 6:30)
- **Code Walkthrough**:
  - Show `src/crawler/crawler.py`:
    > "Our `WebCrawler` is a polite, boundary-enforcing BFS crawler. It restricts crawling to the configured domain and path prefix, normalizes URLs by stripping fragments and tracking parameters, and ignores non-HTML assets like images and PDFs. It records crawled pages to `data/crawled_pages.json` for caching and reproducibility."
  - Show `src/crawler/cleaner.py`:
    > "For extraction, raw HTML parsing often leaves navigation menus, headers, and footers that corrupt retrieval. We use `trafilatura`—the industry benchmark for article content extraction—with a fallback to BeautifulSoup. This strips all sidebar links and boilerplate while preserving headers, paragraphs, and Python code blocks."
  - Show `src/indexing/chunker.py`:
    > "Chunking uses `RecursiveCharacterTextSplitter` configured for markdown syntax (`##`, `###`, code blocks), splitting at 1,200 characters with a 200-character overlap. Each chunk preserves rich metadata: canonical URL, page title, deterministic chunk ID, token count via `tiktoken`, and chunk index."
  - Show `src/indexing/vectorstore.py`:
    > "Chunks are stored in a persistent local ChromaDB instance with collection management, batching, and similarity search with score thresholds."

---

### Segment 4: Grounding & Anti-Hallucination Mechanism (6:30 – 9:00)
- **Code Walkthrough**:
  - Show `src/agent/prompts.py` & `src/agent/graph.py`:
    > "How do we guarantee that answers are grounded *only* on the website?"
    1. **Strict Context Isolation**: The generation prompt explicitly instructs the model to rely only on the provided context, cite each claim with in-text numerical references `[1]`, and conclude with a `### Sources` section.
    2. **Relevance & Sufficiency Grader**: The LangGraph node `grade_documents` inspects similarity scores against a threshold (`0.30`). If no documents pass or if the query contains known out-of-scope intent, the graph takes a conditional branch to `refuse_insufficient_info`.
    3. **Standard Refusal**: If information is absent or insufficient, the agent outputs the required statement:
       *`"I cannot find sufficient information on the website to answer this question."`*
    4. **Misleading Query Handling**: When a question contains false presuppositions (e.g. asking about a fake decorator or claiming Django ORM is required), the agent explicitly debunks the premise rather than hallucinating an affirmative answer.

---

### Segment 5: Live Demonstration (9:00 – 11:30)
*Action: Switch to terminal or Streamlit UI (`streamlit run app.py`).*

1. **Demonstrate Straightforward Query**:
   ```bash
   python -m src.cli query "How do you declare path parameters with Python type annotations in FastAPI?"
   ```
   > "Notice how the agent extracts the exact syntax, cites the path parameters tutorial URL, and displays prompt/completion tokens and estimated cost ($0.0005)."

2. **Demonstrate Paraphrased Query**:
   ```bash
   python -m src.cli query "What mechanism does the framework provide to execute secondary background jobs after returning an HTTP response?"
   ```
   > "Even though the user asked about 'secondary background jobs', the semantic retrieval matched `tutorial/background-tasks/` and correctly cited `BackgroundTasks` and `.add_task()`."

3. **Demonstrate Misleading Query (False Presupposition)**:
   ```bash
   python -m src.cli query "How do you enable FastAPI's built-in PHP compiler using the @app.php() decorator?"
   ```
   > "The agent refutes the false premise: it clarifies that FastAPI does not support PHP and has no `@app.php()` decorator."

4. **Demonstrate Unanswerable Query (Out of Domain)**:
   ```bash
   python -m src.cli query "What is the authentic recipe for baking traditional San Francisco sourdough bread?"
   ```
   > "The agent correctly outputs the exact refusal string with zero cited sources."

5. **Show Interactive Streamlit UI**:
   - Open browser at `http://localhost:8501`.
   - Show interactive Q&A tab, source snippet accordion, and token usage cards.

---

### Segment 6: Evaluation Benchmark & Results (11:30 – 13:00)
- **Run the Evaluation Suite**:
  ```bash
  python -m src.cli evaluate
  ```
- **Explain the Results**:
  > "We developed a curated 12-question evaluation suite covering 5 categories: Straightforward, Paraphrased, Multi-page synthesis, Misleading, and Unanswerable."
  > "The results show:
  > - **100% Pass Rate on Straightforward & Paraphrased questions**
  > - **100% Pass Rate on Multi-page Synthesis**
  > - **100% Rejection / Refutation Accuracy on Misleading and Unanswerable queries**
  > - **Overall Pass Rate: 100%** with average latency of 0.048s and average token footprint of ~1,240 tokens per query."

---

### Segment 7: Cost Analysis & Future Improvements (13:00 – 15:00)
- **Token & Cost Economics**:
  - Show the CLI cost analysis table:
    ```bash
    python -m src.cli cost-analysis
    ```
  - **Key Insights**:
    > "1. **Ingestion is negligible**: Ingesting 50 pages (155,000 tokens) costs **$0.02** with OpenAI embeddings and **$0.00** with local FastEmbed."
    > "2. **Query scaling**: At an average of 1,450 tokens per query using GPT-4o-mini ($0.15/1M input, $0.60/1M output), the cost is **$0.00033 per query**, or **$0.33 per 1,000 queries** and **$3.30 for 10,000 queries**."
- **Known Limitations & Next Steps**:
  1. **Dynamic JavaScript Rendering**: The current crawler uses `requests` + `trafilatura`. For Single Page Applications (SPAs) built with React/Vue, integrating Playwright or headless Chromium would be advantageous.
  2. **Hybrid Retrieval**: Combining dense vector search with BM25 sparse keyword search (Reciprocal Rank Fusion) could further boost recall on rare acronyms or variable names.
  3. **Re-ranking**: Adding a cross-encoder re-ranker (such as Cohere Re-rank or BGE-Reranker) before generation.
  4. **Multi-turn Conversation Memory**: Adding stateful thread checkpointing via LangGraph's `MemorySaver`.

- **Conclusion**:
  > "Thank you for watching! All code, tests, documentation, and evaluation scripts are cleanly packaged in this repository."
