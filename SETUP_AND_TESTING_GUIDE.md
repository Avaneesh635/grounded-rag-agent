# Quickstart: API Key Configuration & Real-Website Testing Guide

This guide explains how to add your OpenAI API key and test the Website-Grounded RAG Agent on any real website.

---

## 1. Add Your OpenAI API Key

You can configure your API key in **either of two ways**:

### Method A: Using a `.env` file (Recommended)
In the project root, create or edit `.env` (it is already ignored by git, so your keys will never be accidentally committed):

```bash
# In /home/avaneesh/projects/two
cp .env.example .env
```

Open `.env` and set your key and preferred provider:
```ini
OPENAI_API_KEY=sk-proj-yourActualOpenAiKeyHere...

# LLM Provider
LLM_PROVIDER=openai
OPENAI_MODEL=gpt-4o-mini

# Embedding Provider (choose either "fastembed" for free local CPU, or "openai")
EMBEDDING_PROVIDER=openai
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

### Method B: Directly in your terminal session
```bash
export OPENAI_API_KEY="sk-proj-yourActualOpenAiKeyHere..."
```

---

## 2. Test It on the Included Real Documentation (FastAPI)

The repository comes pre-indexed with **50 real tutorial pages** from [FastAPI](https://fastapi.tiangolo.com/tutorial/). You can immediately ask questions:

```bash
# 1. Single question via CLI
.venv/bin/python -m src.cli query "How do background tasks work in FastAPI?"

# 2. Test a misleading question (verifies grounding / anti-hallucination)
.venv/bin/python -m src.cli query "How do you enable FastAPI's PHP compiler using @app.php()?"

# 3. Test an unanswerable question (verifies strict refusal)
.venv/bin/python -m src.cli query "What is the authentic recipe for baking sourdough bread?"

# 4. Interactive chat session
.venv/bin/python -m src.cli interactive
```

---

## 3. Crawl & Test on Any Other Real Website

To point the agent to **any public website** (e.g. Pydantic, Click, Typer, Stripe docs, etc.):

### Step 1: Run the Crawler
Pass the target website's seed URL. The crawler will automatically detect the domain and crawl up to the specified `--max-pages`:

#### Example A: Crawl Pydantic Documentation
```bash
.venv/bin/python -m src.cli crawl --url "https://docs.pydantic.dev/latest/" --max-pages 25 --force
```

#### Example B: Crawl Click CLI Documentation
```bash
.venv/bin/python -m src.cli crawl --url "https://click.palletsprojects.com/en/latest/" --max-pages 25 --force
```

#### Example C: Crawl Any Custom Documentation / Blog
```bash
.venv/bin/python -m src.cli crawl --url "https://your-target-site.com/docs/" --max-pages 30 --force
```

*(Optional flags: `--allowed-domain your-target-site.com` and `--path-prefix /docs/` if you want strict custom boundaries).*

### Step 2: Index the Crawled Pages into ChromaDB
```bash
.venv/bin/python -m src.cli index
```
*This splits the extracted text into markdown chunks, computes embeddings, and stores them in ChromaDB.*

### Step 3: Query Your New Website
```bash
.venv/bin/python -m src.cli query "your question here"
```

---

## 4. Test via Interactive Web UI (Streamlit)

Launch the web interface:

```bash
.venv/bin/streamlit run app.py
```

Then open your browser at `http://localhost:8501`:
1. **💬 Interactive Q&A Tab**: Ask questions, view markdown answers, click expandable source links, and view token usage and cost per query.
2. **🧪 Evaluation Suite Tab**: Run automated benchmarks with live progress bars and scorecards.
3. **📊 Token & Cost Analysis Tab**: Use the interactive slider to project query costs at 100, 1k, and 10k queries.
4. **📚 Knowledge Base Tab**: Inspect all crawled pages, word counts, and direct links.
