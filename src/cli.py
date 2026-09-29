"""Command Line Interface (CLI) for the Website-Grounded RAG Agent."""

import argparse
import sys
from pathlib import Path
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from src.agent.graph import GroundedRAGAgent
from src.config import settings
from src.crawler.crawler import WebCrawler
from src.evaluation.runner import EvaluationRunner
from src.indexing.chunker import TextChunker
from src.indexing.embedder import get_embeddings
from src.indexing.vectorstore import VectorStoreManager
from src.tracking.cost_model import CostModel
from src.tracking.token_tracker import tracker

console = Console()


def cmd_crawl(args: argparse.Namespace) -> None:
    """Crawl the website pages."""
    crawler = WebCrawler(
        seed_url=args.url,
        max_pages=args.max_pages,
        delay_seconds=args.delay
    )
    pages = crawler.crawl(force_recrawl=args.force)
    console.print(f"[bold green]Crawl complete. Total pages crawled: {len(pages)}[/bold green]")


def cmd_index(args: argparse.Namespace) -> None:
    """Chunk and index crawled pages into ChromaDB."""
    crawler = WebCrawler()
    pages = crawler.load_crawled_data()
    if not pages:
        console.print("[yellow]No crawled pages found. Running crawler first...[/yellow]")
        pages = crawler.crawl()

    chunker = TextChunker()
    documents, total_tokens = chunker.chunk_pages(pages)
    console.print(f"Created [cyan]{len(documents)}[/cyan] chunks totaling [cyan]{total_tokens:,}[/cyan] tokens.")

    vec_mgr = VectorStoreManager()
    indexed_count = vec_mgr.index_documents(documents, clear_existing=args.clear)

    # Track ingestion
    stats = tracker.record_ingestion(
        pages_count=len(pages),
        chunks_count=indexed_count,
        total_tokens=total_tokens,
        model_name=settings.EMBEDDING_PROVIDER
    )

    console.print(Panel(
        f"[bold]Knowledge Base Indexing Summary[/bold]\n"
        f"• Total Pages: {stats.total_pages}\n"
        f"• Total Chunks: {stats.total_chunks}\n"
        f"• Total Ingestion Tokens: {stats.total_tokens:,}\n"
        f"• Embedding Provider: {stats.embedding_model}\n"
        f"• Estimated Ingestion Cost: [green]${stats.estimated_cost_usd:.5f} USD[/green]",
        title="Ingestion Metrics",
        border_style="green"
    ))


def cmd_query(args: argparse.Namespace) -> None:
    """Run a single query through the RAG Agent."""
    query = args.question
    if not query:
        console.print("[red]Error: Please specify a question using --question or positional argument.[/red]")
        sys.exit(1)

    console.print(f"\n[bold cyan]Query:[/bold cyan] {query}")
    agent = GroundedRAGAgent()

    with console.status("[bold green]Thinking & retrieving from website knowledge base...[/bold green]"):
        result = agent.ask(query)

    console.print("\n[bold green]Answer:[/bold green]")
    console.print(Markdown(result["answer"]))

    sources = result.get("sources", [])
    if sources:
        console.print("\n[bold cyan]Retrieved & Cited Sources:[/bold cyan]")
        table = Table(show_header=True, header_style="bold blue")
        table.add_column("#", justify="right", width=4)
        table.add_column("Page Title", style="bold")
        table.add_column("URL", style="underline")

        for s in sources:
            table.add_row(str(s.get("index", 1)), s.get("title", ""), s.get("url", ""))
        console.print(table)
    else:
        console.print("\n[dim]No source citations (query was either refused or ungrounded).[/dim]")

    tokens = result.get("token_usage", {})
    cost = result.get("cost_usd", 0.0)
    console.print(f"\n[dim]Tokens: {tokens.get('total_tokens', 0)} (Prompt: {tokens.get('prompt_tokens', 0)}, Completion: {tokens.get('completion_tokens', 0)}) | Estimated Cost: ${cost:.6f} USD[/dim]\n")


def cmd_interactive(args: argparse.Namespace) -> None:
    """Interactive chat REPL."""
    console.print(Panel(
        "[bold cyan]Website-Grounded RAG Interactive Assistant[/bold cyan]\n"
        "Ask questions grounded strictly in the crawled documentation.\n"
        "Type [bold yellow]'exit'[/bold yellow] or [bold yellow]'quit'[/bold yellow] to terminate.",
        border_style="cyan"
    ))

    agent = GroundedRAGAgent()

    while True:
        try:
            query = console.input("\n[bold yellow]Ask a question > [/bold yellow]").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                console.print("[dim]Goodbye![/dim]")
                break

            with console.status("[bold green]Searching knowledge base...[/bold green]"):
                res = agent.ask(query)

            console.print("\n[bold green]Answer:[/bold green]")
            console.print(Markdown(res["answer"]))

            sources = res.get("sources", [])
            if sources:
                console.print("\n[bold cyan]Sources:[/bold cyan]")
                for s in sources:
                    console.print(f"  • [{s.get('index', 1)}] {s.get('title')}: [underline]{s.get('url')}[/underline]")

            t = res.get("token_usage", {})
            c = res.get("cost_usd", 0.0)
            console.print(f"[dim]Tokens: {t.get('total_tokens', 0)} | Cost: ${c:.6f}[/dim]")

        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Session terminated.[/dim]")
            break


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Run the 12-question evaluation suite."""
    runner = EvaluationRunner()
    output_path = Path(args.output) if args.output else None
    runner.run_suite(output_path=output_path)


def cmd_cost_analysis(args: argparse.Namespace) -> None:
    """Display comprehensive cost analysis and projections."""
    console.print(Panel(
        "[bold cyan]Website-Grounded RAG: Token & Cost Analysis[/bold cyan]\n"
        "Detailed cost breakdown for document ingestion and multi-tier query scaling.",
        border_style="cyan"
    ))

    # Calculate default projections
    projections = CostModel.generate_projections(
        avg_prompt_tokens=args.prompt_tokens,
        avg_completion_tokens=args.completion_tokens,
        ingestion_tokens=args.ingestion_tokens,
        query_volumes=[1, 100, 1_000, 10_000],
        llm_model=args.model,
        embedding_model=args.embedding_model
    )

    table = CostModel.format_projections_table(projections)
    console.print(table)

    console.print("\n[bold]Model Pricing Assumptions:[/bold]")
    console.print(f"• LLM Model: [cyan]{args.model}[/cyan] ($0.15/1M input, $0.60/1M output)")
    console.print(f"• Embedding Model: [cyan]{args.embedding_model}[/cyan] ($0.02/1M tokens, or $0.00 for local FastEmbed)")
    console.print(f"• Ingestion Knowledge Base Size: [cyan]{args.ingestion_tokens:,}[/cyan] tokens")
    console.print(f"• Average Query Footprint: [cyan]{args.prompt_tokens + args.completion_tokens:,}[/cyan] tokens (Prompt: {args.prompt_tokens}, Completion: {args.completion_tokens})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Website-Grounded RAG Agent CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Crawl command
    crawl_p = subparsers.add_parser("crawl", help="Crawl website pages")
    crawl_p.add_argument("--url", default=settings.SEED_URL, help="Seed URL to crawl")
    crawl_p.add_argument("--max-pages", type=int, default=settings.MAX_PAGES, help="Max pages to crawl")
    crawl_p.add_argument("--delay", type=float, default=settings.CRAWL_DELAY_SECONDS, help="Delay between requests")
    crawl_p.add_argument("--force", action="store_true", help="Force re-crawl ignoring cache")
    crawl_p.set_defaults(func=cmd_crawl)

    # Index command
    index_p = subparsers.add_parser("index", help="Chunk and index crawled pages into vector DB")
    index_p.add_argument("--clear", action="store_true", default=True, help="Clear existing index before indexing")
    index_p.set_defaults(func=cmd_index)

    # Query command
    query_p = subparsers.add_parser("query", help="Ask a single question")
    query_p.add_argument("question", nargs="?", default="", help="Natural language question")
    query_p.set_defaults(func=cmd_query)

    # Interactive command
    subparsers.add_parser("interactive", help="Start interactive terminal chat").set_defaults(func=cmd_interactive)

    # Evaluate command
    eval_p = subparsers.add_parser("evaluate", help="Run evaluation suite on test dataset")
    eval_p.add_argument("--output", default="eval_results.json", help="Path to save evaluation JSON results")
    eval_p.set_defaults(func=cmd_evaluate)

    # Cost analysis command
    cost_p = subparsers.add_parser("cost-analysis", help="Show cost breakdown and scaling projections")
    cost_p.add_argument("--prompt-tokens", type=int, default=1200, help="Average prompt tokens per query")
    cost_p.add_argument("--completion-tokens", type=int, default=250, help="Average completion tokens per query")
    cost_p.add_argument("--ingestion-tokens", type=int, default=45000, help="Total ingestion tokens")
    cost_p.add_argument("--model", default=settings.OPENAI_MODEL, help="LLM model name")
    cost_p.add_argument("--embedding-model", default=settings.OPENAI_EMBEDDING_MODEL, help="Embedding model name")
    cost_p.set_defaults(func=cmd_cost_analysis)

    parsed_args = parser.parse_args()
    parsed_args.func(parsed_args)


if __name__ == "__main__":
    main()
