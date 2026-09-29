"""Automated evaluation runner for assessing RAG retrieval, grounding, and rejection."""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from rich.console import Console
from rich.table import Table

from src.agent.graph import GroundedRAGAgent
from src.evaluation.dataset import EVAL_QUESTIONS, EvalSample

console = Console()


class EvaluationRunner:
    """Executes the benchmark evaluation suite and produces a performance scorecard."""

    def __init__(self, agent: Optional[GroundedRAGAgent] = None):
        self.agent = agent or GroundedRAGAgent()

    def evaluate_sample(self, sample: EvalSample) -> Dict:
        """Run single evaluation sample through the RAG pipeline."""
        start_time = time.perf_counter()
        result = self.agent.ask(sample.question)
        latency_sec = time.perf_counter() - start_time

        answer = result.get("answer", "")
        sources = result.get("sources", [])
        source_urls = [s.get("url", "") for s in sources]
        token_usage = result.get("token_usage", {})
        cost_usd = result.get("cost_usd", 0.0)

        # 1. Assess Refusal / Rejection
        refusal_markers = [
            "cannot find sufficient information",
            "not enough information",
            "does not mention",
            "not support",
            "does not require",
            "not mandatory",
            "not available",
            "no information",
        ]
        is_refusal = any(m in answer.lower() for m in refusal_markers) or len(sources) == 0

        refusal_correct = (is_refusal == sample.expected_refusal) or (
            sample.category == "misleading" and any(k.lower() in answer.lower() for k in sample.expected_answer_keywords)
        )

        # 2. Assess Retrieval Recall
        retrieval_hit = False
        if sample.expected_refusal:
            # For unanswerable, not retrieving or retrieving low-relevance docs is acceptable
            retrieval_hit = True
        else:
            for url in source_urls:
                if any(kw in url for kw in sample.expected_url_keywords):
                    retrieval_hit = True
                    break

        # 3. Assess Answer Content Quality
        keyword_hits = sum(
            1 for kw in sample.expected_answer_keywords if kw.lower() in answer.lower()
        )
        content_pass = (
            keyword_hits > 0 if sample.expected_answer_keywords else True
        )

        # Overall sample passed
        sample_passed = refusal_correct and (sample.expected_refusal or retrieval_hit)

        return {
            "id": sample.id,
            "category": sample.category,
            "question": sample.question,
            "description": sample.description,
            "expected_refusal": sample.expected_refusal,
            "actual_refusal": is_refusal,
            "refusal_correct": refusal_correct,
            "retrieval_hit": retrieval_hit,
            "source_urls": source_urls,
            "answer_preview": answer[:220] + "..." if len(answer) > 220 else answer,
            "answer_full": answer,
            "sample_passed": sample_passed,
            "latency_sec": round(latency_sec, 3),
            "tokens": token_usage.get("total_tokens", 0),
            "cost_usd": cost_usd,
        }

    def run_suite(self, output_path: Optional[Path] = None) -> Dict:
        """Run the full 12-question evaluation suite."""
        console.print("[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]")
        console.print("[bold cyan]       Running Website-Grounded RAG Evaluation Suite        [/bold cyan]")
        console.print("[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]\n")

        results: List[Dict] = []
        for sample in EVAL_QUESTIONS:
            console.print(f"Evaluating Q{sample.id} [{sample.category.upper()}]: {sample.question[:65]}...")
            sample_res = self.evaluate_sample(sample)
            status_symbol = "[bold green]PASS ✓[/bold green]" if sample_res["sample_passed"] else "[bold red]FAIL ✗[/bold red]"
            console.print(f"  Result: {status_symbol} (Latency: {sample_res['latency_sec']}s, Cost: ${sample_res['cost_usd']:.6f})")
            results.append(sample_res)

        # Compute aggregate metrics
        total = len(results)
        passed = sum(1 for r in results if r["sample_passed"])
        pass_rate = (passed / total) * 100.0 if total > 0 else 0.0

        refusal_tests = [r for r in results if r["expected_refusal"]]
        refusal_correct = sum(1 for r in refusal_tests if r["refusal_correct"])
        refusal_rate = (refusal_correct / len(refusal_tests)) * 100.0 if refusal_tests else 0.0

        answerable_tests = [r for r in results if not r["expected_refusal"]]
        retrieval_hits = sum(1 for r in answerable_tests if r["retrieval_hit"])
        retrieval_recall = (retrieval_hits / len(answerable_tests)) * 100.0 if answerable_tests else 0.0

        avg_latency = sum(r["latency_sec"] for r in results) / total if total > 0 else 0.0
        avg_tokens = sum(r["tokens"] for r in results) / total if total > 0 else 0.0
        total_cost = sum(r["cost_usd"] for r in results)

        # Categorical Breakdown
        categories = ["straightforward", "paraphrased", "multi-page", "misleading", "unanswerable"]
        category_stats = {}
        for cat in categories:
            cat_results = [r for r in results if r["category"] == cat]
            if cat_results:
                cat_passed = sum(1 for r in cat_results if r["sample_passed"])
                category_stats[cat] = {
                    "total": len(cat_results),
                    "passed": cat_passed,
                    "pass_rate": (cat_passed / len(cat_results)) * 100.0,
                }

        summary = {
            "total_questions": total,
            "passed_questions": passed,
            "overall_pass_rate_pct": round(pass_rate, 2),
            "refusal_accuracy_pct": round(refusal_rate, 2),
            "retrieval_recall_pct": round(retrieval_recall, 2),
            "avg_latency_sec": round(avg_latency, 3),
            "avg_tokens_per_query": round(avg_tokens, 1),
            "total_eval_cost_usd": round(total_cost, 6),
            "category_breakdown": category_stats,
            "detailed_results": results,
        }

        # Print Rich Summary Table
        self._print_results_table(summary)

        # Save to JSON
        save_path = output_path or Path("eval_results.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        console.print(f"\n[green]Saved evaluation results to {save_path}[/green]")

        return summary

    def _print_results_table(self, summary: Dict) -> None:
        """Render a formatted Rich table with evaluation summary."""
        console.print("\n[bold magenta]Evaluation Summary by Category:[/bold magenta]")
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Category", style="yellow")
        table.add_column("Total", justify="right")
        table.add_column("Passed", justify="right")
        table.add_column("Pass Rate", justify="right", style="bold green")

        for cat, stats in summary["category_breakdown"].items():
            table.add_row(
                cat.capitalize(),
                str(stats["total"]),
                str(stats["passed"]),
                f"{stats['pass_rate']:.1f}%"
            )
        console.print(table)

        console.print(f"\n[bold]Overall Pass Rate:[/bold] [bold green]{summary['overall_pass_rate_pct']}%[/bold green]")
        console.print(f"[bold]Grounding / Refusal Accuracy:[/bold] [bold green]{summary['refusal_accuracy_pct']}%[/bold green]")
        console.print(f"[bold]Retrieval Recall (Answerable):[/bold] [bold green]{summary['retrieval_recall_pct']}%[/bold green]")
        console.print(f"[bold]Average Latency:[/bold] {summary['avg_latency_sec']}s")
        console.print(f"[bold]Average Tokens / Query:[/bold] {summary['avg_tokens_per_query']}")
        console.print(f"[bold]Total Evaluation Cost:[/bold] ${summary['total_eval_cost_usd']:.6f}")
