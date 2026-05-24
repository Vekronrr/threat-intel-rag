"""
RAGAS evaluation for VulnMind RAG system.
Metrics: faithfulness, answer_relevancy, context_precision, context_recall.
Outputs a scorecard table.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import structlog
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from rag.retriever import HybridRetriever
from rag.reranker import CrossEncoderReranker
from graph.risk_scorer import RiskScorer
from rag.prompt_builder import build_prompt
from openai import AsyncOpenAI

logger = structlog.get_logger(__name__)

TEST_QUERIES_PATH = Path(__file__).parent / "test_queries.json"
RESULTS_PATH = Path("./eval/benchmark_results.json")


async def _answer_question(
    question: str,
    retriever: HybridRetriever,
    reranker: CrossEncoderReranker,
    scorer: RiskScorer | None,
    oai: AsyncOpenAI,
) -> tuple[str, list[str]]:
    """Run retrieval + LLM for a single query, return (answer, contexts)."""
    chunks = await retriever.retrieve(question)
    reranked = reranker.rerank(question, chunks, top_n=5)

    risk_scores = {}
    if scorer:
        for chunk in reranked:
            cve_id = chunk.metadata.get("cve_id", "")
            if cve_id.startswith("CVE-"):
                risk_scores[cve_id] = scorer.score(cve_id)

    system_prompt, user_message = build_prompt(question, reranked, risk_scores)

    resp = await oai.chat.completions.create(
        model=os.getenv("LLM_MODEL", "gpt-4o"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0,
    )
    answer = resp.choices[0].message.content
    contexts = [c.text for c in reranked]
    return answer, contexts


async def run_benchmark(
    test_queries_path: Path = TEST_QUERIES_PATH,
    output_path: Path = RESULTS_PATH,
    max_queries: int | None = None,
) -> dict[str, Any]:
    """Run full RAGAS benchmark. Returns metric scores."""
    from ingestion.embedder import CHROMA_COLLECTION_CVES

    oai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    retriever = HybridRetriever(CHROMA_COLLECTION_CVES)
    reranker = CrossEncoderReranker()

    graph_path = Path("./data/vulnmind_graph.graphml")
    scorer = RiskScorer(graph_path) if graph_path.exists() else None

    queries = json.loads(test_queries_path.read_text())
    if max_queries:
        queries = queries[:max_queries]

    logger.info("Running benchmark", queries=len(queries))

    questions, answers, contexts, ground_truths = [], [], [], []

    for i, q in enumerate(queries):
        logger.info("Evaluating query", id=q["id"], question=q["question"][:60])
        try:
            answer, ctx = await _answer_question(q["question"], retriever, reranker, scorer, oai)
            questions.append(q["question"])
            answers.append(answer)
            contexts.append(ctx)
            ground_truths.append(q["ground_truth"])
        except Exception as e:
            logger.error("Query failed", id=q["id"], error=str(e))
            # Add placeholder so dataset stays aligned
            questions.append(q["question"])
            answers.append("Error during evaluation")
            contexts.append([""])
            ground_truths.append(q["ground_truth"])

        # Rate limit: 2s between queries
        await asyncio.sleep(2)

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    logger.info("Running RAGAS metrics")
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    scores = {
        "faithfulness": float(result["faithfulness"]),
        "answer_relevancy": float(result["answer_relevancy"]),
        "context_precision": float(result["context_precision"]),
        "context_recall": float(result["context_recall"]),
        "composite": float(
            (result["faithfulness"] + result["answer_relevancy"]
             + result["context_precision"] + result["context_recall"]) / 4
        ),
    }

    scorecard = {
        "scores": scores,
        "n_queries": len(queries),
        "per_query": [
            {
                "id": queries[i]["id"],
                "question": questions[i][:80],
                "answer_preview": answers[i][:200],
            }
            for i in range(len(queries))
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(scorecard, indent=2))

    _print_scorecard(scores)
    return scores


def _print_scorecard(scores: dict[str, float]) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="VulnMind RAG Benchmark Scorecard", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Score", style="green")
    table.add_column("Grade", style="yellow")

    def grade(s: float) -> str:
        if s >= 0.85:
            return "A"
        if s >= 0.75:
            return "B"
        if s >= 0.65:
            return "C"
        return "F"

    for metric, score in scores.items():
        table.add_row(metric.replace("_", " ").title(), f"{score:.4f}", grade(score))

    console.print(table)


if __name__ == "__main__":
    asyncio.run(run_benchmark())
