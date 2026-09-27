from __future__ import annotations

import argparse
from pathlib import Path

from app.evaluation.dataset import EvaluationDataset
from app.evaluation.evaluator import RAGEvaluator
from app.evaluation.reports import EvaluationReport
from app.indexing.bm25_store import BM25Store
from app.pipeline.rag_pipeline import RAGPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the Hybrid-RAG pipeline."
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to the evaluation dataset JSON file.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Directory where evaluation reports are saved.",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="K value used for Recall@K and Precision@K.",
    )

    return parser.parse_args()


def build_pipeline() -> RAGPipeline:
    """Build the production RAG pipeline used for evaluation."""

    bm25_store = BM25Store()

    return RAGPipeline.create(
        bm25_store=bm25_store,
    )


def main() -> None:
    args = parse_args()

    if args.k <= 0:
        raise ValueError("--k must be greater than 0")

    dataset = EvaluationDataset.from_json(
        args.dataset
    )

    rag_pipeline = build_pipeline()

    evaluator = RAGEvaluator(
        rag_pipeline=rag_pipeline,
        k=args.k,
    )

    summary = evaluator.evaluate(dataset)

    report = EvaluationReport(summary)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = report.save_json(
        args.output_dir / "evaluation.json"
    )

    text_path = report.save_text(
        args.output_dir / "evaluation.txt"
    )

    print(report.to_text())

    print()
    print("Reports saved:")
    print(f"  JSON: {json_path}")
    print(f"  Text: {text_path}")


if __name__ == "__main__":
    main()