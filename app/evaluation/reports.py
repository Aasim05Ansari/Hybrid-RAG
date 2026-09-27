from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from app.evaluation.evaluator import EvaluationSummary


class EvaluationReport:
    """Formats and persists an EvaluationSummary."""

    def __init__(self, summary: EvaluationSummary) -> None:
        if summary is None:
            raise ValueError("summary is required")

        self.summary = summary

    def to_dict(self) -> dict:
        """Convert the evaluation summary into a serializable dictionary."""

        return asdict(self.summary)

    def to_json(self, indent: int = 2) -> str:
        """Serialize the evaluation summary as JSON."""

        if indent < 0:
            raise ValueError("indent cannot be negative")

        return json.dumps(
            self.to_dict(),
            indent=indent,
        )

    def save_json(
        self,
        path: str | Path,
        indent: int = 2,
    ) -> Path:
        """Save the evaluation report as a JSON file."""

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        path.write_text(
            self.to_json(indent=indent),
            encoding="utf-8",
        )

        return path

    def to_text(self) -> str:
        """Create a human-readable evaluation report."""

        summary = self.summary

        lines = [
            "RAG Evaluation Report",
            "=====================",
            "",
            "Retrieval Metrics",
            "-----------------",
            f"Recall@K: {summary.recall_at_k:.4f}",
            f"Precision@K: {summary.precision_at_k:.4f}",
            f"Mean Reciprocal Rank: "
            f"{summary.mean_reciprocal_rank:.4f}",
            "",
            "Grounding Metrics",
            "-----------------",
            f"Citation Accuracy: "
            f"{summary.citation_accuracy:.4f}",
            f"Citation Presence: "
            f"{summary.citation_presence:.4f}",
            f"Citation Coverage: "
            f"{summary.citation_coverage:.4f}",
            f"Answer Completeness: "
            f"{summary.answer_completeness:.4f}",
            "",
            "Confidence Metrics",
            "------------------",
            f"Retrieval Confidence: "
            f"{summary.retrieval_confidence:.4f}",
            f"Composite Confidence: "
            f"{summary.composite_confidence:.4f}",
            "",
            "Safety",
            "------",
            f"Abstention Rate: "
            f"{summary.abstention_rate:.4f}",
            "",
            "Performance",
            "-----------",
            f"Average Latency: "
            f"{summary.average_latency:.4f} seconds",
            "",
            f"Evaluated Cases: {len(summary.results)}",
        ]

        return "\n".join(lines)

    def save_text(
        self,
        path: str | Path,
    ) -> Path:
        """Save the human-readable report as a text file."""

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        path.write_text(
            self.to_text(),
            encoding="utf-8",
        )

        return path