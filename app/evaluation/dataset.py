from dataclasses import dataclass, field
from pathlib import Path
import json


@dataclass(frozen=True)
class EvaluationCase:
    """Represents one RAG evaluation question and its ground truth."""

    id: str
    question: str
    expected_answer: str
    expected_sources: list[str] = field(default_factory=list)
    expected_chunk_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("Evaluation case id cannot be empty")

        if not self.question.strip():
            raise ValueError("Evaluation case question cannot be empty")

        if not self.expected_answer.strip():
            raise ValueError("Evaluation case expected_answer cannot be empty")


class EvaluationDataset:
    """Collection of evaluation cases."""

    def __init__(self, cases: list[EvaluationCase]) -> None:
        if not cases:
            raise ValueError("Evaluation dataset cannot be empty")

        self.cases = cases

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self):
        return iter(self.cases)

    @classmethod
    def from_json(cls, path: str | Path) -> "EvaluationDataset":
        """Load an evaluation dataset from a JSON file."""

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Evaluation dataset not found: {path}")

        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, list):
            raise ValueError("Evaluation dataset JSON must contain a list")

        cases = [
            EvaluationCase(
                id=item["id"],
                question=item["question"],
                expected_answer=item["expected_answer"],
                expected_sources=item.get("expected_sources", []),
                expected_chunk_ids=item.get("expected_chunk_ids", []),
            )
            for item in data
        ]

        return cls(cases)