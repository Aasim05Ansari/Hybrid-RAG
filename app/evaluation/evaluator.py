from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from app.evaluation.dataset import EvaluationCase, EvaluationDataset
from app.evaluation.metrics import (
    answer_completeness,
    average_latency,
    citation_accuracy,
    citation_coverage,
    citation_presence,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)
from app.pipeline.rag_pipeline import RAGPipeline


@dataclass(frozen=True)
class EvaluationResult:
    """Evaluation result for a single question."""

    case_id: str
    question: str
    answer: str

    recall_at_k: float
    precision_at_k: float
    reciprocal_rank: float

    citation_accuracy: float
    citation_presence: float
    citation_coverage: float
    answer_completeness: float

    retrieval_confidence: float
    composite_confidence: float

    abstained: bool
    latency_seconds: float


@dataclass(frozen=True)
class EvaluationSummary:
    """Aggregated evaluation results for a dataset."""

    results: list[EvaluationResult]

    recall_at_k: float
    precision_at_k: float
    mean_reciprocal_rank: float

    citation_accuracy: float
    citation_presence: float
    citation_coverage: float
    answer_completeness: float

    retrieval_confidence: float
    composite_confidence: float

    abstention_rate: float
    average_latency: float


class RAGEvaluator:
    """Evaluates a RAGPipeline against an EvaluationDataset."""

    def __init__(self, rag_pipeline: RAGPipeline, k: int = 5) -> None:
        if rag_pipeline is None:
            raise ValueError("rag_pipeline is required")

        if k <= 0:
            raise ValueError("k must be greater than 0")

        self.rag_pipeline = rag_pipeline
        self.k = k

    def evaluate_case(self, case: EvaluationCase) -> EvaluationResult:
        """Evaluate one question."""

        if case is None:
            raise ValueError("case is required")

        start = perf_counter()

        grounded_answer = self.rag_pipeline.run(case.question)

        latency = perf_counter() - start

        retrieved_ids = [
            result.chunk_id
            for result in grounded_answer.retrieval_results
        ]

        expected_ids = case.expected_chunk_ids

        retrieval_recall = recall_at_k(
            retrieved_ids,
            expected_ids,
            self.k,
        )

        retrieval_precision = precision_at_k(
            retrieved_ids,
            expected_ids,
            self.k,
        )

        reciprocal_rank = mean_reciprocal_rank(
            retrieved_ids,
            expected_ids,
        )

        citation_evaluation = grounded_answer.citation_evaluation

        citation_acc = citation_accuracy(
            citation_evaluation.supported_claims,
            citation_evaluation.cited_claims,
        )

        citation_pres = citation_presence(
            citation_evaluation.cited_claims,
            citation_evaluation.total_claims,
        )

        citation_cov = citation_coverage(
            citation_evaluation.supported_claims,
            citation_evaluation.total_claims,
        )

        completeness = grounded_answer.answer_completeness.score

        return EvaluationResult(
            case_id=case.id,
            question=case.question,
            answer=grounded_answer.answer,
            recall_at_k=retrieval_recall,
            precision_at_k=retrieval_precision,
            reciprocal_rank=reciprocal_rank,
            citation_accuracy=citation_acc,
            citation_presence=citation_pres,
            citation_coverage=citation_cov,
            answer_completeness=completeness,
            retrieval_confidence=grounded_answer.retrieval_confidence,
            composite_confidence=grounded_answer.composite_confidence.score,
            abstained=grounded_answer.abstention.abstained,
            latency_seconds=latency,
        )

    def evaluate(
        self,
        dataset: EvaluationDataset,
    ) -> EvaluationSummary:
        """Evaluate the complete dataset."""

        if dataset is None:
            raise ValueError("dataset is required")

        results = [
            self.evaluate_case(case)
            for case in dataset
        ]

        if not results:
            raise ValueError("Evaluation dataset cannot be empty")

        return EvaluationSummary(
            results=results,
            recall_at_k=sum(
                result.recall_at_k for result in results
            ) / len(results),
            precision_at_k=sum(
                result.precision_at_k for result in results
            ) / len(results),
            mean_reciprocal_rank=sum(
                result.reciprocal_rank for result in results
            ) / len(results),
            citation_accuracy=sum(
                result.citation_accuracy for result in results
            ) / len(results),
            citation_presence=sum(
                result.citation_presence for result in results
            ) / len(results),
            citation_coverage=sum(
                result.citation_coverage for result in results
            ) / len(results),
            answer_completeness=sum(
                result.answer_completeness for result in results
            ) / len(results),
            retrieval_confidence=sum(
                result.retrieval_confidence for result in results
            ) / len(results),
            composite_confidence=sum(
                result.composite_confidence for result in results
            ) / len(results),
            abstention_rate=sum(
                result.abstained for result in results
            ) / len(results),
            average_latency=average_latency(
                [result.latency_seconds for result in results]
            ),
        )