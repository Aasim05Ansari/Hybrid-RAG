from types import SimpleNamespace

import pytest

from app.evaluation.dataset import EvaluationCase, EvaluationDataset
from app.evaluation.evaluator import (
    EvaluationResult,
    EvaluationSummary,
    RAGEvaluator,
)


class FakeRetrievalResult:
    def __init__(self, chunk_id: str):
        self.chunk_id = chunk_id


class FakeRAGPipeline:
    def __init__(self, grounded_answer):
        self.grounded_answer = grounded_answer
        self.queries = []

    def run(self, query: str):
        self.queries.append(query)
        return self.grounded_answer


def make_grounded_answer(
    retrieved_ids=None,
    answer="Employees may take 12 days of sick leave. [1]",
    retrieval_confidence=0.8,
    composite_confidence=0.85,
    supported_claims=1,
    cited_claims=1,
    total_claims=1,
    completeness=1.0,
    abstained=False,
):
    retrieved_ids = retrieved_ids or ["chunk-1"]

    return SimpleNamespace(
        query="What is the sick leave policy?",
        answer=answer,
        retrieval_results=[
            FakeRetrievalResult(chunk_id)
            for chunk_id in retrieved_ids
        ],
        retrieval_confidence=retrieval_confidence,
        citation_evaluation=SimpleNamespace(
            supported_claims=supported_claims,
            cited_claims=cited_claims,
            total_claims=total_claims,
        ),
        answer_completeness=SimpleNamespace(
            score=completeness,
        ),
        composite_confidence=SimpleNamespace(
            score=composite_confidence,
        ),
        abstention=SimpleNamespace(
            abstained=abstained,
        ),
    )


def make_case(
    case_id="q001",
    expected_chunk_ids=None,
):
    return EvaluationCase(
        id=case_id,
        question="What is the sick leave policy?",
        expected_answer="Employees may take 12 days of sick leave.",
        expected_sources=["leave.txt"],
        expected_chunk_ids=expected_chunk_ids or ["chunk-1"],
    )


def test_evaluator_rejects_missing_pipeline():
    with pytest.raises(
        ValueError,
        match="rag_pipeline is required",
    ):
        RAGEvaluator(None)


def test_evaluator_rejects_invalid_k():
    pipeline = FakeRAGPipeline(
        make_grounded_answer()
    )

    with pytest.raises(
        ValueError,
        match="k must be greater than 0",
    ):
        RAGEvaluator(pipeline, k=0)


def test_evaluator_evaluates_single_case():
    grounded_answer = make_grounded_answer(
        retrieved_ids=["chunk-1", "chunk-2"],
    )

    pipeline = FakeRAGPipeline(grounded_answer)
    evaluator = RAGEvaluator(pipeline, k=2)

    case = make_case(
        expected_chunk_ids=["chunk-1"],
    )

    result = evaluator.evaluate_case(case)

    assert isinstance(result, EvaluationResult)

    assert result.case_id == "q001"
    assert result.answer == (
        "Employees may take 12 days of sick leave. [1]"
    )

    assert result.recall_at_k == 1.0
    assert result.precision_at_k == 0.5
    assert result.reciprocal_rank == 1.0

    assert result.citation_accuracy == 1.0
    assert result.citation_presence == 1.0
    assert result.citation_coverage == 1.0

    assert result.answer_completeness == 1.0
    assert result.retrieval_confidence == 0.8
    assert result.composite_confidence == 0.85
    assert result.abstained is False

    assert result.latency_seconds >= 0.0


def test_evaluator_passes_question_to_pipeline():
    grounded_answer = make_grounded_answer()

    pipeline = FakeRAGPipeline(grounded_answer)
    evaluator = RAGEvaluator(pipeline)

    case = make_case()

    evaluator.evaluate_case(case)

    assert pipeline.queries == [
        "What is the sick leave policy?"
    ]


def test_evaluator_handles_missing_relevant_result():
    grounded_answer = make_grounded_answer(
        retrieved_ids=["chunk-2", "chunk-3"],
    )

    pipeline = FakeRAGPipeline(grounded_answer)
    evaluator = RAGEvaluator(pipeline, k=2)

    case = make_case(
        expected_chunk_ids=["chunk-1"],
    )

    result = evaluator.evaluate_case(case)

    assert result.recall_at_k == 0.0
    assert result.precision_at_k == 0.0
    assert result.reciprocal_rank == 0.0


def test_evaluator_reads_grounding_signals():
    grounded_answer = make_grounded_answer(
        supported_claims=2,
        cited_claims=3,
        total_claims=4,
        completeness=0.75,
        retrieval_confidence=0.65,
        composite_confidence=0.72,
        abstained=True,
    )

    pipeline = FakeRAGPipeline(grounded_answer)
    evaluator = RAGEvaluator(pipeline)

    result = evaluator.evaluate_case(make_case())

    assert result.citation_accuracy == pytest.approx(2 / 3)
    assert result.citation_presence == pytest.approx(3 / 4)
    assert result.citation_coverage == pytest.approx(2 / 4)

    assert result.answer_completeness == 0.75
    assert result.retrieval_confidence == 0.65
    assert result.composite_confidence == 0.72
    assert result.abstained is True


def test_evaluator_rejects_missing_case():
    pipeline = FakeRAGPipeline(
        make_grounded_answer()
    )

    evaluator = RAGEvaluator(pipeline)

    with pytest.raises(
        ValueError,
        match="case is required",
    ):
        evaluator.evaluate_case(None)


def test_evaluator_rejects_missing_dataset():
    pipeline = FakeRAGPipeline(
        make_grounded_answer()
    )

    evaluator = RAGEvaluator(pipeline)

    with pytest.raises(
        ValueError,
        match="dataset is required",
    ):
        evaluator.evaluate(None)


def test_evaluator_returns_dataset_summary():
    grounded_answer = make_grounded_answer(
        retrieved_ids=["chunk-1"],
    )

    pipeline = FakeRAGPipeline(grounded_answer)
    evaluator = RAGEvaluator(pipeline, k=1)

    dataset = EvaluationDataset(
        [
            make_case("q001", ["chunk-1"]),
            make_case("q002", ["chunk-1"]),
        ]
    )

    summary = evaluator.evaluate(dataset)

    assert isinstance(summary, EvaluationSummary)

    assert len(summary.results) == 2

    assert summary.recall_at_k == 1.0
    assert summary.precision_at_k == 1.0
    assert summary.mean_reciprocal_rank == 1.0

    assert summary.citation_accuracy == 1.0
    assert summary.citation_presence == 1.0
    assert summary.citation_coverage == 1.0
    assert summary.answer_completeness == 1.0

    assert summary.retrieval_confidence == 0.8
    assert summary.composite_confidence == 0.85

    assert summary.abstention_rate == 0.0
    assert summary.average_latency >= 0.0


def test_evaluator_calculates_abstention_rate():
    first = make_grounded_answer(abstained=True)
    second = make_grounded_answer(abstained=False)

    class SequentialFakePipeline:
        def __init__(self):
            self.answers = [first, second]
            self.index = 0

        def run(self, query):
            answer = self.answers[self.index]
            self.index += 1
            return answer

    pipeline = SequentialFakePipeline()
    evaluator = RAGEvaluator(pipeline)

    dataset = EvaluationDataset(
        [
            make_case("q001"),
            make_case("q002"),
        ]
    )

    summary = evaluator.evaluate(dataset)

    assert summary.abstention_rate == 0.5