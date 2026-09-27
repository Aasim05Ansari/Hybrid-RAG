import json

import pytest

from app.evaluation.evaluator import (
    EvaluationResult,
    EvaluationSummary,
)
from app.evaluation.reports import EvaluationReport


def make_result(
    case_id="q001",
    recall=1.0,
    precision=0.8,
    reciprocal_rank=1.0,
):
    return EvaluationResult(
        case_id=case_id,
        question="What is the leave policy?",
        answer="Employees may take leave. [1]",
        recall_at_k=recall,
        precision_at_k=precision,
        reciprocal_rank=reciprocal_rank,
        citation_accuracy=1.0,
        citation_presence=1.0,
        citation_coverage=1.0,
        answer_completeness=0.9,
        retrieval_confidence=0.8,
        composite_confidence=0.85,
        abstained=False,
        latency_seconds=0.25,
    )


def make_summary():
    results = [
        make_result("q001"),
        make_result("q002"),
    ]

    return EvaluationSummary(
        results=results,
        recall_at_k=0.9,
        precision_at_k=0.8,
        mean_reciprocal_rank=0.95,
        citation_accuracy=0.9,
        citation_presence=0.95,
        citation_coverage=0.85,
        answer_completeness=0.9,
        retrieval_confidence=0.8,
        composite_confidence=0.85,
        abstention_rate=0.1,
        average_latency=0.25,
    )


def test_report_requires_summary():
    with pytest.raises(
        ValueError,
        match="summary is required",
    ):
        EvaluationReport(None)


def test_report_to_dict():
    summary = make_summary()
    report = EvaluationReport(summary)

    data = report.to_dict()

    assert isinstance(data, dict)

    assert data["recall_at_k"] == 0.9
    assert data["precision_at_k"] == 0.8
    assert data["mean_reciprocal_rank"] == 0.95

    assert data["citation_accuracy"] == 0.9
    assert data["citation_presence"] == 0.95
    assert data["citation_coverage"] == 0.85

    assert data["answer_completeness"] == 0.9
    assert data["retrieval_confidence"] == 0.8
    assert data["composite_confidence"] == 0.85

    assert data["abstention_rate"] == 0.1
    assert data["average_latency"] == 0.25

    assert len(data["results"]) == 2


def test_report_to_json():
    summary = make_summary()
    report = EvaluationReport(summary)

    output = report.to_json()

    data = json.loads(output)

    assert data["recall_at_k"] == 0.9
    assert len(data["results"]) == 2


def test_report_to_json_supports_custom_indent():
    summary = make_summary()
    report = EvaluationReport(summary)

    output = report.to_json(indent=4)

    assert output.startswith("{\n")
    assert "    \"results\"" in output


def test_report_rejects_negative_indent():
    summary = make_summary()
    report = EvaluationReport(summary)

    with pytest.raises(
        ValueError,
        match="indent cannot be negative",
    ):
        report.to_json(indent=-1)


def test_report_saves_json(tmp_path):
    summary = make_summary()
    report = EvaluationReport(summary)

    output_path = tmp_path / "reports" / "evaluation.json"

    returned_path = report.save_json(output_path)

    assert returned_path == output_path
    assert output_path.exists()

    data = json.loads(
        output_path.read_text(encoding="utf-8")
    )

    assert data["recall_at_k"] == 0.9
    assert len(data["results"]) == 2


def test_report_to_text():
    summary = make_summary()
    report = EvaluationReport(summary)

    output = report.to_text()

    assert "RAG Evaluation Report" in output

    assert "Recall@K: 0.9000" in output
    assert "Precision@K: 0.8000" in output
    assert "Mean Reciprocal Rank: 0.9500" in output

    assert "Citation Accuracy: 0.9000" in output
    assert "Citation Presence: 0.9500" in output
    assert "Citation Coverage: 0.8500" in output

    assert "Answer Completeness: 0.9000" in output

    assert "Retrieval Confidence: 0.8000" in output
    assert "Composite Confidence: 0.8500" in output

    assert "Abstention Rate: 0.1000" in output
    assert "Average Latency: 0.2500 seconds" in output

    assert "Evaluated Cases: 2" in output


def test_report_saves_text(tmp_path):
    summary = make_summary()
    report = EvaluationReport(summary)

    output_path = tmp_path / "evaluation.txt"

    returned_path = report.save_text(output_path)

    assert returned_path == output_path
    assert output_path.exists()

    content = output_path.read_text(
        encoding="utf-8"
    )

    assert "RAG Evaluation Report" in content
    assert "Recall@K: 0.9000" in content