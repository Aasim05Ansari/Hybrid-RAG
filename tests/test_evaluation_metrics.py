import pytest

from app.evaluation.metrics import (
    abstention_rate,
    answer_completeness,
    average_latency,
    citation_accuracy,
    citation_coverage,
    citation_presence,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------


def test_recall_at_k():
    retrieved = ["chunk-1", "chunk-2", "chunk-3"]
    expected = ["chunk-1", "chunk-3"]

    assert recall_at_k(retrieved, expected, 2) == 0.5


def test_recall_at_k_all_relevant():
    retrieved = ["chunk-1", "chunk-2", "chunk-3"]
    expected = ["chunk-1", "chunk-2"]

    assert recall_at_k(retrieved, expected, 3) == 1.0


def test_recall_at_k_no_expected_ids():
    assert recall_at_k(["chunk-1"], [], 5) == 0.0


def test_recall_at_k_invalid_k():
    with pytest.raises(ValueError, match="greater than 0"):
        recall_at_k(["chunk-1"], ["chunk-1"], 0)


def test_precision_at_k():
    retrieved = ["chunk-1", "chunk-2", "chunk-3"]
    expected = ["chunk-1", "chunk-3"]

    assert precision_at_k(retrieved, expected, 2) == 0.5


def test_precision_at_k_fewer_results_than_k():
    retrieved = ["chunk-1"]
    expected = ["chunk-1"]

    assert precision_at_k(retrieved, expected, 5) == 1.0


def test_precision_at_k_no_results():
    assert precision_at_k([], ["chunk-1"], 5) == 0.0


def test_precision_at_k_invalid_k():
    with pytest.raises(ValueError, match="greater than 0"):
        precision_at_k(["chunk-1"], ["chunk-1"], 0)


def test_mean_reciprocal_rank():
    retrieved = ["chunk-3", "chunk-2", "chunk-1"]
    expected = ["chunk-1"]

    assert mean_reciprocal_rank(retrieved, expected) == pytest.approx(1 / 3)


def test_mean_reciprocal_rank_first_result():
    retrieved = ["chunk-1", "chunk-2", "chunk-3"]
    expected = ["chunk-1"]

    assert mean_reciprocal_rank(retrieved, expected) == 1.0


def test_mean_reciprocal_rank_no_match():
    retrieved = ["chunk-2", "chunk-3"]
    expected = ["chunk-1"]

    assert mean_reciprocal_rank(retrieved, expected) == 0.0


# ---------------------------------------------------------------------------
# Citation metrics
# ---------------------------------------------------------------------------


def test_citation_accuracy():
    assert citation_accuracy(
        supported_claims=3,
        cited_claims=4,
    ) == 0.75


def test_citation_accuracy_no_citations():
    assert citation_accuracy(0, 0) == 0.0


def test_citation_accuracy_invalid_counts():
    with pytest.raises(ValueError):
        citation_accuracy(5, 4)


def test_citation_presence():
    assert citation_presence(
        cited_claims=3,
        total_claims=4,
    ) == 0.75


def test_citation_presence_no_claims():
    assert citation_presence(0, 0) == 0.0


def test_citation_coverage():
    assert citation_coverage(
        supported_claims=3,
        total_claims=4,
    ) == 0.75


def test_citation_coverage_no_claims():
    assert citation_coverage(0, 0) == 0.0


# ---------------------------------------------------------------------------
# Answer completeness
# ---------------------------------------------------------------------------


def test_answer_completeness():
    assert answer_completeness(
        covered_terms=4,
        total_terms=5,
    ) == 0.8


def test_answer_completeness_complete():
    assert answer_completeness(5, 5) == 1.0


def test_answer_completeness_no_terms():
    assert answer_completeness(0, 0) == 1.0


# ---------------------------------------------------------------------------
# Abstention
# ---------------------------------------------------------------------------


def test_abstention_rate():
    assert abstention_rate(
        abstained_count=2,
        total_count=10,
    ) == 0.2


def test_abstention_rate_no_queries():
    assert abstention_rate(0, 0) == 0.0


def test_abstention_rate_invalid_counts():
    with pytest.raises(ValueError):
        abstention_rate(5, 4)


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------


def test_average_latency():
    assert average_latency([1.0, 2.0, 3.0]) == 2.0


def test_average_latency_empty():
    assert average_latency([]) == 0.0


def test_average_latency_rejects_negative_values():
    with pytest.raises(ValueError):
        average_latency([1.0, -0.5, 2.0])