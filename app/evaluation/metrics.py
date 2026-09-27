from __future__ import annotations

from collections.abc import Sequence


def recall_at_k(
    retrieved_ids: Sequence[str],
    expected_ids: Sequence[str],
    k: int,
) -> float:
    """Calculate Recall@K.

    Recall@K = relevant retrieved items in top K / total relevant items.

    Returns 0.0 when there are no expected IDs.
    """

    if k <= 0:
        raise ValueError("k must be greater than 0")

    expected = set(expected_ids)

    if not expected:
        return 0.0

    retrieved = set(retrieved_ids[:k])

    return len(retrieved & expected) / len(expected)


def precision_at_k(
    retrieved_ids: Sequence[str],
    expected_ids: Sequence[str],
    k: int,
) -> float:
    """Calculate Precision@K.

    Precision@K = relevant retrieved items in top K / K.

    The denominator is limited to the number of actually retrieved
    items when fewer than K results are available.
    """

    if k <= 0:
        raise ValueError("k must be greater than 0")

    retrieved = list(retrieved_ids[:k])

    if not retrieved:
        return 0.0

    expected = set(expected_ids)

    if not expected:
        return 0.0

    relevant = sum(item in expected for item in retrieved)

    return relevant / len(retrieved)


def mean_reciprocal_rank(
    retrieved_ids: Sequence[str],
    expected_ids: Sequence[str],
) -> float:
    """Calculate Reciprocal Rank for a single query.

    Returns the reciprocal of the rank of the first relevant result.
    Returns 0.0 when no relevant result is retrieved.
    """

    expected = set(expected_ids)

    if not expected:
        return 0.0

    for rank, item_id in enumerate(retrieved_ids, start=1):
        if item_id in expected:
            return 1.0 / rank

    return 0.0


def citation_accuracy(
    supported_claims: int,
    cited_claims: int,
) -> float:
    """Calculate citation accuracy.

    Accuracy = supported cited claims / cited claims.
    """

    if supported_claims < 0 or cited_claims < 0:
        raise ValueError("Claim counts cannot be negative")

    if supported_claims > cited_claims:
        raise ValueError(
            "supported_claims cannot exceed cited_claims"
        )

    if cited_claims == 0:
        return 0.0

    return supported_claims / cited_claims


def citation_presence(
    cited_claims: int,
    total_claims: int,
) -> float:
    """Calculate the proportion of claims that contain citations."""

    if cited_claims < 0 or total_claims < 0:
        raise ValueError("Claim counts cannot be negative")

    if cited_claims > total_claims:
        raise ValueError(
            "cited_claims cannot exceed total_claims"
        )

    if total_claims == 0:
        return 0.0

    return cited_claims / total_claims


def citation_coverage(
    supported_claims: int,
    total_claims: int,
) -> float:
    """Calculate the proportion of all claims supported by evidence."""

    if supported_claims < 0 or total_claims < 0:
        raise ValueError("Claim counts cannot be negative")

    if supported_claims > total_claims:
        raise ValueError(
            "supported_claims cannot exceed total_claims"
        )

    if total_claims == 0:
        return 0.0

    return supported_claims / total_claims


def answer_completeness(
    covered_terms: int,
    total_terms: int,
) -> float:
    """Calculate answer completeness."""

    if covered_terms < 0 or total_terms < 0:
        raise ValueError("Term counts cannot be negative")

    if covered_terms > total_terms:
        raise ValueError(
            "covered_terms cannot exceed total_terms"
        )

    if total_terms == 0:
        return 1.0

    return covered_terms / total_terms


def abstention_rate(
    abstained_count: int,
    total_count: int,
) -> float:
    """Calculate the proportion of queries where the system abstained."""

    if abstained_count < 0 or total_count < 0:
        raise ValueError("Counts cannot be negative")

    if abstained_count > total_count:
        raise ValueError(
            "abstained_count cannot exceed total_count"
        )

    if total_count == 0:
        return 0.0

    return abstained_count / total_count


def average_latency(
    latencies: Sequence[float],
) -> float:
    """Calculate average latency in seconds."""

    if not latencies:
        return 0.0

    if any(latency < 0 for latency in latencies):
        raise ValueError("Latency values cannot be negative")

    return sum(latencies) / len(latencies)