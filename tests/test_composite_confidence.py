import pytest

from app.generation.composite_confidence import (
    CompositeConfidenceCalculator,
)


def test_perfect_scores_produce_perfect_confidence():
    result = CompositeConfidenceCalculator().calculate(
        retrieval_confidence=1.0,
        citation_accuracy=1.0,
        citation_coverage=1.0,
        answer_completeness=1.0,
    )

    assert result.score == 1.0


def test_zero_scores_produce_zero_confidence():
    result = CompositeConfidenceCalculator().calculate(
        retrieval_confidence=0.0,
        citation_accuracy=0.0,
        citation_coverage=0.0,
        answer_completeness=0.0,
    )

    assert result.score == 0.0


def test_weighted_calculation():
    result = CompositeConfidenceCalculator().calculate(
        retrieval_confidence=0.8,
        citation_accuracy=0.9,
        citation_coverage=0.7,
        answer_completeness=0.6,
    )

    expected = (
        0.30 * 0.8
        + 0.25 * 0.9
        + 0.25 * 0.7
        + 0.20 * 0.6
    )

    assert result.score == pytest.approx(expected)


def test_low_citation_accuracy_caps_confidence():
    result = CompositeConfidenceCalculator().calculate(
        retrieval_confidence=1.0,
        citation_accuracy=0.2,
        citation_coverage=1.0,
        answer_completeness=1.0,
    )

    assert result.score == 0.49


def test_citation_accuracy_at_floor_is_not_capped():
    result = CompositeConfidenceCalculator().calculate(
        retrieval_confidence=1.0,
        citation_accuracy=0.5,
        citation_coverage=1.0,
        answer_completeness=1.0,
    )

    assert result.score > 0.49


def test_scores_are_preserved():
    result = CompositeConfidenceCalculator().calculate(
        retrieval_confidence=0.8,
        citation_accuracy=0.9,
        citation_coverage=0.7,
        answer_completeness=0.6,
    )

    assert result.retrieval_confidence == 0.8
    assert result.citation_accuracy == 0.9
    assert result.citation_coverage == 0.7
    assert result.answer_completeness == 0.6


@pytest.mark.parametrize(
    "field",
    [
        "retrieval_confidence",
        "citation_accuracy",
        "citation_coverage",
        "answer_completeness",
    ],
)
def test_values_cannot_be_below_zero(field):
    values = {
        "retrieval_confidence": 0.5,
        "citation_accuracy": 0.5,
        "citation_coverage": 0.5,
        "answer_completeness": 0.5,
    }

    values[field] = -0.1

    with pytest.raises(ValueError):
        CompositeConfidenceCalculator().calculate(**values)


@pytest.mark.parametrize(
    "field",
    [
        "retrieval_confidence",
        "citation_accuracy",
        "citation_coverage",
        "answer_completeness",
    ],
)
def test_values_cannot_exceed_one(field):
    values = {
        "retrieval_confidence": 0.5,
        "citation_accuracy": 0.5,
        "citation_coverage": 0.5,
        "answer_completeness": 0.5,
    }

    values[field] = 1.1

    with pytest.raises(ValueError):
        CompositeConfidenceCalculator().calculate(**values)
