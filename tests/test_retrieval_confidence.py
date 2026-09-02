from app.generation.retrieval_confidence import (
    RetrievalConfidenceCalculator,
)
from app.retrieval.models import RetrievalResult


def make_result(score):
    return RetrievalResult(
        chunk_id=f"chunk-{score}",
        content="Some document content",
        metadata={"source": "document.pdf"},
        score=score,
    )


def test_empty_results_have_zero_confidence():
    result = RetrievalConfidenceCalculator().calculate([])

    assert result.score == 0.0
    assert result.result_count == 0
    assert result.top_score == 0.0
    assert result.score_spread == 0.0


def test_single_strong_result():
    results = [
        make_result(1.0),
    ]

    result = RetrievalConfidenceCalculator().calculate(results)

    assert result.result_count == 1
    assert result.top_score == 1.0
    assert result.score_spread == 0.0
    assert 0.0 <= result.score <= 1.0


def test_multiple_results():
    results = [
        make_result(0.9),
        make_result(0.7),
        make_result(0.5),
    ]

    result = RetrievalConfidenceCalculator().calculate(results)

    assert result.result_count == 3
    assert result.top_score == 0.9
    assert result.score_spread == 0.4
    assert 0.0 <= result.score <= 1.0


def test_stronger_top_result_increases_confidence():
    calculator = RetrievalConfidenceCalculator()

    weak = calculator.calculate(
        [
            make_result(0.4),
            make_result(0.3),
        ]
    )

    strong = calculator.calculate(
        [
            make_result(0.9),
            make_result(0.3),
        ]
    )

    assert strong.score > weak.score


def test_more_results_do_not_exceed_one():
    results = [
        make_result(1.0),
        make_result(0.9),
        make_result(0.8),
        make_result(0.7),
        make_result(0.6),
        make_result(0.5),
    ]

    result = RetrievalConfidenceCalculator().calculate(results)

    assert 0.0 <= result.score <= 1.0


def test_scores_are_clamped():
    results = [
        make_result(5.0),
        make_result(-2.0),
    ]

    result = RetrievalConfidenceCalculator().calculate(results)

    assert 0.0 <= result.score <= 1.0
