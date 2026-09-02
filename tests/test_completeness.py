import pytest

from app.generation.completeness import (
    AnswerCompletenessEvaluator,
)


def test_complete_answer():
    result = AnswerCompletenessEvaluator().evaluate(
        "What is the annual leave allowance?",
        "Employees receive 24 days of annual leave.",
    )

    assert result.score == 1.0
    assert result.missing_terms == []


def test_partially_complete_answer():
    result = AnswerCompletenessEvaluator().evaluate(
        "What are the annual leave and sick leave allowances?",
        "Employees receive 24 days of annual leave.",
    )

    assert 0.0 < result.score < 1.0
    assert "sick" in result.missing_terms


def test_empty_answer():
    result = AnswerCompletenessEvaluator().evaluate(
        "What is the annual leave allowance?",
        "",
    )

    assert result.score == 0.0
    assert result.covered_terms == []
    assert "annual" in result.missing_terms
    assert "leave" in result.missing_terms
    assert "allowance" in result.missing_terms


def test_unrelated_answer():
    result = AnswerCompletenessEvaluator().evaluate(
        "What is the annual leave allowance?",
        "The office is located in Bangalore.",
    )

    assert result.score == 0.0


def test_stop_words_are_ignored():
    result = AnswerCompletenessEvaluator().evaluate(
        "What is the annual leave?",
        "Annual leave is 24 days.",
    )

    assert result.score == 1.0


def test_empty_question_is_rejected():
    with pytest.raises(ValueError):
        AnswerCompletenessEvaluator().evaluate(
            "",
            "Employees receive 24 days of annual leave.",
        )
