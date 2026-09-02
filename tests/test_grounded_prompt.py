import pytest

from app.generation.prompt import GroundedPromptBuilder


def test_grounded_prompt_contains_context_and_query():
    builder = GroundedPromptBuilder()

    prompt = builder.build(
        query="How many sick leave days are allowed?",
        context=(
            "[1]\n"
            "Source: leave.txt\n"
            "Section: Sick Leave\n"
            "Content:\n"
            "Employees may take up to 12 days of sick leave."
        ),
    )

    assert "How many sick leave days are allowed?" in prompt
    assert "12 days of sick leave" in prompt
    assert "[1]" in prompt


def test_grounded_prompt_contains_grounding_rules():
    builder = GroundedPromptBuilder()

    prompt = builder.build(
        query="What is the leave policy?",
        context="[1]\nContent:\nLeave policy information.",
    )

    assert "ONLY the provided context" in prompt
    assert "Do not use outside knowledge" in prompt
    assert "Do not invent" in prompt
    assert "enough information" in prompt
    assert "cite" in prompt.lower()


def test_grounded_prompt_rejects_empty_query():
    builder = GroundedPromptBuilder()

    with pytest.raises(ValueError, match="Query cannot be empty"):
        builder.build(
            query="",
            context="[1]\nContent:\nSome context.",
        )


def test_grounded_prompt_rejects_empty_context():
    builder = GroundedPromptBuilder()

    with pytest.raises(ValueError, match="Context cannot be empty"):
        builder.build(
            query="What is the policy?",
            context="",
        )
