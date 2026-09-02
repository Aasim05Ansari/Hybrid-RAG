import pytest

from app.generation.llm import (
    LLMProvider,
    OpenAILLMProvider,
)


class FakeLLMProvider(LLMProvider):

    def __init__(self, response="Test answer [1]"):
        self.response = response
        self.received_prompt = None

    def generate(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        self.received_prompt = prompt
        return self.response


def test_fake_llm_provider_generates_answer():
    provider = FakeLLMProvider(
        response="Employees get 12 sick leave days. [1]"
    )

    answer = provider.generate(
        "How many sick leave days?"
    )

    assert answer == (
        "Employees get 12 sick leave days. [1]"
    )


def test_fake_llm_provider_receives_grounded_prompt():
    provider = FakeLLMProvider()

    prompt = (
        "Answer using only the supplied context.\n"
        "[1] Sick leave is 12 days."
    )

    provider.generate(prompt)

    assert provider.received_prompt == prompt


def test_fake_llm_provider_rejects_empty_prompt():
    provider = FakeLLMProvider()

    with pytest.raises(
        ValueError,
        match="Prompt cannot be empty",
    ):
        provider.generate("")


def test_openai_provider_is_llm_provider():
    assert issubclass(
        OpenAILLMProvider,
        LLMProvider,
    )
