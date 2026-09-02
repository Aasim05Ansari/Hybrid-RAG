import pytest

from app.generation.context import ContextBuilder
from app.generation.llm import LLMProvider
from app.generation.pipeline import GenerationPipeline
from app.generation.prompt import GroundedPromptBuilder
from app.retrieval.models import RetrievalResult


class FakeRetrievalPipeline:

    def __init__(self):
        self.received_query = None

    def retrieve(self, query):
        self.received_query = query

        return [
            RetrievalResult(
                chunk_id="sick-leave",
                content=(
                    "Employees may take up to "
                    "12 days of sick leave."
                ),
                metadata={
                    "source": "leave.txt",
                    "section": "Sick Leave",
                },
                score=0.9,
            )
        ]


class FakeLLMProvider(LLMProvider):

    def __init__(self):
        self.received_prompt = None

    def generate(self, prompt):
        self.received_prompt = prompt

        return (
            "Employees may take up to "
            "12 days of sick leave. [1]"
        )


def create_pipeline():

    retrieval_pipeline = FakeRetrievalPipeline()
    llm_provider = FakeLLMProvider()

    pipeline = GenerationPipeline(
        retrieval_pipeline=retrieval_pipeline,
        context_builder=ContextBuilder(),
        prompt_builder=GroundedPromptBuilder(),
        llm_provider=llm_provider,
    )

    return pipeline, retrieval_pipeline, llm_provider


def test_generation_pipeline_connects_all_components():

    pipeline, retrieval, llm = create_pipeline()

    answer = pipeline.generate(
        "How many sick leave days are allowed?"
    )

    assert (
        answer
        == "Employees may take up to 12 days of sick leave. [1]"
    )

    assert (
        retrieval.received_query
        == "How many sick leave days are allowed?"
    )

    assert llm.received_prompt is not None


def test_generation_pipeline_passes_retrieved_context_to_llm():

    pipeline, _, llm = create_pipeline()

    pipeline.generate(
        "How many sick leave days are allowed?"
    )

    assert "12 days of sick leave" in (
        llm.received_prompt
    )

    assert "Source: leave.txt" in (
        llm.received_prompt
    )

    assert "Section: Sick Leave" in (
        llm.received_prompt
    )


def test_generation_pipeline_creates_grounded_prompt():

    pipeline, _, llm = create_pipeline()

    pipeline.generate(
        "How many sick leave days are allowed?"
    )

    prompt = llm.received_prompt

    assert "ONLY the provided context" in prompt
    assert "Do not use outside knowledge" in prompt
    assert "[1]" in prompt


def test_generation_pipeline_rejects_empty_query():

    pipeline, _, _ = create_pipeline()

    with pytest.raises(
        ValueError,
        match="Query cannot be empty",
    ):
        pipeline.generate("")


def test_generation_pipeline_rejects_empty_retrieval_context():

    class EmptyRetrievalPipeline:

        def retrieve(self, query):
            return []

    pipeline = GenerationPipeline(
        retrieval_pipeline=EmptyRetrievalPipeline(),
        context_builder=ContextBuilder(),
        prompt_builder=GroundedPromptBuilder(),
        llm_provider=FakeLLMProvider(),
    )

    with pytest.raises(
        ValueError,
        match="No retrieval context available",
    ):
        pipeline.generate(
            "What is the leave policy?"
        )
