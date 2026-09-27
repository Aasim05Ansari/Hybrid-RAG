import pytest

from app.generation.grounded_pipeline import (
    GroundedAnswer,
)
from app.pipeline.rag_pipeline import RAGPipeline


class FakeGroundedPipeline:

    def __init__(self):
        self.received_query = None

    def run(self, query):
        self.received_query = query
        return "fake-result"


def test_rag_pipeline_delegates_to_grounded_pipeline():

    grounded = FakeGroundedPipeline()

    pipeline = RAGPipeline(
        grounded_generation_pipeline=grounded
    )

    result = pipeline.run(
        "annual leave"
    )

    assert result == "fake-result"

    assert grounded.received_query == (
        "annual leave"
    )


def test_rag_pipeline_rejects_empty_query():

    grounded = FakeGroundedPipeline()

    pipeline = RAGPipeline(
        grounded_generation_pipeline=grounded
    )

    with pytest.raises(
        ValueError,
        match="Query cannot be empty",
    ):
        pipeline.run("")