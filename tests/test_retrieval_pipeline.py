from app.retrieval.models import RetrievalResult
from app.retrieval.pipeline import RetrievalPipeline
from app.retrieval.reranker import Reranker


def result(
    chunk_id: str,
    score: float = 0.5,
):
    return RetrievalResult(
        chunk_id=chunk_id,
        content=f"Content for {chunk_id}",
        metadata={
            "source": "policy.txt"
        },
        score=score,
    )


class FakeHybridRetriever:

    def __init__(self):
        self.received_query = None

    def retrieve(self, query):

        self.received_query = query

        return [
            result("A"),
            result("B"),
            result("C"),
            result("D"),
            result("E"),
            result("F"),
        ]


class FakeReranker(Reranker):

    def __init__(self):
        self.received_query = None
        self.received_results = None
        self.received_top_k = None

    def rerank(
        self,
        query,
        results,
        top_k,
    ):

        self.received_query = query
        self.received_results = results
        self.received_top_k = top_k

        return results[:top_k]


def test_retrieval_pipeline_connects_hybrid_to_reranker():

    hybrid = FakeHybridRetriever()
    reranker = FakeReranker()

    pipeline = RetrievalPipeline(
        hybrid_retriever=hybrid,
        reranker=reranker,
        final_top_k=5,
    )

    results = pipeline.retrieve(
        "annual leave"
    )

    assert len(results) == 5

    assert hybrid.received_query == (
        "annual leave"
    )

    assert reranker.received_query == (
        "annual leave"
    )


def test_pipeline_passes_hybrid_candidates_to_reranker():

    hybrid = FakeHybridRetriever()
    reranker = FakeReranker()

    pipeline = RetrievalPipeline(
        hybrid_retriever=hybrid,
        reranker=reranker,
        final_top_k=5,
    )

    pipeline.retrieve(
        "annual leave"
    )

    assert len(
        reranker.received_results
    ) == 6


def test_pipeline_requests_final_top_k():

    hybrid = FakeHybridRetriever()
    reranker = FakeReranker()

    pipeline = RetrievalPipeline(
        hybrid_retriever=hybrid,
        reranker=reranker,
        final_top_k=5,
    )

    pipeline.retrieve(
        "annual leave"
    )

    assert (
        reranker.received_top_k
        == 5
    )


def test_pipeline_can_return_top_3():

    hybrid = FakeHybridRetriever()
    reranker = FakeReranker()

    pipeline = RetrievalPipeline(
        hybrid_retriever=hybrid,
        reranker=reranker,
        final_top_k=3,
    )

    results = pipeline.retrieve(
        "annual leave"
    )

    assert len(results) == 3


def test_pipeline_rejects_empty_query():

    pipeline = RetrievalPipeline(
        hybrid_retriever=FakeHybridRetriever(),
        reranker=FakeReranker(),
    )

    try:
        pipeline.retrieve("")

        assert False

    except ValueError as exc:

        assert "Query cannot be empty" in str(exc)


def test_pipeline_rejects_invalid_final_top_k():

    try:
        RetrievalPipeline(
            hybrid_retriever=FakeHybridRetriever(),
            reranker=FakeReranker(),
            final_top_k=0,
        )

        assert False

    except ValueError as exc:

        assert "final_top_k" in str(exc)
