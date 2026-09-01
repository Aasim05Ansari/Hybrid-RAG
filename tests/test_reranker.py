from app.retrieval.models import RetrievalResult
from app.retrieval.reranker import (
    LexicalReranker,
    Reranker,
)


def result(
    chunk_id: str,
    content: str,
    score: float = 0.0,
):

    return RetrievalResult(
        chunk_id=chunk_id,
        content=content,
        metadata={
            "source": "policy.txt"
        },
        score=score,
    )


def test_lexical_reranker_is_reranker():

    reranker = LexicalReranker()

    assert isinstance(
        reranker,
        Reranker,
    )


def test_reranker_changes_ranking():

    results = [
        result(
            "A",
            "The company provides sick leave.",
        ),
        result(
            "B",
            "Annual leave is 24 days per year.",
        ),
        result(
            "C",
            "Employees receive annual leave days.",
        ),
    ]

    reranker = LexicalReranker()

    reranked = reranker.rerank(
        query="annual leave days",
        results=results,
        top_k=3,
    )

    assert reranked[0].chunk_id in {
        "B",
        "C",
    }

    assert reranked[0].score >= (
        reranked[1].score
    )


def test_reranker_produces_new_scores():

    results = [
        result(
            "A",
            "Annual leave is 24 days.",
            score=0.001,
        ),
    ]

    reranker = LexicalReranker()

    reranked = reranker.rerank(
        query="annual leave",
        results=results,
        top_k=1,
    )

    assert reranked[0].score == 1.0

    assert (
        reranked[0].score
        != results[0].score
    )


def test_reranker_limits_to_top_k():

    results = [
        result(
            "A",
            "Annual leave days.",
        ),
        result(
            "B",
            "Sick leave days.",
        ),
        result(
            "C",
            "Carry forward leave.",
        ),
    ]

    reranker = LexicalReranker()

    reranked = reranker.rerank(
        query="leave",
        results=results,
        top_k=2,
    )

    assert len(reranked) == 2


def test_reranker_handles_empty_results():

    reranker = LexicalReranker()

    reranked = reranker.rerank(
        query="annual leave",
        results=[],
        top_k=5,
    )

    assert reranked == []


def test_reranker_rejects_empty_query():

    reranker = LexicalReranker()

    try:
        reranker.rerank(
            query="",
            results=[],
            top_k=5,
        )

        assert False

    except ValueError as exc:

        assert "Query cannot be empty" in str(exc)


def test_reranker_rejects_invalid_top_k():

    reranker = LexicalReranker()

    try:
        reranker.rerank(
            query="annual leave",
            results=[],
            top_k=0,
        )

        assert False

    except ValueError as exc:

        assert "top_k" in str(exc)


def test_reranker_preserves_metadata():

    results = [
        result(
            "A",
            "Annual leave is 24 days.",
        ),
    ]

    reranker = LexicalReranker()

    reranked = reranker.rerank(
        query="annual leave",
        results=results,
        top_k=1,
    )

    assert (
        reranked[0].metadata["source"]
        == "policy.txt"
    )
