from app.retrieval.dense import RetrievalResult
from app.retrieval.rrf import RRFFusion


def result(
    chunk_id: str,
    score: float = 1.0,
):
    return RetrievalResult(
        chunk_id=chunk_id,
        content=f"Content for {chunk_id}",
        metadata={
            "source": "policy.txt"
        },
        score=score,
    )


def test_rrf_fuses_multiple_rankings():

    dense_results = [
        result("A"),
        result("B"),
        result("C"),
    ]

    sparse_results = [
        result("C"),
        result("A"),
        result("D"),
    ]

    fusion = RRFFusion(k=60)

    results = fusion.fuse(
        [
            dense_results,
            sparse_results,
        ]
    )

    assert len(results) == 4

    assert results[0].chunk_id == "A"

    assert results[1].chunk_id == "C"


def test_rrf_score_uses_rank_not_original_score():

    dense_results = [
        result("A", score=0.99),
        result("B", score=0.10),
    ]

    sparse_results = [
        result("B", score=100.0),
        result("A", score=1.0),
    ]

    fusion = RRFFusion(k=60)

    results = fusion.fuse(
        [
            dense_results,
            sparse_results,
        ]
    )

    assert (
        results[0].score
        == results[1].score
    )


def test_rrf_duplicate_results_are_merged():

    dense_results = [
        result("A"),
        result("B"),
    ]

    sparse_results = [
        result("A"),
        result("C"),
    ]

    fusion = RRFFusion(k=60)

    results = fusion.fuse(
        [
            dense_results,
            sparse_results,
        ]
    )

    ids = [
        item.chunk_id
        for item in results
    ]

    assert ids.count("A") == 1

    assert set(ids) == {
        "A",
        "B",
        "C",
    }


def test_rrf_top_k_limits_results():

    dense_results = [
        result("A"),
        result("B"),
        result("C"),
        result("D"),
    ]

    fusion = RRFFusion(k=60)

    results = fusion.fuse(
        [dense_results],
        top_k=2,
    )

    assert len(results) == 2

    assert results[0].chunk_id == "A"

    assert results[1].chunk_id == "B"


def test_rrf_empty_input():

    fusion = RRFFusion()

    results = fusion.fuse([])

    assert results == []


def test_rrf_rejects_invalid_k():

    try:
        RRFFusion(k=0)

        assert False

    except ValueError as exc:

        assert "k" in str(exc)


def test_rrf_rejects_invalid_top_k():

    fusion = RRFFusion()

    try:
        fusion.fuse(
            [[result("A")]],
            top_k=0,
        )

        assert False

    except ValueError as exc:

        assert "top_k" in str(exc)


def test_rrf_applies_weights():

    dense_results = [
        result("A"),
    ]

    sparse_results = [
        result("B"),
    ]

    fusion = RRFFusion(k=60)

    results = fusion.fuse(
        [
            dense_results,
            sparse_results,
        ],
        weights=[
            0.7,
            0.3,
        ],
    )

    assert results[0].chunk_id == "A"

    assert results[0].score == 0.7 / 61

    assert results[1].score == 0.3 / 61


def test_rrf_rejects_mismatched_weights():

    fusion = RRFFusion()

    try:
        fusion.fuse(
            [
                [result("A")],
                [result("B")],
            ],
            weights=[0.7],
        )

        assert False

    except ValueError as exc:

        assert "weights" in str(exc)


def test_rrf_rejects_negative_weights():

    fusion = RRFFusion()

    try:
        fusion.fuse(
            [[result("A")]],
            weights=[-0.1],
        )

        assert False

    except ValueError as exc:

        assert "negative" in str(exc)
