from app.indexing.bm25_store import BM25Store
from app.retrieval.dense import RetrievalResult
from app.retrieval.sparse import SparseRetriever


def create_store():

    store = BM25Store()

    store.add(
        ids=[
            "chunk_1",
            "chunk_2",
            "chunk_3",
        ],
        documents=[
            "Annual leave is 24 days per year.",
            "Sick leave is 12 days per year.",
            "Employees can carry forward unused leave.",
        ],
        metadatas=[
            {"source": "leave.txt"},
            {"source": "leave.txt"},
            {"source": "leave.txt"},
        ],
    )

    return store


def test_sparse_retriever_returns_results():

    store = create_store()

    retriever = SparseRetriever(
        bm25_store=store
    )

    results = retriever.retrieve(
        "annual leave",
        top_k=2,
    )

    assert len(results) == 2

    assert isinstance(
        results[0],
        RetrievalResult,
    )

    assert results[0].chunk_id == "chunk_1"

    assert (
        "Annual leave"
        in results[0].content
    )


def test_sparse_retriever_preserves_metadata():

    store = create_store()

    retriever = SparseRetriever(
        bm25_store=store
    )

    results = retriever.retrieve(
        "annual leave",
        top_k=1,
    )

    assert (
        results[0].metadata["source"]
        == "leave.txt"
    )


def test_sparse_retriever_preserves_bm25_score():

    store = create_store()

    retriever = SparseRetriever(
        bm25_store=store
    )

    results = retriever.retrieve(
        "annual leave",
        top_k=3,
    )

    assert isinstance(
        results[0].score,
        float,
    )

    assert results[0].score >= 0.0


def test_sparse_retriever_rejects_empty_query():

    store = create_store()

    retriever = SparseRetriever(
        bm25_store=store
    )

    try:
        retriever.retrieve("")

        assert False

    except ValueError as exc:

        assert (
            "Query cannot be empty"
            in str(exc)
        )


def test_sparse_retriever_rejects_invalid_top_k():

    store = create_store()

    retriever = SparseRetriever(
        bm25_store=store
    )

    try:
        retriever.retrieve(
            "annual leave",
            top_k=0,
        )

        assert False

    except ValueError as exc:

        assert "top_k" in str(exc)


def test_sparse_retriever_handles_empty_store():

    store = BM25Store()

    retriever = SparseRetriever(
        bm25_store=store
    )

    results = retriever.retrieve(
        "annual leave",
        top_k=10,
    )

    assert results == []
