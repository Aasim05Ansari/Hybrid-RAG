from app.retrieval.dense import RetrievalResult
from app.retrieval.hybrid import HybridRetriever


class FakeDenseRetriever:

    def __init__(self):
        self.received_top_k = None
        self.received_query = None

    def retrieve(self, query, top_k):

        self.received_query = query
        self.received_top_k = top_k

        return [
            RetrievalResult(
                chunk_id="dense_1",
                content="Dense result 1",
                metadata={"source": "dense.txt"},
                score=0.9,
            ),
            RetrievalResult(
                chunk_id="shared",
                content="Shared result",
                metadata={"source": "shared.txt"},
                score=0.8,
            ),
        ]


class FakeSparseRetriever:

    def __init__(self):
        self.received_top_k = None
        self.received_query = None

    def retrieve(self, query, top_k):

        self.received_query = query
        self.received_top_k = top_k

        return [
            RetrievalResult(
                chunk_id="sparse_1",
                content="Sparse result 1",
                metadata={"source": "sparse.txt"},
                score=5.0,
            ),
            RetrievalResult(
                chunk_id="shared",
                content="Shared result",
                metadata={"source": "shared.txt"},
                score=4.0,
            ),
        ]


def test_hybrid_retriever_combines_dense_and_sparse():

    dense = FakeDenseRetriever()
    sparse = FakeSparseRetriever()

    retriever = HybridRetriever(
        dense_retriever=dense,
        sparse_retriever=sparse,
        dense_top_k=10,
        sparse_top_k=10,
        fusion_top_k=20,
        dense_weight=0.7,
        sparse_weight=0.3,
    )

    results = retriever.retrieve(
        "annual leave"
    )

    ids = [
        result.chunk_id
        for result in results
    ]

    assert set(ids) == {
        "dense_1",
        "sparse_1",
        "shared",
    }


def test_hybrid_retriever_uses_configured_top_k():

    dense = FakeDenseRetriever()
    sparse = FakeSparseRetriever()

    retriever = HybridRetriever(
        dense_retriever=dense,
        sparse_retriever=sparse,
        dense_top_k=7,
        sparse_top_k=8,
        fusion_top_k=20,
    )

    retriever.retrieve(
        "annual leave"
    )

    assert dense.received_top_k == 7

    assert sparse.received_top_k == 8


def test_hybrid_retriever_limits_candidates_to_top_20():

    dense = FakeDenseRetriever()
    sparse = FakeSparseRetriever()

    retriever = HybridRetriever(
        dense_retriever=dense,
        sparse_retriever=sparse,
        fusion_top_k=1,
    )

    results = retriever.retrieve(
        "annual leave"
    )

    assert len(results) == 1


def test_hybrid_retriever_passes_query_to_both():

    dense = FakeDenseRetriever()
    sparse = FakeSparseRetriever()

    retriever = HybridRetriever(
        dense_retriever=dense,
        sparse_retriever=sparse,
    )

    query = "How many annual leave days?"

    retriever.retrieve(query)

    assert dense.received_query == query

    assert sparse.received_query == query


def test_hybrid_retriever_rejects_empty_query():

    retriever = HybridRetriever(
        dense_retriever=FakeDenseRetriever(),
        sparse_retriever=FakeSparseRetriever(),
    )

    try:
        retriever.retrieve("")

        assert False

    except ValueError as exc:

        assert "Query cannot be empty" in str(exc)


def test_hybrid_retriever_rejects_zero_weights():

    try:
        HybridRetriever(
            dense_retriever=FakeDenseRetriever(),
            sparse_retriever=FakeSparseRetriever(),
            dense_weight=0.0,
            sparse_weight=0.0,
        )

        assert False

    except ValueError as exc:

        assert "weight" in str(exc)
