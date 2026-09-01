from app.retrieval.dense import DenseRetriever


class FakeEmbeddingProvider:

    def embed_query(self, text):

        if "annual" in text.lower():
            return [1.0, 0.0, 0.0]

        return [0.0, 1.0, 0.0]


class FakeVectorStore:

    def __init__(self):

        self.received_query_embedding = None
        self.received_top_k = None

    def search(
        self,
        query_embedding,
        top_k,
    ):

        self.received_query_embedding = (
            query_embedding
        )

        self.received_top_k = top_k

        return [
            {
                "id": "chunk_1",
                "document": (
                    "Annual leave is 24 days."
                ),
                "metadata": {
                    "source": "policy.txt"
                },
                "distance": 0.1,
            },
            {
                "id": "chunk_2",
                "document": (
                    "Sick leave is 12 days."
                ),
                "metadata": {
                    "source": "policy.txt"
                },
                "distance": 0.3,
            },
        ]


def test_dense_retriever_returns_results():

    embedding_provider = (
        FakeEmbeddingProvider()
    )

    vector_store = FakeVectorStore()

    retriever = DenseRetriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    results = retriever.retrieve(
        "How much annual leave?",
        top_k=10,
    )

    assert len(results) == 2

    assert results[0].chunk_id == "chunk_1"

    assert (
        results[0].content
        == "Annual leave is 24 days."
    )

    assert (
        results[0].metadata["source"]
        == "policy.txt"
    )


def test_dense_retriever_converts_distance_to_score():

    retriever = DenseRetriever(
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=FakeVectorStore(),
    )

    results = retriever.retrieve(
        "annual leave",
        top_k=5,
    )

    assert results[0].score == 0.9
    assert results[1].score == 0.7


def test_dense_retriever_passes_top_k():

    vector_store = FakeVectorStore()

    retriever = DenseRetriever(
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=vector_store,
    )

    retriever.retrieve(
        "annual leave",
        top_k=7,
    )

    assert vector_store.received_top_k == 7


def test_dense_retriever_rejects_empty_query():

    retriever = DenseRetriever(
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=FakeVectorStore(),
    )

    try:
        retriever.retrieve("")
        assert False
    except ValueError as exc:
        assert "Query cannot be empty" in str(exc)


def test_dense_retriever_rejects_invalid_top_k():

    retriever = DenseRetriever(
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=FakeVectorStore(),
    )

    try:
        retriever.retrieve(
            "annual leave",
            top_k=0,
        )
        assert False
    except ValueError as exc:
        assert "top_k" in str(exc)
