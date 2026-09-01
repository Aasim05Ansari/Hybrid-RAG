from app.embeddings.embedder import EmbeddingProvider
from app.indexing.bm25_store import BM25Store
from app.indexing.indexer import Indexer
from app.indexing.vector_store import ChromaVectorStore
from app.ingestion.metadata import Chunk


class FakeEmbeddingProvider(EmbeddingProvider):

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:

        embeddings = []

        for text in texts:

            if "annual" in text.lower():
                embeddings.append(
                    [1.0, 0.0, 0.0]
                )

            else:
                embeddings.append(
                    [0.0, 1.0, 0.0]
                )

        return embeddings

    def embed_query(
        self,
        text: str,
    ) -> list[float]:

        return [1.0, 0.0, 0.0]


def test_indexer_indexes_chunks_into_both_stores(
    tmp_path,
):

    vector_store = ChromaVectorStore(
        persist_directory=str(tmp_path),
        collection_name="test_indexer",
    )

    bm25_store = BM25Store()

    indexer = Indexer(
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=vector_store,
        bm25_store=bm25_store,
    )

    chunks = [
        Chunk(
            chunk_id="chunk_1",
            content="Annual leave is 24 days.",
            metadata={
                "source": "leave.txt",
                "chunk_index": 0,
                "chunking_strategy": "fixed",
            },
        ),
        Chunk(
            chunk_id="chunk_2",
            content="Sick leave is 12 days.",
            metadata={
                "source": "leave.txt",
                "chunk_index": 1,
                "chunking_strategy": "fixed",
            },
        ),
    ]

    result = indexer.index(chunks)

    assert result["indexed"] == [
        "chunk_1",
        "chunk_2",
    ]

    assert result["duplicates"] == []

    vector_results = vector_store.search(
        query_embedding=[1.0, 0.0, 0.0],
        top_k=2,
    )

    bm25_results = bm25_store.search(
        query="annual leave",
        top_k=2,
    )

    assert len(vector_results) == 2
    assert len(bm25_results) == 2


def test_indexer_skips_duplicate_against_existing_store(
    tmp_path,
):

    vector_store = ChromaVectorStore(
        persist_directory=str(tmp_path),
        collection_name="test_dedup_existing",
    )

    bm25_store = BM25Store()

    provider = FakeEmbeddingProvider()

    indexer = Indexer(
        embedding_provider=provider,
        vector_store=vector_store,
        bm25_store=bm25_store,
    )

    first_chunk = Chunk(
        chunk_id="chunk_1",
        content="Annual leave is 24 days.",
        metadata={
            "source": "leave.txt",
        },
    )

    first_result = indexer.index(
        [first_chunk]
    )

    assert first_result["indexed"] == [
        "chunk_1"
    ]

    duplicate_chunk = Chunk(
        chunk_id="chunk_2",
        content="Annual leave is 24 days.",
        metadata={
            "source": "another_leave.txt",
        },
    )

    second_result = indexer.index(
        [duplicate_chunk]
    )

    assert second_result["indexed"] == []

    assert second_result["duplicates"] == [
        "chunk_2"
    ]


def test_indexer_skips_duplicate_inside_same_batch(
    tmp_path,
):

    vector_store = ChromaVectorStore(
        persist_directory=str(tmp_path),
        collection_name="test_dedup_batch",
    )

    bm25_store = BM25Store()

    indexer = Indexer(
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=vector_store,
        bm25_store=bm25_store,
    )

    chunks = [
        Chunk(
            chunk_id="chunk_1",
            content="Annual leave is 24 days.",
            metadata={"source": "leave.txt"},
        ),
        Chunk(
            chunk_id="chunk_2",
            content="Annual leave is 24 days.",
            metadata={"source": "leave.txt"},
        ),
        Chunk(
            chunk_id="chunk_3",
            content="Sick leave is 12 days.",
            metadata={"source": "leave.txt"},
        ),
    ]

    result = indexer.index(chunks)

    assert result["indexed"] == [
        "chunk_1",
        "chunk_3",
    ]

    assert result["duplicates"] == [
        "chunk_2"
    ]


def test_indexer_rejects_invalid_duplicate_threshold(
    tmp_path,
):

    vector_store = ChromaVectorStore(
        persist_directory=str(tmp_path),
        collection_name="test_invalid_threshold",
    )

    bm25_store = BM25Store()

    try:

        Indexer(
            embedding_provider=FakeEmbeddingProvider(),
            vector_store=vector_store,
            bm25_store=bm25_store,
            duplicate_similarity_threshold=1.5,
        )

        assert False

    except ValueError as error:

        assert (
            "between 0 and 1"
            in str(error)
        )
