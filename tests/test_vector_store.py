from app.indexing.vector_store import ChromaVectorStore


def test_chroma_vector_store_add_and_search(tmp_path):

    store = ChromaVectorStore(
        persist_directory=str(tmp_path),
        collection_name="test_collection",
    )

    ids = [
        "chunk_1",
        "chunk_2",
    ]

    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]

    documents = [
        "Annual leave is 24 days.",
        "Sick leave is 12 days.",
    ]

    metadatas = [
        {
            "source": "leave.txt",
            "chunk_index": 0,
        },
        {
            "source": "leave.txt",
            "chunk_index": 1,
        },
    ]

    store.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    results = store.search(
        query_embedding=[1.0, 0.0, 0.0],
        top_k=1,
    )

    assert len(results) == 1

    assert results[0]["id"] == "chunk_1"

    assert (
        results[0]["document"]
        == "Annual leave is 24 days."
    )


def test_chroma_vector_store_rejects_mismatched_lengths(
    tmp_path,
):

    store = ChromaVectorStore(
        persist_directory=str(tmp_path),
        collection_name="test_collection",
    )

    try:

        store.add(
            ids=["chunk_1"],
            embeddings=[
                [1.0, 0.0]
            ],
            documents=[],
            metadatas=[
                {}
            ],
        )

        assert False

    except ValueError as error:

        assert (
            "same length"
            in str(error)
        )