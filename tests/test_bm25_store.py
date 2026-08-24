from app.indexing.bm25_store import BM25Store


def test_bm25_store_add_and_search():

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
            "Annual leave may be carried forward.",
        ],
        metadatas=[
            {"source": "leave.txt", "chunk_index": 0},
            {"source": "leave.txt", "chunk_index": 1},
            {"source": "leave.txt", "chunk_index": 2},
        ],
    )

    results = store.search(
        query="annual leave",
        top_k=2,
    )

    assert len(results) == 2

    result_ids = {
        result["id"]
        for result in results
    }

    assert result_ids.issubset(
        {
            "chunk_1",
            "chunk_2",
            "chunk_3",
        }
    )

    assert all(
        "document" in result
        and "metadata" in result
        and "score" in result
        for result in results
    )

    assert all(
        isinstance(result["score"], float)
        for result in results
    )

def test_bm25_exact_keyword_matching():

    store = BM25Store()

    store.add(
        ids=[
            "chunk_1",
            "chunk_2",
        ],
        documents=[
            "KCSR Rule 27 defines employee leave.",
            "Employees can request annual leave.",
        ],
        metadatas=[
            {"source": "rules.txt"},
            {"source": "leave.txt"},
        ],
    )

    results = store.search(
        query="KCSR Rule 27",
        top_k=1,
    )

    assert results[0]["id"] == "chunk_1"


def test_bm25_empty_query():

    store = BM25Store()

    store.add(
        ids=["chunk_1"],
        documents=["Annual leave is 24 days."],
        metadatas=[{"source": "leave.txt"}],
    )

    try:

        store.search(
            query="",
            top_k=1,
        )

        assert False

    except ValueError as error:

        assert (
            "Query cannot be empty"
            in str(error)
        )