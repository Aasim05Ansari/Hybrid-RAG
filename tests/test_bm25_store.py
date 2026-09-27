from app.indexing.bm25_store import BM25Store


def test_bm25_store_persists_across_instances(tmp_path):

    store_path = tmp_path / "bm25"

    first_store = BM25Store(
        persist_directory=store_path
    )

    first_store.add(
        ids=[
            "chunk-1",
            "chunk-2",
            "chunk-3",
        ],
        documents=[
            "Employees receive 24 days of annual leave.",
            "Employees receive 12 days of sick leave.",
            "Employees can work remotely 2 days per week.",
        ],
        metadatas=[
            {"source": "leave.txt"},
            {"source": "leave.txt"},
            {"source": "remote-work.txt"},
        ],
    )

    assert first_store.document_count == 3
    assert first_store.is_ready is True

    second_store = BM25Store(
        persist_directory=store_path
    )

    assert second_store.document_count == 3
    assert second_store.is_ready is True

    results = second_store.search(
        query="sick leave",
        top_k=1,
    )

    assert results
    assert results[0]["id"] == "chunk-2"


def test_bm25_store_creates_persistence_directory(tmp_path):

    store_path = (
        tmp_path
        / "nested"
        / "bm25"
    )

    store = BM25Store(
        persist_directory=store_path
    )

    store.add(
        ids=["chunk-1"],
        documents=["Annual leave is 24 days."],
        metadatas=[{"source": "leave.txt"}],
    )

    assert store_path.exists()
    assert (store_path / "index.json").exists()


def test_bm25_store_upserts_existing_id(tmp_path):

    store = BM25Store(
        persist_directory=tmp_path / "bm25"
    )

    store.add(
        ids=["chunk-1"],
        documents=["Annual leave is 24 days."],
        metadatas=[{"source": "leave.txt"}],
    )

    store.add(
        ids=["chunk-1"],
        documents=["Annual leave is 30 days."],
        metadatas=[{"source": "updated.txt"}],
    )

    assert store.document_count == 1

    results = store.search(
        query="annual leave",
        top_k=1,
    )

    assert results[0]["id"] == "chunk-1"
    assert results[0]["document"] == (
        "Annual leave is 30 days."
    )
    assert results[0]["metadata"]["source"] == (
        "updated.txt"
    )


def test_bm25_store_empty_persistent_store_is_not_ready(
    tmp_path,
):

    store = BM25Store(
        persist_directory=tmp_path / "bm25"
    )

    assert store.document_count == 0
    assert store.is_ready is False

    assert store.search(
        query="annual leave",
        top_k=5,
    ) == []


def test_bm25_store_rejects_corrupt_persistence_file(
    tmp_path,
):

    store_path = tmp_path / "bm25"
    store_path.mkdir(parents=True)

    index_path = store_path / "index.json"

    index_path.write_text(
        "{not-valid-json",
        encoding="utf-8",
    )

    try:
        BM25Store(
            persist_directory=store_path
        )
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "Invalid BM25 index file" in str(exc)