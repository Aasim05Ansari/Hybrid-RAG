from pathlib import Path

import pytest

from app.ingestion.raw_store import RawDocumentStore


def test_raw_document_store_preserves_original_file(tmp_path):

    source = tmp_path / "policy.txt"

    original_content = (
        "Annual leave is 24 days per year."
    )

    source.write_text(
        original_content,
        encoding="utf-8",
    )

    raw_directory = tmp_path / "raw"

    store = RawDocumentStore(
        storage_directory=raw_directory
    )

    preserved = store.preserve(source)

    assert preserved.exists()

    assert preserved.name == "policy.txt"

    assert preserved.read_text(
        encoding="utf-8"
    ) == original_content

    assert source.exists()

    assert source.read_text(
        encoding="utf-8"
    ) == original_content


def test_raw_document_store_creates_storage_directory(tmp_path):

    source = tmp_path / "policy.txt"

    source.write_text(
        "Employee leave policy",
        encoding="utf-8",
    )

    raw_directory = (
        tmp_path
        / "nested"
        / "raw"
    )

    assert not raw_directory.exists()

    store = RawDocumentStore(
        storage_directory=raw_directory
    )

    assert raw_directory.exists()

    preserved = store.preserve(source)

    assert preserved.exists()


def test_raw_document_store_rejects_missing_file(tmp_path):

    source = tmp_path / "missing.txt"

    store = RawDocumentStore(
        storage_directory=tmp_path / "raw"
    )

    with pytest.raises(FileNotFoundError):

        store.preserve(source)


def test_raw_document_store_rejects_directory(tmp_path):

    source_directory = tmp_path / "documents"

    source_directory.mkdir()

    store = RawDocumentStore(
        storage_directory=tmp_path / "raw"
    )

    with pytest.raises(ValueError):

        store.preserve(source_directory)
