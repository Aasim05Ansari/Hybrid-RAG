from pathlib import Path

from app.chunking.fixed import FixedSizeChunker
from app.ingestion.raw_store import RawDocumentStore
from app.pipeline.ingestion_pipeline import IngestionPipeline


def test_ingestion_pipeline(tmp_path):

    file_path = tmp_path / "policy.txt"

    file_path.write_text(
        (
            "Annual leave is 24 days per year. "
            "Sick leave is 12 days per year. "
            "Unused annual leave may be carried forward."
        ),
        encoding="utf-8",
    )

    chunker = FixedSizeChunker(
        chunk_size=50,
        overlap=10,
    )

    raw_directory = tmp_path / "raw"

    raw_store = RawDocumentStore(
        storage_directory=raw_directory
    )

    pipeline = IngestionPipeline(
        chunker=chunker,
        raw_store=raw_store,
    )

    chunks = pipeline.process(
        file_path
    )

    assert len(chunks) > 0

    assert all(
        chunk.content
        for chunk in chunks
    )

    assert all(
        chunk.metadata["source"]
        == str(file_path)
        for chunk in chunks
    )


def test_ingestion_pipeline_preserves_chunk_strategy(
    tmp_path,
):

    file_path = tmp_path / "policy.txt"

    file_path.write_text(
        "Annual leave is 24 days per year.",
        encoding="utf-8",
    )

    chunker = FixedSizeChunker(
        chunk_size=100,
        overlap=10,
    )

    raw_store = RawDocumentStore(
        storage_directory=tmp_path / "raw"
    )

    pipeline = IngestionPipeline(
        chunker=chunker,
        raw_store=raw_store,
    )

    chunks = pipeline.process(
        file_path
    )

    assert chunks[0].metadata[
        "chunking_strategy"
    ] == "fixed"


def test_ingestion_pipeline_preserves_raw_document(
    tmp_path,
):

    file_path = tmp_path / "policy.txt"

    original_content = (
        "Annual leave is 24 days per year.\n"
        "Sick leave is 12 days per year.\n"
    )

    file_path.write_text(
        original_content,
        encoding="utf-8",
    )

    raw_directory = tmp_path / "raw"

    raw_store = RawDocumentStore(
        storage_directory=raw_directory
    )

    chunker = FixedSizeChunker(
        chunk_size=100,
        overlap=10,
    )

    pipeline = IngestionPipeline(
        chunker=chunker,
        raw_store=raw_store,
    )

    chunks = pipeline.process(
        file_path
    )

    raw_path = raw_directory / "policy.txt"

    assert raw_path.exists()

    assert raw_path.read_text(
        encoding="utf-8"
    ) == original_content

    assert all(
        chunk.metadata["raw_path"]
        == str(raw_path)
        for chunk in chunks
    )


def test_ingestion_does_not_modify_original_document(
    tmp_path,
):

    file_path = tmp_path / "policy.txt"

    original_content = (
        "Annual leave is 24 days per year."
    )

    file_path.write_text(
        original_content,
        encoding="utf-8",
    )

    original_bytes = file_path.read_bytes()

    raw_store = RawDocumentStore(
        storage_directory=tmp_path / "raw"
    )

    pipeline = IngestionPipeline(
        chunker=FixedSizeChunker(
            chunk_size=100,
            overlap=10,
        ),
        raw_store=raw_store,
    )

    pipeline.process(file_path)

    assert file_path.read_bytes() == original_bytes
