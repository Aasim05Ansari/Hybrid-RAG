from pathlib import Path

from app.chunking.fixed import FixedSizeChunker
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

    pipeline = IngestionPipeline(
        chunker=chunker,
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

    pipeline = IngestionPipeline(
        chunker=chunker,
    )

    chunks = pipeline.process(
        file_path
    )

    assert chunks[0].metadata[
        "chunking_strategy"
    ] == "fixed"