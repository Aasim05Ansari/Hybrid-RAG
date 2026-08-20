from app.chunking.base import Chunker
from app.chunking.fixed import FixedSizeChunker
from app.chunking.recursive import RecursiveChunker
from app.chunking.semantic import SemanticChunker
from app.indexing.embeddings import EmbeddingProvider


def get_chunker(
    strategy: str,
    chunk_size: int = 800,
    overlap: int = 120,
    embedding_provider: EmbeddingProvider | None = None,
) -> Chunker:

    strategy = strategy.lower()

    if strategy == "fixed":
        return FixedSizeChunker(
            chunk_size=chunk_size,
            overlap=overlap,
        )

    if strategy == "recursive":
        return RecursiveChunker(
            chunk_size=chunk_size,
            overlap=overlap,
        )

    if strategy == "semantic":

        if embedding_provider is None:
            raise ValueError(
                "embedding_provider is required "
                "for semantic chunking"
            )

        return SemanticChunker(
            embedding_provider=embedding_provider,
        )

    raise ValueError(
        f"Unsupported chunking strategy: {strategy}"
    )