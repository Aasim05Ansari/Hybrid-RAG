import math

from app.chunking.base import Chunker
from app.ingestion.metadata import Document, Chunk
from app.indexing.embeddings import EmbeddingProvider


def cosine_similarity(
    a: list[float],
    b: list[float],
) -> float:

    dot_product = sum(
        x * y
        for x, y in zip(a, b)
    )

    norm_a = math.sqrt(
        sum(x * x for x in a)
    )

    norm_b = math.sqrt(
        sum(x * x for x in b)
    )

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


class SemanticChunker(Chunker):

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        similarity_threshold: float = 0.70,
    ):
        self.embedding_provider = embedding_provider
        self.similarity_threshold = similarity_threshold

    def chunk(
        self,
        document: Document,
    ) -> list[Chunk]:

        paragraphs = [
            paragraph.strip()
            for paragraph in document.content.split("\n\n")
            if paragraph.strip()
        ]

        if not paragraphs:
            return []

        embeddings = (
            self.embedding_provider
            .embed_documents(paragraphs)
        )

        chunks = []
        current_chunk = [paragraphs[0]]
        chunk_index = 0

        for index in range(1, len(paragraphs)):

            similarity = cosine_similarity(
                embeddings[index - 1],
                embeddings[index],
            )

            if similarity >= self.similarity_threshold:

                current_chunk.append(
                    paragraphs[index]
                )

            else:

                content = "\n\n".join(
                    current_chunk
                )

                chunks.append(
                    Chunk(
                        chunk_id=(
                            f"{document.metadata.get('document_id', 'doc')}"
                            f"_chunk_{chunk_index}"
                        ),
                        content=content,
                        metadata={
                            **document.metadata,
                            "chunk_index": chunk_index,
                            "chunking_strategy": "semantic",
                            "character_count": len(content),
                        },
                    )
                )

                chunk_index += 1

                current_chunk = [
                    paragraphs[index]
                ]

        if current_chunk:

            content = "\n\n".join(
                current_chunk
            )

            chunks.append(
                Chunk(
                    chunk_id=(
                        f"{document.metadata.get('document_id', 'doc')}"
                        f"_chunk_{chunk_index}"
                    ),
                    content=content,
                    metadata={
                        **document.metadata,
                        "chunk_index": chunk_index,
                        "chunking_strategy": "semantic",
                        "character_count": len(content),
                    },
                )
            )

        return chunks
    
    