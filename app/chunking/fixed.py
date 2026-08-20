from app.chunking.base import Chunker
from app.ingestion.metadata import Document, Chunk


class FixedSizeChunker(Chunker):

    def __init__(
        self,
        chunk_size: int = 800,
        overlap: int = 120,
    ):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.content.strip()

        if not text:
            return []

        chunks = []

        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + self.chunk_size

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunk_id = (
                    f"{document.metadata.get('document_id', 'doc')}"
                    f"_chunk_{chunk_index}"
                )

                metadata = {
                    **document.metadata,
                    "chunk_index": chunk_index,
                    "chunking_strategy": "fixed",
                    "character_count": len(chunk_text),
                }

                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        content=chunk_text,
                        metadata=metadata,
                    )
                )

                chunk_index += 1

            start += self.chunk_size - self.overlap

        return chunks