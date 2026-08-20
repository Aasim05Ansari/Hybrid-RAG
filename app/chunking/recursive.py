from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.chunking.base import Chunker
from app.ingestion.metadata import Document, Chunk


class RecursiveChunker(Chunker):

    def __init__(
        self,
        chunk_size: int = 800,
        overlap: int = 120,
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ],
        )

    def chunk(self, document: Document) -> list[Chunk]:
        texts = self.splitter.split_text(document.content)

        chunks = []

        for index, text in enumerate(texts):
            metadata = {
                **document.metadata,
                "chunk_index": index,
                "chunking_strategy": "recursive",
                "character_count": len(text),
            }

            chunks.append(
                Chunk(
                    chunk_id=(
                        f"{document.metadata.get('document_id', 'doc')}"
                        f"_chunk_{index}"
                    ),
                    content=text,
                    metadata=metadata,
                )
            )

        return chunks