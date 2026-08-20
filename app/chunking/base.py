from abc import ABC, abstractmethod

from app.ingestion.metadata import Document, Chunk


class Chunker(ABC):

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """Split a document into chunks."""
        raise NotImplementedError