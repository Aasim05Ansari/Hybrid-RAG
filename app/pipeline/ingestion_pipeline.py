from pathlib import Path

from app.chunking.base import Chunker
from app.ingestion.cleaner import DocumentCleaner
from app.ingestion.loader import get_loader
from app.ingestion.metadata import Chunk


class IngestionPipeline:

    def __init__(
        self,
        chunker: Chunker,
        cleaner: DocumentCleaner | None = None,
    ):
        self.chunker = chunker
        self.cleaner = cleaner or DocumentCleaner()

    def process(
        self,
        path: Path,
    ) -> list[Chunk]:

        loader = get_loader(path)

        document = loader.load(path)

        document = self.cleaner.clean(document)

        chunks = self.chunker.chunk(document)

        return chunks