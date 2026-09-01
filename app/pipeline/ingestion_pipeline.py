from pathlib import Path

from app.chunking.base import Chunker
from app.ingestion.cleaner import DocumentCleaner
from app.ingestion.loader import get_loader
from app.ingestion.metadata import Chunk
from app.ingestion.raw_store import RawDocumentStore


class IngestionPipeline:

    def __init__(
        self,
        chunker: Chunker,
        cleaner: DocumentCleaner | None = None,
        raw_store: RawDocumentStore | None = None,
    ):
        self.chunker = chunker
        self.cleaner = cleaner or DocumentCleaner()
        self.raw_store = raw_store or RawDocumentStore()

    def process(
        self,
        path: Path,
    ) -> list[Chunk]:

        path = Path(path)

        # Preserve the untouched source before processing.
        raw_path = self.raw_store.preserve(path)

        # Load the source document.
        loader = get_loader(path)

        document = loader.load(path)

        # Add a reference to the preserved raw document.
        document.metadata["raw_path"] = str(raw_path)

        # Clean the loaded document.
        document = self.cleaner.clean(document)

        # Split the cleaned document into chunks.
        chunks = self.chunker.chunk(document)

        return chunks
