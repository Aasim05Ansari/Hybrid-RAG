from pathlib import Path
import shutil

from app.config.settings import settings


class RawDocumentStore:
    """
    Stores an untouched copy of every ingested source document.
    """

    def __init__(
        self,
        storage_directory: str | Path | None = None,
    ):
        self.storage_directory = Path(
            storage_directory
            or (Path(settings.chroma_persist_dir).parent / "raw")
        )

        self.storage_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    def preserve(self, source_path: Path) -> Path:
        """
        Copy the original source file into raw storage.

        Returns:
            Path to the preserved raw document.
        """

        source_path = Path(source_path)

        if not source_path.exists():
            raise FileNotFoundError(
                f"Source file not found: {source_path}"
            )

        if not source_path.is_file():
            raise ValueError(
                f"Source path is not a file: {source_path}"
            )

        destination = (
            self.storage_directory
            / source_path.name
        )

        shutil.copy2(
            source_path,
            destination,
        )

        return destination
