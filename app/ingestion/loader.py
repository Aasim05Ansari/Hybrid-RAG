from pypdf import PdfReader
from docx import Document as DocxDocument
from abc import ABC, abstractmethod
from pathlib import Path

from app.ingestion.metadata import Document


class DocumentLoader(ABC):

    @abstractmethod
    def load(self, path: Path) -> Document:
        """Load a file and return a Document."""
        raise NotImplementedError
    
class TextLoader(DocumentLoader):

    def load(self, path: Path) -> Document:

        if not path.exists():
            raise FileNotFoundError(
                f"File not found: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Path is not a file: {path}"
            )

        content = path.read_text(
            encoding="utf-8"
        )

        return Document(
            content=content,
            metadata={
                "source": str(path),
                "filename": path.name,
                "file_type": path.suffix.lower(),
            },
        )
        
class PDFLoader(DocumentLoader):

    def load(self, path: Path) -> Document:

        if not path.exists():
            raise FileNotFoundError(
                f"File not found: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Path is not a file: {path}"
            )

        reader = PdfReader(str(path))

        pages = []

        for page in reader.pages:

            text = page.extract_text()

            if text:
                pages.append(text)

        content = "\n\n".join(pages)

        return Document(
            content=content,
            metadata={
                "source": str(path),
                "filename": path.name,
                "file_type": path.suffix.lower(),
                "page_count": len(reader.pages),
            },
        )
        
class DOCXLoader(DocumentLoader):

    def load(self, path: Path) -> Document:

        if not path.exists():
            raise FileNotFoundError(
                f"File not found: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Path is not a file: {path}"
            )

        docx = DocxDocument(str(path))

        paragraphs = []

        for paragraph in docx.paragraphs:

            text = paragraph.text.strip()

            if text:
                paragraphs.append(text)

        content = "\n\n".join(paragraphs)

        return Document(
            content=content,
            metadata={
                "source": str(path),
                "filename": path.name,
                "file_type": path.suffix.lower(),
            },
        )
        
def get_loader(path: Path) -> DocumentLoader:

    extension = path.suffix.lower()

    if extension in {".txt", ".md", ".markdown"}:
        return TextLoader()

    if extension == ".pdf":
        return PDFLoader()
    
    if extension == ".docx":
        return DOCXLoader()

    raise ValueError(
        f"Unsupported file type: {extension}"
    )
