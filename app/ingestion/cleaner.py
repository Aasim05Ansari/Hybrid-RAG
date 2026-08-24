import re

from app.ingestion.metadata import Document


class DocumentCleaner:

    def clean(self, document: Document) -> Document:

        content = document.content

        # Normalize Windows and old Mac line endings
        content = content.replace("\r\n", "\n")
        content = content.replace("\r", "\n")

        # Remove trailing whitespace from each line
        content = "\n".join(
            line.rstrip()
            for line in content.split("\n")
        )

        # Collapse repeated spaces/tabs
        content = re.sub(
            r"[ \t]+",
            " ",
            content,
        )

        # Collapse excessive blank lines
        content = re.sub(
            r"\n{3,}",
            "\n\n",
            content,
        )

        # Remove leading/trailing whitespace
        content = content.strip()

        return Document(
            content=content,
            metadata=document.metadata.copy(),
        )