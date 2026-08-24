from app.ingestion.cleaner import DocumentCleaner
from app.ingestion.metadata import Document


def test_document_cleaner():

    document = Document(
        content=(
            "  Annual leave is 24 days.  \r\n"
            "\r\n"
            "\r\n"
            "Employees may carry forward leave.   "
        ),
        metadata={
            "document_id": "test_doc",
            "source": "test.txt",
        },
    )

    cleaner = DocumentCleaner()

    cleaned = cleaner.clean(document)

    assert cleaned.content == (
        "Annual leave is 24 days.\n\n"
        "Employees may carry forward leave."
    )


def test_cleaner_preserves_metadata():

    document = Document(
        content="  Hello world.  ",
        metadata={
            "document_id": "doc_1",
            "source": "test.txt",
        },
    )

    cleaner = DocumentCleaner()

    cleaned = cleaner.clean(document)

    assert cleaned.metadata == document.metadata