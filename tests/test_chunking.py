from app.chunking.factory import get_chunker
from app.ingestion.metadata import Document


def test_fixed_chunking():

    document = Document(
        content="A" * 2000,
        metadata={
            "document_id": "test_doc",
            "source": "test.txt",
        },
    )

    chunker = get_chunker(
        strategy="fixed",
        chunk_size=500,
        overlap=50,
    )

    chunks = chunker.chunk(document)

    assert len(chunks) > 1

    assert all(chunk.content for chunk in chunks)

    assert all(
        chunk.metadata["chunking_strategy"] == "fixed"
        for chunk in chunks
    )


def test_recursive_chunking():

    document = Document(
        content=(
            "Annual leave is 24 days per year.\n\n"
            "Sick leave is 12 days per year.\n\n"
            "Unused annual leave may be carried forward."
        ),
        metadata={
            "document_id": "leave_policy",
            "source": "leave_policy.txt",
        },
    )

    chunker = get_chunker(
        strategy="recursive",
        chunk_size=100,
        overlap=20,
    )

    chunks = chunker.chunk(document)

    assert len(chunks) > 0

    assert all(
        chunk.metadata["chunking_strategy"] == "recursive"
        for chunk in chunks
    )
    
def test_semantic_chunking():

    document = Document(
        content=(
            "Annual leave is 24 days per year. "
            "Employees may carry forward unused annual leave. "
            "The company provides health insurance."
        ),
        metadata={
            "document_id": "semantic_test",
            "source": "test.txt",
        },
    )

    class FakeEmbeddingProvider:

        def embed_documents(self, texts):

            return [
                [1.0, 0.0],
                [0.99, 0.01],
                [0.0, 1.0],
            ]

        def embed_query(self, text):

            return [1.0, 0.0]

    chunker = get_chunker(
        strategy="semantic",
        embedding_provider=FakeEmbeddingProvider(),
    )

    chunks = chunker.chunk(document)

    assert len(chunks) == 2

    assert chunks[0].metadata["chunking_strategy"] == "semantic"
    assert chunks[1].metadata["chunking_strategy"] == "semantic"