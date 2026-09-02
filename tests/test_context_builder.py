from app.generation.context import ContextBuilder
from app.retrieval.models import RetrievalResult


def test_context_builder_numbers_results():
    results = [
        RetrievalResult(
            chunk_id="chunk_1",
            content="Employees may take up to 12 days of sick leave.",
            metadata={
                "source": "leave.txt",
                "section": "Sick Leave",
            },
            score=0.9,
        ),
        RetrievalResult(
            chunk_id="chunk_2",
            content="Employees are entitled to 24 days of annual leave.",
            metadata={
                "source": "leave.txt",
                "section": "Annual Leave",
            },
            score=0.8,
        ),
    ]

    context = ContextBuilder().build(results)

    assert "[1]" in context
    assert "[2]" in context

    assert "12 days of sick leave" in context
    assert "24 days of annual leave" in context


def test_context_builder_preserves_source_and_section():
    result = RetrievalResult(
        chunk_id="chunk_1",
        content="Test content.",
        metadata={
            "source": "policy.pdf",
            "section": "Leave Policy",
        },
        score=0.9,
    )

    context = ContextBuilder().build([result])

    assert "Source: policy.pdf" in context
    assert "Section: Leave Policy" in context
    assert "Test content." in context


def test_context_builder_empty_results():
    context = ContextBuilder().build([])

    assert context == ""
