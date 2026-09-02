from app.generation.citation_verifier import (
    CitationVerifier,
)
from app.retrieval.models import RetrievalResult


def make_results():
    return [
        RetrievalResult(
            chunk_id="chunk_1",
            content="Annual leave is 24 days.",
            metadata={"source": "leave.txt"},
            score=0.9,
        ),
        RetrievalResult(
            chunk_id="chunk_2",
            content="Sick leave is 12 days.",
            metadata={"source": "leave.txt"},
            score=0.8,
        ),
        RetrievalResult(
            chunk_id="chunk_3",
            content="Unused leave may be carried forward.",
            metadata={"source": "leave.txt"},
            score=0.7,
        ),
    ]


def test_verifies_valid_citations():
    verifier = CitationVerifier()

    result = verifier.verify(
        citations=[1, 2],
        results=make_results(),
    )

    assert result.valid == [1, 2]
    assert result.invalid == []


def test_identifies_invalid_citations():
    verifier = CitationVerifier()

    result = verifier.verify(
        citations=[1, 4, 7],
        results=make_results(),
    )

    assert result.valid == [1]
    assert result.invalid == [4, 7]


def test_handles_empty_citations():
    verifier = CitationVerifier()

    result = verifier.verify(
        citations=[],
        results=make_results(),
    )

    assert result.valid == []
    assert result.invalid == []


def test_handles_no_results():
    verifier = CitationVerifier()

    result = verifier.verify(
        citations=[1, 2],
        results=[],
    )

    assert result.valid == []
    assert result.invalid == [1, 2]


def test_returns_sorted_citations():
    verifier = CitationVerifier()

    result = verifier.verify(
        citations=[3, 1, 2],
        results=make_results(),
    )

    assert result.valid == [1, 2, 3]
    assert result.invalid == []


def test_verification_result_is_structured():
    verifier = CitationVerifier()

    result = verifier.verify(
        citations=[1, 5],
        results=make_results(),
    )

    assert hasattr(result, "valid")
    assert hasattr(result, "invalid")
