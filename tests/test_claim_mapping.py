from app.generation.claim_mapping import ClaimCitationMapper
from app.generation.claims import AnswerClaim
from app.retrieval.models import RetrievalResult


def make_result(chunk_id, content, source):
    return RetrievalResult(
        chunk_id=chunk_id,
        content=content,
        metadata={"source": source},
        score=0.9,
    )


def test_maps_claim_to_cited_result():
    claims = [
        AnswerClaim(
            claim="Employees receive 24 days of annual leave",
            citation=1,
        )
    ]

    results = [
        make_result(
            "chunk-1",
            "Employees are entitled to 24 days of annual leave.",
            "employee_policy.pdf",
        )
    ]

    mapped = ClaimCitationMapper().map(claims, results)

    assert len(mapped) == 1
    assert mapped[0].claim == "Employees receive 24 days of annual leave"
    assert mapped[0].citation == 1
    assert mapped[0].evidence == results[0]


def test_maps_multiple_claims_to_different_results():
    claims = [
        AnswerClaim(
            claim="Employees receive 24 days of annual leave",
            citation=1,
        ),
        AnswerClaim(
            claim="Employees may take 12 days of sick leave",
            citation=2,
        ),
    ]

    results = [
        make_result(
            "chunk-1",
            "Employees receive 24 days of annual leave.",
            "leave_policy.pdf",
        ),
        make_result(
            "chunk-2",
            "Employees may take 12 days of sick leave.",
            "leave_policy.pdf",
        ),
    ]

    mapped = ClaimCitationMapper().map(claims, results)

    assert mapped[0].evidence == results[0]
    assert mapped[1].evidence == results[1]


def test_uncited_claim_has_no_evidence():
    claims = [
        AnswerClaim(
            claim="The company was founded in 1990",
            citation=None,
        )
    ]

    results = [
        make_result(
            "chunk-1",
            "Employees receive 24 days of annual leave.",
            "leave_policy.pdf",
        )
    ]

    mapped = ClaimCitationMapper().map(claims, results)

    assert mapped[0].citation is None
    assert mapped[0].evidence is None


def test_invalid_citation_has_no_evidence():
    claims = [
        AnswerClaim(
            claim="Employees receive 50 days of leave",
            citation=5,
        )
    ]

    results = [
        make_result(
            "chunk-1",
            "Employees receive 24 days of annual leave.",
            "leave_policy.pdf",
        )
    ]

    mapped = ClaimCitationMapper().map(claims, results)

    assert mapped[0].citation == 5
    assert mapped[0].evidence is None


def test_empty_claims_returns_empty():
    assert ClaimCitationMapper().map([], []) == []
