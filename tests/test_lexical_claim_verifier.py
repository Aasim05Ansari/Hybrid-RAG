from app.generation.claim_mapping import ClaimEvidence
from app.generation.claim_verifier import VerificationStatus
from app.generation.lexical_claim_verifier import LexicalClaimVerifier
from app.retrieval.models import RetrievalResult


def make_evidence(
    claim,
    content,
    citation=1,
):
    result = RetrievalResult(
        chunk_id="chunk-1",
        content=content,
        metadata={"source": "policy.pdf"},
        score=0.9,
    )

    return ClaimEvidence(
        claim=claim,
        citation=citation,
        evidence=result,
    )


def test_supported_claim():
    claims = [
        make_evidence(
            "Employees receive 24 days of annual leave",
            "Employees receive 24 days of annual leave per calendar year.",
        )
    ]

    results = LexicalClaimVerifier().verify(claims)

    assert results[0].status == VerificationStatus.SUPPORTED


def test_contradicted_numeric_claim():
    claims = [
        make_evidence(
            "Employees receive 30 days of annual leave",
            "Employees receive 24 days of annual leave per calendar year.",
        )
    ]

    results = LexicalClaimVerifier().verify(claims)

    assert results[0].status == VerificationStatus.CONTRADICTED


def test_insufficient_evidence():
    claims = [
        make_evidence(
            "Employees receive 24 days of annual leave",
            "Employees may take sick leave when they are unwell.",
        )
    ]

    results = LexicalClaimVerifier().verify(claims)

    assert results[0].status == VerificationStatus.INSUFFICIENT


def test_missing_evidence_is_insufficient():
    claims = [
        ClaimEvidence(
            claim="The company was founded in 1990",
            citation=5,
            evidence=None,
        )
    ]

    results = LexicalClaimVerifier().verify(claims)

    assert results[0].status == VerificationStatus.INSUFFICIENT
    assert results[0].evidence is None


def test_multiple_claims():
    claims = [
        make_evidence(
            "Employees receive 24 days of annual leave",
            "Employees receive 24 days of annual leave per year.",
            citation=1,
        ),
        make_evidence(
            "Employees can work remotely every Friday",
            "Employees receive 12 days of sick leave.",
            citation=2,
        ),
    ]

    results = LexicalClaimVerifier().verify(claims)

    assert results[0].status == VerificationStatus.SUPPORTED
    assert results[1].status == VerificationStatus.INSUFFICIENT


def test_invalid_threshold():
    try:
        LexicalClaimVerifier(support_threshold=0)
        assert False
    except ValueError:
        pass


def test_decimal_numbers_can_be_detected():
    claims = [
        make_evidence(
            "The threshold is 0.8",
            "The threshold is 0.5.",
        )
    ]

    results = LexicalClaimVerifier().verify(claims)

    assert results[0].status == VerificationStatus.CONTRADICTED
