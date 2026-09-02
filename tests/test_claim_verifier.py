from app.generation.claim_mapping import ClaimEvidence
from app.generation.claim_verifier import (
    ClaimVerifier,
    ClaimVerification,
    VerificationStatus,
)


def test_verification_status_values():
    assert VerificationStatus.SUPPORTED.value == "supported"
    assert VerificationStatus.CONTRADICTED.value == "contradicted"
    assert VerificationStatus.INSUFFICIENT.value == "insufficient"


def test_claim_verification_dataclass():
    result = ClaimVerification(
        claim="Employees receive 24 days of annual leave",
        citation=1,
        status=VerificationStatus.SUPPORTED,
        evidence="Employees are entitled to 24 days of annual leave.",
    )

    assert result.claim == "Employees receive 24 days of annual leave"
    assert result.citation == 1
    assert result.status == VerificationStatus.SUPPORTED
    assert result.evidence == (
        "Employees are entitled to 24 days of annual leave."
    )


def test_claim_verifier_is_abstract():
    try:
        ClaimVerifier()
        assert False, "ClaimVerifier should be abstract"
    except TypeError:
        pass


def test_uncited_claim_can_be_represented():
    claim = ClaimEvidence(
        claim="The company was founded in 1990",
        citation=None,
        evidence=None,
    )

    result = ClaimVerification(
        claim=claim.claim,
        citation=claim.citation,
        status=VerificationStatus.INSUFFICIENT,
        evidence=claim.evidence,
    )

    assert result.status == VerificationStatus.INSUFFICIENT
    assert result.citation is None
    assert result.evidence is None
