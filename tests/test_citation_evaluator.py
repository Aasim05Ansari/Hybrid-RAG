from app.generation.citation_evaluator import CitationEvaluator
from app.generation.claim_verifier import (
    ClaimVerification,
    VerificationStatus,
)


def make_verification(
    claim,
    citation,
    status,
):
    return ClaimVerification(
        claim=claim,
        citation=citation,
        status=status,
        evidence="Evidence",
    )


def test_all_claims_supported():
    verifications = [
        make_verification(
            "Claim one",
            1,
            VerificationStatus.SUPPORTED,
        ),
        make_verification(
            "Claim two",
            2,
            VerificationStatus.SUPPORTED,
        ),
    ]

    result = CitationEvaluator().evaluate(verifications)

    assert result.citation_accuracy == 1.0
    assert result.citation_presence == 1.0
    assert result.citation_coverage == 1.0

    assert result.supported_claims == 2
    assert result.contradicted_claims == 0
    assert result.insufficient_claims == 0
    assert result.cited_claims == 2
    assert result.total_claims == 2


def test_one_contradicted_citation():
    verifications = [
        make_verification(
            "Claim one",
            1,
            VerificationStatus.SUPPORTED,
        ),
        make_verification(
            "Claim two",
            2,
            VerificationStatus.CONTRADICTED,
        ),
    ]

    result = CitationEvaluator().evaluate(verifications)

    assert result.citation_accuracy == 0.5
    assert result.citation_presence == 1.0
    assert result.citation_coverage == 0.5

    assert result.supported_claims == 1
    assert result.contradicted_claims == 1


def test_uncited_claim_reduces_presence_and_coverage():
    verifications = [
        make_verification(
            "Claim one",
            1,
            VerificationStatus.SUPPORTED,
        ),
        make_verification(
            "Claim two",
            None,
            VerificationStatus.INSUFFICIENT,
        ),
    ]

    result = CitationEvaluator().evaluate(verifications)

    # One of two claims has a citation.
    assert result.citation_presence == 0.5

    # The one citation that exists is correct.
    assert result.citation_accuracy == 1.0

    # One of two claims is actually supported.
    assert result.citation_coverage == 0.5

    assert result.cited_claims == 1
    assert result.total_claims == 2


def test_all_citations_wrong():
    verifications = [
        make_verification(
            "Claim one",
            1,
            VerificationStatus.CONTRADICTED,
        ),
        make_verification(
            "Claim two",
            2,
            VerificationStatus.INSUFFICIENT,
        ),
    ]

    result = CitationEvaluator().evaluate(verifications)

    assert result.citation_accuracy == 0.0
    assert result.citation_presence == 1.0
    assert result.citation_coverage == 0.0


def test_no_claims():
    result = CitationEvaluator().evaluate([])

    assert result.citation_accuracy == 0.0
    assert result.citation_presence == 0.0
    assert result.citation_coverage == 0.0

    assert result.supported_claims == 0
    assert result.contradicted_claims == 0
    assert result.insufficient_claims == 0
    assert result.cited_claims == 0
    assert result.total_claims == 0


def test_mixed_claims():
    verifications = [
        make_verification(
            "Claim one",
            1,
            VerificationStatus.SUPPORTED,
        ),
        make_verification(
            "Claim two",
            2,
            VerificationStatus.SUPPORTED,
        ),
        make_verification(
            "Claim three",
            3,
            VerificationStatus.CONTRADICTED,
        ),
        make_verification(
            "Claim four",
            None,
            VerificationStatus.INSUFFICIENT,
        ),
    ]

    result = CitationEvaluator().evaluate(verifications)

    # 2 of 3 cited claims are supported.
    assert result.citation_accuracy == 2 / 3

    # 3 of 4 claims have citations.
    assert result.citation_presence == 3 / 4

    # 2 of 4 total claims are supported.
    assert result.citation_coverage == 2 / 4

    assert result.supported_claims == 2
    assert result.contradicted_claims == 1
    assert result.insufficient_claims == 1
    assert result.cited_claims == 3
    assert result.total_claims == 4
