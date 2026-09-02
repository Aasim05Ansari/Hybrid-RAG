from app.generation.claims import (
    AnswerClaim,
    ClaimExtractor,
)


def test_extracts_claim_with_citation():

    extractor = ClaimExtractor()

    claims = extractor.extract(
        "Employees receive 24 days of annual leave [1]."
    )

    assert claims == [
        AnswerClaim(
            claim="Employees receive 24 days of annual leave",
            citation=1,
        )
    ]


def test_extracts_multiple_claims():

    extractor = ClaimExtractor()

    claims = extractor.extract(
        "Employees receive 24 days of annual leave [1]. "
        "Employees may take 12 days of sick leave [2]."
    )

    assert claims == [
        AnswerClaim(
            claim="Employees receive 24 days of annual leave",
            citation=1,
        ),
        AnswerClaim(
            claim="Employees may take 12 days of sick leave",
            citation=2,
        ),
    ]


def test_extracts_uncited_claim():

    extractor = ClaimExtractor()

    claims = extractor.extract(
        "Employees receive 24 days of annual leave."
    )

    assert claims == [
        AnswerClaim(
            claim="Employees receive 24 days of annual leave",
            citation=None,
        )
    ]


def test_handles_multiple_citations_for_one_sentence():

    extractor = ClaimExtractor()

    claims = extractor.extract(
        "Employees receive annual leave and sick leave [1] [2]."
    )

    assert claims == [
        AnswerClaim(
            claim="Employees receive annual leave and sick leave",
            citation=1,
        ),
        AnswerClaim(
            claim="Employees receive annual leave and sick leave",
            citation=2,
        ),
    ]


def test_returns_empty_for_empty_answer():

    extractor = ClaimExtractor()

    assert extractor.extract("") == []


def test_removes_citation_from_claim_text():

    extractor = ClaimExtractor()

    claims = extractor.extract(
        "Annual leave is 24 days [1]."
    )

    assert "[1]" not in claims[0].claim
    assert claims[0].citation == 1
