from app.generation.claim_mapping import ClaimEvidence
from app.generation.claim_verifier import VerificationStatus
from app.generation.llm_claim_verifier import LLMClaimVerifier
from app.generation.llm import LLMProvider
from app.retrieval.models import RetrievalResult


class FakeLLMProvider(LLMProvider):

    def __init__(self, responses):
        self.responses = responses
        self.prompts = []
        self.index = 0

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)

        response = self.responses[self.index]
        self.index += 1

        return response


def make_claim(claim, evidence, citation=1):
    result = RetrievalResult(
        chunk_id="chunk-1",
        content=evidence,
        metadata={"source": "policy.pdf"},
        score=0.9,
    )

    return ClaimEvidence(
        claim=claim,
        citation=citation,
        evidence=result,
    )


def test_supported_claim():
    llm = FakeLLMProvider(
        ['{"status": "supported"}']
    )

    claims = [
        make_claim(
            "Employees receive 24 days of annual leave.",
            "Employees are entitled to 24 days of annual leave.",
        )
    ]

    results = LLMClaimVerifier(llm).verify(claims)

    assert results[0].status == VerificationStatus.SUPPORTED


def test_contradicted_claim():
    llm = FakeLLMProvider(
        ['{"status": "contradicted"}']
    )

    claims = [
        make_claim(
            "Employees receive 30 days of annual leave.",
            "Employees receive 24 days of annual leave.",
        )
    ]

    results = LLMClaimVerifier(llm).verify(claims)

    assert results[0].status == VerificationStatus.CONTRADICTED


def test_insufficient_claim():
    llm = FakeLLMProvider(
        ['{"status": "insufficient"}']
    )

    claims = [
        make_claim(
            "Employees can work remotely every Friday.",
            "Employees receive 24 days of annual leave.",
        )
    ]

    results = LLMClaimVerifier(llm).verify(claims)

    assert results[0].status == VerificationStatus.INSUFFICIENT


def test_uncited_claim_is_insufficient_without_calling_llm():
    llm = FakeLLMProvider([])

    claims = [
        ClaimEvidence(
            claim="The company was founded in 1990.",
            citation=None,
            evidence=None,
        )
    ]

    results = LLMClaimVerifier(llm).verify(claims)

    assert results[0].status == VerificationStatus.INSUFFICIENT
    assert llm.prompts == []


def test_invalid_llm_response_raises_error():
    llm = FakeLLMProvider(
        ["This is not JSON"]
    )

    claims = [
        make_claim(
            "Employees receive 24 days of annual leave.",
            "Employees receive 24 days of annual leave.",
        )
    ]

    try:
        LLMClaimVerifier(llm).verify(claims)
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "invalid verification response" in str(exc)


def test_invalid_status_raises_error():
    llm = FakeLLMProvider(
        ['{"status": "maybe"}']
    )

    claims = [
        make_claim(
            "Employees receive 24 days of annual leave.",
            "Employees receive 24 days of annual leave.",
        )
    ]

    try:
        LLMClaimVerifier(llm).verify(claims)
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "invalid verification response" in str(exc)


def test_prompt_contains_claim_and_evidence():
    llm = FakeLLMProvider(
        ['{"status": "supported"}']
    )

    claims = [
        make_claim(
            "Employees receive 24 days of annual leave.",
            "Employees are entitled to 24 days of annual leave.",
        )
    ]

    LLMClaimVerifier(llm).verify(claims)

    assert len(llm.prompts) == 1
    assert "Employees receive 24 days" in llm.prompts[0]
    assert "Employees are entitled to 24 days" in llm.prompts[0]
    assert "Do not use outside knowledge" in llm.prompts[0]
