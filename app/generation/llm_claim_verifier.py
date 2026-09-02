import json

from app.generation.claim_mapping import ClaimEvidence
from app.generation.claim_verifier import (
    ClaimVerification,
    ClaimVerifier,
    VerificationStatus,
)
from app.generation.llm import LLMProvider


class LLMClaimVerifier(ClaimVerifier):
    """
    Semantic claim verifier using an LLMProvider.

    The LLM receives a claim and its cited evidence and must classify
    the relationship as:

        supported
        contradicted
        insufficient
    """

    VERIFICATION_PROMPT = """You are a citation verification system.

Determine whether the evidence supports the claim.

Return ONLY valid JSON in exactly this format:

{{
  "status": "supported"
}}

The status must be exactly one of:

- "supported"
- "contradicted"
- "insufficient"

Definitions:

SUPPORTED:
The evidence directly supports the claim.

CONTRADICTED:
The evidence directly conflicts with the claim.

INSUFFICIENT:
The evidence does not provide enough information to establish
or contradict the claim.

Do not use outside knowledge.

CLAIM:
{claim}

EVIDENCE:
{evidence}
"""

    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider

    def verify(
        self,
        claims: list[ClaimEvidence],
    ) -> list[ClaimVerification]:

        results = []

        for item in claims:

            if item.evidence is None:
                results.append(
                    ClaimVerification(
                        claim=item.claim,
                        citation=item.citation,
                        status=VerificationStatus.INSUFFICIENT,
                        evidence=None,
                    )
                )
                continue

            evidence_text = item.evidence.content

            prompt = self.VERIFICATION_PROMPT.format(
                claim=item.claim,
                evidence=evidence_text,
            )

            response = self.llm_provider.generate(prompt)

            status = self._parse_status(response)

            results.append(
                ClaimVerification(
                    claim=item.claim,
                    citation=item.citation,
                    status=status,
                    evidence=evidence_text,
                )
            )

        return results

    @staticmethod
    def _parse_status(response: str) -> VerificationStatus:
        try:
            data = json.loads(response)
            value = data["status"].strip().lower()
            return VerificationStatus(value)

        except (json.JSONDecodeError, KeyError, ValueError):
            raise ValueError(
                "LLM verifier returned an invalid verification response"
            )
