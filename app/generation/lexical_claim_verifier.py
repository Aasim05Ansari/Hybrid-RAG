import re

from app.generation.claim_mapping import ClaimEvidence
from app.generation.claim_verifier import (
    ClaimVerification,
    ClaimVerifier,
    VerificationStatus,
)


class LexicalClaimVerifier(ClaimVerifier):
    """
    Deterministic baseline verifier.

    This verifier is intentionally simple. It uses lexical overlap
    to detect support and explicit numeric conflicts to detect
    straightforward contradictions.

    It is a development/test baseline, not a semantic verifier.
    """

    def __init__(self, support_threshold: float = 0.6):
        if not 0 < support_threshold <= 1:
            raise ValueError("support_threshold must be between 0 and 1")

        self.support_threshold = support_threshold

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

            if self._has_numeric_conflict(
                item.claim,
                evidence_text,
            ):
                status = VerificationStatus.CONTRADICTED

            else:
                claim_tokens = self._tokens(item.claim)
                evidence_tokens = self._tokens(evidence_text)

                if not claim_tokens or not evidence_tokens:
                    status = VerificationStatus.INSUFFICIENT

                else:
                    overlap = len(
                        claim_tokens.intersection(evidence_tokens)
                    ) / len(claim_tokens)

                    if overlap >= self.support_threshold:
                        status = VerificationStatus.SUPPORTED
                    else:
                        status = VerificationStatus.INSUFFICIENT

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
    def _tokens(text: str) -> set[str]:
        return set(re.findall(r"\b\w+\b", text.lower()))

    @staticmethod
    def _has_numeric_conflict(
        claim: str,
        evidence: str,
    ) -> bool:
        """
        Detect simple numeric contradictions.

        Example:

        Claim:
            Employees receive 30 days of annual leave.

        Evidence:
            Employees receive 24 days of annual leave.

        Returns:
            True
        """

        claim_numbers = re.findall(r"\b\d+(?:\.\d+)?\b", claim)
        evidence_numbers = re.findall(r"\b\d+(?:\.\d+)?\b", evidence)

        if not claim_numbers or not evidence_numbers:
            return False

        claim_tokens = LexicalClaimVerifier._tokens(claim)
        evidence_tokens = LexicalClaimVerifier._tokens(evidence)

        shared_terms = claim_tokens.intersection(evidence_tokens)

        # Ignore numbers when determining whether the two statements
        # talk about the same subject.
        shared_terms -= set(claim_numbers)
        shared_terms -= set(evidence_numbers)

        if not shared_terms:
            return False

        return any(
            claim_number not in evidence_numbers
            for claim_number in claim_numbers
        )
