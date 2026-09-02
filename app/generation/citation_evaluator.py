from dataclasses import dataclass

from app.generation.claim_verifier import (
    ClaimVerification,
    VerificationStatus,
)


@dataclass
class CitationEvaluation:
    citation_accuracy: float
    citation_presence: float
    citation_coverage: float
    supported_claims: int
    contradicted_claims: int
    insufficient_claims: int
    cited_claims: int
    total_claims: int


class CitationEvaluator:

    def evaluate(
        self,
        verifications: list[ClaimVerification],
    ) -> CitationEvaluation:

        if not verifications:
            return CitationEvaluation(
                citation_accuracy=0.0,
                citation_presence=0.0,
                citation_coverage=0.0,
                supported_claims=0,
                contradicted_claims=0,
                insufficient_claims=0,
                cited_claims=0,
                total_claims=0,
            )

        supported = sum(
            1
            for item in verifications
            if item.status == VerificationStatus.SUPPORTED
        )

        contradicted = sum(
            1
            for item in verifications
            if item.status == VerificationStatus.CONTRADICTED
        )

        insufficient = sum(
            1
            for item in verifications
            if item.status == VerificationStatus.INSUFFICIENT
        )

        total = len(verifications)

        cited_claims = sum(
            1
            for item in verifications
            if item.citation is not None
        )

        # Accuracy:
        # Of the claims that contain citations,
        # what percentage are actually supported?
        if cited_claims:
            citation_accuracy = supported / cited_claims
        else:
            citation_accuracy = 0.0

        # Presence:
        # What percentage of all claims contain citations?
        citation_presence = cited_claims / total

        # Coverage:
        # What percentage of all claims are actually
        # supported by cited evidence?
        citation_coverage = supported / total

        return CitationEvaluation(
            citation_accuracy=citation_accuracy,
            citation_presence=citation_presence,
            citation_coverage=citation_coverage,
            supported_claims=supported,
            contradicted_claims=contradicted,
            insufficient_claims=insufficient,
            cited_claims=cited_claims,
            total_claims=total,
        )
