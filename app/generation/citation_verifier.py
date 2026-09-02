from dataclasses import dataclass

from app.retrieval.models import RetrievalResult


@dataclass
class CitationVerificationResult:
    valid: list[int]
    invalid: list[int]


class CitationVerifier:

    def verify(
        self,
        citations: list[int],
        results: list[RetrievalResult],
    ) -> CitationVerificationResult:

        if not citations:
            return CitationVerificationResult(
                valid=[],
                invalid=[],
            )

        valid_range = set(
            range(1, len(results) + 1)
        )

        valid = sorted(
            citation
            for citation in citations
            if citation in valid_range
        )

        invalid = sorted(
            citation
            for citation in citations
            if citation not in valid_range
        )

        return CitationVerificationResult(
            valid=valid,
            invalid=invalid,
        )
