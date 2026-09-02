from dataclasses import dataclass

from app.generation.claims import AnswerClaim
from app.retrieval.models import RetrievalResult


@dataclass
class ClaimEvidence:
    claim: str
    citation: int | None
    evidence: RetrievalResult | None


class ClaimCitationMapper:
    def map(
        self,
        claims: list[AnswerClaim],
        results: list[RetrievalResult],
    ) -> list[ClaimEvidence]:
        mapped = []

        for claim in claims:
            evidence = None

            if claim.citation is not None:
                index = claim.citation - 1

                if 0 <= index < len(results):
                    evidence = results[index]

            mapped.append(
                ClaimEvidence(
                    claim=claim.claim,
                    citation=claim.citation,
                    evidence=evidence,
                )
            )

        return mapped
