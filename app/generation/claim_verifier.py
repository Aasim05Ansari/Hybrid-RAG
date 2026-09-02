from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from app.generation.claim_mapping import ClaimEvidence


class VerificationStatus(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    INSUFFICIENT = "insufficient"


@dataclass
class ClaimVerification:
    claim: str
    citation: int | None
    status: VerificationStatus
    evidence: str | None


class ClaimVerifier(ABC):

    @abstractmethod
    def verify(
        self,
        claims: list[ClaimEvidence],
    ) -> list[ClaimVerification]:
        raise NotImplementedError
