from dataclasses import dataclass
from typing import Any


@dataclass
class RetrievalResult:
    """
    Common result object used by dense,
    sparse, fusion, and reranking stages.
    """

    chunk_id: str
    content: str
    metadata: dict[str, Any]
    score: float
