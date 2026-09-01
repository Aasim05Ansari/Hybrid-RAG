import re

from app.retrieval.models import RetrievalResult


class Reranker:

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        raise NotImplementedError


class LexicalReranker(Reranker):
    """
    Deterministic reranker used for development and testing.

    It scores candidates according to the proportion of
    query terms appearing in the candidate content.
    """

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:

        if not query.strip():
            raise ValueError(
                "Query cannot be empty"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0"
            )

        if not results:
            return []

        query_terms = self._tokenize(query)

        if not query_terms:
            return results[:top_k]

        scored_results = []

        for result in results:

            document_terms = self._tokenize(
                result.content
            )

            document_term_set = set(
                document_terms
            )

            matched_terms = sum(
                1
                for term in query_terms
                if term in document_term_set
            )

            lexical_score = (
                matched_terms
                / len(query_terms)
            )

            scored_results.append(
                (
                    lexical_score,
                    result,
                )
            )

        scored_results.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        reranked = []

        for lexical_score, result in scored_results:

            reranked.append(
                RetrievalResult(
                    chunk_id=result.chunk_id,
                    content=result.content,
                    metadata=result.metadata,
                    score=lexical_score,
                )
            )

        return reranked[:top_k]

    @staticmethod
    def _tokenize(text: str) -> list[str]:

        return re.findall(
            r"\b\w+\b",
            text.lower(),
        )
