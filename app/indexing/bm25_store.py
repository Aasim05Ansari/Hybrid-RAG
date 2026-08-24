from typing import Any

from rank_bm25 import BM25Okapi


class BM25Store:

    def __init__(self):
        self.ids: list[str] = []
        self.documents: list[str] = []
        self.metadatas: list[dict[str, Any]] = []
        self.bm25: BM25Okapi | None = None

    def add(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:

        if not ids:
            return

        if not (
            len(ids)
            == len(documents)
            == len(metadatas)
        ):
            raise ValueError(
                "ids, documents, and metadatas "
                "must have the same length"
            )

        self.ids.extend(ids)
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)

        tokenized_documents = [
            self._tokenize(document)
            for document in self.documents
        ]

        self.bm25 = BM25Okapi(
            tokenized_documents
        )

    def search(
        self,
        query: str,
        top_k: int,
    ) -> list[dict[str, Any]]:

        if not query.strip():
            raise ValueError(
                "Query cannot be empty"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero"
            )

        if self.bm25 is None:
            return []

        tokenized_query = self._tokenize(query)

        scores = self.bm25.get_scores(
            tokenized_query
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        results = []

        for index in ranked_indices[:top_k]:

            results.append(
                {
                    "id": self.ids[index],
                    "document": self.documents[index],
                    "metadata": self.metadatas[index],
                    "score": float(scores[index]),
                }
            )

        return results

    @staticmethod
    def _tokenize(text: str) -> list[str]:

        return text.lower().split()