from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from app.config.settings import settings


class BM25Store:
    """Persistent sparse retrieval store backed by BM25Okapi."""

    FORMAT_VERSION = 1
    TOKENIZER_VERSION = "regex_word_v2"
    INDEX_FILENAME = "index.json"

    def __init__(
        self,
        persist_directory: str | Path | None = None,
    ):
        self.persist_directory = Path(
            persist_directory
            or settings.bm25_persist_dir
        )

        self.index_path = (
            self.persist_directory / self.INDEX_FILENAME
        )

        self.ids: list[str] = []
        self.documents: list[str] = []
        self.metadatas: list[dict[str, Any]] = []
        self.bm25: BM25Okapi | None = None

        self._load()

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

        existing = {
            chunk_id: index
            for index, chunk_id in enumerate(self.ids)
        }

        for chunk_id, document, metadata in zip(
            ids,
            documents,
            metadatas,
        ):
            if chunk_id in existing:
                index = existing[chunk_id]

                self.documents[index] = document
                self.metadatas[index] = metadata

            else:
                existing[chunk_id] = len(self.ids)

                self.ids.append(chunk_id)
                self.documents.append(document)
                self.metadatas.append(metadata)

        self._rebuild()

        self.save()

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

    def save(self) -> None:
        """Persist the BM25 corpus state to disk."""

        self.persist_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "format_version": self.FORMAT_VERSION,
            "tokenizer_version": self.TOKENIZER_VERSION,
            "ids": self.ids,
            "documents": self.documents,
            "metadatas": self.metadatas,
        }

        temporary_path = (
            self.persist_directory
            / f"{self.INDEX_FILENAME}.tmp"
        )

        temporary_path.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        temporary_path.replace(self.index_path)

    def load(self) -> None:
        """Load persisted BM25 corpus state and rebuild the index."""

        self._load()

    @property
    def document_count(self) -> int:
        return len(self.ids)

    @property
    def is_ready(self) -> bool:
        return self.bm25 is not None

    def _load(self) -> None:

        if not self.index_path.exists():
            return
        
        if self.index_path.stat().st_size == 0:
            return

        try:
            payload = json.loads(
                self.index_path.read_text(
                    encoding="utf-8"
                )
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid BM25 index file: {self.index_path}"
            ) from exc

        if payload.get("format_version") != self.FORMAT_VERSION:
            raise ValueError(
                "Unsupported BM25 index format version"
            )

        if (
            payload.get("tokenizer_version")
            != self.TOKENIZER_VERSION
        ):
            raise ValueError(
                "Unsupported BM25 tokenizer version"
            )

        ids = payload.get("ids")
        documents = payload.get("documents")
        metadatas = payload.get("metadatas")

        if not isinstance(ids, list):
            raise ValueError(
                "BM25 index ids must be a list"
            )

        if not isinstance(documents, list):
            raise ValueError(
                "BM25 index documents must be a list"
            )

        if not isinstance(metadatas, list):
            raise ValueError(
                "BM25 index metadatas must be a list"
            )

        if not (
            len(ids)
            == len(documents)
            == len(metadatas)
        ):
            raise ValueError(
                "Persisted BM25 index contains "
                "mismatched lengths"
            )

        self.ids = ids
        self.documents = documents
        self.metadatas = metadatas

        self._rebuild()

    def _rebuild(self) -> None:

        if not self.documents:
            self.bm25 = None
            return

        tokenized_documents = [
            self._tokenize(document)
            for document in self.documents
        ]

        self.bm25 = BM25Okapi(
            tokenized_documents
        )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(
            r"\b\w+\b",
            text.lower(),
        )