from math import sqrt

from app.chunking.base import Chunker
from app.ingestion.metadata import Document, Chunk
from app.indexing.embeddings import EmbeddingProvider


class SemanticChunker(Chunker):

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        similarity_threshold: float = 0.75,
    ):
        if not 0 <= similarity_threshold <= 1:
            raise ValueError(
                "similarity_threshold must be between 0 and 1"
            )

        self.embedding_provider = embedding_provider
        self.similarity_threshold = similarity_threshold

    def chunk(self, document: Document) -> list[Chunk]:

        text = document.content.strip()

        if not text:
            return []

        sentences = self._split_sentences(text)

        if not sentences:
            return []

        if len(sentences) == 1:
            return [
                self._create_chunk(
                    document=document,
                    text=sentences[0],
                    index=0,
                )
            ]

        embeddings = self.embedding_provider.embed_documents(
            sentences
        )

        chunks = []
        current_sentences = [sentences[0]]
        chunk_index = 0

        for index in range(1, len(sentences)):

            similarity = self._cosine_similarity(
                embeddings[index - 1],
                embeddings[index],
            )

            if similarity < self.similarity_threshold:

                chunks.append(
                    self._create_chunk(
                        document=document,
                        text=" ".join(current_sentences),
                        index=chunk_index,
                    )
                )

                chunk_index += 1
                current_sentences = [sentences[index]]

            else:
                current_sentences.append(sentences[index])

        if current_sentences:

            chunks.append(
                self._create_chunk(
                    document=document,
                    text=" ".join(current_sentences),
                    index=chunk_index,
                )
            )

        return chunks

    @staticmethod
    def _split_sentences(text: str) -> list[str]:

        sentences = []

        for sentence in text.replace("!", ".").replace("?", ".").split("."):

            sentence = sentence.strip()

            if sentence:
                sentences.append(sentence)

        return sentences

    @staticmethod
    def _cosine_similarity(
        vector_a: list[float],
        vector_b: list[float],
    ) -> float:

        if len(vector_a) != len(vector_b):
            raise ValueError(
                "Embedding vectors must have the same dimension"
            )

        dot_product = sum(
            a * b
            for a, b in zip(vector_a, vector_b)
        )

        magnitude_a = sqrt(
            sum(a * a for a in vector_a)
        )

        magnitude_b = sqrt(
            sum(b * b for b in vector_b)
        )

        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0

        return dot_product / (
            magnitude_a * magnitude_b
        )

    @staticmethod
    def _create_chunk(
        document: Document,
        text: str,
        index: int,
    ) -> Chunk:

        chunk_id = (
            f"{document.metadata.get('document_id', 'doc')}"
            f"_chunk_{index}"
        )

        metadata = {
            **document.metadata,
            "chunk_index": index,
            "chunking_strategy": "semantic",
            "character_count": len(text),
        }

        return Chunk(
            chunk_id=chunk_id,
            content=text,
            metadata=metadata,
        )