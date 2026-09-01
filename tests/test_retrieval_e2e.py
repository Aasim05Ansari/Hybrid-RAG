from pathlib import Path

from app.indexing.bm25_store import BM25Store
from app.indexing.vector_store import ChromaVectorStore
from app.retrieval.dense import DenseRetriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.pipeline import RetrievalPipeline
from app.retrieval.reranker import LexicalReranker
from app.retrieval.rrf import RRFFusion
from app.retrieval.sparse import SparseRetriever


class FakeEmbeddingProvider:
    """
    Deterministic local embeddings for tests.
    No OpenAI API calls and no API credits required.
    """

    def __init__(self):
        self.vectors = {
            "employee leave policy": [1.0, 0.0, 0.0],
            "annual leave employees 24 days": [0.9, 0.1, 0.0],
            "sick leave employees 12 days": [0.85, 0.15, 0.0],
            "leave carry forward": [0.8, 0.2, 0.0],
            "office attendance policy": [0.0, 1.0, 0.0],
            "remote work policy": [0.0, 0.9, 0.1],
        }

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text):
        text = text.lower()

        if "sick" in text:
            return [0.85, 0.15, 0.0]

        if "annual" in text or "leave" in text:
            return [0.9, 0.1, 0.0]

        if "carry" in text:
            return [0.8, 0.2, 0.0]

        if "remote" in text or "office" in text:
            return [0.0, 0.9, 0.1]

        return [0.5, 0.5, 0.0]


def build_pipeline(tmp_path):
    embedding_provider = FakeEmbeddingProvider()

    vector_store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma",
        collection_name="e2e_test",
    )

    bm25_store = BM25Store()

    documents = [
        "Employees are entitled to 24 days of annual leave per calendar year.",
        "Employees may take up to 12 days of sick leave per calendar year.",
        "Unused annual leave may be carried forward subject to company policy.",
        "Employees must follow the office attendance policy.",
        "Employees may work remotely according to the remote work policy.",
    ]

    ids = [
        "annual-leave",
        "sick-leave",
        "carry-forward",
        "office-policy",
        "remote-work",
    ]

    metadatas = [
        {"source": "leave.txt", "section": "Annual Leave"},
        {"source": "leave.txt", "section": "Sick Leave"},
        {"source": "leave.txt", "section": "Carry Forward"},
        {"source": "office.txt", "section": "Attendance"},
        {"source": "remote.txt", "section": "Remote Work"},
    ]

    embeddings = embedding_provider.embed_documents(documents)

    vector_store.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    bm25_store.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
    )

    dense_retriever = DenseRetriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    sparse_retriever = SparseRetriever(
        bm25_store=bm25_store,
    )

    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        fusion=RRFFusion(),
        dense_top_k=3,
        sparse_top_k=3,
        fusion_top_k=5,
        dense_weight=0.7,
        sparse_weight=0.3,
    )

    reranker = LexicalReranker()

    return RetrievalPipeline(
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        final_top_k=5,
    )


def test_end_to_end_retrieval_pipeline(tmp_path):
    pipeline = build_pipeline(tmp_path)

    results = pipeline.retrieve(
        "How many sick leave days can employees take?"
    )

    assert results
    assert len(results) <= 5

    result_ids = [result.chunk_id for result in results]

    assert "sick-leave" in result_ids

    top_result = results[0]

    assert top_result.chunk_id == "sick-leave"
    assert "12 days" in top_result.content
    assert top_result.metadata["section"] == "Sick Leave"


def test_end_to_end_retrieval_returns_metadata(tmp_path):
    pipeline = build_pipeline(tmp_path)

    results = pipeline.retrieve(
        "How many annual leave days are employees entitled to?"
    )

    assert results

    annual_result = next(
        result for result in results
        if result.chunk_id == "annual-leave"
    )

    assert annual_result.metadata["source"] == "leave.txt"
    assert annual_result.metadata["section"] == "Annual Leave"


def test_end_to_end_retrieval_respects_final_top_k(tmp_path):
    pipeline = build_pipeline(tmp_path)

    pipeline.final_top_k = 3

    results = pipeline.retrieve(
        "employee leave policy"
    )

    assert len(results) <= 3
