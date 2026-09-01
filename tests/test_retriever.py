"""Unit tests for legal semantic retrieval orchestration."""

import pytest

from legal_rag.config import AppConfig
from legal_rag.rag import LegalRetriever, RetrievalResult


def _result(article_number: int, similarity: float) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=f"article-{article_number}",
        article_number=article_number,
        text=f"Text {article_number}",
        language="ar",
        book="Book",
        chapter=None,
        section=None,
        topic="Topic",
        is_repealed=False,
        source_page=article_number,
        citation=f"Article {article_number}",
        similarity=similarity,
    )


class FakeEmbedder:
    def __init__(self) -> None:
        self.queries: list[str] = []
        self.embedding = [0.5] * 384

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return self.embedding


class FakeRepository:
    def __init__(self, results: list[RetrievalResult] | None = None) -> None:
        self.results = results or []
        self.calls: list[tuple[list[float], int]] = []

    def semantic_search(
        self,
        embedding: list[float],
        top_k: int,
    ) -> list[RetrievalResult]:
        self.calls.append((embedding, top_k))
        return self.results


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_retriever_rejects_empty_query(query: str) -> None:
    embedder = FakeEmbedder()
    repository = FakeRepository()
    retriever = LegalRetriever(embedder, repository)

    with pytest.raises(ValueError, match="must not be empty"):
        retriever.search(query)

    assert embedder.queries == []
    assert repository.calls == []


@pytest.mark.parametrize("top_k", [0, -1])
def test_retriever_rejects_invalid_top_k(top_k: int) -> None:
    embedder = FakeEmbedder()
    repository = FakeRepository()
    retriever = LegalRetriever(embedder, repository)

    with pytest.raises(ValueError, match="top_k must be greater than zero"):
        retriever.search("valid query", top_k=top_k)

    assert embedder.queries == []


def test_retriever_passes_embedding_and_requested_top_k() -> None:
    embedder = FakeEmbedder()
    expected = [_result(2, 0.9), _result(1, 0.8)]
    repository = FakeRepository(expected)
    retriever = LegalRetriever(embedder, repository)

    results = retriever.search("ما هو العقد؟", top_k=2)

    assert embedder.queries == ["ما هو العقد؟"]
    assert repository.calls == [(embedder.embedding, 2)]
    assert results == expected
    assert [result.article_number for result in results] == [2, 1]
    assert [result.similarity for result in results] == [0.9, 0.8]


def test_retriever_uses_configured_default_top_k() -> None:
    embedder = FakeEmbedder()
    repository = FakeRepository()
    retriever = LegalRetriever(
        embedder,
        repository,
        AppConfig(retrieval_top_k=7),
    )

    retriever.search("query")

    assert repository.calls == [(embedder.embedding, 7)]
