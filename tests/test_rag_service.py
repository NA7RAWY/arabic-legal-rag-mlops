"""Unit tests for RAG service orchestration."""

import pytest

from legal_rag.rag import LegalRAGService, NoRetrievedContextError
from legal_rag.rag.generator import build_legal_context
from legal_rag.storage import RetrievalResult


def _result(article_number: int, similarity: float) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=f"article-{article_number}",
        article_number=article_number,
        text=f"Text {article_number}",
        language="en",
        book=None,
        chapter=None,
        section=None,
        topic=None,
        is_repealed=False,
        source_page=1,
        citation=f"Article {article_number}",
        similarity=similarity,
    )


class FakeRetriever:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results
        self.calls: list[tuple[str, int | None]] = []

    def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        self.calls.append((query, top_k))
        return self.results


class FakeGenerator:
    def __init__(self, answer: str = "Grounded answer") -> None:
        self.answer = answer
        self.calls: list[tuple[str, str]] = []
        self.error: Exception | None = None

    def generate(self, question: str, context: str) -> str:
        self.calls.append((question, context))
        if self.error is not None:
            raise self.error
        return self.answer


def test_rag_service_orchestrates_and_preserves_sources() -> None:
    sources = [_result(148, 0.9), _result(149, 0.8)]
    retriever = FakeRetriever(sources)
    generator = FakeGenerator("Answer citing Articles 148 and 149")
    service = LegalRAGService(retriever, generator)

    result = service.answer("What governs contracts?", top_k=2)

    assert retriever.calls == [("What governs contracts?", 2)]
    assert generator.calls == [
        ("What governs contracts?", build_legal_context(sources))
    ]
    assert result.question == "What governs contracts?"
    assert result.answer == "Answer citing Articles 148 and 149"
    assert result.retrieved_sources == tuple(sources)


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_rag_service_rejects_empty_question(question: str) -> None:
    retriever = FakeRetriever([])
    generator = FakeGenerator()
    service = LegalRAGService(retriever, generator)

    with pytest.raises(ValueError, match="Question must not be empty"):
        service.answer(question)

    assert retriever.calls == []
    assert generator.calls == []


def test_rag_service_rejects_invalid_top_k() -> None:
    retriever = FakeRetriever([])
    generator = FakeGenerator()

    with pytest.raises(ValueError, match="top_k must be greater than zero"):
        LegalRAGService(retriever, generator).answer("question", top_k=0)

    assert retriever.calls == []


def test_rag_service_rejects_missing_retrieved_context() -> None:
    retriever = FakeRetriever([])
    generator = FakeGenerator()

    with pytest.raises(NoRetrievedContextError, match="No legal context"):
        LegalRAGService(retriever, generator).answer("question")

    assert generator.calls == []


def test_rag_service_propagates_generator_failure() -> None:
    source = _result(148, 0.9)
    generator = FakeGenerator()
    generator.error = RuntimeError("generation failed")
    service = LegalRAGService(FakeRetriever([source]), generator)

    with pytest.raises(RuntimeError, match="generation failed"):
        service.answer("question")

    assert generator.calls == [("question", build_legal_context([source]))]
