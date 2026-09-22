"""Unit tests for RAG service orchestration."""

from collections.abc import Iterator

import pytest
from prometheus_client import CollectorRegistry

from legal_rag.monitoring import PrometheusMetrics
from legal_rag.rag import LegalRAGService, NoRetrievedContextError
from legal_rag.rag.generator import (
    GenerationChunk,
    GenerationResult,
    LLMUsage,
    build_legal_context,
)
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
    def __init__(
        self,
        answer: str = "Grounded answer",
        chunks: tuple[str, ...] = ("Grounded ", "answer"),
    ) -> None:
        self.answer = answer
        self.chunks = chunks
        self.calls: list[tuple[str, str]] = []
        self.stream_calls: list[tuple[str, str]] = []
        self.error: Exception | None = None

    def generate(self, question: str, context: str) -> str:
        self.calls.append((question, context))
        if self.error is not None:
            raise self.error
        return self.answer

    def stream_generate(self, question: str, context: str) -> Iterator[str]:
        self.stream_calls.append((question, context))
        if self.error is not None:
            raise self.error
        return iter(self.chunks)


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


def test_rag_service_streams_with_one_retrieval_and_preserves_sources() -> None:
    sources = [_result(148, 0.9), _result(149, 0.8)]
    retriever = FakeRetriever(sources)
    generator = FakeGenerator(chunks=("one", "two"))
    service = LegalRAGService(retriever, generator)

    result = service.stream_answer("What governs contracts?", top_k=2)

    assert retriever.calls == [("What governs contracts?", 2)]
    assert generator.stream_calls == [
        ("What governs contracts?", build_legal_context(sources))
    ]
    assert list(result.chunks) == ["onetwo"]
    assert result.retrieved_sources == tuple(sources)
    assert retriever.calls == [("What governs contracts?", 2)]


def test_service_records_authoritative_stream_usage_once() -> None:
    class UsageGenerator(FakeGenerator):
        def stream_generate_with_usage(
            self, question: str, context: str
        ) -> Iterator[GenerationChunk]:
            self.stream_calls.append((question, context))
            yield GenerationChunk("answer", LLMUsage(10, 1, 11))
            yield GenerationChunk(" text", LLMUsage(10, 2, 12))

    metrics = PrometheusMetrics(CollectorRegistry())
    service = LegalRAGService(
        FakeRetriever([_result(148, 0.9)]),
        UsageGenerator(),
        metrics=metrics,
        provider="gemini",
        input_cost_per_million_tokens_usd=1.0,
        output_cost_per_million_tokens_usd=2.0,
    )

    result = service.stream_answer("question")

    assert list(result.chunks) == ["answer ", "text"]
    assert metrics.registry.get_sample_value(
        "legal_rag_llm_tokens_total",
        {"provider": "gemini", "token_type": "input", "mode": "stream"},
    ) == pytest.approx(10)
    assert metrics.registry.get_sample_value(
        "legal_rag_llm_tokens_total",
        {"provider": "gemini", "token_type": "output", "mode": "stream"},
    ) == pytest.approx(2)


def test_service_records_authoritative_full_usage_without_estimating_text() -> None:
    class UsageGenerator(FakeGenerator):
        def generate_with_usage(self, question: str, context: str) -> GenerationResult:
            self.calls.append((question, context))
            return GenerationResult("a very long answer", LLMUsage(total_tokens=7))

    metrics = PrometheusMetrics(CollectorRegistry())
    service = LegalRAGService(
        FakeRetriever([_result(148, 0.9)]),
        UsageGenerator(),
        metrics=metrics,
        provider="vllm",
        input_cost_per_million_tokens_usd=100.0,
        output_cost_per_million_tokens_usd=100.0,
    )

    service.answer("question")

    assert metrics.registry.get_sample_value(
        "legal_rag_llm_tokens_total",
        {"provider": "vllm", "token_type": "total", "mode": "full"},
    ) == pytest.approx(7)
    assert (
        metrics.registry.get_sample_value(
            "legal_rag_llm_usage_cost_usd_total",
            {"provider": "vllm", "mode": "full"},
        )
        is None
    )
