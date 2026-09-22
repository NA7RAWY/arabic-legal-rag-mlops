"""Prometheus monitoring tests without external services."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry, generate_latest

from legal_rag.api.app import create_app
from legal_rag.api.dependencies import get_rag_service
from legal_rag.config import AppConfig
from legal_rag.monitoring import PrometheusMetrics
from legal_rag.rag import (
    GenerationError,
    LegalRAGResult,
    LegalRAGService,
    LegalRAGStreamResult,
)
from legal_rag.storage import RetrievalResult


def _source(article_number: int = 164) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=f"article-{article_number}",
        article_number=article_number,
        text="كل خطأ سبب ضرراً للغير يلزم من ارتكبه بالتعويض.",
        language="ar",
        book="Book I",
        chapter="Chapter III",
        section=None,
        topic=None,
        is_repealed=False,
        source_page=24,
        citation=f"Egyptian Civil Code, Article {article_number}",
        similarity=0.91,
    )


class FakeAPIService:
    def answer(self, question: str, top_k: int | None = None) -> LegalRAGResult:
        return LegalRAGResult(question, "إجابة قانونية.", (_source(),))

    def stream_answer(
        self,
        question: str,
        top_k: int | None = None,
    ) -> LegalRAGStreamResult:
        return LegalRAGStreamResult(question, iter(("إجابة ", "متدفقة")), (_source(),))


class FakeRetriever:
    def __init__(self, sources: list[RetrievalResult]) -> None:
        self.sources = sources

    def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        return self.sources


class FakeGenerator:
    def __init__(self, *, stream_error: bool = False) -> None:
        self.stream_error = stream_error

    def generate(self, question: str, context: str) -> str:
        return "Grounded answer"

    def stream_generate(self, question: str, context: str) -> Iterator[str]:
        yield "first"
        if self.stream_error:
            raise GenerationError("provider failed")
        yield "second"


class FakeClock:
    def __init__(self, values: list[float]) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)


def _metrics() -> PrometheusMetrics:
    return PrometheusMetrics(CollectorRegistry())


def test_metrics_endpoint_exposes_low_cardinality_http_and_ask_metrics() -> None:
    metrics = _metrics()
    app = create_app(AppConfig(retrieval_top_k=5), metrics=metrics)
    app.dependency_overrides[get_rag_service] = FakeAPIService
    client = TestClient(app)

    assert client.get("/health").status_code == 200
    assert client.post("/ask", json={"question": "سؤال سري"}).status_code == 200
    assert client.post("/ask", json={"question": " "}).status_code == 422
    stream = client.post("/ask/stream", json={"question": "سؤال آخر"})
    assert stream.status_code == 200

    response = client.get("/metrics")
    body = response.text

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert (
        'legal_rag_http_requests_total{method="GET",route="/health",status="200"} 1.0'
        in body
    )
    assert (
        'legal_rag_http_requests_total{method="POST",route="/ask",status="422"} 1.0'
        in body
    )
    assert (
        'legal_rag_http_errors_total{method="POST",route="/ask",status="422"} 1.0'
        in body
    )
    assert 'legal_rag_ask_requests_total{mode="full"} 1.0' in body
    assert 'legal_rag_ask_requests_total{mode="stream"} 1.0' in body
    assert "legal_rag_http_request_duration_seconds" in body
    assert "سؤال سري" not in body
    assert "إجابة قانونية" not in body


def test_rag_service_records_retrieval_generation_and_source_count() -> None:
    metrics = _metrics()
    service = LegalRAGService(
        FakeRetriever([_source(148), _source(149)]),
        FakeGenerator(),
        metrics=metrics,
        provider="gemini",
        clock=FakeClock([1.0, 1.2, 2.0, 2.5]),
    )

    result = service.answer("question", top_k=2)

    assert len(result.retrieved_sources) == 2
    assert metrics.registry.get_sample_value(
        "legal_rag_retrieval_duration_seconds_count"
    ) == pytest.approx(1)
    assert metrics.registry.get_sample_value(
        "legal_rag_retrieval_duration_seconds_sum"
    ) == pytest.approx(0.2)
    assert metrics.registry.get_sample_value(
        "legal_rag_retrieved_sources_sum"
    ) == pytest.approx(2)
    assert metrics.registry.get_sample_value(
        "legal_rag_generation_duration_seconds_sum",
        {"provider": "gemini", "mode": "full"},
    ) == pytest.approx(0.5)


def test_streaming_provider_failure_is_counted_after_stream_consumption() -> None:
    metrics = _metrics()
    service = LegalRAGService(
        FakeRetriever([_source()]),
        FakeGenerator(stream_error=True),
        metrics=metrics,
        provider="gemini",
        clock=FakeClock([1.0, 1.1, 2.0, 2.4]),
    )

    result = service.stream_answer("question")
    with pytest.raises(GenerationError, match="provider failed"):
        list(result.chunks)

    assert metrics.registry.get_sample_value(
        "legal_rag_llm_provider_failures_total",
        {"provider": "gemini", "mode": "stream"},
    ) == pytest.approx(1)
    assert metrics.registry.get_sample_value(
        "legal_rag_generation_duration_seconds_sum",
        {"provider": "gemini", "mode": "stream"},
    ) == pytest.approx(0.4)


def test_metric_recording_failure_does_not_break_application_work() -> None:
    metrics = _metrics()

    class BrokenCounter:
        def labels(self, *labels: str) -> object:
            raise RuntimeError("metrics backend failed")

    metrics.ask_requests = BrokenCounter()  # type: ignore[assignment]
    metrics.http_requests = BrokenCounter()  # type: ignore[assignment]
    app = create_app(AppConfig(), metrics=metrics)
    app.dependency_overrides[get_rag_service] = FakeAPIService

    response = TestClient(app).post("/ask", json={"question": "سؤال"})

    assert response.status_code == 200


def test_metrics_endpoint_does_not_initialize_rag_service() -> None:
    metrics = _metrics()
    app = create_app(AppConfig(), metrics=metrics)

    def fail_if_initialized() -> None:
        pytest.fail("metrics must not construct the RAG service")

    app.dependency_overrides[get_rag_service] = fail_if_initialized

    response = TestClient(app).get("/metrics")

    assert response.status_code == 200


def test_token_metric_is_not_fabricated_without_provider_usage() -> None:
    metrics = _metrics()

    exposition = generate_latest(metrics.registry).decode("utf-8")

    assert "# HELP legal_rag_llm_tokens_total" in exposition
    assert 'legal_rag_llm_tokens_total{provider="vllm"' not in exposition

    metrics.record_token_usage(
        "vllm",
        mode="full",
        input_tokens=12,
        output_tokens=4,
        total_tokens=16,
        input_cost_per_million_tokens_usd=1.0,
        output_cost_per_million_tokens_usd=2.0,
    )

    assert metrics.registry.get_sample_value(
        "legal_rag_llm_tokens_total",
        {"provider": "vllm", "token_type": "input", "mode": "full"},
    ) == pytest.approx(12)
    assert metrics.registry.get_sample_value(
        "legal_rag_llm_tokens_total",
        {"provider": "vllm", "token_type": "output", "mode": "full"},
    ) == pytest.approx(4)
    assert metrics.registry.get_sample_value(
        "legal_rag_llm_tokens_total",
        {"provider": "vllm", "token_type": "total", "mode": "full"},
    ) == pytest.approx(16)
    assert metrics.registry.get_sample_value(
        "legal_rag_llm_usage_cost_usd_total",
        {"provider": "vllm", "mode": "full"},
    ) == pytest.approx(0.00002)


def test_missing_pricing_does_not_emit_a_cost_sample() -> None:
    metrics = _metrics()

    metrics.record_token_usage(
        "gemini", mode="stream", input_tokens=10, output_tokens=5
    )

    assert (
        metrics.registry.get_sample_value(
            "legal_rag_llm_usage_cost_usd_total",
            {"provider": "gemini", "mode": "stream"},
        )
        is None
    )


def test_unclassified_provider_tokens_do_not_emit_partial_cost() -> None:
    metrics = _metrics()

    metrics.record_token_usage(
        "gemini",
        mode="full",
        input_tokens=10,
        output_tokens=5,
        total_tokens=18,
        input_cost_per_million_tokens_usd=1.0,
        output_cost_per_million_tokens_usd=2.0,
    )

    assert (
        metrics.registry.get_sample_value(
            "legal_rag_llm_usage_cost_usd_total",
            {"provider": "gemini", "mode": "full"},
        )
        is None
    )
