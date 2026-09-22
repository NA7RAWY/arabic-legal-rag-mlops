"""Deterministic PII output guardrail tests with synthetic values only."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry, generate_latest

from legal_rag.api.app import create_app
from legal_rag.api.dependencies import get_rag_service
from legal_rag.config import AppConfig
from legal_rag.guardrails import PIICategory, PIIGuardrail, StreamingPIIRedactor
from legal_rag.monitoring import PrometheusMetrics
from legal_rag.rag import GenerationError, LegalRAGService
from legal_rag.storage import RetrievalResult

SYNTHETIC_EMAIL = "person@example.test"
SYNTHETIC_PHONE = "010-0000-0000"
SYNTHETIC_NATIONAL_ID = "30001018800000"


def _source() -> RetrievalResult:
    return RetrievalResult(
        chunk_id="article-164",
        article_number=164,
        text="نص قانوني لا يخضع لتنقية إجابة النموذج.",
        language="ar",
        book=None,
        chapter=None,
        section=None,
        topic=None,
        is_repealed=False,
        source_page=1,
        citation="Egyptian Civil Code, Article 164",
        similarity=0.9,
    )


class FakeRetriever:
    def search(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        return [_source()]


class FakeGenerator:
    def __init__(self, answer: str, chunks: tuple[str, ...] | None = None) -> None:
        self.answer = answer
        self.chunks = chunks or (answer,)

    def generate(self, question: str, context: str) -> str:
        return self.answer

    def stream_generate(self, question: str, context: str) -> Iterator[str]:
        return iter(self.chunks)


class RecordingObservation:
    def __init__(self) -> None:
        self.success_metadata: dict[str, Any] | None = None

    def succeed(self, metadata: dict[str, Any] | None = None) -> None:
        self.success_metadata = metadata

    def fail(self, status: str) -> None:
        pass

    def end(self) -> None:
        pass


class RecordingTrace(RecordingObservation):
    def __init__(self) -> None:
        super().__init__()
        self.retrieval = RecordingObservation()
        self.generation = RecordingObservation()

    def start_retrieval(self) -> RecordingObservation:
        return self.retrieval

    def start_generation(self) -> RecordingObservation:
        return self.generation


class RecordingTracer:
    def __init__(self) -> None:
        self.trace = RecordingTrace()

    def start_request(self, **kwargs: Any) -> RecordingTrace:
        return self.trace


@pytest.mark.parametrize(
    ("value", "marker", "category"),
    [
        (SYNTHETIC_EMAIL, "[REDACTED_EMAIL]", PIICategory.EMAIL),
        (SYNTHETIC_PHONE, "[REDACTED_PHONE]", PIICategory.PHONE),
        (
            SYNTHETIC_NATIONAL_ID,
            "[REDACTED_NATIONAL_ID]",
            PIICategory.NATIONAL_ID,
        ),
    ],
)
def test_redacts_supported_high_confidence_pii(
    value: str,
    marker: str,
    category: PIICategory,
) -> None:
    result = PIIGuardrail().redact(f"Synthetic value: {value}")

    assert value not in result.text
    assert marker in result.text
    assert result.categories == (category,)
    assert result.total == 1


def test_redacts_multiple_pii_types_without_retaining_values() -> None:
    text = f"{SYNTHETIC_EMAIL} {SYNTHETIC_PHONE} {SYNTHETIC_NATIONAL_ID}"

    result = PIIGuardrail().redact(text)

    assert all(value not in result.text for value in text.split())
    assert result.total == 3
    assert set(result.categories) == set(PIICategory)


def test_does_not_redact_articles_dates_or_invalid_national_id_shapes() -> None:
    text = "Articles 164 and 1022; dates 2024-01-01 and 01/01/2024; 39913321234567"

    result = PIIGuardrail().redact(text)

    assert result.text == text
    assert result.triggered is False


def test_ask_api_returns_sanitized_answer_and_preserves_sources() -> None:
    service = LegalRAGService(
        FakeRetriever(),
        FakeGenerator(f"Contact {SYNTHETIC_EMAIL}"),
    )
    app = create_app(AppConfig())
    app.dependency_overrides[get_rag_service] = lambda: service

    response = TestClient(app).post("/ask", json={"question": "سؤال"})

    assert response.status_code == 200
    assert SYNTHETIC_EMAIL not in response.text
    assert response.json()["answer"] == "Contact [REDACTED_EMAIL]"
    assert response.json()["sources"][0]["article_number"] == 164


def test_stream_redacts_split_pii_while_emitting_incrementally() -> None:
    service = LegalRAGService(
        FakeRetriever(),
        FakeGenerator(
            "unused",
            chunks=("Contact person@", "example.test or 010-", "0000-0000"),
        ),
    )
    app = create_app(AppConfig())
    app.dependency_overrides[get_rag_service] = lambda: service

    response = TestClient(app).post("/ask/stream", json={"question": "سؤال"})

    assert response.status_code == 200
    assert SYNTHETIC_EMAIL not in response.text
    assert SYNTHETIC_PHONE not in response.text
    assert "person@" not in response.text
    assert "event: sources" in response.text
    assert response.text.count("event: token") == 3
    assert "[REDACTED_EMAIL]" in response.text
    assert "[REDACTED_PHONE]" in response.text
    assert response.text.index("event: sources") < response.text.index("event: token")
    assert response.text.endswith("event: done\ndata: {}\n\n")


@pytest.mark.parametrize(
    ("chunks", "value", "marker"),
    [
        (("person@", "example.test", " "), SYNTHETIC_EMAIL, "[REDACTED_EMAIL]"),
        (("010-", "0000-", "0000", " "), SYNTHETIC_PHONE, "[REDACTED_PHONE]"),
        (
            ("300", "010", "188", "000", "00", " "),
            SYNTHETIC_NATIONAL_ID,
            "[REDACTED_NATIONAL_ID]",
        ),
    ],
)
def test_streaming_redactor_protects_pii_across_adversarial_chunks(
    chunks: tuple[str, ...],
    value: str,
    marker: str,
) -> None:
    redactor = StreamingPIIRedactor()
    emitted = [redactor.feed(chunk).text for chunk in chunks]
    emitted.append(redactor.finish().text)
    output = "".join(emitted)

    assert value not in output
    assert marker in output
    assert all(value not in partial for partial in emitted)


@pytest.mark.parametrize(
    ("value", "marker"),
    [
        (SYNTHETIC_EMAIL, "[REDACTED_EMAIL]"),
        (SYNTHETIC_PHONE, "[REDACTED_PHONE]"),
        (SYNTHETIC_NATIONAL_ID, "[REDACTED_NATIONAL_ID]"),
    ],
)
def test_streaming_redactor_is_safe_at_every_character_boundary(
    value: str, marker: str
) -> None:
    redactor = StreamingPIIRedactor()

    output = "".join(redactor.feed(char).text for char in f"{value} ")
    output += redactor.finish().text

    assert value not in output
    assert marker in output


def test_streaming_redactor_handles_multiple_values_and_preserves_legal_numbers() -> (
    None
):
    redactor = StreamingPIIRedactor()
    chunks = (
        "Articles 164 and 1022. Contact person@",
        "example.test or 010-",
        "0000-0000; ID 3000101",
        "8800000.",
    )

    output = "".join(redactor.feed(chunk).text for chunk in chunks)
    output += redactor.finish().text

    assert "Articles 164 and 1022" in output
    assert SYNTHETIC_EMAIL not in output
    assert SYNTHETIC_PHONE not in output
    assert SYNTHETIC_NATIONAL_ID not in output
    assert output.count("[REDACTED_") == 3


def test_streaming_redactor_flushes_safe_pending_suffix_at_eof() -> None:
    redactor = StreamingPIIRedactor()

    assert redactor.feed("ordinary").text == ""
    assert redactor.finish().text == "ordinary"
    assert redactor.max_buffered_characters == 254


def test_service_stream_is_incremental_before_provider_completion() -> None:
    events: list[str] = []

    class IncrementalGenerator(FakeGenerator):
        def stream_generate(self, question: str, context: str) -> Iterator[str]:
            events.append("provider-first")
            yield "مرحبا "
            events.append("provider-second")
            yield "بكم"
            events.append("provider-complete")

    service = LegalRAGService(FakeRetriever(), IncrementalGenerator("unused"))
    stream = service.stream_answer("question").chunks

    assert next(stream) == "مرحبا "
    assert events == ["provider-first"]
    assert next(stream) == "بكم"
    assert events == ["provider-first", "provider-second"]
    with pytest.raises(StopIteration):
        next(stream)
    assert events[-1] == "provider-complete"


def test_provider_failure_discards_unsafe_suffix_and_emits_safe_error() -> None:
    class FailingGenerator(FakeGenerator):
        def stream_generate(self, question: str, context: str) -> Iterator[str]:
            yield "Safe text "
            yield "person@"
            raise GenerationError("synthetic provider detail")

    metrics = PrometheusMetrics(CollectorRegistry())
    service = LegalRAGService(
        FakeRetriever(),
        FailingGenerator("unused"),
        metrics=metrics,
        provider="gemini",
    )
    app = create_app(AppConfig(), metrics=metrics)
    app.dependency_overrides[get_rag_service] = lambda: service

    response = TestClient(app).post("/ask/stream", json={"question": "سؤال"})

    assert response.status_code == 200
    assert "Safe text " in response.text
    assert "person@" not in response.text
    assert "example.test" not in response.text
    assert "event: error" in response.text
    assert "event: done" not in response.text
    assert response.text.index("event: sources") < response.text.index("event: token")
    assert response.text.index("event: token") < response.text.index("event: error")
    assert metrics.registry.get_sample_value(
        "legal_rag_llm_provider_failures_total",
        {"provider": "gemini", "mode": "stream"},
    ) == pytest.approx(1)


def test_stream_redaction_metrics_and_trace_metadata_are_bounded() -> None:
    metrics = PrometheusMetrics(CollectorRegistry())
    tracer = RecordingTracer()
    service = LegalRAGService(
        FakeRetriever(),
        FakeGenerator(
            "unused",
            chunks=("person@", "example.test and 010-", "0000-0000 "),
        ),
        metrics=metrics,
        tracer=tracer,
    )

    output = "".join(service.stream_answer("question").chunks)
    exposition = generate_latest(metrics.registry).decode("utf-8")
    metadata = tracer.trace.generation.success_metadata

    assert SYNTHETIC_EMAIL not in output
    assert SYNTHETIC_PHONE not in output
    assert 'category="email",mode="stream"' in exposition
    assert 'category="phone",mode="stream"' in exposition
    assert metrics.registry.get_sample_value(
        "legal_rag_pii_redactions_total",
        {"category": "email", "mode": "stream"},
    ) == pytest.approx(1)
    assert metrics.registry.get_sample_value(
        "legal_rag_pii_redactions_total",
        {"category": "phone", "mode": "stream"},
    ) == pytest.approx(1)
    assert metadata == {
        "pii_guardrail_triggered": True,
        "pii_redaction_count": 2,
        "pii_categories": ["email", "phone"],
    }
    assert SYNTHETIC_EMAIL not in repr(metadata)
    assert SYNTHETIC_PHONE not in repr(metadata)


def test_prometheus_and_langfuse_metadata_are_bounded_and_contain_no_pii() -> None:
    metrics = PrometheusMetrics(CollectorRegistry())
    tracer = RecordingTracer()
    service = LegalRAGService(
        FakeRetriever(),
        FakeGenerator(f"{SYNTHETIC_EMAIL} and {SYNTHETIC_PHONE}"),
        metrics=metrics,
        tracer=tracer,
    )

    result = service.answer("question")
    exposition = generate_latest(metrics.registry).decode("utf-8")
    metadata = tracer.trace.generation.success_metadata

    assert SYNTHETIC_EMAIL not in result.answer
    assert SYNTHETIC_PHONE not in result.answer
    assert SYNTHETIC_EMAIL not in exposition
    assert SYNTHETIC_PHONE not in exposition
    assert 'category="email",mode="full"' in exposition
    assert 'category="phone",mode="full"' in exposition
    assert metadata == {
        "pii_guardrail_triggered": True,
        "pii_redaction_count": 2,
        "pii_categories": ["email", "phone"],
    }
    assert SYNTHETIC_EMAIL not in repr(metadata)
    assert SYNTHETIC_PHONE not in repr(metadata)
