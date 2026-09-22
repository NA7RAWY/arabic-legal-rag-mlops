"""Langfuse tracing tests using in-memory v4-shaped SDK fakes."""

import logging
from collections.abc import Iterator
from typing import Any

import pytest

from legal_rag.config import AppConfig
from legal_rag.monitoring.tracing import LangfuseTracer, NoOpTracer, build_rag_tracer
from legal_rag.rag import GenerationError, LegalRAGService
from legal_rag.storage import RetrievalResult


def _source() -> RetrievalResult:
    return RetrievalResult(
        chunk_id="article-164",
        article_number=164,
        text="sensitive legal context",
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
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls = 0

    def search(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        self.calls += 1
        if self.error:
            raise self.error
        return [_source()]


class FakeGenerator:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error

    def generate(self, question: str, context: str) -> str:
        if self.error:
            raise self.error
        return "sensitive generated answer"

    def stream_generate(self, question: str, context: str) -> Iterator[str]:
        if self.error:
            raise self.error
        return iter(("first", "second"))


class FakeObservation:
    def __init__(self, creation: dict[str, Any], *, fail_operations: bool = False):
        self.creation = creation
        self.fail_operations = fail_operations
        self.children: list[FakeObservation] = []
        self.updates: list[dict[str, Any]] = []
        self.ended = False

    def start_observation(self, **kwargs: Any) -> "FakeObservation":
        if self.fail_operations:
            raise RuntimeError("sdk secret must not leak")
        child = FakeObservation(kwargs)
        self.children.append(child)
        return child

    def update(self, **kwargs: Any) -> None:
        if self.fail_operations:
            raise RuntimeError("sdk secret must not leak")
        self.updates.append(kwargs)

    def end(self) -> None:
        if self.fail_operations:
            raise RuntimeError("sdk secret must not leak")
        self.ended = True


class FakeClient:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.roots: list[FakeObservation] = []

    def start_observation(self, **kwargs: Any) -> FakeObservation:
        if self.error:
            raise self.error
        root = FakeObservation(kwargs)
        self.roots.append(root)
        return root


def _service(
    client: FakeClient,
    *,
    retriever: FakeRetriever | None = None,
    generator: FakeGenerator | None = None,
) -> LegalRAGService:
    return LegalRAGService(
        retriever or FakeRetriever(),
        generator or FakeGenerator(),
        tracer=LangfuseTracer(client),
        provider="gemini",
        embedding_model="intfloat/multilingual-e5-small",
        generator_model="gemini-test",
    )


def test_disabled_tracing_is_noop_without_constructing_client() -> None:
    called = False

    def client_factory(**kwargs: Any) -> Any:
        nonlocal called
        called = True
        raise AssertionError("client must not be constructed")

    tracer = build_rag_tracer(
        AppConfig(langfuse_enabled=False), client_factory=client_factory
    )

    assert isinstance(tracer, NoOpTracer)
    assert called is False


def test_enabled_tracer_initializes_default_v4_client_without_keyed_lookup() -> None:
    client = FakeClient()
    calls: list[dict[str, Any]] = []

    def client_factory(**kwargs: Any) -> FakeClient:
        calls.append(kwargs)
        return client

    tracer = build_rag_tracer(
        AppConfig(
            langfuse_enabled=True,
            langfuse_public_key="public-test-key",
            langfuse_secret_key="secret-test-key",
            langfuse_base_url="https://example.langfuse.test",
        ),
        client_factory=client_factory,
    )

    assert isinstance(tracer, LangfuseTracer)
    assert calls == [{}]


def test_real_langfuse_v4_sdk_exports_nested_observations_in_memory() -> None:
    from langfuse import Langfuse
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    client = Langfuse(
        public_key="test-public-key",
        secret_key="test-secret-key",
        base_url="http://127.0.0.1:9",
        span_exporter=exporter,
    )
    try:
        result = _service(client).answer("private question", top_k=5)
        client.flush()

        spans = exporter.get_finished_spans()
        assert result.answer == "sensitive generated answer"
        assert [span.name for span in spans] == [
            "retrieval",
            "generation",
            "legal-rag-request",
        ]
        root = spans[-1]
        assert all(span.parent.span_id == root.context.span_id for span in spans[:2])
        assert [span.attributes["langfuse.observation.type"] for span in spans] == [
            "retriever",
            "generation",
            "chain",
        ]
        serialized = repr([span.attributes for span in spans])
        assert "private question" not in serialized
        assert "sensitive legal context" not in serialized
        assert "sensitive generated answer" not in serialized
    finally:
        client.shutdown()


def test_langfuse_environment_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "public-test-key")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "secret-test-key")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://example.langfuse.test")

    config = AppConfig()

    assert config.langfuse_enabled is True
    assert config.langfuse_public_key == "public-test-key"
    assert config.langfuse_secret_key == "secret-test-key"
    assert config.langfuse_base_url == "https://example.langfuse.test"


def test_invalid_langfuse_enabled_value_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGFUSE_ENABLED", "sometimes")

    with pytest.raises(ValueError, match="LANGFUSE_ENABLED must be a boolean"):
        AppConfig()


def test_full_answer_creates_private_root_retrieval_generation_hierarchy() -> None:
    client = FakeClient()

    result = _service(client).answer("private legal question", top_k=5)

    assert result.answer == "sensitive generated answer"
    root = client.roots[0]
    assert root.creation["name"] == "legal-rag-request"
    assert root.creation["as_type"] == "chain"
    assert root.creation["metadata"] == {
        "endpoint": "/ask",
        "mode": "full",
        "top_k": 5,
        "provider": "gemini",
        "embedding_model": "intfloat/multilingual-e5-small",
        "content_capture": "disabled",
    }
    assert [child.creation["as_type"] for child in root.children] == [
        "retriever",
        "generation",
    ]
    retrieval, generation = root.children
    assert retrieval.updates[-1]["metadata"] == {
        "mode": "full",
        "requested_top_k": 5,
        "status": "success",
        "source_count": 1,
        "article_numbers": [164],
        "chunk_ids": ["article-164"],
    }
    assert generation.creation["model"] == "gemini-test"
    assert root.ended and retrieval.ended and generation.ended
    serialized = repr([root.creation, retrieval.updates, generation.updates])
    assert "private legal question" not in serialized
    assert "sensitive legal context" not in serialized
    assert "sensitive generated answer" not in serialized


def test_stream_trace_stays_open_until_iteration_completes() -> None:
    client = FakeClient()
    service = _service(client)

    result = service.stream_answer("private question", top_k=5)
    root = client.roots[0]
    generation = root.children[1]
    chunks = iter(result.chunks)

    assert root.ended is False
    assert generation.ended is False
    assert next(chunks) == "firstsecond"
    assert root.ended is False
    assert list(chunks) == []
    assert root.ended is True
    assert generation.ended is True
    assert generation.updates[-1]["metadata"]["status"] == "success"


@pytest.mark.parametrize(
    ("retriever", "generator", "failed_type"),
    [
        (FakeRetriever(RuntimeError("database detail")), FakeGenerator(), "retriever"),
        (
            FakeRetriever(),
            FakeGenerator(GenerationError("provider detail")),
            "generation",
        ),
    ],
)
def test_failures_are_recorded_without_sensitive_exception_details(
    retriever: FakeRetriever,
    generator: FakeGenerator,
    failed_type: str,
) -> None:
    client = FakeClient()
    service = _service(client, retriever=retriever, generator=generator)

    with pytest.raises((RuntimeError, GenerationError)):
        service.answer("question")

    root = client.roots[0]
    failed = next(
        child for child in root.children if child.creation["as_type"] == failed_type
    )
    assert failed.updates[-1]["level"] == "ERROR"
    serialized = repr([root.updates, failed.updates])
    assert "database detail" not in serialized
    assert "provider detail" not in serialized


def test_langfuse_sdk_failure_does_not_break_rag_or_log_secret(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = FakeClient(error=RuntimeError("private-sdk-secret"))

    with caplog.at_level(logging.WARNING):
        result = _service(client).answer("question")

    assert result.answer == "sensitive generated answer"
    assert "private-sdk-secret" not in caplog.text
    assert "Langfuse request observation creation failed" in caplog.text


def test_incomplete_langfuse_credentials_fail_closed_to_noop(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        tracer = build_rag_tracer(
            AppConfig(
                langfuse_enabled=True,
                langfuse_public_key="public-value",
                langfuse_secret_key=None,
            )
        )

    assert isinstance(tracer, NoOpTracer)
    assert "public-value" not in caplog.text
