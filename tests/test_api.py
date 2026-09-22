"""API tests with all external RAG dependencies replaced by fakes."""

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
from typing import Any

import pytest
from fastapi.testclient import TestClient

from legal_rag.api import dependencies
from legal_rag.api.app import create_app
from legal_rag.api.dependencies import RAGConfigurationError, get_rag_service
from legal_rag.config import AppConfig
from legal_rag.rag import GenerationError, LegalRAGResult, LegalRAGStreamResult
from legal_rag.storage import RetrievalResult


def _source() -> RetrievalResult:
    return RetrievalResult(
        chunk_id="article-164",
        article_number=164,
        text="يكون الشخص مسئولا عن أعماله غير المشروعة.",
        language="ar",
        book="Book I",
        chapter="Chapter III",
        section=None,
        topic=None,
        is_repealed=False,
        source_page=24,
        citation="Egyptian Civil Code, Article 164",
        similarity=0.84723,
    )


class FakeRAGService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int | None]] = []
        self.stream_calls: list[tuple[str, int | None]] = []
        self.error: Exception | None = None
        self.stream_error_after_first = False

    def answer(self, question: str, top_k: int | None = None) -> LegalRAGResult:
        self.calls.append((question, top_k))
        if self.error is not None:
            raise self.error
        return LegalRAGResult(
            question=question,
            answer="يكون الشخص مسؤولاً وفقاً للمادة 164.",
            retrieved_sources=(_source(),),
        )

    def stream_answer(
        self,
        question: str,
        top_k: int | None = None,
    ) -> LegalRAGStreamResult:
        self.stream_calls.append((question, top_k))

        def chunks() -> Iterator[str]:
            if self.error is not None and not self.stream_error_after_first:
                raise self.error
            yield "إجابة "
            if self.error is not None:
                raise self.error
            yield "متدفقة"

        return LegalRAGStreamResult(question, chunks(), (_source(),))


@pytest.fixture
def fake_service() -> FakeRAGService:
    return FakeRAGService()


@pytest.fixture
def client(fake_service: FakeRAGService) -> TestClient:
    application = create_app(AppConfig(retrieval_top_k=5))
    application.dependency_overrides[get_rag_service] = lambda: fake_service
    return TestClient(application)


def test_health_is_lightweight_and_does_not_initialize_rag_service() -> None:
    application = create_app()

    def fail_if_initialized() -> Any:
        pytest.fail("RAG service must not be initialized by health endpoint")

    application.dependency_overrides[get_rag_service] = fail_if_initialized

    response = TestClient(application).get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "app": "Arabic Legal RAG",
        "version": "0.4.0",
    }


def test_ask_returns_answer_and_concise_source_mapping(
    client: TestClient,
    fake_service: FakeRAGService,
) -> None:
    response = client.post(
        "/ask",
        json={"question": "متى يكون الشخص مسؤولاً عن التعويض؟", "top_k": 5},
    )

    assert response.status_code == 200
    assert fake_service.calls == [("متى يكون الشخص مسؤولاً عن التعويض؟", 5)]
    assert response.json() == {
        "question": "متى يكون الشخص مسؤولاً عن التعويض؟",
        "answer": "يكون الشخص مسؤولاً وفقاً للمادة 164.",
        "sources": [
            {
                "chunk_id": "article-164",
                "article_number": 164,
                "citation": "Egyptian Civil Code, Article 164",
                "language": "ar",
                "similarity": 0.84723,
            }
        ],
    }


def test_ask_uses_configured_default_top_k(
    client: TestClient,
    fake_service: FakeRAGService,
) -> None:
    response = client.post("/ask", json={"question": "سؤال"})

    assert response.status_code == 200
    assert fake_service.calls == [("سؤال", 5)]


def test_ask_rejects_missing_question(client: TestClient) -> None:
    response = client.post("/ask", json={"top_k": 5})

    assert response.status_code == 422


def test_ask_rejects_blank_question(client: TestClient) -> None:
    response = client.post("/ask", json={"question": "   "})

    assert response.status_code == 422


@pytest.mark.parametrize("top_k", [0, -1])
def test_ask_rejects_non_positive_top_k(client: TestClient, top_k: int) -> None:
    response = client.post(
        "/ask",
        json={"question": "valid question", "top_k": top_k},
    )

    assert response.status_code == 422


def test_ask_maps_generation_failure_without_internal_details(
    client: TestClient,
    fake_service: FakeRAGService,
) -> None:
    fake_service.error = GenerationError("provider leaked internal detail")

    response = client.post("/ask", json={"question": "question"})

    assert response.status_code == 502
    assert response.json() == {"detail": "Answer generation provider failed"}
    assert "leaked" not in response.text


def test_ask_maps_missing_generation_configuration() -> None:
    application = create_app()

    def missing_configuration() -> Any:
        raise RAGConfigurationError("secret configuration detail")

    application.dependency_overrides[get_rag_service] = missing_configuration

    response = TestClient(application).post(
        "/ask",
        json={"question": "question"},
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "RAG service is not configured"}
    assert "secret" not in response.text


def test_streaming_ask_emits_sources_tokens_and_done_in_order(
    client: TestClient,
    fake_service: FakeRAGService,
) -> None:
    response = client.post(
        "/ask/stream",
        json={"question": "متى يجب التعويض؟", "top_k": 5},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert fake_service.stream_calls == [("متى يجب التعويض؟", 5)]
    assert response.text.index("event: sources") < response.text.index("event: token")
    assert response.text.count("event: sources") == 1
    assert response.text.count("event: token") == 2
    assert response.text.endswith("event: done\ndata: {}\n\n")
    assert '"article_number": 164' in response.text


def test_streaming_ask_does_not_emit_sources_before_first_provider_chunk(
    client: TestClient,
    fake_service: FakeRAGService,
) -> None:
    fake_service.error = GenerationError("private provider detail")

    response = client.post("/ask/stream", json={"question": "سؤال"})

    assert response.status_code == 200
    assert "event: error" in response.text
    assert "Answer generation provider failed" in response.text
    assert "private provider detail" not in response.text
    assert "event: sources" not in response.text
    assert "event: token" not in response.text
    assert "event: done" not in response.text


def test_streaming_ask_maps_failure_after_commit_to_safe_error_without_retry(
    client: TestClient,
    fake_service: FakeRAGService,
) -> None:
    fake_service.error = GenerationError("private provider detail")
    fake_service.stream_error_after_first = True

    response = client.post("/ask/stream", json={"question": "سؤال"})

    assert response.status_code == 200
    assert fake_service.stream_calls == [("سؤال", 5)]
    assert response.text.count("event: sources") == 1
    assert response.text.count("event: token") == 1
    assert response.text.index("event: sources") < response.text.index("event: token")
    assert response.text.index("event: token") < response.text.index("event: error")
    assert "private provider detail" not in response.text
    assert "event: done" not in response.text


def test_concurrent_dependency_cold_start_builds_service_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = FakeRAGService()
    build_started = Event()
    release_build = Event()
    count_lock = Lock()
    build_count = 0

    def controlled_build() -> FakeRAGService:
        nonlocal build_count
        with count_lock:
            build_count += 1
        build_started.set()
        assert release_build.wait(timeout=2)
        return service

    monkeypatch.setattr(dependencies, "_service", None)
    monkeypatch.setattr(dependencies, "_build_rag_service", controlled_build)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(dependencies.get_rag_service)
        assert build_started.wait(timeout=2)
        second = executor.submit(dependencies.get_rag_service)
        release_build.set()

    assert first.result() is service
    assert second.result() is service
    assert build_count == 1
