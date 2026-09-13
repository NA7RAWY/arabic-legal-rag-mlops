"""Tests for the BentoML ASGI adapter without importing BentoML itself."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from legal_rag.api.dependencies import get_rag_service
from legal_rag.config import AppConfig
from legal_rag.rag import GenerationError, LegalRAGResult, LegalRAGStreamResult
from legal_rag.serving import create_serving_app
from legal_rag.storage import RetrievalResult

SERVICE_PATH = Path("src/legal_rag/serving/service.py")


class FakeRAGService:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[tuple[str, int | None]] = []

    def answer(self, question: str, top_k: int | None = None) -> LegalRAGResult:
        self.calls.append((question, top_k))
        if self.error is not None:
            raise self.error
        source = RetrievalResult(
            chunk_id="article-164",
            article_number=164,
            text="كل خطأ سبب ضرراً للغير يلزم من ارتكبه بالتعويض.",
            language="ar",
            book="Book I",
            chapter="Chapter III",
            section=None,
            topic=None,
            is_repealed=False,
            source_page=24,
            citation="Egyptian Civil Code, Article 164",
            similarity=0.91,
        )
        return LegalRAGResult(question, "إجابة مستندة إلى المادة 164.", (source,))

    def stream_answer(
        self,
        question: str,
        top_k: int | None = None,
    ) -> LegalRAGStreamResult:
        self.calls.append((question, top_k))

        def chunks() -> Iterator[str]:
            yield "إجابة "
            yield "متدفقة"

        result = self.answer(question, top_k)
        self.calls.pop()
        return LegalRAGStreamResult(question, chunks(), result.retrieved_sources)


def _client(service: FakeRAGService) -> TestClient:
    application = create_serving_app(AppConfig(retrieval_top_k=5))
    application.dependency_overrides[get_rag_service] = lambda: service
    return TestClient(application)


def test_bentoml_health_contract_does_not_resolve_rag_dependency() -> None:
    application = create_serving_app()

    def fail_if_resolved() -> None:
        pytest.fail("health must not construct the RAG service")

    application.dependency_overrides[get_rag_service] = fail_if_resolved

    response = TestClient(application).get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "app": "Arabic Legal RAG",
        "version": "0.1.0",
    }


def test_bentoml_ask_reuses_rag_service_and_response_contract() -> None:
    service = FakeRAGService()

    response = _client(service).post(
        "/ask",
        json={"question": "متى يجب التعويض؟", "top_k": 5},
    )

    assert response.status_code == 200
    assert service.calls == [("متى يجب التعويض؟", 5)]
    assert response.json() == {
        "question": "متى يجب التعويض؟",
        "answer": "إجابة مستندة إلى المادة 164.",
        "sources": [
            {
                "chunk_id": "article-164",
                "article_number": 164,
                "citation": "Egyptian Civil Code, Article 164",
                "language": "ar",
                "similarity": 0.91,
            }
        ],
    }


@pytest.mark.parametrize(
    ("payload", "status_code"),
    [
        ({}, 422),
        ({"question": "   "}, 422),
        ({"question": "valid", "top_k": 0}, 422),
    ],
)
def test_bentoml_ask_preserves_request_validation(
    payload: dict[str, object],
    status_code: int,
) -> None:
    response = _client(FakeRAGService()).post("/ask", json=payload)

    assert response.status_code == status_code


def test_bentoml_ask_preserves_safe_provider_error_mapping() -> None:
    response = _client(FakeRAGService(GenerationError("private provider detail"))).post(
        "/ask",
        json={"question": "سؤال"},
    )

    assert response.status_code == 502
    assert response.json() == {"detail": "Answer generation provider failed"}
    assert "private provider detail" not in response.text


def test_service_uses_current_bentoml_asgi_api() -> None:
    source = SERVICE_PATH.read_text(encoding="utf-8")

    assert '@bentoml.asgi_app(asgi_application, path="/")' in source
    assert "@bentoml.service(" in source
    assert "class LegalRAGBentoService:" in source


def test_bentoml_mount_exposes_streaming_fastapi_route() -> None:
    service = FakeRAGService()

    response = _client(service).post(
        "/ask/stream",
        json={"question": "متى يجب التعويض؟", "top_k": 5},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: sources" in response.text
    assert response.text.count("event: token") == 2
    assert "event: done" in response.text
