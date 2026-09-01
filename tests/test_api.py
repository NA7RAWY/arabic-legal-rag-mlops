"""API tests with all external RAG dependencies replaced by fakes."""

from typing import Any

import pytest
from fastapi.testclient import TestClient

from legal_rag.api.app import create_app
from legal_rag.api.dependencies import RAGConfigurationError, get_rag_service
from legal_rag.config import AppConfig
from legal_rag.rag import GenerationError, LegalRAGResult
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
        self.error: Exception | None = None

    def answer(self, question: str, top_k: int | None = None) -> LegalRAGResult:
        self.calls.append((question, top_k))
        if self.error is not None:
            raise self.error
        return LegalRAGResult(
            question=question,
            answer="يكون الشخص مسؤولاً وفقاً للمادة 164.",
            retrieved_sources=(_source(),),
        )


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
        "version": "0.1.0",
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
