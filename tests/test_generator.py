"""Unit tests for grounded answer generation providers."""

from types import SimpleNamespace
from typing import Any

import pytest

from legal_rag.config import AppConfig
from legal_rag.rag.generator import (
    GeminiGenerator,
    GenerationError,
    SYSTEM_INSTRUCTION,
    build_legal_context,
)
from legal_rag.storage import RetrievalResult


def _result(article_number: int = 148) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=f"article-{article_number}",
        article_number=article_number,
        text=f"نص المادة {article_number}",
        language="ar",
        book=None,
        chapter=None,
        section=None,
        topic=None,
        is_repealed=False,
        source_page=10,
        citation=f"Egyptian Civil Code, Article {article_number}",
        similarity=0.9,
    )


class FakeModels:
    def __init__(self, response_text: str = "الإجابة [المادة 148]") -> None:
        self.response_text = response_text
        self.calls: list[dict[str, Any]] = []
        self.error: Exception | None = None

    def generate_content(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(text=self.response_text)


class FakeClient:
    def __init__(self, models: FakeModels) -> None:
        self.models = models


def test_build_legal_context_is_deterministic() -> None:
    context = build_legal_context([_result(148), _result(149)])

    assert context == (
        "[Source 1]\n"
        "Article number: 148\n"
        "Citation: Egyptian Civil Code, Article 148\n"
        "Language: ar\n"
        "Text:\n"
        "نص المادة 148\n\n"
        "[Source 2]\n"
        "Article number: 149\n"
        "Citation: Egyptian Civil Code, Article 149\n"
        "Language: ar\n"
        "Text:\n"
        "نص المادة 149"
    )


def test_gemini_generator_requires_api_key() -> None:
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        GeminiGenerator(AppConfig(gemini_api_key=None))


def test_gemini_generator_passes_grounded_prompt_and_system_instruction() -> None:
    models = FakeModels()
    generator = GeminiGenerator(
        AppConfig(gemini_api_key="test-key", gemini_model="test-model"),
        client=FakeClient(models),
    )

    answer = generator.generate("ما حكم العقد؟", "legal context")

    assert answer == "الإجابة [المادة 148]"
    assert models.calls[0]["model"] == "test-model"
    assert models.calls[0]["contents"] == (
        "User question:\nما حكم العقد؟\n\n"
        "Retrieved legal context:\nlegal context"
    )
    assert models.calls[0]["config"].system_instruction == SYSTEM_INSTRUCTION


def test_gemini_generator_wraps_provider_failure() -> None:
    models = FakeModels()
    models.error = RuntimeError("provider unavailable")
    generator = GeminiGenerator(api_key="test-key", client=FakeClient(models))

    with pytest.raises(GenerationError, match="Gemini generation failed") as exc:
        generator.generate("question", "context")

    assert isinstance(exc.value.__cause__, RuntimeError)


def test_gemini_generator_rejects_empty_provider_response() -> None:
    generator = GeminiGenerator(
        api_key="test-key",
        client=FakeClient(FakeModels("   ")),
    )

    with pytest.raises(GenerationError, match="empty response"):
        generator.generate("question", "context")
