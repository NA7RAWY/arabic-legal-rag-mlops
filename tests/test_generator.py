"""Unit tests for grounded answer generation providers."""

import json
from types import SimpleNamespace
from typing import Any
from urllib.request import Request

import pytest

from legal_rag.config import AppConfig
from legal_rag.rag.generator import (
    SYSTEM_INSTRUCTION,
    GeminiGenerator,
    GenerationError,
    OpenAICompatibleGenerator,
    build_generator,
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
        self.stream_calls: list[dict[str, Any]] = []
        self.stream_texts = ["الإجابة ", "[المادة 148]"]
        self.error: Exception | None = None
        self.failures: list[Exception] = []

    def _raise_failure(self) -> None:
        if self.failures:
            raise self.failures.pop(0)
        if self.error is not None:
            raise self.error

    def generate_content(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        self._raise_failure()
        return SimpleNamespace(text=self.response_text)

    def generate_content_stream(self, **kwargs: Any) -> list[SimpleNamespace]:
        self.stream_calls.append(kwargs)
        self._raise_failure()
        return [SimpleNamespace(text=text) for text in self.stream_texts]


class FakeClient:
    def __init__(self, models: FakeModels) -> None:
        self.models = models


class FakeAPIError(RuntimeError):
    def __init__(self, code: int) -> None:
        super().__init__(f"provider status {code}")
        self.code = code


class FakeHTTPResponse:
    def __init__(
        self,
        payload: dict[str, object],
        lines: list[bytes] | None = None,
    ) -> None:
        self.payload = payload
        self.lines = lines or []

    def __enter__(self) -> "FakeHTTPResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")

    def __iter__(self) -> Any:
        return iter(self.lines)


class FakeURLOpen:
    def __init__(
        self,
        payload: dict[str, object] | None = None,
        lines: list[bytes] | None = None,
    ) -> None:
        self.payload = payload or {
            "choices": [{"message": {"content": "إجابة محلية مؤسسة."}}]
        }
        self.lines = lines
        self.calls: list[tuple[Request, float]] = []
        self.error: Exception | None = None

    def __call__(self, request: Request, *, timeout: float) -> FakeHTTPResponse:
        self.calls.append((request, timeout))
        if self.error is not None:
            raise self.error
        return FakeHTTPResponse(self.payload, self.lines)


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
        "User question:\nما حكم العقد؟\n\nRetrieved legal context:\nlegal context"
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


def test_default_generator_provider_remains_gemini(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LLM_PROVIDER", raising=False)

    generator = build_generator(AppConfig(gemini_api_key="test-key"))

    assert isinstance(generator, GeminiGenerator)


def test_vllm_provider_selection_uses_configured_openai_endpoint() -> None:
    generator = build_generator(
        AppConfig(
            llm_provider="vllm",
            vllm_base_url="http://vllm.internal:9000/v1/",
            vllm_model="local-model",
            vllm_api_key="test-placeholder",
        )
    )

    assert isinstance(generator, OpenAICompatibleGenerator)
    assert generator.base_url == "http://vllm.internal:9000/v1"
    assert generator.model_name == "local-model"


def test_openai_compatible_generator_preserves_grounded_prompt() -> None:
    opener = FakeURLOpen()
    generator = OpenAICompatibleGenerator(
        base_url="http://vllm:8000/v1/",
        model_name="local-model",
        api_key="EMPTY",
        timeout=30,
        urlopen_callable=opener,
    )

    answer = generator.generate("ما حكم العقد؟", "legal context")

    assert answer == "إجابة محلية مؤسسة."
    request, timeout = opener.calls[0]
    payload = json.loads(request.data.decode("utf-8"))
    assert request.full_url == "http://vllm:8000/v1/chat/completions"
    assert request.method == "POST"
    assert request.get_header("Authorization") == "Bearer EMPTY"
    assert timeout == 30
    assert payload == {
        "model": "local-model",
        "messages": [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {
                "role": "user",
                "content": (
                    "User question:\nما حكم العقد؟\n\n"
                    "Retrieved legal context:\nlegal context"
                ),
            },
        ],
        "stream": False,
    }


def test_openai_compatible_generator_maps_network_failure() -> None:
    opener = FakeURLOpen()
    opener.error = OSError("connection refused")
    generator = OpenAICompatibleGenerator(
        base_url="http://vllm:8000/v1",
        model_name="local-model",
        urlopen_callable=opener,
    )

    with pytest.raises(
        GenerationError, match="OpenAI-compatible generation failed"
    ) as exc:
        generator.generate("question", "context")

    assert isinstance(exc.value.__cause__, OSError)


def test_invalid_generator_provider_fails_clearly() -> None:
    with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER 'unknown'"):
        build_generator(AppConfig(llm_provider="unknown"))


def test_gemini_generator_streams_grounded_chunks_in_order() -> None:
    models = FakeModels()
    generator = GeminiGenerator(
        AppConfig(gemini_api_key="test-key", gemini_model="test-model"),
        client=FakeClient(models),
    )

    chunks = list(generator.stream_generate("ما حكم العقد؟", "legal context"))

    assert chunks == ["الإجابة ", "[المادة 148]"]
    assert models.stream_calls[0]["model"] == "test-model"
    assert models.stream_calls[0]["contents"] == (
        "User question:\nما حكم العقد؟\n\nRetrieved legal context:\nlegal context"
    )
    assert models.stream_calls[0]["config"].system_instruction == SYSTEM_INSTRUCTION


def test_gemini_streaming_maps_provider_failure() -> None:
    models = FakeModels()
    models.error = RuntimeError("provider unavailable")
    generator = GeminiGenerator(api_key="test-key", client=FakeClient(models))

    with pytest.raises(GenerationError, match="Gemini streaming generation failed"):
        list(generator.stream_generate("question", "context"))


def test_gemini_retries_transient_failure_then_succeeds() -> None:
    models = FakeModels()
    models.failures = [FakeAPIError(503)]
    delays: list[float] = []
    generator = GeminiGenerator(
        api_key="test-key",
        client=FakeClient(models),
        sleep_callable=delays.append,
    )

    answer = generator.generate("question", "context")

    assert answer == "الإجابة [المادة 148]"
    assert len(models.calls) == 2
    assert delays == [0.25]


def test_gemini_transient_retries_are_bounded() -> None:
    models = FakeModels()
    models.failures = [FakeAPIError(503) for _ in range(3)]
    delays: list[float] = []
    generator = GeminiGenerator(
        api_key="test-key",
        client=FakeClient(models),
        max_attempts=3,
        backoff_seconds=0.1,
        sleep_callable=delays.append,
    )

    with pytest.raises(GenerationError, match="Gemini generation failed"):
        generator.generate("question", "context")

    assert len(models.calls) == 3
    assert delays == [0.1, 0.2]


def test_gemini_does_not_retry_non_transient_failure() -> None:
    models = FakeModels()
    models.failures = [FakeAPIError(401)]
    delays: list[float] = []
    generator = GeminiGenerator(
        api_key="test-key",
        client=FakeClient(models),
        sleep_callable=delays.append,
    )

    with pytest.raises(GenerationError, match="Gemini generation failed"):
        generator.generate("question", "context")

    assert len(models.calls) == 1
    assert delays == []


def test_gemini_stream_retries_transient_failure_before_first_chunk() -> None:
    models = FakeModels()
    models.failures = [FakeAPIError(429)]
    delays: list[float] = []
    generator = GeminiGenerator(
        api_key="test-key",
        client=FakeClient(models),
        sleep_callable=delays.append,
    )

    chunks = list(generator.stream_generate("question", "context"))

    assert chunks == ["الإجابة ", "[المادة 148]"]
    assert len(models.stream_calls) == 2
    assert delays == [0.25]


def test_gemini_stream_does_not_retry_after_first_chunk() -> None:
    delays: list[float] = []

    class FailingAfterFirstChunkModels(FakeModels):
        def generate_content_stream(self, **kwargs: Any) -> Any:
            self.stream_calls.append(kwargs)

            def responses() -> Any:
                yield SimpleNamespace(text="first")
                raise FakeAPIError(503)

            return responses()

    models = FailingAfterFirstChunkModels()
    generator = GeminiGenerator(
        api_key="test-key",
        client=FakeClient(models),
        sleep_callable=delays.append,
    )
    stream = generator.stream_generate("question", "context")

    assert next(stream) == "first"
    with pytest.raises(GenerationError, match="Gemini streaming generation failed"):
        next(stream)

    assert len(models.stream_calls) == 1
    assert delays == []


def test_openai_compatible_generator_parses_sse_until_done() -> None:
    opener = FakeURLOpen(
        lines=[
            b'data: {"choices":[{"delta":{"content":"first "}}]}\n',
            b"\n",
            b'data: {"choices":[{"delta":{"content":"second"}}]}\n',
            b"data: [DONE]\n",
            b'data: {"choices":[{"delta":{"content":"ignored"}}]}\n',
        ]
    )
    generator = OpenAICompatibleGenerator(
        base_url="http://vllm:8000/v1",
        model_name="local-model",
        urlopen_callable=opener,
    )

    chunks = list(generator.stream_generate("question", "context"))

    assert chunks == ["first ", "second"]
    request, _ = opener.calls[0]
    payload = json.loads(request.data.decode("utf-8"))
    assert request.full_url == "http://vllm:8000/v1/chat/completions"
    assert payload["model"] == "local-model"
    assert payload["stream"] is True
    assert payload["messages"][0] == {
        "role": "system",
        "content": SYSTEM_INSTRUCTION,
    }


def test_openai_compatible_streaming_maps_network_failure() -> None:
    opener = FakeURLOpen()
    opener.error = OSError("connection refused")
    generator = OpenAICompatibleGenerator(
        base_url="http://vllm:8000/v1",
        model_name="local-model",
        urlopen_callable=opener,
    )

    with pytest.raises(
        GenerationError, match="OpenAI-compatible streaming generation failed"
    ):
        list(generator.stream_generate("question", "context"))


@pytest.mark.parametrize(
    "lines",
    [
        [b"data: not-json\n"],
        [b'data: {"choices": []}\n'],
        [b'data: {"choices":[{"delta":{"content":"partial"}}]}\n'],
    ],
)
def test_openai_compatible_streaming_rejects_malformed_stream(
    lines: list[bytes],
) -> None:
    generator = OpenAICompatibleGenerator(
        base_url="http://vllm:8000/v1",
        model_name="local-model",
        urlopen_callable=FakeURLOpen(lines=lines),
    )

    with pytest.raises(GenerationError):
        list(generator.stream_generate("question", "context"))
