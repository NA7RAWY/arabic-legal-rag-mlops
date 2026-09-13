"""Grounded answer generation providers."""

import json
import logging
from collections.abc import Callable, Iterator
from time import sleep
from typing import Any, Protocol
from urllib.request import Request, urlopen

from legal_rag.config import AppConfig, get_config
from legal_rag.storage import RetrievalResult

logger = logging.getLogger(__name__)

SYSTEM_INSTRUCTION = """You answer questions about the Egyptian Civil Code.
Use only the supplied retrieved legal context.
Do not invent or rely on articles, laws, citations, legal facts, or general legal knowledge outside that context.
If the context is insufficient, explicitly state that the retrieved materials are insufficient to answer.
Cite the relevant article numbers in the answer.
Prefer Arabic when the user's question is in Arabic.
Keep any disclaimer concise and state that the answer is informational, not professional legal advice.
"""


class LLMGenerator(Protocol):
    """Provider-independent grounded text generation contract."""

    def generate(self, question: str, context: str) -> str: ...

    def stream_generate(self, question: str, context: str) -> Iterator[str]: ...


class GenerationError(RuntimeError):
    """Raised when an LLM provider cannot generate an answer."""


def build_generation_prompt(question: str, context: str) -> str:
    """Build the shared grounded user prompt for every generation provider."""

    return f"User question:\n{question}\n\nRetrieved legal context:\n{context}"


def build_legal_context(results: list[RetrievalResult]) -> str:
    """Build deterministic, citation-ready context from retrieved chunks."""

    sources = []
    for index, result in enumerate(results, start=1):
        sources.append(
            "\n".join(
                (
                    f"[Source {index}]",
                    f"Article number: {result.article_number}",
                    f"Citation: {result.citation}",
                    f"Language: {result.language}",
                    "Text:",
                    result.text,
                )
            )
        )
    return "\n\n".join(sources)


class GeminiGenerator:
    """Generate grounded answers with Google's Gemini API."""

    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        api_key: str | None = None,
        model_name: str | None = None,
        client: Any | None = None,
        max_attempts: int = 3,
        backoff_seconds: float = 0.25,
        sleep_callable: Callable[[float], None] = sleep,
    ) -> None:
        active_config = config if config is not None else get_config()
        self.api_key = api_key or active_config.gemini_api_key
        if not self.api_key:
            raise ValueError(
                "Gemini API key is required; set the GEMINI_API_KEY environment variable"
            )
        self.model_name = model_name or active_config.gemini_model
        self._client = client
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        if backoff_seconds < 0:
            raise ValueError("backoff_seconds must not be negative")
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds
        self._sleep = sleep_callable

    def _get_client(self) -> Any:
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self.api_key)
        return self._client

    @staticmethod
    def _is_transient(exc: Exception) -> bool:
        status_code = getattr(exc, "code", None)
        return isinstance(status_code, int) and (
            status_code == 429 or 500 <= status_code < 600
        )

    def _retry_delay(self, attempt: int) -> float:
        return self.backoff_seconds * (2 ** (attempt - 1))

    def _log_retry(self, attempt: int, exc: Exception) -> None:
        logger.warning(
            "Retrying Gemini request after transient status %s; attempt %d of %d",
            getattr(exc, "code", "unknown"),
            attempt + 1,
            self.max_attempts,
        )

    def generate(self, question: str, context: str) -> str:
        """Generate an answer grounded only in the supplied legal context."""

        from google.genai import types

        prompt = build_generation_prompt(question, context)
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self._get_client().models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION
                    ),
                )
                answer = response.text
                break
            except Exception as exc:
                if self._is_transient(exc) and attempt < self.max_attempts:
                    self._log_retry(attempt, exc)
                    self._sleep(self._retry_delay(attempt))
                    continue
                raise GenerationError("Gemini generation failed") from exc

        if not answer or not answer.strip():
            raise GenerationError("Gemini returned an empty response")
        return answer.strip()

    def stream_generate(self, question: str, context: str) -> Iterator[str]:
        """Yield grounded Gemini response chunks as they arrive."""

        from google.genai import types

        prompt = build_generation_prompt(question, context)
        for attempt in range(1, self.max_attempts + 1):
            yielded = False
            try:
                responses = self._get_client().models.generate_content_stream(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION
                    ),
                )
                for response in responses:
                    text = response.text
                    if text:
                        yielded = True
                        yield text
                if not yielded:
                    raise GenerationError("Gemini returned an empty response stream")
                return
            except GenerationError:
                raise
            except Exception as exc:
                if (
                    not yielded
                    and self._is_transient(exc)
                    and attempt < self.max_attempts
                ):
                    self._log_retry(attempt, exc)
                    self._sleep(self._retry_delay(attempt))
                    continue
                raise GenerationError("Gemini streaming generation failed") from exc


class OpenAICompatibleGenerator:
    """Generate through an OpenAI-compatible chat-completions endpoint."""

    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        base_url: str | None = None,
        model_name: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
        urlopen_callable: Any = urlopen,
    ) -> None:
        active_config = config if config is not None else get_config()
        configured_base_url = (
            active_config.vllm_base_url if base_url is None else base_url
        )
        self.base_url = configured_base_url.rstrip("/")
        self.model_name = active_config.vllm_model if model_name is None else model_name
        self.api_key = active_config.vllm_api_key if api_key is None else api_key
        self.timeout = timeout
        self._urlopen = urlopen_callable
        if not self.base_url:
            raise ValueError("VLLM_BASE_URL must not be empty")
        if not self.model_name:
            raise ValueError("VLLM_MODEL must not be empty")

    def _request(self, question: str, context: str, *, stream: bool) -> Request:
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {
                    "role": "user",
                    "content": build_generation_prompt(question, context),
                },
            ],
            "stream": stream,
        }
        return Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

    def generate(self, question: str, context: str) -> str:
        """Call the OpenAI-compatible ``/chat/completions`` API."""

        request = self._request(question, context, stream=False)
        try:
            with self._urlopen(request, timeout=self.timeout) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
            answer = response_payload["choices"][0]["message"]["content"]
        except Exception as exc:
            raise GenerationError("OpenAI-compatible generation failed") from exc

        if not isinstance(answer, str) or not answer.strip():
            raise GenerationError(
                "OpenAI-compatible provider returned an empty response"
            )
        return answer.strip()

    def stream_generate(self, question: str, context: str) -> Iterator[str]:
        """Yield deltas from an OpenAI-compatible SSE response."""

        request = self._request(question, context, stream=True)
        yielded = False
        completed = False
        try:
            with self._urlopen(request, timeout=self.timeout) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line or line.startswith(":"):
                        continue
                    if not line.startswith("data:"):
                        continue
                    data = line.removeprefix("data:").strip()
                    if data == "[DONE]":
                        completed = True
                        break
                    event = json.loads(data)
                    delta = event["choices"][0]["delta"].get("content")
                    if delta:
                        if not isinstance(delta, str):
                            raise TypeError("stream content delta must be text")
                        yielded = True
                        yield delta
        except GenerationError:
            raise
        except Exception as exc:
            raise GenerationError(
                "OpenAI-compatible streaming generation failed"
            ) from exc
        if not completed:
            raise GenerationError(
                "OpenAI-compatible response stream ended before [DONE]"
            )
        if not yielded:
            raise GenerationError(
                "OpenAI-compatible provider returned an empty response stream"
            )


def build_generator(config: AppConfig | None = None) -> LLMGenerator:
    """Construct the configured generation provider."""

    active_config = config if config is not None else get_config()
    provider = active_config.llm_provider.strip().lower()
    if provider == "gemini":
        return GeminiGenerator(active_config)
    if provider == "vllm":
        return OpenAICompatibleGenerator(active_config)
    raise ValueError(
        f"Unsupported LLM_PROVIDER {active_config.llm_provider!r}; "
        "expected 'gemini' or 'vllm'"
    )
