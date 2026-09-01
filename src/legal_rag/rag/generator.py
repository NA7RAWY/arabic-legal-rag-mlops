"""Grounded answer generation providers."""

from typing import Any, Protocol

from legal_rag.config import AppConfig, get_config
from legal_rag.storage import RetrievalResult

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


class GenerationError(RuntimeError):
    """Raised when an LLM provider cannot generate an answer."""


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
    ) -> None:
        active_config = config if config is not None else get_config()
        self.api_key = api_key or active_config.gemini_api_key
        if not self.api_key:
            raise ValueError(
                "Gemini API key is required; set the GEMINI_API_KEY environment variable"
            )
        self.model_name = model_name or active_config.gemini_model
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def generate(self, question: str, context: str) -> str:
        """Generate an answer grounded only in the supplied legal context."""

        from google.genai import types

        prompt = (
            f"User question:\n{question}\n\n"
            f"Retrieved legal context:\n{context}"
        )
        try:
            response = self._get_client().models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION
                ),
            )
            answer = response.text
        except Exception as exc:
            raise GenerationError("Gemini generation failed") from exc

        if not answer or not answer.strip():
            raise GenerationError("Gemini returned an empty response")
        return answer.strip()
