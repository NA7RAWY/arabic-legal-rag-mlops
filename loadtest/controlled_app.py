"""Deterministic ASGI target for application-layer load testing only."""

from collections.abc import Iterator

from legal_rag.api.app import create_app
from legal_rag.api.dependencies import get_rag_service
from legal_rag.rag import LegalRAGResult, LegalRAGStreamResult
from legal_rag.storage import RetrievalResult


def _controlled_source() -> RetrievalResult:
    return RetrievalResult(
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
        similarity=1.0,
    )


class ControlledRAGService:
    """Stable fake boundary; never used by the production application entrypoint."""

    def answer(self, question: str, top_k: int | None = None) -> LegalRAGResult:
        return LegalRAGResult(
            question=question,
            answer="إجابة اختبارية متحكم بها مستندة إلى المادة 164.",
            retrieved_sources=(_controlled_source(),),
        )

    def stream_answer(
        self,
        question: str,
        top_k: int | None = None,
    ) -> LegalRAGStreamResult:
        def chunks() -> Iterator[str]:
            yield "إجابة اختبارية "
            yield "متحكم بها "
            yield "مستندة إلى المادة 164."

        return LegalRAGStreamResult(
            question=question,
            chunks=chunks(),
            retrieved_sources=(_controlled_source(),),
        )


app = create_app()
app.dependency_overrides[get_rag_service] = ControlledRAGService
