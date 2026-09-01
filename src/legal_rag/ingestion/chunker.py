"""Convert legal articles into retrieval-ready article chunks."""

from dataclasses import dataclass

from legal_rag.ingestion.loader import LegalArticle


@dataclass(frozen=True, slots=True)
class LegalChunk:
    """A single article-sized legal text chunk with source metadata."""

    chunk_id: str
    article_number: int
    text: str
    language: str
    book: str | None
    chapter: str | None
    section: str | None
    topic: str | None
    is_repealed: bool
    source_page: int
    citation: str


def article_to_chunk(article: LegalArticle) -> LegalChunk:
    """Convert one legal article into one chunk, preferring Arabic text."""

    if article.text_ar and article.text_ar.strip():
        text = article.text_ar
        language = "ar"
    elif article.text_en and article.text_en.strip():
        text = article.text_en
        language = "en"
    else:
        raise ValueError(
            f"Article {article.article_number} has no usable Arabic or English text"
        )

    return LegalChunk(
        chunk_id=f"article-{article.article_number}",
        article_number=article.article_number,
        text=text,
        language=language,
        book=article.book,
        chapter=article.chapter,
        section=article.section,
        topic=article.topic,
        is_repealed=article.is_repealed,
        source_page=article.source_page,
        citation=article.citation,
    )


def chunk_articles(articles: list[LegalArticle]) -> list[LegalChunk]:
    """Convert legal articles to chunks while preserving input order."""

    return [article_to_chunk(article) for article in articles]
