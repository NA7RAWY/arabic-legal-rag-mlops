"""Tests for article-level legal corpus chunking."""

from dataclasses import replace
from pathlib import Path

import pytest

from legal_rag.ingestion import (
    LegalArticle,
    article_to_chunk,
    chunk_articles,
    load_articles,
)

CANONICAL_CORPUS = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed"
    / "civil_code_articles_clean_v2.json"
)


def _article(article_number: int = 1) -> LegalArticle:
    return LegalArticle(
        article_number=article_number,
        book="Test Book",
        chapter="Test Chapter",
        section="Test Section",
        topic="Test Topic",
        text_ar="نص عربي",
        text_en="English text",
        is_repealed=True,
        source_page=7,
        citation=f"Egyptian Civil Code, Article {article_number}",
    )


def test_article_to_chunk_prefers_arabic_and_preserves_metadata() -> None:
    article = _article()

    chunk = article_to_chunk(article)

    assert chunk.chunk_id == "article-1"
    assert chunk.article_number == article.article_number
    assert chunk.text == article.text_ar
    assert chunk.language == "ar"
    assert chunk.book == article.book
    assert chunk.chapter == article.chapter
    assert chunk.section == article.section
    assert chunk.topic == article.topic
    assert chunk.is_repealed == article.is_repealed
    assert chunk.source_page == article.source_page
    assert chunk.citation == article.citation


def test_article_to_chunk_falls_back_to_english() -> None:
    article = replace(_article(1022), text_ar=None)

    chunk = article_to_chunk(article)

    assert chunk.text == article.text_en
    assert chunk.language == "en"
    assert chunk.chunk_id == "article-1022"


def test_article_to_chunk_falls_back_for_empty_arabic() -> None:
    article = replace(_article(), text_ar="")

    chunk = article_to_chunk(article)

    assert chunk.text == article.text_en
    assert chunk.language == "en"


def test_article_to_chunk_rejects_missing_text() -> None:
    article = replace(_article(), text_ar=None, text_en="")

    with pytest.raises(ValueError, match="Article 1 has no usable"):
        article_to_chunk(article)


def test_chunk_articles_preserves_count_and_order() -> None:
    articles = [_article(3), _article(1), _article(2)]

    chunks = chunk_articles(articles)

    assert len(chunks) == len(articles)
    assert [chunk.article_number for chunk in chunks] == [3, 1, 2]
    assert [chunk.chunk_id for chunk in chunks] == [
        "article-3",
        "article-1",
        "article-2",
    ]


@pytest.mark.skipif(
    not CANONICAL_CORPUS.is_file(),
    reason="canonical corpus is a DVC-managed integration fixture",
)
def test_chunk_canonical_corpus() -> None:
    articles = load_articles(CANONICAL_CORPUS)

    chunks = chunk_articles(articles)
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    chunk_1022 = next(chunk for chunk in chunks if chunk.article_number == 1022)
    article_1022 = next(
        article for article in articles if article.article_number == 1022
    )

    assert len(chunks) == 1149
    assert chunks[0].chunk_id == "article-1"
    assert chunks[-1].chunk_id == "article-1149"
    assert chunk_1022.language == "en"
    assert chunk_1022.text == article_1022.text_en
    assert len(chunk_ids) == len(set(chunk_ids))
