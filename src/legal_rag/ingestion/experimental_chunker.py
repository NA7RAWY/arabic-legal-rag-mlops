"""Experimental long-article chunking without changing production defaults."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil
from statistics import median

from legal_rag.ingestion.chunker import LegalChunk, article_to_chunk
from legal_rag.ingestion.loader import LegalArticle


@dataclass(frozen=True, slots=True)
class SplitChunkConfig:
    split_threshold: int = 600
    chunk_size: int = 500
    chunk_overlap: int = 75

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap must be between zero and chunk_size")
        if self.split_threshold < self.chunk_size:
            raise ValueError("split_threshold must be at least chunk_size")


@dataclass(frozen=True, slots=True)
class CorpusLengthAnalysis:
    article_count: int
    minimum: int
    median: float
    p75: int
    p90: int
    p95: int
    maximum: int
    articles_split: int
    split_percentage: float


def _selected_text(article: LegalArticle) -> str:
    return article_to_chunk(article).text


def analyze_corpus_lengths(
    articles: list[LegalArticle],
    *,
    split_threshold: int,
) -> CorpusLengthAnalysis:
    """Compute deterministic nearest-rank character-length statistics."""

    if not articles:
        raise ValueError("At least one article is required")
    lengths = sorted(len(_selected_text(article)) for article in articles)

    def percentile(value: float) -> int:
        return lengths[ceil(value * len(lengths)) - 1]

    split_count = sum(length > split_threshold for length in lengths)
    return CorpusLengthAnalysis(
        article_count=len(lengths),
        minimum=lengths[0],
        median=float(median(lengths)),
        p75=percentile(0.75),
        p90=percentile(0.90),
        p95=percentile(0.95),
        maximum=lengths[-1],
        articles_split=split_count,
        split_percentage=100.0 * split_count / len(lengths),
    )


def split_long_article(
    article: LegalArticle,
    config: SplitChunkConfig,
) -> list[LegalChunk]:
    """Split one qualifying article with deterministic character overlap."""

    base = article_to_chunk(article)
    if len(base.text) <= config.split_threshold:
        return [base]

    step = config.chunk_size - config.chunk_overlap
    pieces = []
    for part, start in enumerate(range(0, len(base.text), step), start=1):
        text = base.text[start : start + config.chunk_size]
        pieces.append(
            replace(
                base,
                chunk_id=f"article-{article.article_number}-part-{part}",
                text=text,
            )
        )
        if start + config.chunk_size >= len(base.text):
            break
    return pieces


def split_long_articles(
    articles: list[LegalArticle],
    config: SplitChunkConfig | None = None,
) -> list[LegalChunk]:
    """Apply experimental splitting while preserving article order."""

    active_config = config or SplitChunkConfig()
    return [
        chunk
        for article in articles
        for chunk in split_long_article(article, active_config)
    ]
