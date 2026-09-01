"""Load legal articles from the canonical JSON corpus."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from legal_rag.config import AppConfig, get_config

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LegalArticle:
    """A single article from the Egyptian Civil Code corpus."""

    article_number: int
    book: str | None
    chapter: str | None
    section: str | None
    topic: str | None
    text_ar: str | None
    text_en: str
    is_repealed: bool
    source_page: int
    citation: str


_FIELD_TYPES: dict[str, type | tuple[type, ...]] = {
    "article_number": int,
    "book": (str, type(None)),
    "chapter": (str, type(None)),
    "section": (str, type(None)),
    "topic": (str, type(None)),
    "text_ar": (str, type(None)),
    "text_en": str,
    "is_repealed": bool,
    "source_page": int,
    "citation": str,
}


def _parse_article(record: Any, index: int) -> LegalArticle:
    if not isinstance(record, dict):
        raise ValueError(f"Malformed article at index {index}: expected an object")

    missing_fields = _FIELD_TYPES.keys() - record.keys()
    if missing_fields:
        missing = ", ".join(sorted(missing_fields))
        raise ValueError(
            f"Malformed article at index {index}: missing fields: {missing}"
        )

    unexpected_fields = record.keys() - _FIELD_TYPES.keys()
    if unexpected_fields:
        unexpected = ", ".join(sorted(unexpected_fields))
        raise ValueError(
            f"Malformed article at index {index}: unexpected fields: {unexpected}"
        )

    for field_name, expected_type in _FIELD_TYPES.items():
        value = record[field_name]
        if not isinstance(value, expected_type):
            raise ValueError(
                f"Malformed article at index {index}: field {field_name!r} "
                f"has invalid type {type(value).__name__}"
            )

    return LegalArticle(**record)


def load_articles(path: Path) -> list[LegalArticle]:
    """Read and validate legal articles from a UTF-8 JSON file."""

    try:
        with path.open(encoding="utf-8") as corpus_file:
            records = json.load(corpus_file)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Corpus file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in corpus file {path}: {exc.msg} "
            f"(line {exc.lineno}, column {exc.colno})"
        ) from exc

    if not isinstance(records, list):
        raise ValueError(
            f"Invalid corpus format in {path}: JSON root must be a list"
        )

    return [_parse_article(record, index) for index, record in enumerate(records)]


def load_corpus(config: AppConfig | None = None) -> list[LegalArticle]:
    """Load the corpus configured for the application."""

    active_config = config if config is not None else get_config()
    articles = load_articles(active_config.corpus_path)
    logger.info(
        "Loaded %d articles from corpus %s",
        len(articles),
        active_config.corpus_path,
    )
    return articles
