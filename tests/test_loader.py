"""Tests for the legal corpus loading layer."""

import json
from pathlib import Path
from typing import Any

import pytest

from legal_rag.ingestion import LegalArticle, load_articles

CANONICAL_CORPUS = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed"
    / "civil_code_articles_clean_v2.json"
)


def _valid_record() -> dict[str, Any]:
    return {
        "article_number": 1,
        "book": "Test Book",
        "chapter": "Test Chapter",
        "section": None,
        "topic": "Test Topic",
        "text_ar": "نص تجريبي",
        "text_en": "Test text",
        "is_repealed": False,
        "source_page": 3,
        "citation": "Egyptian Civil Code, Article 1",
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def test_load_canonical_corpus() -> None:
    articles = load_articles(CANONICAL_CORPUS)
    article_numbers = [article.article_number for article in articles]
    article_1022 = next(
        article for article in articles if article.article_number == 1022
    )

    assert len(articles) == 1149
    assert articles[0].article_number == 1
    assert articles[-1].article_number == 1149
    assert article_1022.text_ar is None
    assert len(article_numbers) == len(set(article_numbers))


def test_load_valid_temporary_corpus(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.json"
    record = _valid_record()
    _write_json(corpus_path, [record])

    articles = load_articles(corpus_path)

    assert articles == [LegalArticle(**record)]
    assert isinstance(articles[0], LegalArticle)
    assert articles[0].text_ar == "نص تجريبي"


def test_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.json"

    with pytest.raises(FileNotFoundError, match="Corpus file not found"):
        load_articles(missing_path)


def test_invalid_json_raises_value_error(tmp_path: Path) -> None:
    corpus_path = tmp_path / "invalid.json"
    corpus_path.write_text("{invalid", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON in corpus file"):
        load_articles(corpus_path)


def test_non_list_root_raises_value_error(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.json"
    _write_json(corpus_path, _valid_record())

    with pytest.raises(ValueError, match="JSON root must be a list"):
        load_articles(corpus_path)


def test_missing_required_field_raises_value_error(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.json"
    record = _valid_record()
    del record["citation"]
    _write_json(corpus_path, [record])

    with pytest.raises(ValueError, match=r"missing fields: citation"):
        load_articles(corpus_path)


def test_incorrect_field_type_raises_value_error(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.json"
    record = _valid_record()
    record["article_number"] = "1"
    _write_json(corpus_path, [record])

    with pytest.raises(ValueError, match=r"field 'article_number'.*invalid type str"):
        load_articles(corpus_path)


def test_unexpected_field_raises_value_error(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.json"
    record = _valid_record()
    record["unexpected"] = "value"
    _write_json(corpus_path, [record])

    with pytest.raises(ValueError, match=r"unexpected fields: unexpected"):
        load_articles(corpus_path)
