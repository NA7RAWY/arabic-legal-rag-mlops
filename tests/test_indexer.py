"""Unit tests for corpus indexing orchestration."""

import json
from pathlib import Path

import pytest

from legal_rag.config import AppConfig
from legal_rag.ingestion import LegalChunk, index_corpus


def _write_corpus(path: Path) -> None:
    records = [
        {
            "article_number": article_number,
            "book": None,
            "chapter": None,
            "section": None,
            "topic": None,
            "text_ar": f"نص {article_number}",
            "text_en": f"Text {article_number}",
            "is_repealed": False,
            "source_page": article_number,
            "citation": f"Article {article_number}",
        }
        for article_number in (2, 1)
    ]
    path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")


class FakeEmbedder:
    def __init__(self, embedding_count: int = 2) -> None:
        self.embedding_count = embedding_count
        self.received_chunks: list[LegalChunk] = []

    def embed_chunks(self, chunks: list[LegalChunk]) -> list[list[float]]:
        self.received_chunks = chunks
        return [[float(index)] * 384 for index in range(self.embedding_count)]


class FakeRepository:
    def __init__(self) -> None:
        self.received_chunks: list[LegalChunk] = []
        self.received_embeddings: list[list[float]] = []

    def upsert_chunks(
        self,
        chunks: list[LegalChunk],
        embeddings: list[list[float]],
    ) -> int:
        self.received_chunks = chunks
        self.received_embeddings = embeddings
        return len(chunks)


def test_index_corpus_preserves_chunk_and_embedding_order(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.json"
    _write_corpus(corpus_path)
    repository = FakeRepository()
    embedder = FakeEmbedder()

    count = index_corpus(
        repository,
        embedder,
        AppConfig(corpus_path=corpus_path),
    )

    assert count == 2
    assert [chunk.chunk_id for chunk in embedder.received_chunks] == [
        "article-2",
        "article-1",
    ]
    assert repository.received_chunks == embedder.received_chunks
    assert repository.received_embeddings == [[0.0] * 384, [1.0] * 384]


def test_index_corpus_rejects_embedding_count_mismatch(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.json"
    _write_corpus(corpus_path)
    repository = FakeRepository()

    with pytest.raises(ValueError, match="2 chunks, 1 embeddings"):
        index_corpus(
            repository,
            FakeEmbedder(embedding_count=1),
            AppConfig(corpus_path=corpus_path),
        )

    assert repository.received_chunks == []
