"""Unit tests for PostgreSQL legal chunk persistence."""

from typing import Any

import pytest

from legal_rag.config import AppConfig, get_config
from legal_rag.ingestion import LegalChunk
from legal_rag.storage import PostgresChunkRepository


def _chunk(article_number: int = 1) -> LegalChunk:
    return LegalChunk(
        chunk_id=f"article-{article_number}",
        article_number=article_number,
        text="نص قانوني",
        language="ar",
        book="Book",
        chapter="Chapter",
        section="Section",
        topic="Topic",
        is_repealed=False,
        source_page=5,
        citation=f"Article {article_number}",
    )


class FakeCursor:
    def __init__(
        self,
        fetch_result: tuple[Any, ...] | None = None,
        fetch_results: list[tuple[Any, ...]] | None = None,
    ) -> None:
        self.fetch_result = fetch_result
        self.fetch_results = fetch_results or []
        self.executed: list[tuple[str, Any]] = []
        self.executemany_call: tuple[str, list[tuple[Any, ...]]] | None = None

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def execute(self, query: str, parameters: Any = None) -> None:
        self.executed.append((query, parameters))

    def executemany(
        self,
        query: str,
        parameters: list[tuple[Any, ...]],
    ) -> None:
        self.executemany_call = (query, parameters)

    def fetchone(self) -> tuple[Any, ...] | None:
        return self.fetch_result

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self.fetch_results


class FakeConnection:
    def __init__(self, cursor: FakeCursor) -> None:
        self.fake_cursor = cursor

    def __enter__(self) -> "FakeConnection":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def cursor(self) -> FakeCursor:
        return self.fake_cursor


class FakeConnectionFactory:
    def __init__(self, cursor: FakeCursor) -> None:
        self.cursor = cursor
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> FakeConnection:
        self.calls.append(kwargs)
        return FakeConnection(self.cursor)


def test_upsert_chunks_uses_conflict_update_and_preserves_values() -> None:
    cursor = FakeCursor()
    factory = FakeConnectionFactory(cursor)
    repository = PostgresChunkRepository(
        AppConfig(),
        connection_factory=factory,
    )
    chunk = _chunk()

    count = repository.upsert_chunks([chunk], [[0.5] * 384])

    assert count == 1
    assert cursor.executemany_call is not None
    query, parameters = cursor.executemany_call
    assert "ON CONFLICT (chunk_id) DO UPDATE" in query
    assert "%s::vector" in query
    assert parameters[0][:11] == (
        chunk.chunk_id,
        chunk.article_number,
        chunk.text,
        chunk.language,
        chunk.book,
        chunk.chapter,
        chunk.section,
        chunk.topic,
        chunk.is_repealed,
        chunk.source_page,
        chunk.citation,
    )
    assert parameters[0][11].startswith("[0.5,0.5")
    assert factory.calls[0]["dbname"] == "legal_rag"


def test_upsert_rejects_incorrect_embedding_dimension() -> None:
    repository = PostgresChunkRepository(
        AppConfig(),
        connection_factory=lambda **kwargs: pytest.fail("must not connect"),
    )

    with pytest.raises(ValueError, match="384, received 383"):
        repository.upsert_chunks([_chunk()], [[0.0] * 383])


def test_upsert_rejects_chunk_embedding_count_mismatch() -> None:
    repository = PostgresChunkRepository(AppConfig())

    with pytest.raises(ValueError, match="1 chunks, 0 embeddings"):
        repository.upsert_chunks([_chunk()], [])


def test_count_chunks_and_get_chunk_use_parameterized_queries() -> None:
    count_cursor = FakeCursor((7,))
    count_repository = PostgresChunkRepository(
        AppConfig(),
        connection_factory=FakeConnectionFactory(count_cursor),
    )
    stored_chunk = _chunk(1022)
    chunk_cursor = FakeCursor(
        (
            stored_chunk.chunk_id,
            stored_chunk.article_number,
            stored_chunk.text,
            stored_chunk.language,
            stored_chunk.book,
            stored_chunk.chapter,
            stored_chunk.section,
            stored_chunk.topic,
            stored_chunk.is_repealed,
            stored_chunk.source_page,
            stored_chunk.citation,
        )
    )
    chunk_repository = PostgresChunkRepository(
        AppConfig(),
        connection_factory=FakeConnectionFactory(chunk_cursor),
    )

    assert count_repository.count_chunks() == 7
    assert chunk_repository.get_chunk("article-1022") == stored_chunk
    assert chunk_cursor.executed[0][1] == ("article-1022",)


def test_config_reads_postgres_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSTGRES_DB", "custom_db")
    monkeypatch.setenv("POSTGRES_USER", "custom_user")
    monkeypatch.setenv("POSTGRES_PASSWORD", "custom_password")
    monkeypatch.setenv("POSTGRES_HOST", "database")
    monkeypatch.setenv("POSTGRES_PORT", "5544")

    config = get_config()

    assert config.postgres_db == "custom_db"
    assert config.postgres_user == "custom_user"
    assert config.postgres_password == "custom_password"
    assert config.postgres_host == "database"
    assert config.postgres_port == 5544


def test_semantic_search_uses_cosine_distance_and_parameters() -> None:
    chunk = _chunk(9)
    cursor = FakeCursor(
        fetch_results=[
            (
                chunk.chunk_id,
                chunk.article_number,
                chunk.text,
                chunk.language,
                chunk.book,
                chunk.chapter,
                chunk.section,
                chunk.topic,
                chunk.is_repealed,
                chunk.source_page,
                chunk.citation,
                0.875,
            )
        ]
    )
    repository = PostgresChunkRepository(
        AppConfig(),
        connection_factory=FakeConnectionFactory(cursor),
    )

    results = repository.semantic_search([0.25] * 384, top_k=3)

    query, parameters = cursor.executed[0]
    assert "%s::vector" in query
    assert "1 - (chunks.embedding <=> query_vector.embedding)" in query
    assert "ORDER BY chunks.embedding <=> query_vector.embedding" in query
    assert "LIMIT %s" in query
    assert parameters[1] == 3
    assert parameters[0].startswith("[0.25,0.25")
    assert results[0].chunk_id == "article-9"
    assert results[0].similarity == 0.875


def test_semantic_search_validates_dimension_and_top_k() -> None:
    repository = PostgresChunkRepository(AppConfig())

    with pytest.raises(ValueError, match="384, received 2"):
        repository.semantic_search([0.1, 0.2], top_k=5)

    with pytest.raises(ValueError, match="top_k must be greater than zero"):
        repository.semantic_search([0.0] * 384, top_k=0)


def test_repository_uses_validated_alternative_table_name() -> None:
    cursor = FakeCursor((0,))
    repository = PostgresChunkRepository(
        AppConfig(),
        connection_factory=FakeConnectionFactory(cursor),
        table_name="legal_chunks_experiment",
    )

    assert repository.count_chunks() == 0
    assert "FROM legal_chunks_experiment" in cursor.executed[0][0]

    with pytest.raises(ValueError, match="safe lowercase SQL identifier"):
        PostgresChunkRepository(AppConfig(), table_name="legal_chunks; DROP TABLE")
