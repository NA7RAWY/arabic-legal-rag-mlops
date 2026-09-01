"""PostgreSQL persistence for embedded legal chunks."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import psycopg

from legal_rag.config import AppConfig, get_config
from legal_rag.ingestion import LegalChunk

EMBEDDING_DIMENSION = 384

_UPSERT_SQL = """
INSERT INTO legal_chunks (
    chunk_id,
    article_number,
    text,
    language,
    book,
    chapter,
    section,
    topic,
    is_repealed,
    source_page,
    citation,
    embedding
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
ON CONFLICT (chunk_id) DO UPDATE SET
    article_number = EXCLUDED.article_number,
    text = EXCLUDED.text,
    language = EXCLUDED.language,
    book = EXCLUDED.book,
    chapter = EXCLUDED.chapter,
    section = EXCLUDED.section,
    topic = EXCLUDED.topic,
    is_repealed = EXCLUDED.is_repealed,
    source_page = EXCLUDED.source_page,
    citation = EXCLUDED.citation,
    embedding = EXCLUDED.embedding
"""

_SEMANTIC_SEARCH_SQL = """
WITH query_vector AS (
    SELECT %s::vector AS embedding
)
SELECT
    chunks.chunk_id,
    chunks.article_number,
    chunks.text,
    chunks.language,
    chunks.book,
    chunks.chapter,
    chunks.section,
    chunks.topic,
    chunks.is_repealed,
    chunks.source_page,
    chunks.citation,
    1 - (chunks.embedding <=> query_vector.embedding) AS similarity
FROM legal_chunks AS chunks
CROSS JOIN query_vector
ORDER BY chunks.embedding <=> query_vector.embedding
LIMIT %s
"""


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """A legal chunk returned by semantic search."""

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
    similarity: float


class PostgresChunkRepository:
    """Store and inspect legal chunks in PostgreSQL with pgvector."""

    def __init__(
        self,
        config: AppConfig | None = None,
        connection_factory: Callable[..., Any] = psycopg.connect,
    ) -> None:
        self.config = config if config is not None else get_config()
        self._connection_factory = connection_factory

    def _connect(self) -> Any:
        return self._connection_factory(
            dbname=self.config.postgres_db,
            user=self.config.postgres_user,
            password=self.config.postgres_password,
            host=self.config.postgres_host,
            port=self.config.postgres_port,
        )

    @staticmethod
    def _validate_embedding(embedding: list[float]) -> None:
        if len(embedding) != EMBEDDING_DIMENSION:
            raise ValueError(
                "Embedding dimension must be "
                f"{EMBEDDING_DIMENSION}, received {len(embedding)}"
            )

    @classmethod
    def _upsert_parameters(
        cls,
        chunk: LegalChunk,
        embedding: list[float],
    ) -> tuple[Any, ...]:
        cls._validate_embedding(embedding)
        vector = "[" + ",".join(str(float(value)) for value in embedding) + "]"
        return (
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
            vector,
        )

    def upsert_chunks(
        self,
        chunks: list[LegalChunk],
        embeddings: list[list[float]],
    ) -> int:
        """Insert or update chunks and embeddings in one transaction."""

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Chunk and embedding counts must match: "
                f"{len(chunks)} chunks, {len(embeddings)} embeddings"
            )

        parameters = [
            self._upsert_parameters(chunk, embedding)
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        if not parameters:
            return 0

        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.executemany(_UPSERT_SQL, parameters)
        return len(parameters)

    def count_chunks(self) -> int:
        """Return the number of stored legal chunks."""

        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT count(*) FROM legal_chunks")
                row = cursor.fetchone()
        if row is None:
            raise RuntimeError("PostgreSQL did not return a chunk count")
        return int(row[0])

    def get_chunk(self, chunk_id: str) -> LegalChunk | None:
        """Fetch one stored chunk by its deterministic identifier."""

        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        chunk_id,
                        article_number,
                        text,
                        language,
                        book,
                        chapter,
                        section,
                        topic,
                        is_repealed,
                        source_page,
                        citation
                    FROM legal_chunks
                    WHERE chunk_id = %s
                    """,
                    (chunk_id,),
                )
                row = cursor.fetchone()
        return LegalChunk(*row) if row is not None else None

    def semantic_search(
        self,
        embedding: list[float],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Return the nearest chunks ordered by cosine similarity."""

        self._validate_embedding(embedding)
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        vector = "[" + ",".join(str(float(value)) for value in embedding) + "]"
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(_SEMANTIC_SEARCH_SQL, (vector, top_k))
                rows = cursor.fetchall()

        return [
            RetrievalResult(*row[:-1], similarity=float(row[-1]))
            for row in rows
        ]
