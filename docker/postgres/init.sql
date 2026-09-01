CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS legal_chunks (
    chunk_id TEXT PRIMARY KEY,
    article_number INTEGER NOT NULL,
    text TEXT NOT NULL,
    language TEXT NOT NULL,
    book TEXT,
    chapter TEXT,
    section TEXT,
    topic TEXT,
    is_repealed BOOLEAN NOT NULL,
    source_page INTEGER NOT NULL,
    citation TEXT NOT NULL,
    embedding VECTOR(384) NOT NULL
);
