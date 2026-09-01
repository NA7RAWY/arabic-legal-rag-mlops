"""HTTP request and response schemas."""

from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str


class AskRequest(BaseModel):
    question: str
    top_k: int | None = Field(default=None, gt=0)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        question = value.strip()
        if not question:
            raise ValueError("question must not be empty or whitespace-only")
        return question


class SourceResponse(BaseModel):
    chunk_id: str
    article_number: int
    citation: str
    language: str
    similarity: float


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceResponse]
