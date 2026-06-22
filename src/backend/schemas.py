from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str = Field(..., examples=["user"])
    content: str


class ChatRequest(BaseModel):
    question: str
    session_id: str | None = None
    history: list[Message] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    latency_ms: float
    sources: list[str] = Field(default_factory=list)


class FeedbackRequest(BaseModel):
    session_id: str
    question: str
    answer: str
    rating: int = Field(..., ge=1, le=5)
    comment: str | None = None


class FeedbackResponse(BaseModel):
    status: str


class HealthResponse(BaseModel):
    status: str
    service: str


class StatsResponse(BaseModel):
    sessions: int
    messages: int
    feedback_items: int


class SqlQueryResult(BaseModel):
    sql: str
    rows: list[dict]
    latency_ms: float
    error: str | None = None


class TokenRequest(BaseModel):
    username: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

