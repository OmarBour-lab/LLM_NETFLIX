import time
import uuid
from collections import defaultdict

from fastapi import Depends, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

try:
    from ..chain import ask_netflix
    from .auth import create_access_token, get_current_subject
    from .schemas import (
        ChatRequest,
        ChatResponse,
        FeedbackRequest,
        FeedbackResponse,
        HealthResponse,
        Message,
        StatsResponse,
        TokenRequest,
        TokenResponse,
    )
except ImportError:
    from src.chain import ask_netflix
    from src.backend.auth import create_access_token, get_current_subject
    from src.backend.schemas import (
        ChatRequest,
        ChatResponse,
        FeedbackRequest,
        FeedbackResponse,
        HealthResponse,
        Message,
        StatsResponse,
        TokenRequest,
        TokenResponse,
    )


app = FastAPI(
    title="Netflix RAG + NL2SQL API",
    version="1.0.0",
    description="Local Netflix catalog chatbot API powered by RAG, NL2SQL, and Ollama.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SESSION_HISTORY: dict[str, list[Message]] = defaultdict(list)
FEEDBACK: list[FeedbackRequest] = []


def message_to_dict(message: Message) -> dict:
    if hasattr(message, "model_dump"):
        return message.model_dump()
    return message.dict()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="netflix-rag-nl2sql-api")


@app.post("/token", response_model=TokenResponse)
def token(request: TokenRequest) -> TokenResponse:
    return TokenResponse(access_token=create_access_token(request.username))


@app.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    subject: str = Depends(get_current_subject),
) -> ChatResponse:
    del subject
    start = time.perf_counter()
    session_id = request.session_id or str(uuid.uuid4())
    history = [message_to_dict(message) for message in (request.history or SESSION_HISTORY[session_id])]
    answer = ask_netflix(request.question, history=history)
    latency_ms = (time.perf_counter() - start) * 1000

    SESSION_HISTORY[session_id].append(Message(role="user", content=request.question))
    SESSION_HISTORY[session_id].append(Message(role="assistant", content=answer))

    return ChatResponse(
        answer=answer,
        session_id=session_id,
        latency_ms=latency_ms,
        sources=[],
    )


@app.get("/history", response_model=list[Message])
def history(
    session_id: str,
    subject: str = Depends(get_current_subject),
) -> list[Message]:
    del subject
    return SESSION_HISTORY.get(session_id, [])


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(
    request: FeedbackRequest,
    subject: str = Depends(get_current_subject),
) -> FeedbackResponse:
    del subject
    FEEDBACK.append(request)
    return FeedbackResponse(status="received")


@app.get("/stats", response_model=StatsResponse)
def stats(subject: str = Depends(get_current_subject)) -> StatsResponse:
    del subject
    return StatsResponse(
        sessions=len(SESSION_HISTORY),
        messages=sum(len(messages) for messages in SESSION_HISTORY.values()),
        feedback_items=len(FEEDBACK),
    )


@app.websocket("/ws/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    try:
        while True:
            question = await websocket.receive_text()
            history = [message_to_dict(message) for message in SESSION_HISTORY[session_id]]
            answer = ask_netflix(question, history=history)
            SESSION_HISTORY[session_id].append(Message(role="user", content=question))
            SESSION_HISTORY[session_id].append(Message(role="assistant", content=answer))

            for token in answer.split():
                await websocket.send_text(token + " ")
            await websocket.send_text("[DONE]")
    except WebSocketDisconnect:
        return
