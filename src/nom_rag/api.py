"""HTTP API for the chat UI.

The embedding model and index load once at startup, not per request — a cold
load is ~10s and would dominate every answer otherwise.

The response deliberately exposes the machinery, not just the answer: which
chunks were retrieved and with what score, whether the question was rewritten
and into what, the trace id, and per-stage latency. A compliance answer you
cannot audit is not worth much, and the same fields are what make the UI able
to show its work.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from threading import BoundedSemaphore

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from nom_rag.config import settings
from nom_rag.conversation import Turn
from nom_rag.corpus.sources import SOURCES
from nom_rag.refs import chunk_covers

WEB_DIR = Path(__file__).parent / "web"

_state: dict = {}
_chat_slots = BoundedSemaphore(settings.max_concurrent_queries)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from nom_rag.index.embed import Embedder
    from nom_rag.index.store import Retriever
    from nom_rag.llm.base import get_client

    _state["retriever"] = Retriever(Embedder(settings.embedding_model))
    _state["client"] = get_client(settings.provider, settings.generation_model)
    yield
    _state.clear()


app = FastAPI(title="nom-rag", lifespan=lifespan)


class TurnIn(BaseModel):
    question: str
    answer: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[TurnIn] = Field(default_factory=list, max_length=20)


class Citation(BaseModel):
    ref: str  # "NOM-035:7.9"
    nom: str
    numeral: str
    text: str  # the clause text, so the UI can show the source without a round trip


class RetrievedOut(BaseModel):
    chunk_id: str
    score: float
    section_title: str
    text: str
    cited: bool


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    retrieved: list[RetrievedOut]
    trace_id: str
    model: str
    latency_ms: dict
    rewritten: bool
    retrieval_query: str


@app.get("/api/health")
def health() -> dict:
    return {
        "ok": bool(_state),
        "provider": settings.provider,
        "generation_model": settings.generation_model,
        "corpus": [
            {"short_id": s.short_id, "nom_id": s.nom_id, "title": s.title}
            for s in SOURCES.values()
        ],
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    from nom_rag.pipeline import ask

    if not _state:
        raise HTTPException(503, "El índice todavía no termina de cargar.")

    history = [Turn(t.question, t.answer) for t in request.history]
    acquired = _chat_slots.acquire(timeout=settings.query_wait_timeout_seconds)
    if not acquired:
        raise HTTPException(
            503,
            "Hay varias consultas en curso. Inténtalo nuevamente en un momento.",
        )
    try:
        result = ask(
            request.question, _state["retriever"], _state["client"], history=history
        )
    except Exception as exc:
        # surface the provider's own message; quota errors are the common case
        raise HTTPException(502, f"El modelo no respondió: {exc}") from exc
    finally:
        _chat_slots.release()

    citations = []
    for ref in result.citations:
        nom, _, numeral = ref.partition(":")
        source = next(
            (c for c in result.retrieved if chunk_covers(c.short_id, c.numeral, c.text, ref)),
            None,
        )
        citations.append(
            Citation(
                ref=ref,
                nom=nom,
                numeral=numeral,
                text=source.text if source else "",
            )
        )

    return ChatResponse(
        answer=result.answer,
        citations=citations,
        retrieved=[
            RetrievedOut(
                chunk_id=c.chunk_id,
                score=c.score,
                section_title=c.section_title,
                text=c.text,
                cited=any(
                    chunk_covers(c.short_id, c.numeral, c.text, ref)
                    for ref in result.citations
                ),
            )
            for c in result.retrieved
        ],
        trace_id=result.trace_id,
        model=result.model,
        latency_ms=result.latency_ms,
        rewritten=result.rewritten,
        retrieval_query=result.retrieval_query,
    )


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> FileResponse:
    return FileResponse(WEB_DIR / "assistant-cat-cutout.png", media_type="image/png")


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
