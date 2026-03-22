from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal, cast

from fastapi import FastAPI

from src.api_models import AgentRequest, AgentResponse
from src.rag_service import RAGRuntime
from src.settings import load_settings

_runtime: RAGRuntime | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _runtime
    settings = load_settings()
    _runtime = RAGRuntime(settings)
    _runtime.load_index()
    yield

app = FastAPI(title="RAG Agent System", lifespan=lifespan)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}


def _get_runtime() -> RAGRuntime:
    global _runtime
    if _runtime is None:
        settings = load_settings()
        _runtime = RAGRuntime(settings)
        _runtime.load_index()
    return _runtime


@app.post("/inference", response_model=AgentResponse)
def inference(request: AgentRequest) -> AgentResponse:
    runtime = _get_runtime()
    answer = runtime.infer(request.query)
    return AgentResponse(answer=cast(Literal["A", "B", "C", "D"], answer))


# Keep legacy misspelled route for compatibility.
@app.post("/infernce", response_model=AgentResponse)
def inference_legacy(request: AgentRequest) -> AgentResponse:
    return inference(request)
