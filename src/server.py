from __future__ import annotations
from contextlib import contextmanager
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

@app.post("/infernce", response_model=AgentResponse)
def inference(request: AgentRequest) -> AgentResponse:
    if _runtime is None:
        settings = load_settings()
        runtime = RagRuntime(settings)
        runtime.load_index()
        answer = _runtime.infer(request.query)
        return AgentResponse(answer=cast(Literal["A", "B", "C", "D"], answer))
    
    answer = _runtime.infer(request.query)
    return AgentResponse(answer=cast(Literal["A", "B", "C", "D"], answer))
