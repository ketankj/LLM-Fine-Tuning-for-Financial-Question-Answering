"""FastAPI service for the fine-tuned financial QA model.

Run:
    uvicorn app.main:app --host 0.0.0.0 --port 8000

Env vars:
    FINQA_BASE_MODEL   base model id      (default Qwen/Qwen2.5-1.5B-Instruct)
    FINQA_ADAPTER      LoRA adapter path  (default: none → base model)
    FINQA_LOAD_4BIT    "1"/"0"            (default 1; ignored without a GPU)
"""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    context: str | None = Field(None, max_length=12000)
    max_new_tokens: int = Field(128, ge=1, le=512)


class AskResponse(BaseModel):
    answer: str
    model: str
    adapter: str | None
    latency_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = None
    app.state.base_model = os.getenv("FINQA_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    app.state.adapter = os.getenv("FINQA_ADAPTER") or None
    if os.getenv("FINQA_SKIP_LOAD") != "1":  # tests set this and inject a fake model
        from finqa_ft.inference import FinQAModel

        app.state.model = FinQAModel.load(
            app.state.base_model, app.state.adapter, os.getenv("FINQA_LOAD_4BIT", "1") == "1"
        )
    yield


app = FastAPI(title="Financial QA (fine-tuned LLM)", version="0.1.0", lifespan=lifespan)


def get_model(request: Request):
    model = request.app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model


@app.get("/health")
def health(request: Request):
    return {
        "status": "ok",
        "model_loaded": request.app.state.model is not None,
        "base_model": request.app.state.base_model,
        "adapter": request.app.state.adapter,
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, request: Request, model=Depends(get_model)):
    start = time.perf_counter()
    answer = model.answer(req.question, req.context, req.max_new_tokens)
    return AskResponse(
        answer=answer,
        model=request.app.state.base_model,
        adapter=request.app.state.adapter,
        latency_ms=round((time.perf_counter() - start) * 1000, 2),
    )
