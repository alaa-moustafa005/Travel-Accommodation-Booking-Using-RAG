"""
app.py — FastAPI web layer for the Safari travel RAG system.

This file owns HTTP concerns only: routes, request/response models, CORS,
and serving the frontend. All AI/retrieval logic lives in rag.py and is
imported here as a single function, rag_answer(). Splitting it this way
means the RAG pipeline can be tested or reused independently of the web
server (e.g. called directly from a script or notebook).
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from rag import rag_answer

# ════════════════════════════════════════════════
# FastAPI app
# ════════════════════════════════════════════════
app = FastAPI(title="Safari Travel RAG System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Q(BaseModel):
    message: str
    history: list = []


@app.post("/ask")
async def ask(q: Q):
    return {"answer": rag_answer(q.message, q.history)}


@app.get("/health")
async def health():
    return {"status": "ok"}


# Serve the frontend (static/index.html) at the root URL, same origin as the API
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse("static/index.html")
