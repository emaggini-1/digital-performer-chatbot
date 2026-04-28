import json
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from chatbot.rag import retrieve, generate_answer, generate_answer_stream

app = FastAPI(title="Digital Performer Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)

STATIC_DIR = Path(__file__).parent / "static"
INPUT_DIR = Path(__file__).parent.parent / "input"

app.mount("/docs", StaticFiles(directory=str(INPUT_DIR)), name="docs")

class ChatRequest(BaseModel):
    message: str

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.post("/api/chat")
def chat(req: ChatRequest):
    chunks = retrieve(req.message)
    return generate_answer(req.message, chunks)

@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest):
    chunks = retrieve(req.message)

    def event_stream():
        try:
            sources = None
            for item in generate_answer_stream(req.message, chunks):
                if item is None:
                    continue
                if isinstance(item, list):
                    sources = item
                    break
                yield f"data: {json.dumps({'token': item})}\n\n"
            if sources is not None:
                yield f"data: {json.dumps({'sources': sources, 'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC_DIR / "index.html").read_text()
