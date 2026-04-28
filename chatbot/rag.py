import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
from chatbot.config import INDEX_DIR, EMBEDDING_MODEL, LLM_MODEL, TOP_K, MAX_CONTEXT_TOKENS

# probably not what you want in production,  but locally we want the key in the project, not environment
load_dotenv(override=True)

_embed_model = None
_embeddings = None
_chunks = None
_client = None

def _load():
    global _embed_model, _embeddings, _chunks, _client
    if _embeddings is None:
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)
        _embeddings = np.load(os.path.join(INDEX_DIR, "embeddings.npy"))
        with open(os.path.join(INDEX_DIR, "chunks.json")) as f:
            _chunks = json.load(f)
        _client = OpenAI()

def retrieve(query: str) -> list[dict]:
    _load()
    q_vec = _embed_model.encode([query])[0]
    q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-10)
    scores = _embeddings @ q_norm
    top_indices = np.argsort(scores)[::-1][:TOP_K]
    results = []
    for i in top_indices:
        chunk = dict(_chunks[i])
        chunk["score"] = float(scores[i])
        results.append(chunk)
    return results

SYSTEM_PROMPT = """\
You are a helpful assistant for Digital Performer, the professional DAW (Digital Audio Workstation) software by MOTU.

Answer the user's question using ONLY the User Guide excerpts provided below.
- Cite your sources using [page X] format.
- If the excerpts do not contain enough information, say so clearly and suggest the user check the full manual or MOTU support at motu.com/support.
- Be specific and practical — the user is trying to accomplish a task in Digital Performer.
- Quote relevant menu names, window names, and keyboard shortcuts exactly as they appear in the manual.

USER GUIDE EXCERPTS:
{context}"""

def _build_context(chunks: list[dict]) -> tuple[str, list[dict]]:
    parts = []
    sources = []
    token_total = 0
    for chunk in chunks:
        tokens = chunk.get("token_count", 0)
        if token_total + tokens > MAX_CONTEXT_TOKENS:
            break
        parts.append(f"---\n{chunk['text']}")
        sources.append({
            "chunk_id": chunk.get("chunk_id", ""),
            "chunk_index": chunk.get("chunk_index", 0),
            "score": round(chunk.get("score", 0), 3),
        })
        token_total += tokens
    return "\n\n".join(parts), sources

def generate_answer(query: str, chunks: list[dict]) -> dict:
    _load()
    context, sources = _build_context(chunks)
    prompt = SYSTEM_PROMPT.format(context=context)
    response = _client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": query},
        ],
        temperature=0.2,
    )
    return {
        "answer": response.choices[0].message.content,
        "sources": sources,
    }

def generate_answer_stream(query: str, chunks: list[dict]):
    _load()
    context, sources = _build_context(chunks)
    prompt = SYSTEM_PROMPT.format(context=context)
    stream = _client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": query},
        ],
        temperature=0.2,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
    yield None  # sentinel — signals end of stream, caller sends sources
    yield sources
