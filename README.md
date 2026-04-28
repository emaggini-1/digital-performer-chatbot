# Digital Performer Chatbot

A RAG (Retrieval-Augmented Generation) chatbot that answers questions about Digital Performer by searching the official MOTU User Guide and generating grounded, cited answers via an LLM.

## How it works

1. The User Guide PDF is chunked into 1,550 text segments and embedded as 384-dimensional vectors
2. When you ask a question, it is embedded with the same model and compared against all chunks using cosine similarity
3. The top 8 most relevant chunks are retrieved and sent to GPT-4o-mini as grounding context
4. The model answers using only those excerpts and cites page numbers — citations are clickable links into the PDF

See [`rag-explainer.md`](./rag-explainer.md) for a detailed technical explanation.

---

## Prerequisites

- Python 3.9+
- An OpenAI API key ([platform.openai.com/api-keys](https://platform.openai.com/api-keys))
- The Digital Performer User Guide PDF placed at `input/Digital+Performer+User+Guide.pdf`

---

## Quick start

If the search index has already been built (i.e. `chatbot/search_index/` exists with `embeddings.npy` and `chunks.json`), you can skip straight to running the server.

### 1. Install dependencies

```bash
pip3 install fastapi "uvicorn[standard]" openai sentence-transformers \
             numpy python-dotenv tiktoken pandas pyarrow \
             pdfplumber pymupdf datasketch
```

### 2. Set your OpenAI API key

```bash
cp .env.example .env
# Then edit .env and replace the placeholder with your real key
```

Or create `.env` manually:

```
OPENAI_API_KEY=sk-proj-...
```

### 3. Run the server

```bash
uvicorn chatbot.server:app --port 8000
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

### Stopping the server

If you started the server in the foreground, press `Ctrl+C`.

If it is running in the background (e.g. started with `&` or from a previous session), find and kill it by port:

```bash
kill $(lsof -ti :8000)
```

To confirm nothing is left listening:

```bash
lsof -i :8000
```

---

## Rebuilding the search index from scratch

Only needed if you replace the PDF or want to re-run the full pipeline.

### Step 1 — Extract text from the PDF

```bash
python3 extract_text.py
```

Reads `input/*.pdf` → writes per-document JSON to `training_data/raw/`.

### Step 2 — Clean and normalize text

```bash
python3 clean_text.py
```

Fixes OCR artifacts (soft hyphens, ligatures, excess whitespace) → writes to `training_data/cleaned/`.

### Step 3 — Chunk into token-sized segments

```bash
python3 chunk_text.py
```

Splits each document into 512-token chunks with 100-token overlap → writes `training_data/chunks/all_chunks.jsonl`.

### Step 4 — Enrich with metadata

```bash
python3 enrich_metadata.py
```

Attaches corpus name, extraction date, and document metadata → writes `training_data/chunks/enriched_chunks.jsonl`.

### Step 5 — Export to Parquet (optional)

```bash
python3 export_parquet.py
```

Columnar format for downstream ML tooling → writes `training_data/chunks/enriched_chunks.parquet`.

### Step 6 — Generate embeddings

```bash
python3 generate_embeddings.py
```

Runs `all-MiniLM-L6-v2` locally to produce 384-dim vectors → writes `training_data/chunks/embeddings.npz` and `embedding_ids.json`.

### Step 7 — Deduplicate

```bash
python3 dedup_chunks.py
```

MinHash LSH near-duplicate detection → writes `training_data/chunks/deduped_chunks.jsonl`.

### Step 8 — Build the server index

```bash
python3 chatbot/build_index.py
```

Packages the deduped chunks and embeddings into `chatbot/search_index/` for use by the server.

### Run the server

```bash
uvicorn chatbot.server:app --port 8000
```

---

## Project structure

```
dp-chatbot/
├── chatbot/
│   ├── build_index.py        # Packages training data into the search index
│   ├── config.py             # Model names, chunk params, TOP_K, etc.
│   ├── rag.py                # Retrieval + LLM prompting logic
│   ├── server.py             # FastAPI app (chat API + static UI)
│   ├── search_index/
│   │   ├── embeddings.npy    # (1550, 384) float32 matrix
│   │   └── chunks.json       # Chunk text + metadata
│   └── static/
│       └── index.html        # Chat UI
├── input/
│   └── Digital+Performer+User+Guide.pdf
├── training_data/
│   ├── raw/                  # Per-page JSON extracted from PDF
│   ├── cleaned/              # OCR-normalized text
│   └── chunks/               # JSONL, Parquet, embeddings
├── extract_text.py           # Pipeline step 1
├── clean_text.py             # Pipeline step 2
├── chunk_text.py             # Pipeline step 3
├── enrich_metadata.py        # Pipeline step 4
├── export_parquet.py         # Pipeline step 5
├── generate_embeddings.py    # Pipeline step 6
├── dedup_chunks.py           # Pipeline step 7
├── .env                      # API keys — never commit this
└── rag-explainer.md          # Technical deep-dive into the RAG pipeline
```

---

## Configuration

Edit `chatbot/config.py` to change runtime behaviour:

| Setting | Default | Description |
|---------|---------|-------------|
| `LLM_MODEL` | `gpt-4o-mini` | OpenAI model for answer generation |
| `TOP_K` | `8` | Number of chunks retrieved per query |
| `MAX_CONTEXT_TOKENS` | `6000` | Max tokens sent as context to the LLM |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model for query embedding |

To use the full model, set `LLM_MODEL = "gpt-4o"` or override via environment:

```bash
LLM_MODEL=gpt-4o uvicorn chatbot.server:app --port 8000
```

### Free local alternative (no API key needed)

Install [Ollama](https://ollama.com/) and run a local model:

```bash
brew install ollama
ollama pull llama3
```

Add to `.env`:

```
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama
LLM_MODEL=llama3
```

---

## API

The server exposes three endpoints:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Chat UI |
| `GET` | `/api/health` | Health check → `{"status": "ok"}` |
| `POST` | `/api/chat` | Single response → `{"answer": "...", "sources": [...]}` |
| `POST` | `/api/chat/stream` | SSE streaming → tokens + sources on completion |

**Chat request body:**

```json
{ "message": "How do I set the tempo?" }
```

**Non-streaming response:**

```json
{
  "answer": "To set the tempo, go to Project > Conductor Track... [page 208]",
  "sources": [
    { "chunk_id": "...", "chunk_index": 279, "score": 0.703 }
  ]
}
```

---

## Cost estimate

| Component | Cost |
|-----------|------|
| Query embedding | $0 — runs locally |
| GPT-4o-mini answer generation | ~$0.001–0.003 per query |
| ~100 queries/month | ~$0.10–0.30/mo |

Set a spending limit at [platform.openai.com/settings/organization/billing](https://platform.openai.com/settings/organization/billing).
