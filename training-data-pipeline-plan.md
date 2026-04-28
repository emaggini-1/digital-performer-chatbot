# Plan: Training Data Pipeline 
---

## Overview

This plan extracts, cleans, structures, and exports the OCR'd text from the 34 PDFs into formats suitable for LLM fine-tuning, RAG retrieval, embedding pipelines, and general ML training data workflows.

```
output/*.pdf (searchable PDFs)
    │
    ▼
┌─────────────────────────────┐
│  1. Extract raw text        │
│  2. Clean & normalize       │
│  3. Chunk into segments     │
│  4. Attach metadata         │
│  5. Export to pipeline fmt  │
│  6. Validate & QA           │
└─────────────────────────────┘
    │
    ▼
training_data/  (JSONL, Parquet, or vector-ready chunks)
```

---

## Step 1 — Install Additional Dependencies

```bash
# Python text extraction & processing
pip3 install pdfplumber pymupdf tiktoken

# Structured output formats
pip3 install pandas pyarrow

# Optional: for embedding generation
pip3 install sentence-transformers

# Optional: for deduplication
pip3 install datasketch
```

| Tool | Role |
|------|------|
| `pdfplumber` | High-fidelity text extraction preserving layout/tables |
| `pymupdf` (fitz) | Fast text extraction with page-level metadata |
| `tiktoken` | Token counting (for chunk size targeting) |
| `pandas` + `pyarrow` | DataFrame manipulation and Parquet export |
| `sentence-transformers` | Generate embeddings for vector search / RAG |
| `datasketch` | MinHash-based near-duplicate detection |

---

## Step 2 — Extract Raw Text from OCR'd PDFs

Extract text on a per-page basis to preserve document structure.

```python
#!/usr/bin/env python3
"""extract_text.py — Extract text from OCR'd PDFs into structured JSON."""

import json
import os
import fitz  # pymupdf

INPUT_DIR = "input"
OUTPUT_DIR = "training_data/raw"

os.makedirs(OUTPUT_DIR, exist_ok=True)

documents = []

for filename in sorted(os.listdir(INPUT_DIR)):
    if not filename.endswith(".pdf"):
        continue
    filepath = os.path.join(INPUT_DIR, filename)
    doc = fitz.open(filepath)

    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        pages.append({
            "page": page_num,
            "text": text,
            "char_count": len(text),
            "word_count": len(text.split()),
        })

    record = {
        "source_file": filename,
        "total_pages": len(pages),
        "pages": pages,
        "full_text": "\n\n".join(p["text"] for p in pages),
    }
    documents.append(record)
    doc.close()

# Write per-document JSON files
for doc_rec in documents:
    out_path = os.path.join(OUTPUT_DIR, doc_rec["source_file"].replace(".pdf", ".json"))
    with open(out_path, "w") as f:
        json.dump(doc_rec, f, indent=2)

print(f"Extracted {len(documents)} documents to {OUTPUT_DIR}/")
```

```bash
mkdir -p training_data/raw
python3 extract_text.py
```

**Verify:** Spot-check a few `.json` files to confirm text is present and readable.

---

## Step 3 — Clean and Normalize Text

OCR output commonly has artifacts. Clean before chunking.

```python
#!/usr/bin/env python3
"""clean_text.py — Normalize OCR text artifacts."""

import json
import os
import re

RAW_DIR = "training_data/raw"
CLEAN_DIR = "training_data/cleaned"

os.makedirs(CLEAN_DIR, exist_ok=True)

def clean(text: str) -> str:
    # Fix common OCR artifacts
    text = text.replace("\u00ad", "-")       # soft hyphens
    text = text.replace("\ufb01", "fi")      # ligature fi
    text = text.replace("\ufb02", "fl")      # ligature fl
    text = re.sub(r"[^\S\n]+", " ", text)    # collapse whitespace (preserve newlines)
    text = re.sub(r"\n{3,}", "\n\n", text)   # collapse excessive blank lines
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # rejoin hyphenated line breaks
    text = text.strip()
    return text

for filename in sorted(os.listdir(RAW_DIR)):
    if not filename.endswith(".json"):
        continue
    with open(os.path.join(RAW_DIR, filename)) as f:
        doc = json.load(f)

    for page in doc["pages"]:
        page["text"] = clean(page["text"])
    doc["full_text"] = clean(doc["full_text"])

    with open(os.path.join(CLEAN_DIR, filename), "w") as f:
        json.dump(doc, f, indent=2)

print(f"Cleaned {len(os.listdir(CLEAN_DIR))} documents to {CLEAN_DIR}/")
```

```bash
python3 clean_text.py
```

---

## Step 4 — Chunk Text for Training / Retrieval

Different pipelines need different chunk strategies:

| Use Case | Recommended Chunk Size | Overlap |
|----------|----------------------|---------|
| RAG / vector search | 256–512 tokens | 50–100 tokens |
| LLM fine-tuning | 1024–2048 tokens | 0 |
| Classification | Full document | N/A |

```python
#!/usr/bin/env python3
"""chunk_text.py — Split cleaned text into token-sized chunks with metadata."""

import json
import os
import tiktoken

CLEAN_DIR = "training_data/cleaned"
CHUNK_DIR = "training_data/chunks"

os.makedirs(CHUNK_DIR, exist_ok=True)

# Configure for your target model
ENCODING = tiktoken.encoding_for_model("gpt-4")
CHUNK_SIZE = 512      # tokens
CHUNK_OVERLAP = 100   # tokens

def chunk_text(text: str, chunk_size: int, overlap: int):
    tokens = ENCODING.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = ENCODING.decode(chunk_tokens)
        chunks.append({
            "text": chunk_text,
            "token_count": len(chunk_tokens),
            "start_token": start,
            "end_token": min(end, len(tokens)),
        })
        start += chunk_size - overlap
    return chunks

all_chunks = []

for filename in sorted(os.listdir(CLEAN_DIR)):
    if not filename.endswith(".json"):
        continue
    with open(os.path.join(CLEAN_DIR, filename)) as f:
        doc = json.load(f)

    source = doc["source_file"]
    chunks = chunk_text(doc["full_text"], CHUNK_SIZE, CHUNK_OVERLAP)

    for i, chunk in enumerate(chunks):
        chunk["chunk_id"] = f"{source}::chunk_{i:04d}"
        chunk["source_file"] = source
        chunk["chunk_index"] = i
        chunk["total_chunks"] = len(chunks)
        all_chunks.append(chunk)

# Write JSONL (one chunk per line — standard for ML pipelines)
jsonl_path = os.path.join(CHUNK_DIR, "all_chunks.jsonl")
with open(jsonl_path, "w") as f:
    for chunk in all_chunks:
        f.write(json.dumps(chunk) + "\n")

print(f"Created {len(all_chunks)} chunks across all documents → {jsonl_path}")
```

```bash
python3 chunk_text.py
```

---

## Step 5 — Add Metadata and Enrich

Attach provenance metadata to each chunk for filtering and traceability.

```python
#!/usr/bin/env python3
"""enrich_metadata.py — Add document-level metadata to chunks."""

import json
import os
from datetime import datetime

CHUNK_FILE = "training_data/chunks/all_chunks.jsonl"
ENRICHED_FILE = "training_data/chunks/enriched_chunks.jsonl"

# Define any known metadata per source file (manually or from a manifest)
# Extend this as needed
METADATA_MAP = {
    # "1.pdf": {"doc_type": "legal_filing", "date": "2025-01-15", "parties": ["..."]},
}

enriched = []
with open(CHUNK_FILE) as f:
    for line in f:
        chunk = json.loads(line)
        source = chunk["source_file"]

        # Attach metadata
        chunk["corpus"] = "munroe_documents"
        chunk["extraction_date"] = datetime.now().isoformat()
        chunk["pipeline_version"] = "1.0"

        if source in METADATA_MAP:
            chunk["metadata"] = METADATA_MAP[source]

        enriched.append(chunk)

with open(ENRICHED_FILE, "w") as f:
    for chunk in enriched:
        f.write(json.dumps(chunk) + "\n")

print(f"Enriched {len(enriched)} chunks → {ENRICHED_FILE}")
```

```bash
python3 enrich_metadata.py
```

---

## Step 6 — Export to Target Formats

### 6a. JSONL (already done — universal format)

The `enriched_chunks.jsonl` file is ready for most pipelines (OpenAI fine-tuning, LangChain, LlamaIndex, etc.).

### 6b. Parquet (columnar — ideal for large-scale processing)

```python
#!/usr/bin/env python3
"""export_parquet.py — Convert JSONL chunks to Parquet."""

import json
import pandas as pd

ENRICHED_FILE = "training_data/chunks/enriched_chunks.jsonl"
PARQUET_FILE = "training_data/chunks/enriched_chunks.parquet"

records = []
with open(ENRICHED_FILE) as f:
    for line in f:
        records.append(json.loads(line))

df = pd.DataFrame(records)
df.to_parquet(PARQUET_FILE, index=False)
print(f"Exported {len(df)} rows → {PARQUET_FILE}")
print(f"Columns: {list(df.columns)}")
```

### 6c. Embeddings (for vector databases / RAG)

```python
#!/usr/bin/env python3
"""generate_embeddings.py — Create vector embeddings for each chunk."""

import json
import numpy as np
from sentence_transformers import SentenceTransformer

ENRICHED_FILE = "training_data/chunks/enriched_chunks.jsonl"
EMBEDDINGS_FILE = "training_data/chunks/embeddings.npz"
IDS_FILE = "training_data/chunks/embedding_ids.json"

model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim, fast

chunks = []
with open(ENRICHED_FILE) as f:
    for line in f:
        chunks.append(json.loads(line))

texts = [c["text"] for c in chunks]
ids = [c["chunk_id"] for c in chunks]

print(f"Generating embeddings for {len(texts)} chunks...")
embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

np.savez_compressed(EMBEDDINGS_FILE, embeddings=embeddings)
with open(IDS_FILE, "w") as f:
    json.dump(ids, f)

print(f"Saved embeddings ({embeddings.shape}) → {EMBEDDINGS_FILE}")
```

---

## Step 7 — Quality Assurance & Validation

```bash
# 1. Count chunks and verify none are empty
python3 -c "
import json
empty = 0
total = 0
with open('training_data/chunks/enriched_chunks.jsonl') as f:
    for line in f:
        total += 1
        chunk = json.loads(line)
        if len(chunk['text'].strip()) < 10:
            empty += 1
            print(f'  WARNING: near-empty chunk: {chunk[\"chunk_id\"]}')
print(f'Total: {total} chunks, {empty} near-empty')
"

# 2. Spot-check random chunks
python3 -c "
import json, random
chunks = [json.loads(l) for l in open('training_data/chunks/enriched_chunks.jsonl')]
for c in random.sample(chunks, min(3, len(chunks))):
    print(f'--- {c[\"chunk_id\"]} ({c[\"token_count\"]} tokens) ---')
    print(c['text'][:300])
    print()
"
```

**Red flags to watch for:**
- Chunks with very few characters → may indicate failed OCR on that page
- Garbled/nonsensical text → re-OCR with different DPI or preprocessing flags
- Duplicate chunks across files → run dedup (see Step 8)

---

## Step 8 — Deduplication (Optional)

Near-duplicate detection across chunks using MinHash:

```python
#!/usr/bin/env python3
"""dedup_chunks.py — Remove near-duplicate chunks via MinHash LSH."""

import json
from datasketch import MinHash, MinHashLSH

ENRICHED_FILE = "training_data/chunks/enriched_chunks.jsonl"
DEDUPED_FILE = "training_data/chunks/deduped_chunks.jsonl"

THRESHOLD = 0.8  # Jaccard similarity threshold

chunks = []
with open(ENRICHED_FILE) as f:
    for line in f:
        chunks.append(json.loads(line))

# Build MinHash for each chunk
lsh = MinHashLSH(threshold=THRESHOLD, num_perm=128)
minhashes = {}

for chunk in chunks:
    mh = MinHash(num_perm=128)
    for word in chunk["text"].lower().split():
        mh.update(word.encode("utf-8"))
    minhashes[chunk["chunk_id"]] = mh

# Insert and detect duplicates
seen = set()
deduped = []

for chunk in chunks:
    cid = chunk["chunk_id"]
    mh = minhashes[cid]
    result = lsh.query(mh)
    if any(r in seen for r in result if r != cid):
        continue  # skip near-duplicate
    lsh.insert(cid, mh)
    seen.add(cid)
    deduped.append(chunk)

with open(DEDUPED_FILE, "w") as f:
    for chunk in deduped:
        f.write(json.dumps(chunk) + "\n")

removed = len(chunks) - len(deduped)
print(f"Deduplication: {len(chunks)} → {len(deduped)} chunks ({removed} removed)")
```

---

## Final Directory Structure

```
munroe/
├── images/              # Original flat-image PDFs (34 files)
├── output/              # OCR'd searchable PDFs (from ocr-conversion-plan.md)
├── training_data/
│   ├── raw/             # Per-document JSON with page-level text
│   ├── cleaned/         # Normalized text (OCR artifact removal)
│   └── chunks/
│       ├── all_chunks.jsonl          # Raw chunks
│       ├── enriched_chunks.jsonl     # Chunks with metadata
│       ├── deduped_chunks.jsonl      # After deduplication
│       ├── enriched_chunks.parquet   # Columnar format
│       ├── embeddings.npz            # Vector embeddings
│       └── embedding_ids.json        # Chunk ID ↔ embedding index map
├── ocr-conversion-plan.md
└── training-data-pipeline-plan.md
```

---

## Quick Reference: Full Pipeline Commands

```bash
# 0. OCR the PDFs first (see ocr-conversion-plan.md)
mkdir -p output
for pdf in images/*.pdf; do
  ocrmypdf --output-type pdf --image-dpi 200 --optimize 1 \
    --skip-text --clean --deskew "$pdf" "output/$(basename $pdf)"
done

# 1. Extract text
python3 extract_text.py

# 2. Clean text
python3 clean_text.py

# 3. Chunk text
python3 chunk_text.py

# 4. Enrich metadata
python3 enrich_metadata.py

# 5. Export to Parquet
python3 export_parquet.py

# 6. Generate embeddings (optional)
python3 generate_embeddings.py

# 7. Deduplicate (optional)
python3 dedup_chunks.py
```
