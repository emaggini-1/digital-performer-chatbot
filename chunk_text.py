#!/usr/bin/env python3
"""chunk_text.py — Split cleaned text into token-sized chunks with metadata."""

import json
import os
import tiktoken

CLEAN_DIR = "training_data/cleaned"
CHUNK_DIR = "training_data/chunks"

os.makedirs(CHUNK_DIR, exist_ok=True)

ENCODING = tiktoken.encoding_for_model("gpt-4")
CHUNK_SIZE = 512
CHUNK_OVERLAP = 100

def chunk_text(text: str, chunk_size: int, overlap: int):
    tokens = ENCODING.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_str = ENCODING.decode(chunk_tokens)
        chunks.append({
            "text": chunk_str,
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

jsonl_path = os.path.join(CHUNK_DIR, "all_chunks.jsonl")
with open(jsonl_path, "w") as f:
    for chunk in all_chunks:
        f.write(json.dumps(chunk) + "\n")

print(f"Created {len(all_chunks)} chunks across all documents → {jsonl_path}")
