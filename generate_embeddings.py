#!/usr/bin/env python3
"""generate_embeddings.py — Create vector embeddings for each chunk."""

import json
import numpy as np
from sentence_transformers import SentenceTransformer

ENRICHED_FILE = "training_data/chunks/enriched_chunks.jsonl"
EMBEDDINGS_FILE = "training_data/chunks/embeddings.npz"
IDS_FILE = "training_data/chunks/embedding_ids.json"

model = SentenceTransformer("all-MiniLM-L6-v2")

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

print(f"Saved embeddings {embeddings.shape} → {EMBEDDINGS_FILE}")
