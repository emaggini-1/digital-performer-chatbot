#!/usr/bin/env python3
"""build_index.py — Package training data into a portable search index."""

import json
import os
import numpy as np

CHUNKS_FILE = "training_data/chunks/deduped_chunks.jsonl"
EMBEDDINGS_FILE = "training_data/chunks/embeddings.npz"
INDEX_DIR = "chatbot/search_index"

os.makedirs(INDEX_DIR, exist_ok=True)

chunks = []
with open(CHUNKS_FILE) as f:
    for line in f:
        chunks.append(json.loads(line))

data = np.load(EMBEDDINGS_FILE)
embeddings = data["embeddings"]

assert len(chunks) <= embeddings.shape[0], "More chunks than embeddings"
embeddings = embeddings[:len(chunks)]

np.save(os.path.join(INDEX_DIR, "embeddings.npy"), embeddings)
with open(os.path.join(INDEX_DIR, "chunks.json"), "w") as f:
    json.dump(chunks, f)

print(f"Index built: {len(chunks)} chunks, embeddings shape {embeddings.shape}")
