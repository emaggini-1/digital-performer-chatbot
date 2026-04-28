#!/usr/bin/env python3
"""dedup_chunks.py — Remove near-duplicate chunks via MinHash LSH."""

import json
from datasketch import MinHash, MinHashLSH

ENRICHED_FILE = "training_data/chunks/enriched_chunks.jsonl"
DEDUPED_FILE = "training_data/chunks/deduped_chunks.jsonl"

THRESHOLD = 0.8

chunks = []
with open(ENRICHED_FILE) as f:
    for line in f:
        chunks.append(json.loads(line))

lsh = MinHashLSH(threshold=THRESHOLD, num_perm=128)
minhashes = {}

for chunk in chunks:
    mh = MinHash(num_perm=128)
    for word in chunk["text"].lower().split():
        mh.update(word.encode("utf-8"))
    minhashes[chunk["chunk_id"]] = mh

seen = set()
deduped = []

for chunk in chunks:
    cid = chunk["chunk_id"]
    mh = minhashes[cid]
    result = lsh.query(mh)
    if any(r in seen for r in result if r != cid):
        continue
    lsh.insert(cid, mh)
    seen.add(cid)
    deduped.append(chunk)

with open(DEDUPED_FILE, "w") as f:
    for chunk in deduped:
        f.write(json.dumps(chunk) + "\n")

removed = len(chunks) - len(deduped)
print(f"Deduplication: {len(chunks)} → {len(deduped)} chunks ({removed} removed)")
