#!/usr/bin/env python3
"""enrich_metadata.py — Add document-level metadata to chunks."""

import json
import os
from datetime import datetime

CHUNK_FILE = "training_data/chunks/all_chunks.jsonl"
ENRICHED_FILE = "training_data/chunks/enriched_chunks.jsonl"

METADATA_MAP = {
    "Digital+Performer+User+Guide.pdf": {
        "doc_type": "user_guide",
        "product": "Digital Performer",
        "publisher": "MOTU",
        "subject": "DAW software documentation",
    },
}

enriched = []
with open(CHUNK_FILE) as f:
    for line in f:
        chunk = json.loads(line)
        source = chunk["source_file"]

        chunk["corpus"] = "digital_performer_docs"
        chunk["extraction_date"] = datetime.now().isoformat()
        chunk["pipeline_version"] = "1.0"

        if source in METADATA_MAP:
            chunk["metadata"] = METADATA_MAP[source]

        enriched.append(chunk)

with open(ENRICHED_FILE, "w") as f:
    for chunk in enriched:
        f.write(json.dumps(chunk) + "\n")

print(f"Enriched {len(enriched)} chunks → {ENRICHED_FILE}")
