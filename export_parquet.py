#!/usr/bin/env python3
"""export_parquet.py — Convert JSONL chunks to Parquet."""

import json
import pandas as pd

ENRICHED_FILE = "training_data/chunks/enriched_chunks.jsonl"
PARQUET_FILE = "training_data/chunks/enriched_chunks.parquet"

records = []
with open(ENRICHED_FILE) as f:
    for line in f:
        r = json.loads(line)
        # Flatten nested metadata dict for columnar storage
        meta = r.pop("metadata", {})
        r.update({f"meta_{k}": v for k, v in meta.items()})
        records.append(r)

df = pd.DataFrame(records)
df.to_parquet(PARQUET_FILE, index=False)
print(f"Exported {len(df)} rows → {PARQUET_FILE}")
print(f"Columns: {list(df.columns)}")
