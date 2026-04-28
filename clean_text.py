#!/usr/bin/env python3
"""clean_text.py — Normalize OCR text artifacts."""

import json
import os
import re

RAW_DIR = "training_data/raw"
CLEAN_DIR = "training_data/cleaned"

os.makedirs(CLEAN_DIR, exist_ok=True)

def clean(text: str) -> str:
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
