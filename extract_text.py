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

for doc_rec in documents:
    out_path = os.path.join(OUTPUT_DIR, doc_rec["source_file"].replace(".pdf", ".json"))
    with open(out_path, "w") as f:
        json.dump(doc_rec, f, indent=2)

print(f"Extracted {len(documents)} documents to {OUTPUT_DIR}/")
