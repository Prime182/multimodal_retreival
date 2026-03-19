
import pymupdf4llm
from pathlib import Path
import json

pdf_path = "pdfs/AO_5c05577.pdf"
if not Path(pdf_path).exists():
    print(f"PDF {pdf_path} not found")
else:
    print(f"Analyzing {pdf_path}...")
    chunks = pymupdf4llm.to_markdown(
        pdf_path,
        page_chunks=True,
        show_progress=False,
    )
    print(f"Got {len(chunks)} chunks")
    for i, chunk in enumerate(chunks[:2]):
        print(f"\n--- Chunk {i} ---")
        print(f"Metadata keys: {list(chunk.get('metadata', {}).keys())}")
        print(f"Top-level keys: {list(chunk.keys())}")
        if "page_boxes" in chunk:
            print(f"Number of page_boxes: {len(chunk['page_boxes'])}")
            if chunk["page_boxes"]:
                print(f"First box sample: {chunk['page_boxes'][0]}")
        else:
            print("page_boxes NOT FOUND in chunk")
