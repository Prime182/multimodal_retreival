# Multimodal PDF Retrieval — Upgrade Implementation Plan
## For Codex Sub-Agent Execution

---

## Overview

This plan upgrades the multimodal PDF retrieval system from a flat text+image pipeline to a
structured, section-aware pipeline that handles four content types: **text**, **equations**,
**tables**, and **images**. IDs are now derived from the PDF filename (e.g. `BJ_100833.pdf` →
`journal_id=BJ`, `article_id=100833`) rather than auto-generated.

Each task below is designed to be handed to a single Codex sub-agent as an isolated, testable
unit of work. Run them in the order listed — later tasks depend on earlier ones.

---

## Prerequisites (do once before running any agent)

```bash
# 1. Install updated Python dependencies
pip install -r requirements.txt

# 2. Confirm Poppler is installed (provides pdftotext + pdfimages)
pdftotext -v
pdfimages -v

# 3. Copy the output files from the previous session into the repo
cp outputs/backend/src/multimodal/types.py      backend/src/multimodal/types.py
cp outputs/backend/src/multimodal/ingestion.py  backend/src/multimodal/ingestion.py
cp outputs/backend/src/multimodal/storage.py    backend/src/multimodal/storage.py
cp outputs/backend/src/multimodal/service.py    backend/src/multimodal/service.py
cp outputs/backend/src/multimodal/api.py        backend/src/multimodal/api.py
cp outputs/frontend/src/types.ts                frontend/src/types.ts
cp outputs/frontend/src/App.tsx                 frontend/src/App.tsx
cp outputs/frontend/src/components/SearchResults.tsx       frontend/src/components/SearchResults.tsx
cp outputs/frontend/src/components/search-results.css      frontend/src/components/search-results.css
```

---

## Task Map

```
Task 1 ──► types.py          (data contracts)
Task 2 ──► ingestion.py      (depends on Task 1)
Task 3 ──► storage.py        (depends on Task 1)
Task 4 ──► service.py        (depends on Task 2 + 3)
Task 5 ──► api.py            (depends on Task 4)
Task 6 ──► frontend types    (independent)
Task 7 ──► frontend UI       (depends on Task 6)
Task 8 ──► integration test  (depends on all)
```

---

## Task 1 — Update `types.py`

### What changes
- Add `parse_pdf_filename(path)` helper that splits `BJ_100833.pdf` into `("BJ", "100833")`.
- Add `journal_id` and `article_id` fields to every dataclass.
- Add `EquationChunk` dataclass with `latex`, `context`, `embed_text` property.
- Add `TableChunk` dataclass with `csv_data`, `header`, `row_index`, `embed_text` property.
- Add `section` field to `ExtractedImage` and `TextChunk`.
- Add `content_type` field (literal `"text" | "equation" | "table" | "image"`) to all chunks.
- Update `IngestedDocument` to hold `equation_chunks` and `table_chunks` lists.

### Codex prompt
```
You are working in the file backend/src/multimodal/types.py.

Replace the entire file with the version provided in outputs/backend/src/multimodal/types.py.

After replacing, verify:
1. `from backend.src.multimodal.types import parse_pdf_filename` works in a Python REPL.
2. `parse_pdf_filename("BJ_100833.pdf")` returns `("BJ", "100833")`.
3. `EquationChunk`, `TableChunk`, `ExtractedImage`, `TextChunk`, `IngestedDocument`,
   `SearchResult` are all importable.
4. `EquationChunk(chunk_id="x", journal_id="BJ", article_id="100833",
   source_path="/tmp/a.pdf", latex="E=mc^2", context="Energy equation").embed_text`
   returns a non-empty string containing the latex.

Fix any import errors before finishing.
```

---

## Task 2 — Rewrite `ingestion.py`

### What changes
- Replace flat `pdftotext` → text chunks pipeline with a 6-step pipeline:
  1. `_extract_page_text` — unchanged pdftotext call, returns `list[_PageText]`
  2. `_build_section_map` — heuristic detection of numbered headings, known section names,
     ALL-CAPS lines; returns `dict[page_number, section_name]`
  3. `_build_text_chunks` — flushes buffer at section boundaries in addition to token limits;
     attaches `section` and `journal_id`/`article_id` to each chunk
  4. `_extract_equations` — scans lines for math symbols (>15% density), LaTeX markers,
     dense operators; groups consecutive equation lines into atomic `EquationChunk`s;
     captures one sentence of surrounding context; NEVER splits an equation block
  5. `_extract_tables` via `pdfplumber`:
     - ≤ 20 rows → single `TableChunk` with full CSV
     - > 20 rows → one `TableChunk` per data row, header prepended to every row
  6. `_extract_images` — same `pdfimages` call, now correlates each image to the
     section_map using inferred page number; sets `section` on each `ExtractedImage`
- `PDFIngestionAgent.process_pdf` returns `IngestedDocument` with all four lists populated.
- All IDs constructed as `{jid}_{aid}_{type}_{counter:05d}` — no UUID-based chunk IDs.

### Codex prompt
```
You are working in the file backend/src/multimodal/ingestion.py.

Replace the entire file with the version provided in outputs/backend/src/multimodal/ingestion.py.

After replacing, run the following smoke test (replace the path with any small PDF you have):

    from pathlib import Path
    from backend.src.multimodal.ingestion import PDFIngestionAgent

    agent = PDFIngestionAgent()
    doc = agent.process_pdf("pdfs/BJ_100833.pdf", "/tmp/ingest_test/")

    print("document_id:", doc.document_id)
    print("journal_id :", doc.journal_id)
    print("article_id :", doc.article_id)
    print("text chunks:", len(doc.text_chunks))
    print("equations  :", len(doc.equation_chunks))
    print("tables     :", len(doc.table_chunks))
    print("images     :", len(doc.images))

    # Verify IDs follow the convention
    if doc.text_chunks:
        assert doc.text_chunks[0].chunk_id.startswith("BJ_100833_text_")
    if doc.equation_chunks:
        assert doc.equation_chunks[0].chunk_id.startswith("BJ_100833_eq_")
    if doc.table_chunks:
        assert doc.table_chunks[0].chunk_id.startswith("BJ_100833_tbl_")
    if doc.images:
        assert doc.images[0].image_id.startswith("BJ_100833_img_")

    print("All assertions passed.")

Fix any failures before finishing. If pdfplumber is not installed, run `pip install pdfplumber`.
```

---

## Task 3 — Update `storage.py`

### What changes
- Add `add_equations(equations, embeddings)` method.
- Add `add_tables(tables, embeddings)` method.
- New `search(query_embedding, limit, content_types)` method — queries **all** content types
  in the single collection; optional `content_types` filter uses Chroma `$in` operator.
- Old `search_images()` kept as backwards-compatible alias calling `search(..., content_types=["image"])`.
- `_clean_metadata` unchanged — still strips None values and converts Path objects.

### Codex prompt
```
You are working in the file backend/src/multimodal/storage.py.

Replace the entire file with the version provided in outputs/backend/src/multimodal/storage.py.

After replacing, run this test using an in-memory chromadb client:

    import chromadb
    from backend.src.multimodal.storage import PublisherArticleStore
    from backend.src.multimodal.types import TextChunk, EquationChunk, TableChunk, ExtractedImage

    client = chromadb.EphemeralClient()
    store = PublisherArticleStore(persist_directory="/tmp/test_chroma", client=client)

    # Add one of each type with a dummy 4-dim embedding
    store.add_text_chunks(
        [TextChunk("BJ_1_text_00001","BJ","1","/tmp/a.pdf","hello world",2,1,1,"Intro",None)],
        [[0.1, 0.2, 0.3, 0.4]],
    )
    store.add_equations(
        [EquationChunk("BJ_1_eq_00001","BJ","1","/tmp/a.pdf","E=mc^2","Energy is",1,"Results")],
        [[0.5, 0.6, 0.7, 0.8]],
    )
    store.add_tables(
        [TableChunk("BJ_1_tbl_00001","BJ","1","/tmp/a.pdf","col1,val\na,1","col1,val",1,None,"Methods")],
        [[0.2, 0.3, 0.4, 0.5]],
    )

    assert store.collection_count() == 3

    # Search all types
    results = store.search([0.1, 0.2, 0.3, 0.4], limit=3)
    assert len(results) == 3

    # Filter to equation only
    eq_results = store.search([0.5, 0.6, 0.7, 0.8], limit=3, content_types=["equation"])
    assert len(eq_results) == 1
    assert eq_results[0].metadata["kind"] == "equation"

    print("All storage tests passed.")

Fix any failures before finishing.
```

---

## Task 4 — Update `service.py`

### What changes
- `index_pdf` now calls `add_equations`, `add_tables`, `add_images`, and `add_text_chunks`
  for whichever lists are non-empty.
- Equations and tables are embedded as text via `embed_texts([chunk.embed_text for chunk in ...])`.
- Images still embedded as binary via `embed_file(img.file_path)`.
- New `search(query, limit, content_types)` method — embeds query text then calls
  `store.search(...)` with optional type filter.
- `search_images` kept as alias.

### Codex prompt
```
You are working in the file backend/src/multimodal/service.py.

Replace the entire file with the version provided in outputs/backend/src/multimodal/service.py.

Verify the imports resolve cleanly:

    python -c "from backend.src.multimodal.service import MultimodalRetrievalService; print('OK')"

No runtime test needed here — the integration test in Task 8 covers the full flow.
Fix any import errors before finishing.
```

---

## Task 5 — Update `api.py`

### What changes
- `SearchRequest` model gains an optional `content_types: list[str] | None` field.
- `POST /search` passes `content_types` to `service.search()`.
- `POST /index` response now includes `equation_chunks` and `table_chunks` counts.
- CORS middleware already present — no change needed.

### Codex prompt
```
You are working in the file backend/src/multimodal/api.py.

Replace the entire file with the version provided in outputs/backend/src/multimodal/api.py.

Start the server in dry-run mode and confirm the OpenAPI schema is correct:

    PYTHONPATH=backend/src python -c "
    from multimodal.api import app
    import json
    schema = app.openapi()
    search_body = schema['components']['schemas']['SearchRequest']['properties']
    assert 'content_types' in search_body, 'content_types missing from SearchRequest'
    index_resp = schema['paths']['/index']['post']
    print('SearchRequest fields:', list(search_body.keys()))
    print('Schema OK')
    "

Fix any failures before finishing.
```

---

## Task 6 — Update frontend `types.ts`

### What changes
- Add `ContentType = "text" | "equation" | "table" | "image"` union type.
- Add `latex`, `csv_data`, `header`, `row_index`, `context` fields to `SearchResultMetadata`.
- Add `journal_id`, `article_id`, `section` fields.
- Keep all existing fields to avoid breaking existing usage.

### Codex prompt
```
You are working in the file frontend/src/types.ts.

Replace the entire file with the version provided in outputs/frontend/src/types.ts.

Run the TypeScript compiler to verify no errors:

    cd frontend && npx tsc --noEmit

Fix any type errors before finishing.
```

---

## Task 7 — Update frontend UI (`App.tsx`, `SearchResults.tsx`, `search-results.css`, `styles.css`)

### What changes

**`App.tsx`**
- Add four content-type toggle chips (Text / Equation / Table / Image), all active by default.
- At least one chip must remain active at all times.
- Pass `content_types` array (or `null` for all) in the `/search` request body.

**`SearchResults.tsx`**
- Cards render differently per content type:
  - `image` → photo with `<img>` tag
  - `text` → 320-char text preview with blue background tint
  - `equation` → monospace `<pre>` block with context sentence, purple tint
  - `table` → real HTML `<table>` parsed from `csv_data`, cyan tint
- Header shows per-type result counts as coloured badges.
- Kind badge on each card footer.

**`search-results.css`**
- Four CSS custom property sets for accent colours per kind.
- `.sr-table` styling for the HTML table inside table cards.
- `.sr-eq-block` monospace pre styling.

**`styles.css`**
- `.type-chip` and `.type-chip--{type}` classes for the filter chips.

### Codex prompt
```
You are working in the frontend directory.

Replace the following files with the versions provided in outputs/:
  - frontend/src/App.tsx
  - frontend/src/components/SearchResults.tsx
  - frontend/src/components/search-results.css
  - frontend/src/styles.css

Then run:

    cd frontend && npm run build

Fix any TypeScript or CSS errors. Common issues to watch for:
- `import.meta.env` type — cast as `(import.meta as any).env` if TS complains.
- Ensure `ContentType` is imported from `../types` in SearchResults.tsx.
- The `resolveMode` function must handle the case where neither distance nor score exists
  (return "distance" as default).

After a clean build, also run:

    cd frontend && npm run dev

and confirm the dev server starts without errors on http://localhost:5173.
```

---

## Task 8 — End-to-End Integration Test

### What this verifies
1. Backend indexes a real PDF and returns counts for all four chunk types.
2. Backend search returns results with `kind` metadata.
3. Frontend build is clean.
4. Frontend correctly constructs the `/search` request with `content_types`.

### Codex prompt
```
Run the full integration test for the multimodal PDF retrieval upgrade.

Step 1 — Start the backend:
    PYTHONPATH=backend/src GEMINI_API_KEY=<your_key> \
      uvicorn main:app --app-dir backend/src --port 8000 &

Step 2 — Health check:
    curl http://localhost:8000/health
    # Expected: {"status":"ok"}

Step 3 — Index a PDF (use BJ_100833.pdf if available, else any PDF):
    curl -X POST http://localhost:8000/index \
      -H "Content-Type: application/json" \
      -d '{"pdf_path":"pdfs/BJ_100833.pdf"}'
    # Expected response contains: document_id, journal_id, article_id,
    #   text_chunks, equation_chunks, table_chunks, images  (all integer counts)

Step 4 — Search all types:
    curl -X POST http://localhost:8000/search \
      -H "Content-Type: application/json" \
      -d '{"query":"energy balance","limit":6}'
    # Expected: results array; each result has metadata.kind in
    #   ["text","equation","table","image"]

Step 5 — Search equations only:
    curl -X POST http://localhost:8000/search \
      -H "Content-Type: application/json" \
      -d '{"query":"differential equation","limit":4,"content_types":["equation"]}'
    # Expected: all results have metadata.kind == "equation"

Step 6 — Frontend build:
    cd frontend && npm run build
    # Expected: zero errors, dist/ directory created

Step 7 — Verify ID convention on indexed document:
    # From the Step 3 response, check that document_id matches the PDF stem
    # e.g. for BJ_100833.pdf → document_id should be "BJ_100833"

Report the output of each step. If any step fails, diagnose and fix the root cause
before reporting. Do not mark the task complete until all six steps pass.
```

---

## Rollback Instructions

If any task fails and needs to be rolled back:

```bash
# Restore a single file from git
git checkout HEAD -- backend/src/multimodal/types.py

# Or restore all backend files at once
git checkout HEAD -- backend/src/multimodal/

# Wipe the Chroma database (required when schema changes between runs)
rm -rf backend/data/chroma/

# Wipe extracted images (if re-indexing with new ID scheme)
rm -rf backend/data/images/
```

> **Important**: Whenever you change the chunk ID format or add new content types,
> delete `backend/data/chroma/` and re-index all PDFs. Chroma does not migrate
> existing records — stale records from the old ID scheme will cause search results
> to mix old and new formats.

---

## File Change Summary

| File | Change type | Depends on |
|------|-------------|------------|
| `backend/src/multimodal/types.py` | Full rewrite | — |
| `backend/src/multimodal/ingestion.py` | Full rewrite | types.py |
| `backend/src/multimodal/storage.py` | Full rewrite | types.py |
| `backend/src/multimodal/service.py` | Full rewrite | ingestion, storage |
| `backend/src/multimodal/api.py` | Full rewrite | service |
| `requirements.txt` | Add pdfplumber | — |
| `frontend/src/types.ts` | Full rewrite | — |
| `frontend/src/App.tsx` | Full rewrite | types.ts |
| `frontend/src/components/SearchResults.tsx` | Full rewrite | types.ts |
| `frontend/src/components/search-results.css` | Full rewrite | — |
| `frontend/src/styles.css` | Full rewrite | — |

Total files changed: **11**
New dependency: `pdfplumber>=0.11.0`