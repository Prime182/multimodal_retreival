# agents.md
## Multimodal PDF Retrieval System — Agent Guide

This file is the single source of truth for any automated agent (Codex, Claude, CI bot,
or similar) working in this repository. Read it fully before touching any file.

---

## Table of Contents

1. [Repository Map](#1-repository-map)
2. [Environment Setup](#2-environment-setup)
3. [Key Commands](#3-key-commands)
4. [Architecture in Plain Terms](#4-architecture-in-plain-terms)
5. [Data & ID Contracts](#5-data--id-contracts)
6. [Module Responsibilities](#6-module-responsibilities)
7. [Coding Conventions](#7-coding-conventions)
8. [How to Add a New Content Type](#8-how-to-add-a-new-content-type)
9. [Testing Strategy](#9-testing-strategy)
10. [Known Issues & Active Bugs](#10-known-issues--active-bugs)
11. [What Agents Must Never Do](#11-what-agents-must-never-do)
12. [Debugging Playbook](#12-debugging-playbook)
13. [Dependency Notes](#13-dependency-notes)

---

## 1. Repository Map

```
.
├── backend/
│   └── src/
│       ├── main.py                   ← uvicorn entry-point (re-exports app)
│       └── multimodal/
│           ├── __init__.py
│           ├── api.py                ← FastAPI app, routes /index /search /health
│           ├── embeddings.py         ← Gemini embedding client + helpers
│           ├── ingestion.py          ← PDF → TextChunk / EquationChunk / TableChunk / ExtractedImage
│           ├── service.py            ← Orchestration: ingest → embed → store → search
│           ├── storage.py            ← ChromaDB read/write wrapper
│           └── types.py              ← All dataclasses + parse_pdf_filename
├── frontend/
│   ├── src/
│   │   ├── main.tsx                  ← React entry
│   │   ├── App.tsx                   ← Search form + content-type filter chips
│   │   ├── styles.css                ← App-level styles (chips, form, hero)
│   │   ├── types.ts                  ← TypeScript types shared across components
│   │   └── components/
│   │       ├── SearchResults.tsx     ← Result grid, per-kind card rendering
│   │       └── search-results.css   ← Card, table, equation, image styles
│   ├── index.html
│   ├── package.json
│   ├── tsconfig.json
│   └── vite.config.ts
├── backend/data/                     ← GITIGNORED — runtime artefacts
│   ├── chroma/                       ← ChromaDB persistence files
│   └── images/                       ← Extracted PDF images
├── pdfs/                             ← GITIGNORED — source PDFs for indexing
├── requirements.txt
├── .env.example
├── BUGS_AND_SOLUTIONS.md             ← Active bug tracker (read before editing)
└── agents.md                         ← This file
```

---

## 2. Environment Setup

### Python backend

```bash
# 1. Create virtualenv (Python 3.10+ required — strict=True zip is used)
python3.10 -m venv .venv
source .venv/bin/activate

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Verify Poppler is on PATH (required for pdftotext and pdfimages)
pdftotext -v
pdfimages -v
# If missing on Ubuntu/Debian:  sudo apt-get install poppler-utils
# If missing on macOS:          brew install poppler

# 4. Copy and fill in environment variables
cp .env.example .env
# Edit .env and set GEMINI_API_KEY=<your key>
```

### Frontend

```bash
cd frontend
npm install
cp .env.example .env
# VITE_API_BASE_URL defaults to http://localhost:8000 — change if backend runs elsewhere
```

### Required environment variables

| Variable | Where | Required | Description |
|---|---|---|---|
| `GEMINI_API_KEY` | `.env` (backend) | Yes for /index and /search | Google Gemini API key |
| `GEMINI_EMBEDDING_MODEL` | `.env` (backend) | No | Defaults to `gemini-embedding-2-preview` |
| `VITE_API_BASE_URL` | `frontend/.env` | No | Defaults to `http://localhost:8000` |

---

## 3. Key Commands

### Backend

```bash
# Run the development server (hot-reload)
PYTHONPATH=backend/src uvicorn main:app --app-dir backend/src --reload --port 8000

# Run with a real API key inline
PYTHONPATH=backend/src GEMINI_API_KEY=AIza... uvicorn main:app --app-dir backend/src --reload

# Verify imports without starting the server
PYTHONPATH=backend/src python -c "from multimodal.api import app; print('OK')"

# Wipe ChromaDB and start fresh (REQUIRED after any change to chunk ID format or schema)
rm -rf backend/data/chroma/

# Wipe extracted images (required when re-indexing with a new image naming scheme)
rm -rf backend/data/images/

# Index a single PDF
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"pdf_path": "pdfs/BJ_100833.pdf"}'

# Search across all content types
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "energy balance", "limit": 6}'

# Search equations only
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "differential equation", "limit": 4, "content_types": ["equation"]}'

# Health check
curl http://localhost:8000/health
```

### Frontend

```bash
cd frontend

npm run dev      # start Vite dev server on http://localhost:5173
npm run build    # production build — TypeScript compile + Vite bundle
npm run preview  # preview the production build locally
npx tsc --noEmit # type-check only, no output files
```

---

## 4. Architecture in Plain Terms

```
PDF file
  │
  ▼
ingestion.py ─── pdftotext  ──► _PageText list
              ├── pdfplumber ──► TableChunk list  ──┐
              ├── pdfimages  ──► image files         │
              │                                      │
              ▼                                      │
         section_map (page → section name)           │
              │                                      │
              ├──► _build_text_chunks ──► TextChunk list   │
              ├──► _extract_equations ──► EquationChunk list (uses table_exclusion set ◄─┘)
              ├──► _extract_tables ────► TableChunk list
              └──► _extract_images ────► ExtractedImage list
                        │
                        ▼
                  IngestedDocument
                        │
                        ▼
service.py ─── embed_texts (text, equation, table chunks)
            ── embed_file  (images)
                        │
                        ▼
storage.py ─── ChromaDB collection "publisher_articles"
                (all content types in ONE collection, kind field differentiates them)
                        │
                        ▼
api.py ─── POST /index ─► returns counts by type
        ── POST /search ─► returns ranked SearchResult list with metadata
                        │
                        ▼
frontend ─── App.tsx (filter chips) ──► SearchResults.tsx (per-kind card rendering)
```

**The single-collection design is intentional.** All four content types (text, equation,
table, image) live in one ChromaDB collection. The `kind` metadata field is used for
optional filtering via the `$in` operator. This means a single query can retrieve mixed
results ranked by semantic distance across all types.

---

## 5. Data & ID Contracts

### PDF filename convention

PDFs must be named `<JournalID>_<ArticleID>.pdf`, e.g. `BJ_100833.pdf`.

- `JournalID` — 2–6 uppercase letters (e.g. `BJ`, `PLOS`, `NATURE`)
- `ArticleID` — alphanumeric string (e.g. `100833`, `e1005604`)
- Separator — underscore preferred; digits-only article IDs may omit it

Files that do not match fall back to `journal_id = "UNK"`, `article_id = <full stem>`.
They are indexed normally; only journal grouping is lost.

### Chunk ID format

Every stored record uses a deterministic ID constructed from the PDF stem:

| Content type | Pattern | Example |
|---|---|---|
| Text | `{jid}_{aid}_text_{n:05d}` | `BJ_100833_text_00001` |
| Equation | `{jid}_{aid}_eq_{n:05d}` | `BJ_100833_eq_00003` |
| Table | `{jid}_{aid}_tbl_{n:05d}` | `BJ_100833_tbl_00002` |
| Image | `{jid}_{aid}_img_{n:05d}` | `BJ_100833_img_00001` |

`n` is the 1-based sequential counter **within each type** for that document.

**Critical**: If you change the ID format, you MUST wipe `backend/data/chroma/` and
re-index all PDFs. ChromaDB does not migrate existing records.

### `kind` metadata field

Every ChromaDB record has `metadata.kind` set to one of:
`"text"` | `"equation"` | `"table"` | `"image"`

This is the field used by the `$in` filter in `storage.search()`. All dataclass
`metadata()` methods must include `payload["kind"] = self.content_type`.

### `IngestedDocument` structure

```python
@dataclass(slots=True)
class IngestedDocument:
    document_id: str          # = Path(pdf_path).stem  e.g. "BJ_100833"
    journal_id:  str          # e.g. "BJ"
    article_id:  str          # e.g. "100833"
    source_path: str          # absolute path to original PDF
    text_chunks:      list[TextChunk]
    equation_chunks:  list[EquationChunk]
    table_chunks:     list[TableChunk]
    images:           list[ExtractedImage]
```

---

## 6. Module Responsibilities

### `types.py` — Data contracts only

- Defines all dataclasses (`TextChunk`, `EquationChunk`, `TableChunk`, `ExtractedImage`,
  `IngestedDocument`, `SearchResult`).
- Defines `parse_pdf_filename()` and `build_document_id()`.
- **No business logic, no I/O, no imports from other modules in this package.**
- Every dataclass must implement a `metadata() -> dict[str, Any]` method. The dict must
  include a `"kind"` key.
- Every chunk dataclass must expose an `embed_text: str` property used by the embedding
  layer.

### `ingestion.py` — PDF → structured chunks

- Depends on `types.py` only (within this package).
- Uses three system tools: `pdftotext`, `pdfimages`, `pdfplumber`.
- The public interface is `PDFIngestionAgent.process_pdf(pdf_path, assets_dir)` and the
  convenience wrapper `ingest_pdf(...)`.
- Returns a fully populated `IngestedDocument` — no partial state.
- Does NOT embed, does NOT write to storage.

### `embeddings.py` — Gemini API wrapper

- Depends on nothing in this package.
- Exposes `GeminiEmbeddingClient` with methods `embed_text`, `embed_texts`, `embed_file`,
  `embed_files`.
- All embedding calls go through `_with_retry` (add this — see BUGS_AND_SOLUTIONS.md #7).
- `embed_file` must wrap the Part in a `Content` object — see BUGS_AND_SOLUTIONS.md #3.

### `storage.py` — ChromaDB read/write

- Depends on `types.py` only (within this package).
- One collection: `"publisher_articles"`.
- Public write methods: `add_text_chunks`, `add_equations`, `add_tables`, `add_images`.
  All accept `(items, embeddings)` pairs.
- Public read method: `search(query_embedding, *, limit, content_types)`.
- `_clean_metadata` strips `None` values and converts `Path` objects to `str` before
  storing — ChromaDB rejects both.

### `service.py` — Orchestration

- Depends on `ingestion.py`, `embeddings.py`, `storage.py`, `types.py`.
- `index_pdf(pdf_path, image_output_dir)` — runs ingest, assigns `image_url`, embeds
  each content type, stores all chunks, returns the completed `IngestedDocument`.
- `search(query, *, limit, content_types)` — embeds query text, delegates to
  `store.search`.
- Must NOT mutate `IngestedDocument` fields directly — use `dataclasses.replace` for
  `image_url` assignment (see BUGS_AND_SOLUTIONS.md #9).

### `api.py` — HTTP layer

- Depends on `service.py` only (within this package).
- Three routes: `GET /health`, `POST /index`, `POST /search`.
- `ServiceContainer` is a lazy singleton — service is created on first request, not at
  import time, so the app starts without a `GEMINI_API_KEY`.
- All routes must have try/except with structured `HTTPException` responses — see
  BUGS_AND_SOLUTIONS.md #5.

---

## 7. Coding Conventions

### Python

- **Python 3.10+** is required. Use `match`/`case`, `X | Y` union types, and `zip(strict=True)` freely.
- All dataclasses use `@dataclass(slots=True)`. Do not add `frozen=True` unless the class
  has no fields that need post-construction assignment.
- Use `from __future__ import annotations` at the top of every module — this enables
  forward references in type hints without runtime cost.
- Private helpers are prefixed with a single underscore (`_extract_equations`,
  `_build_section_map`, etc.).
- CLI tool calls use `subprocess.run(..., check=False, capture_output=True)` — never
  `shell=True`, never `check=True` (we inspect returncode manually and raise `RuntimeError`
  with structured JSON so the API can return a clean error).
- No `print` statements in library code. Use `import logging; logger = logging.getLogger(__name__)`.
  (Current code uses print in some places — fix as you touch those files.)
- All public functions that accept paths accept `str | Path` and immediately resolve to
  `Path` internally.

### TypeScript / React

- All components are function components with explicit return types.
- `import.meta.env` must be cast as `(import.meta as any).env` — the tsconfig does not
  include Vite's env typings.
- CSS classes follow BEM-style naming: `search-results__card`, `search-results__card--text`.
- Content-type colours are defined as CSS custom properties on `.search-results__card--{kind}`
  (e.g. `--card-accent`, `--card-tint`). Do not hardcode hex values in TSX.
- Never use `localStorage` or `sessionStorage` — this app has no persistent client-side state.

### General

- No auto-generated UUIDs for chunk IDs. All IDs are derived from the PDF stem — this
  makes re-indexing idempotent.
- Never commit `.env`, `backend/data/`, `pdfs/`, or `frontend/dist/` — all are gitignored.
- If you add a new Python dependency, add it to `requirements.txt` with a minimum version
  pin (`>=X.Y.Z`).

---

## 8. How to Add a New Content Type

The pipeline is intentionally layered so that adding a type requires touching exactly five
files. Follow these steps in order.

### Step 1 — `types.py`

1. Add the new literal to `ContentType`: `Literal["text", "equation", "table", "image", "newtype"]`
2. Create a new `@dataclass(slots=True)` with at minimum:
   - `chunk_id: str`
   - `journal_id: str`
   - `article_id: str`
   - `source_path: str`
   - `content_type: ContentType = "newtype"`
   - `embed_text: str` property
   - `metadata() -> dict[str, Any]` method with `payload["kind"] = self.content_type`
3. Add `newtype_chunks: list[NewtypeChunk] = field(default_factory=list)` to `IngestedDocument`.

### Step 2 — `ingestion.py`

1. Write a `_extract_newtypes(*, pages, section_map, journal_id, article_id, source_path) -> list[NewtypeChunk]` function.
2. Call it inside `PDFIngestionAgent.process_pdf` and assign the result to `IngestedDocument.newtype_chunks`.
3. Import the new dataclass from `types.py`.

### Step 3 — `storage.py`

1. Add `add_newtypes(self, items, embeddings) -> None` following the exact same pattern as
   `add_text_chunks`. The `kind` in metadata is the only differentiator.

### Step 4 — `service.py`

1. In `index_pdf`, add a block:
   ```python
   if document.newtype_chunks:
       self.store.add_newtypes(
           document.newtype_chunks,
           self.embedding_client.embed_texts(
               [chunk.embed_text for chunk in document.newtype_chunks]
           ),
       )
   ```

### Step 5 — Frontend

1. Add `"newtype"` to the `ContentType` union in `frontend/src/types.ts`.
2. Add it to `CONTENT_TYPES` arrays in `App.tsx` and `SearchResults.tsx`.
3. Add a `renderNewtypeCard` function in `SearchResults.tsx`.
4. Add `--{newtype}` colour variant CSS rules in `search-results.css` and `styles.css`.

---

## 9. Testing Strategy

There is currently no test suite. When adding tests, follow these conventions.

### Backend — unit tests

Use `pytest`. Place tests in `backend/tests/`.

```
backend/
└── tests/
    ├── conftest.py           ← shared fixtures
    ├── test_types.py         ← parse_pdf_filename, dataclass metadata()
    ├── test_ingestion.py     ← _is_equation_line, _build_section_map, etc.
    ├── test_storage.py       ← add/search with chromadb.EphemeralClient()
    └── test_service.py       ← index_pdf with mocked embedding client
```

**Key fixtures**:

```python
# conftest.py
import pytest
import chromadb
from multimodal.storage import PublisherArticleStore

@pytest.fixture
def ephemeral_store(tmp_path):
    client = chromadb.EphemeralClient()
    return PublisherArticleStore(persist_directory=tmp_path, client=client)
```

**Run backend tests**:

```bash
PYTHONPATH=backend/src pytest backend/tests/ -v
```

### Equation detector — the most critical unit test surface

Every time you modify `_is_equation_line`, run this checklist:

```python
# Should return True
assert _is_equation_line(r"\frac{d}{dt} E = P_{in} - P_{out}")
assert _is_equation_line("E = mc^2")
assert _is_equation_line("y = 3.14 * r^2 + c")
assert _is_equation_line("dV/dt = k_1 * C - k_2 * V")

# Should return False — these are prose / table content
assert not _is_equation_line("The average yield (2019–2023) was 84.3%.")
assert not _is_equation_line("Total = 1,240 participants enrolled.")
assert not _is_equation_line("Group = Treatment (n = 340)")
assert not _is_equation_line("T_max       37.8    °C")   # table row
assert not _is_equation_line("(Smith et al., 2020) > prior estimates")
assert not _is_equation_line("However, the results were consistent with earlier findings.")
```

### Frontend

Use Vitest (matches Vite's build tool) if a test suite is added.

```bash
cd frontend
npx vitest run
```

### Integration test

```bash
# Backend must be running with a real GEMINI_API_KEY
curl -s http://localhost:8000/health | python3 -m json.tool

# Index
curl -s -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"pdf_path":"pdfs/BJ_100833.pdf"}' | python3 -m json.tool

# Expected shape:
# {
#   "document_id": "BJ_100833",
#   "journal_id": "BJ",
#   "article_id": "100833",
#   "text_chunks": <int>,
#   "equation_chunks": <int>,
#   "table_chunks": <int>,
#   "images": <int>
# }

# Search
curl -s -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"energy balance","limit":5}' | python3 -m json.tool
```

---

## 10. Known Issues & Active Bugs

See `BUGS_AND_SOLUTIONS.md` for detailed root-cause analysis and code-level solutions.

Quick reference:

| # | Symptom | File | Status |
|---|---|---|---|
| 1 | Prose lines classified as equations | `ingestion.py` | Open |
| 2 | Table rows with operators classified as equations | `ingestion.py` | Open |
| 3 | Image embeddings silently fail (wrong Content wrapping) | `embeddings.py` | Open |
| 4 | PDF filename regex rejects 3+ letter journal codes | `types.py` | Open |
| 5 | No error handling in /index and /search endpoints | `api.py` | Open |
| 6 | `zip(strict=True)` crashes on Python < 3.10 | `storage.py` | Open |
| 7 | No retry on Gemini rate-limit; large docs fail mid-way | `embeddings.py` | Open |
| 8 | Chroma `$in` filter crashes on empty collection | `storage.py` | Open |
| 9 | `image_url` mutated post-construction on slots dataclass | `service.py` | Open |
| 10 | `resolveMode` can return `undefined` in TypeScript | `SearchResults.tsx` | Open |
| 11 | `moduleResolution: "Node"` outdated for Vite/ESNext | `tsconfig.json` | Open |

**Before starting any task**, check whether the files you need to touch are affected by an
open bug. Fix the bug in the same PR as your feature — do not leave a bug in code you
already have open.

---

## 11. What Agents Must Never Do

These actions will corrupt the repository or break other agents' work.

### Data

- **Never commit `backend/data/`** — it contains runtime ChromaDB files and extracted
  images. These are large binary files that must not be version-controlled.
- **Never commit `pdfs/`** — source PDFs may contain proprietary publisher content.
- **Never commit `.env`** — it contains your API key.

### ChromaDB

- **Never wipe `backend/data/chroma/` silently.** If your change requires a schema
  migration (new chunk ID format, new metadata fields, new content type), document the
  migration in your PR description and note that users must run:
  ```bash
  rm -rf backend/data/chroma/
  # then re-index all PDFs
  ```
- **Never add a second ChromaDB collection** without updating `storage.py`'s `search`
  method to fan out across collections and merge results. The single-collection design
  is what makes mixed-type semantic search possible in one query.

### Embeddings

- **Never change the embedding model** (`GEMINI_EMBEDDING_MODEL`) while any indexed data
  exists in ChromaDB. Vectors from different models are not comparable. A model change
  requires wiping ChromaDB and re-indexing everything.
- **Never batch multiple content types into a single `embed_content` call** — the
  current API contract is one item per call, which makes per-item retry straightforward.

### IDs

- **Never use UUIDs or random IDs for chunk IDs.** All IDs are derived from the PDF
  filename (see §5). This makes re-indexing idempotent — if a PDF is indexed twice, the
  second run overwrites the first because the IDs are identical.
- **Never change the counter variable name** (`len(chunks) + 1`) to something that resets
  per-page. Counters are per-document and must be monotonically increasing across all pages.

### Frontend

- **Never use `localStorage` or `sessionStorage`** — not supported in the Claude.ai
  artifact environment.
- **Never hardcode `http://localhost:8000`** in component code — always read
  `(import.meta as any).env?.VITE_API_BASE_URL`.
- **Never add a CSS file that resets `:root` colours** — `styles.css` owns the global
  palette; component CSS files only add scoped classes.

### Ingestion pipeline

- **Never call `pdfplumber` inside `_extract_equations`** — pdfplumber is already called
  for table extraction and its output is passed in as an exclusion set. Calling it a
  second time inside equation extraction wastes memory and processing time.
- **Never skip the table exclusion set** when modifying `_extract_equations`. The exclusion
  set is what prevents table rows from being mis-classified as equations (Bug #2).

---

## 12. Debugging Playbook

### "No results from /search"

1. Check `GET /health` — is the backend running?
2. Check `/index` was called and returned non-zero counts.
3. Run `store.collection_count()` in a Python REPL — if 0, the collection is empty.
4. Check ChromaDB dimension consistency — if you re-indexed after changing the model,
   old records have a different vector dimension and the query will fail silently.
   → Wipe `backend/data/chroma/` and re-index.

### "All chunks are being classified as equations"

1. Dump the output of `_is_equation_line` on 20 sample lines from a page.
2. Check whether the `_MATH_SYMBOLS` density threshold is triggering — add a temporary
   `print(f"density={density:.2f}, score={score}")` inside `_is_equation_line`.
3. Verify the prose-veto words are present in `_PROSE_VETO_WORDS`.
4. See BUGS_AND_SOLUTIONS.md §1 for the full fix.

### "Images return 500 on /index"

1. Check `pdfimages -v` — is Poppler installed?
2. Check that `backend/data/images/` is writable.
3. Check `embeddings.py` — is `embed_file` wrapping the Part in a `Content` object?
   See BUGS_AND_SOLUTIONS.md §3 for the fix.
4. Check that `GEMINI_API_KEY` is set — image embedding requires a live API call.

### "Frontend shows no results even though /search returns data"

1. Open the browser console — look for CORS errors.
2. Check `VITE_API_BASE_URL` in `frontend/.env` matches the backend port.
3. Check that `result.metadata.kind` exists and is one of the four known values.
   `resolveKind` falls back gracefully but logs nothing — add a console.warn there
   temporarily.

### "TypeScript build fails"

1. Run `npx tsc --noEmit` from `frontend/` and read the full error.
2. The most common cause is a new field added to a backend response that is not reflected
   in `frontend/src/types.ts`.
3. `moduleResolution: "Node"` hides some import errors that only surface at runtime —
   change to `"Bundler"` to surface them at compile time (BUGS_AND_SOLUTIONS.md §11).

---

## 13. Dependency Notes

### Python

| Package | Version | Why pinned |
|---|---|---|
| `chromadb` | `>=1.0.0` | 1.0 changed the client API (`PersistentClient` vs deprecated `Client`) |
| `google-genai` | `>=1.0.0` | 1.0 introduced `genai.Client`; prior versions used a module-level `configure()` |
| `pdfplumber` | `>=0.11.0` | `extract_tables()` return shape changed in 0.11 |
| `Pillow` | `>=11.0.0` | Used only for image dimension metadata; any modern version is fine |
| `fastapi` | `>=0.115.0` | `list[str] | None` annotation in Pydantic models requires this |

### Node / Frontend

| Package | Note |
|---|---|
| `vite` | `^7.x` — requires Node ≥ 20.19 |
| `react` | `^19.x` — uses the new JSX transform, no `import React` needed |
| `typescript` | `^5.8` — required for `"Bundler"` moduleResolution |

### System tools (must be on PATH)

| Tool | Package | Used by |
|---|---|---|
| `pdftotext` | `poppler-utils` | `ingestion._extract_page_text` |
| `pdfimages` | `poppler-utils` | `ingestion._extract_images` |

If either tool is missing, `_require_command` raises a `RuntimeError` with a clear message
before any subprocess is launched.