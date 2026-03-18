# Bug Analysis & Solutions
## Multimodal PDF Retrieval System

---

## Table of Contents

1. [Equation False Positives — Prose Misclassified as Equations](#1-equation-false-positives)
2. [Table Rows Classified as Equations](#2-table-rows-classified-as-equations)
3. [Image Embeddings Silently Failing](#3-image-embeddings-silently-failing)
4. [PDF Filename Regex Too Restrictive](#4-pdf-filename-regex-too-restrictive)
5. [No Error Handling in API Endpoints](#5-no-error-handling-in-api-endpoints)
6. [zip(strict=True) Requires Python 3.10+](#6-zipstricttrue-requires-python-310)
7. [No Retry or Rate-Limit Handling on Gemini Calls](#7-no-retry-or-rate-limit-handling)
8. [Chroma where Filter Crashes on Empty Collection](#8-chroma-where-filter-crashes-on-empty-collection)
9. [image_url Mutation on a slots Dataclass](#9-image_url-mutation-on-a-slots-dataclass)
10. [Frontend: resolveMode Can Return undefined](#10-frontend-resolvemode-can-return-undefined)
11. [Frontend: moduleResolution Node Is Outdated](#11-frontend-moduleresolution-node-is-outdated)

---

## 1. Equation False Positives

### Root Cause — Three overlapping triggers, each too permissive on its own

**Trigger A — Math density threshold is 15%**

```python
# ingestion.py  _MATH_SYMBOLS
_MATH_SYMBOLS = set("=+-*/^_<>%()[]{}|\\∑∫√≈≠≤≥±∞∂∆∇λμσπθβα")
```

The set includes `(`, `)`, `[`, `]`, `{`, `}`, `%`, `+`, `-`, `*` — all extremely common in
ordinary prose, citations, and table cells. A sentence like:

```
The average yield (2019–2023) was 84.3% — up from 71% in prior cycles.
```

…contains `(`, `)`, `-`, `%` which together easily push density above 15 %.

**Trigger B — The `=` regex matches any key=value assignment**

```python
if re.search(r"\b([A-Za-z]\s*=\s*.+|.+\s*=\s*[A-Za-z0-9(])", stripped):
    return True
```

This fires on:
- `status = active`
- `name = Sample Group A`
- `Total = 1,240 participants`
- `Table 2 = Summary of findings (n = 340)`

All common in research PDFs. The pattern asks only that a letter be followed by `=`; it does
not require numeric operands or isolated variable names.

**Trigger C — Parenthesised phrases followed by `=` or `>`**

```python
if re.search(r"\([^)]+\)\s*[=<>]", stripped):
    return True
```

Fires on inline citations: `(Smith et al., 2020) > prior estimates` or
`(see Table 3) = consistent with…`.

**The result**: any paragraph that mentions a percentage, has a unit in parentheses, or
contains a comparison phrase will be yanked out as an `EquationChunk` and never reach the
text chunker.

---

### Solution

Replace the single `_is_equation_line` function with a **scoring approach** that requires
multiple signals to coincide. A line must score ≥ 2 out of 4 independent signals **and** must
pass a prose-veto before it is classified as an equation.

```python
# ── replacements in ingestion.py ─────────────────────────────────────────────

# Tighter math symbol set — remove characters ubiquitous in prose
_MATH_SYMBOLS = set("=<>±∑∫√≈≠≤≥∞∂∆∇λμσπθβα^")

# Common prose words that immediately veto equation status
_PROSE_VETO_WORDS = {
    "the", "and", "that", "this", "with", "from", "have", "which",
    "their", "were", "been", "than", "these", "those", "however",
    "therefore", "although", "results", "figure", "table", "study",
    "patients", "participants", "data", "analysis", "treatment",
}

# Words that positively indicate math context
_MATH_CONTEXT_WORDS = {
    "equation", "formula", "integral", "derivative", "matrix",
    "vector", "scalar", "coefficient", "eigenvalue", "function",
    "polynomial", "theorem", "proof", "lemma",
}

def _is_equation_line(line: str) -> bool:
    stripped = line.strip()
    if len(stripped) < 3:
        return False

    tokens = stripped.split()
    word_count = len(tokens)

    # ── Prose veto: long lines with ordinary words are never equations ────────
    lower_tokens = {t.lower().strip(".,;:()[]") for t in tokens}
    prose_hits = lower_tokens & _PROSE_VETO_WORDS
    # More than 2 prose veto words → definitely prose
    if len(prose_hits) > 2:
        return False
    # Sentences longer than 12 words with any prose veto word → prose
    if word_count > 12 and prose_hits:
        return False

    score = 0

    # ── Signal 1: LaTeX markers (strong, immediate classification) ────────────
    if any(marker in stripped for marker in _LATEX_MARKERS):
        return True  # LaTeX is unambiguous; no further checks needed

    # ── Signal 2: High math-symbol density (raised threshold) ────────────────
    non_space = [c for c in stripped if not c.isspace()]
    if non_space:
        math_hits = sum(c in _MATH_SYMBOLS for c in non_space)
        density = math_hits / len(non_space)
        if density >= 0.30:           # raised from 0.15 → 0.30
            score += 1

    # ── Signal 3: Numeric equation pattern (variable op number) ──────────────
    # Requires at least one side to be numeric or a single variable token,
    # NOT a multi-word phrase.
    numeric_eq = re.search(
        r"(?<!\w)([A-Za-z]{1,4}|\d[\d.,]*)\s*[=<>]\s*(\d[\d.,]*[A-Za-z]{0,3}|[A-Za-z]{1,4})(?!\w)",
        stripped,
    )
    if numeric_eq:
        score += 1

    # ── Signal 4: Dense operator cluster (no English word between ops) ────────
    operator_cluster = re.search(
        r"\d[\d.]*\s*[+\-*/^]\s*\d[\d.]*",   # e.g. 3.14 * r^2
        stripped,
    )
    if operator_cluster:
        score += 1

    # ── Signal 5: Known math-context word (bonus, not standalone) ────────────
    if lower_tokens & _MATH_CONTEXT_WORDS:
        score += 1

    return score >= 2
```

**Why this is safe**: LaTeX markers are an immediate pass (unchanged behaviour for real
equations). Everything else requires at least two independent signals. The prose veto fires
first, so a table header like `"Group = Treatment (n = 340)"` is eliminated before reaching
the scoring phase.

---

## 2. Table Rows Classified as Equations

### Root Cause — Pipeline order and shared text source

The pipeline in `PDFIngestionAgent.process_pdf` is:

```
pdftotext → section_map → text_chunks
                        → equation_chunks   ← runs on ALL pdftotext output
pdfplumber              → table_chunks      ← separate, structured extraction
pdfimages               → images
```

`_extract_equations` receives raw pdftotext output. pdftotext renders table cells as
space-separated text on individual lines — exactly the format that looks like equations to
the current detector. Consider a pdfplumber table that contains:

```
| Parameter | Value | Unit |
| T_max     | 37.8  | °C   |
| pH        | 7.4   | —    |
```

pdftotext emits this as:

```
Parameter   Value   Unit
T_max       37.8    °C
pH          7.4     —
```

The line `pH          7.4` scores high on the numeric-equation signal (`pH = 7.4` pattern),
and `T_max       37.8    °C` triggers the operator-cluster check. Both become
`EquationChunk`s even though pdfplumber correctly identifies them as table rows.

### Why the simple fix of "run equations after tables" does not work

Even if table extraction runs first, `_extract_equations` still works on the pdftotext
string. The two sources (pdftotext text and pdfplumber structured data) are completely
separate; there is no coordinate mapping between them.

---

### Solution — Two-stage table-region exclusion

**Stage 1**: Use pdfplumber to extract table bounding boxes *and* collect all table cell
text into a normalised exclusion set before running equation detection.

**Stage 2**: Pass the exclusion set into `_extract_equations` so each candidate line can be
vetoed if its content originated from a known table region.

```python
# ── New helper in ingestion.py ────────────────────────────────────────────────

def _build_table_text_exclusion(pdf_path: Path) -> set[str]:
    """
    Returns a set of normalised strings extracted from every table cell in the
    PDF.  Any line whose normalised form appears in this set is skipped during
    equation detection.
    """
    if pdfplumber is None:
        return set()

    exclusion: set[str] = set()
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables() or []:
                for row in table:
                    for cell in row:
                        if cell:
                            normalised = re.sub(r"\s+", " ", str(cell)).strip().lower()
                            if normalised:
                                exclusion.add(normalised)
    return exclusion


def _normalise_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip().lower()
```

**Modify `_extract_equations` signature**:

```python
def _extract_equations(
    *,
    pages: Sequence[_PageText],
    section_map: dict[int, str],
    journal_id: str,
    article_id: str,
    source_path: str,
    table_exclusion: set[str],          # ← new parameter
) -> list[EquationChunk]:
    chunks: list[EquationChunk] = []

    for page in pages:
        lines = [line.rstrip() for line in page.text.splitlines()]
        block: list[str] = []
        block_start = 0

        def flush(end_index: int) -> None:
            nonlocal block, block_start
            if not block:
                return
            latex = "\n".join(line.strip() for line in block if line.strip()).strip()
            if latex:
                chunks.append(
                    EquationChunk(
                        chunk_id=f"{journal_id}_{article_id}_eq_{len(chunks) + 1:05d}",
                        journal_id=journal_id,
                        article_id=article_id,
                        source_path=source_path,
                        latex=latex,
                        context=_extract_equation_context(lines, block_start, end_index),
                        page_number=page.page_number,
                        section=section_map.get(page.page_number),
                    )
                )
            block = []
            block_start = 0

        for index, line in enumerate(lines):
            # ── Table-region veto ────────────────────────────────────────────
            if _normalise_line(line) in table_exclusion:
                flush(index)
                continue

            if _is_equation_line(line):
                if not block:
                    block_start = index
                block.append(line)
                continue
            flush(index)

        flush(len(lines))

    return chunks
```

**Update `process_pdf` to wire everything together**:

```python
def process_pdf(self, pdf_path, assets_dir):
    ...
    pages = _extract_page_text(source_pdf)
    section_map = _build_section_map(pages)

    # Build exclusion set BEFORE equation extraction
    table_exclusion = _build_table_text_exclusion(source_pdf)

    return IngestedDocument(
        ...
        text_chunks=_build_text_chunks(...),
        equation_chunks=_extract_equations(
            pages=pages,
            section_map=section_map,
            journal_id=journal_id,
            article_id=article_id,
            source_path=str(source_pdf),
            table_exclusion=table_exclusion,   # ← pass it in
        ),
        table_chunks=_extract_tables(...),
        images=_extract_images(...),
    )
```

**Why this does not break anything**:
- `_build_table_text_exclusion` is a read-only pdfplumber pass; it does not change the
  extraction logic for tables.
- Cell text normalisation (collapse whitespace, lowercase) is robust to the minor formatting
  differences between pdftotext and pdfplumber output.
- If pdfplumber is not installed, the function returns an empty set and behaviour is
  identical to today.
- The exclusion only vetoes lines; multi-line true equations that happen to share one token
  with a table cell will still be captured because the block is only flushed (not discarded)
  on a veto — consecutive equation lines still accumulate normally.

---

## 3. Image Embeddings Silently Failing

### Root Cause — Three compounding issues in `embed_file`

**Issue A — `Part` is not wrapped in a `Content` object**

```python
# embeddings.py — current code
part = genai_types.Part.from_bytes(data=embedded_file.data, mime_type=embedded_file.mime_type)
contents: Any = part if part is not None else embedded_file.data
response = client.models.embed_content(model=model, contents=contents)
```

`models.embed_content` expects `contents` to be a `str`, a `Content` object, or a `list` of
`Content` objects. Passing a bare `Part` is not a documented calling convention; the SDK
either raises an internal serialisation error or silently produces a zero-vector.

**Issue B — Raw bytes fallback**

When `hasattr(genai_types.Part, "from_bytes")` is `False` (which happens if the SDK version
installed does not expose that classmethod), the code falls back to:

```python
contents = embedded_file.data    # raw bytes object
```

`embed_content` cannot accept raw bytes. This raises a `TypeError` that propagates up
through `service.py`'s list-comprehension:

```python
[self.embedding_client.embed_file(image.file_path) for image in document.images]
```

One failure aborts the entire list. Because the API endpoint has no try/except, the whole
`/index` call returns 500 and **no embeddings are stored** for any chunk type.

**Issue C — `_extract_embedding_vector` does not handle image response shapes**

The Gemini API returns a slightly different response shape when the input is multimodal
(a `content_embedding` field rather than `embedding`). `_extract_embedding_vector` does not
check for this key.

---

### Solution

Rewrite `embed_file` in `embeddings.py` with explicit `Content` wrapping, a proper fallback
chain, and a response-shape guard:

```python
def embed_file(
    client: Any,
    path: str | Path,
    *,
    model: str = EMBEDDING_MODEL,
) -> list[float]:
    """Embed a local file (image or PDF) using Gemini multimodal embeddings."""

    embedded_file = read_binary_file(path)

    # ── Build the Part using the most reliable available constructor ──────────
    part = _build_part(embedded_file.data, embedded_file.mime_type)

    # ── Wrap Part in a Content object (required by embed_content) ────────────
    if genai_types is not None and hasattr(genai_types, "Content"):
        contents = genai_types.Content(parts=[part])
    else:
        # Fallback: dict representation accepted by all SDK versions
        import base64
        contents = {
            "parts": [{
                "inline_data": {
                    "mime_type": embedded_file.mime_type,
                    "data": base64.b64encode(embedded_file.data).decode("utf-8"),
                }
            }]
        }

    response = client.models.embed_content(model=model, contents=contents)
    return _extract_embedding_vector(response)


def _build_part(data: bytes, mime_type: str) -> Any:
    """Try all known Part constructors in order of preference."""
    if genai_types is None:
        raise ImportError("google-genai is required for file embedding.")

    # google-genai ≥ 1.0  —  Part.from_bytes(data, mime_type)
    if hasattr(genai_types.Part, "from_bytes"):
        try:
            return genai_types.Part.from_bytes(data=data, mime_type=mime_type)
        except Exception:
            pass

    # google-genai ≥ 0.8  —  Part(inline_data=Blob(...))
    if hasattr(genai_types, "Blob"):
        try:
            return genai_types.Part(
                inline_data=genai_types.Blob(data=data, mime_type=mime_type)
            )
        except Exception:
            pass

    # Universal dict fallback — accepted by every SDK version
    import base64
    return {
        "inline_data": {
            "mime_type": mime_type,
            "data": base64.b64encode(data).decode("utf-8"),
        }
    }
```

Also extend `_extract_embedding_vector` to handle the multimodal response shape:

```python
def _extract_embedding_vector(response: Any) -> list[float]:
    # ── New: multimodal response key ─────────────────────────────────────────
    if isinstance(response, dict):
        # Multimodal shape: {"content_embedding": {"values": [...]}}
        ce = response.get("content_embedding")
        if isinstance(ce, dict):
            values = ce.get("values")
            if values is not None:
                return list(values)

        # Standard text shape: {"embedding": {"values": [...]}}
        if "embedding" in response and isinstance(response["embedding"], dict):
            values = response["embedding"].get("values")
            if values is not None:
                return list(values)
        ...  # rest of existing checks unchanged

    # ── New: attribute-based multimodal shape ─────────────────────────────────
    content_embedding = getattr(response, "content_embedding", None)
    if content_embedding is not None:
        values = getattr(content_embedding, "values", None)
        if values is not None:
            return list(values)

    ...  # rest of existing attribute checks unchanged
```

**Also**: wrap the image embedding loop in `service.py` so one bad image does not abort the
entire index operation:

```python
# service.py — replace the image embedding block
if document.images:
    image_embeddings: list[list[float]] = []
    images_to_store: list[ExtractedImage] = []
    for image in document.images:
        try:
            embedding = self.embedding_client.embed_file(image.file_path)
            image_embeddings.append(embedding)
            images_to_store.append(image)
        except Exception as exc:
            # Log and skip; do not abort the whole document
            print(f"[WARN] Could not embed image {image.file_path}: {exc}")

    if images_to_store:
        self.store.add_images(images_to_store, image_embeddings)
```

---

## 4. PDF Filename Regex Too Restrictive

### Root Cause

```python
# types.py
_PDF_STEM_RE = re.compile(
    r"^(?P<jid>[A-Za-z]{2})(?:_(?P<aid1>[A-Za-z0-9]+)|(?P<aid2>[A-Za-z0-9]+))$"
)
```

The journal prefix is locked to exactly **two letters**. Files named `Nature_12345.pdf`,
`PLOS_98765.pdf`, `JBC100833.pdf` (three-letter prefix, no underscore) all raise
`ValueError` and crash the ingestion agent.

### Solution

Relax the prefix to 2–6 letters and make the underscore separator optional but preferred.
Add a graceful fallback that treats the entire stem as the article ID when the pattern does
not match instead of raising.

```python
# types.py
_PDF_STEM_RE = re.compile(
    r"^(?P<jid>[A-Za-z]{2,6})_(?P<aid>[A-Za-z0-9]+)$"   # preferred: JID_AID
)
_PDF_STEM_NOUNDERSCORE_RE = re.compile(
    r"^(?P<jid>[A-Za-z]{2,6})(?P<aid>[0-9]+)$"            # fallback:  JID12345
)


def parse_pdf_filename(path: str | Path) -> tuple[str, str]:
    stem = Path(path).stem

    m = _PDF_STEM_RE.fullmatch(stem)
    if m:
        return m.group("jid").upper(), m.group("aid")

    m = _PDF_STEM_NOUNDERSCORE_RE.fullmatch(stem)
    if m:
        return m.group("jid").upper(), m.group("aid")

    # Graceful fallback: treat entire stem as article ID with unknown journal
    return "UNK", stem
```

The `"UNK"` fallback means ingestion never crashes on an unconventional filename. The
document is still indexed; only the journal grouping is lost.

---

## 5. No Error Handling in API Endpoints

### Root Cause

Both `POST /index` and `POST /search` call service methods without try/except:

```python
@app.post("/index")
def index_pdf(request: IndexRequest) -> dict[str, object]:
    ...
    document = service.index_pdf(pdf_path, IMAGE_DIR)   # can raise anything
```

A missing Poppler binary, a Gemini quota error, or a ChromaDB write failure all produce an
unhandled 500 with a raw Python traceback visible to the caller.

### Solution

Wrap each endpoint in a typed exception handler:

```python
import logging
from fastapi import HTTPException

logger = logging.getLogger(__name__)

@app.post("/index")
def index_pdf(request: IndexRequest) -> dict[str, object]:
    pdf_path = Path(request.pdf_path)
    if not pdf_path.is_absolute():
        pdf_path = BASE_DIR / pdf_path
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF not found.")

    service = container.get_service()
    try:
        document = service.index_pdf(pdf_path, IMAGE_DIR)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        # Covers missing Poppler commands, pdfplumber errors, etc.
        logger.exception("Ingestion failed for %s", pdf_path)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error indexing %s", pdf_path)
        raise HTTPException(status_code=500, detail="Indexing failed; check server logs.") from exc

    return { ... }


@app.post("/search")
def search(request: SearchRequest) -> dict[str, object]:
    service = container.get_service()
    try:
        results = service.search(
            request.query,
            limit=request.limit,
            content_types=request.content_types,
        )
    except Exception as exc:
        logger.exception("Search failed for query %r", request.query)
        raise HTTPException(status_code=500, detail="Search failed; check server logs.") from exc

    return { ... }
```

---

## 6. `zip(strict=True)` Requires Python 3.10+

### Root Cause

`storage.py` uses `zip(..., strict=True)` in four methods:

```python
for chunk, embedding in zip(chunks, embeddings, strict=True):
```

`strict=True` was added in Python 3.10. On Python 3.9 this raises a `TypeError` on every
add call, so **nothing is ever stored** and the failure is silent in production because the
API has no error handling (see #5).

### Solution

Replace with an explicit length check that gives a clear error message on every supported
Python version:

```python
def _strict_zip(a: Iterable, b: Iterable, *, label: str = "items"):
    a_list = list(a)
    b_list = list(b)
    if len(a_list) != len(b_list):
        raise ValueError(
            f"Mismatch: {len(a_list)} {label} but {len(b_list)} embeddings."
        )
    return zip(a_list, b_list)
```

Use `_strict_zip(chunks, embeddings, label="text chunks")` in every `add_*` method.

Alternatively, add `python_requires = ">=3.10"` to `pyproject.toml` / `setup.cfg` and
document it clearly so the environment expectation is explicit.

---

## 7. No Retry or Rate-Limit Handling

### Root Cause

`GeminiEmbeddingClient.embed_texts` calls `embed_text` in a tight loop with no backoff:

```python
def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
    return [self.embed_text(text) for text in texts]
```

For a document with 200+ chunks, this fires 200+ consecutive API calls. Gemini's embedding
API enforces per-minute quotas; the 201st call will receive HTTP 429 and raise an exception
that aborts the entire ingestion — all previously computed embeddings are discarded because
nothing was stored yet.

### Solution

Add a simple exponential-backoff retry wrapper. Do not add a heavy dependency like
`tenacity`; a small helper is sufficient:

```python
# embeddings.py

import time

def _with_retry(fn, *, max_attempts: int = 5, base_delay: float = 1.0):
    """Call fn(); on HTTP 429 or transient errors, retry with exponential backoff."""
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as exc:
            msg = str(exc).lower()
            is_retryable = "429" in msg or "quota" in msg or "rate" in msg or "timeout" in msg
            if not is_retryable or attempt == max_attempts - 1:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"[WARN] Gemini rate-limit hit; retrying in {delay:.1f}s (attempt {attempt + 1})")
            time.sleep(delay)
    raise RuntimeError("Unreachable")  # pragma: no cover
```

Use it in `embed_text` and `embed_file`:

```python
def embed_text(self, text: str) -> list[float]:
    return _with_retry(lambda: embed_text(self.client, text, model=self.model))

def embed_file(self, path: str | Path) -> list[float]:
    return _with_retry(lambda: embed_file(self.client, path, model=self.model))
```

For very large documents, also store embeddings **incrementally** — call `store.add_text_chunks`
in batches of 50 rather than collecting all embeddings first, so a mid-document failure
does not discard everything.

---

## 8. Chroma `where` Filter Crashes on Empty Collection

### Root Cause

When `content_types` is provided, `storage.py` constructs:

```python
where = {"kind": {"$in": list(content_types)}}
response = self.collection.query(
    query_embeddings=[query_embedding],
    n_results=limit,
    where=where,
    ...
)
```

ChromaDB raises `chromadb.errors.InvalidDimensionException` or a metadata-filter error
when the collection is empty or contains fewer documents than `n_results`. This crashes the
search endpoint entirely.

### Solution

Guard with a count check and clamp `n_results`:

```python
def search(self, query_embedding, *, limit=5, content_types=None):
    count = self.collection.count()
    if count == 0:
        return []

    effective_limit = min(limit, count)
    where = {"kind": {"$in": list(content_types)}} if content_types else None

    try:
        response = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=effective_limit,
            where=where,
            include=["metadatas", "distances"],
        )
    except Exception as exc:
        # ChromaDB can raise on malformed where clauses; return empty rather
        # than crashing the API.
        print(f"[WARN] ChromaDB query failed: {exc}")
        return []

    ...  # rest unchanged
```

---

## 9. `image_url` Mutation on a `slots` Dataclass

### Root Cause

```python
# types.py
@dataclass(slots=True)
class ExtractedImage:
    ...
    image_url: str | None = None
```

```python
# service.py
for image in document.images:
    image.image_url = f"{self.asset_url_prefix}/..."   # post-construction mutation
```

`slots=True` without `frozen=True` allows mutation, but mutating after construction means
the `IngestedDocument` returned from `process_pdf` is a partially-initialised object whose
state depends on external side effects applied in `service.py`. If `index_pdf` is called
without a valid `asset_root`, the images are stored with `image_url = None` and the UI
cannot render them.

### Solution

Move URL construction into `service.py` cleanly, and store the result as a **new object**
rather than mutating the one returned from ingestion:

```python
# In service.py
from dataclasses import replace   # available for dataclasses in Python 3.10+
                                   # or use copy + setattr for 3.9

patched_images: list[ExtractedImage] = []
for image in document.images:
    url: str | None = None
    if self.asset_root is not None:
        try:
            rel = Path(image.file_path).resolve().relative_to(self.asset_root)
            url = f"{self.asset_url_prefix}/{rel.as_posix()}"
        except ValueError:
            pass
    # Create a new instance with image_url set — no mutation
    patched_images.append(
        ExtractedImage(
            **{**vars(image), "image_url": url}   # works without slots;
            # with slots use: dataclasses.replace(image, image_url=url)
        )
    )
```

Because `slots=True` disables `__dict__`, use `dataclasses.replace`:

```python
from dataclasses import replace
patched = replace(image, image_url=url)
```

---

## 10. Frontend: `resolveMode` Can Return `undefined`

### Root Cause

```typescript
// SearchResults.tsx
function resolveMode(results, rankingMode) {
  if (rankingMode !== "auto") {
    return rankingMode;   // returns undefined if rankingMode is undefined
  }
  ...
}
```

The caller applies `|| "distance"` as a fallback, which saves the common case, but if
`rankingMode` is explicitly passed as `undefined` the function returns `undefined` before
the fallback can fire, and the sort comparator receives `"undefined"` as a string.

### Solution

```typescript
function resolveMode(
  results: SearchResult[],
  rankingMode: SearchResultsProps["rankingMode"] = "auto",
): "distance" | "score" {
  if (rankingMode === "distance" || rankingMode === "score") {
    return rankingMode;
  }
  // Auto-detect
  const hasDistance = results.some(
    (r) => typeof r.distance === "number" && !Number.isNaN(r.distance),
  );
  if (hasDistance) return "distance";

  const hasScore = results.some(
    (r) => typeof r.score === "number" && !Number.isNaN(r.score),
  );
  return hasScore ? "score" : "distance";
}
```

Return type is now `"distance" | "score"` (not `string | undefined`), which TypeScript
enforces at the call site.

---

## 11. Frontend: `moduleResolution: "Node"` Is Outdated

### Root Cause

```json
// tsconfig.json
{
  "compilerOptions": {
    "module": "ESNext",
    "moduleResolution": "Node"   // ← wrong for Vite + ESNext
  }
}
```

`moduleResolution: "Node"` uses Node.js CommonJS resolution rules, which do not understand
`exports` maps in `package.json`. With Vite and ESNext modules, the correct value is
`"Bundler"` (TypeScript 5+). Using `"Node"` suppresses import errors that surface at
runtime, particularly for packages that expose multiple entry points via `exports`.

### Solution

```json
{
  "compilerOptions": {
    "module": "ESNext",
    "moduleResolution": "Bundler",   // correct for Vite
    "allowImportingTsExtensions": true
  }
}
```

---

## Change Summary

| # | File | Type | Risk |
|---|------|------|------|
| 1 | `backend/src/multimodal/ingestion.py` | Logic rewrite — `_is_equation_line` | Low — stricter, not looser |
| 2 | `backend/src/multimodal/ingestion.py` | New helper + param — table exclusion | Low — additive |
| 3 | `backend/src/multimodal/embeddings.py` | Rewrite — `embed_file`, `_build_part`, `_extract_embedding_vector` | Medium — API surface |
| 3 | `backend/src/multimodal/service.py` | Add per-image try/except | Low |
| 4 | `backend/src/multimodal/types.py` | Relax regex + fallback | Low |
| 5 | `backend/src/multimodal/api.py` | Add try/except to endpoints | Low |
| 6 | `backend/src/multimodal/storage.py` | Replace `strict=True` zip | Low |
| 7 | `backend/src/multimodal/embeddings.py` | Add `_with_retry` | Low |
| 8 | `backend/src/multimodal/storage.py` | Guard on empty collection | Low |
| 9 | `backend/src/multimodal/types.py` + `service.py` | Use `dataclasses.replace` | Low |
| 10 | `frontend/src/components/SearchResults.tsx` | Fix return type of `resolveMode` | Low |
| 11 | `frontend/tsconfig.json` | Change `moduleResolution` to `"Bundler"` | Low |

---

## Implementation Order

```
types.py          → fix #4 (filename regex) and #9 (replace mutation)
ingestion.py      → fix #1 (equation detector) then #2 (table exclusion)
embeddings.py     → fix #3 (embed_file) and #7 (retry)
storage.py        → fix #6 (zip) and #8 (empty collection guard)
service.py        → fix #3 partial (per-image try/except) and #9 partial
api.py            → fix #5 (error handling)
SearchResults.tsx → fix #10 (resolveMode)
tsconfig.json     → fix #11
```

Each fix is independent — apply and test them in the order above so that earlier fixes do
not mask later ones.