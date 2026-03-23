# Multimodal PDF Retrieval — Integration & Enhancement Plan

> Consolidates three working standalone scripts into the backend ingestion pipeline,
> upgrades caption embedding for tables and images, and removes the pymupdf4llm dependency.

---

## Overview

| Phase | Title | Risk | Effort |
|-------|-------|------|--------|
| 1 | Replace text extraction pipeline | High | Large |
| 2 | Upgrade heading detection | Medium | Small |
| 3 | Verify and patch table.py parity | Low | Small |
| 4 | Verify and patch equation.py parity | Low | Medium |
| 5 | Enrich table caption embedding | Low | Small |
| 6 | Enrich image caption extraction and embedding | Medium | Medium |
| 7 | Dependency cleanup | Low | Small |
| 8 | Re-index and validate | Medium | Medium |

---

## Phase 1 — Replace Text Extraction Pipeline

**Branch:** `feat/pdfplumber-text-pipeline`

**Problem:** `utils.py` calls `pymupdf4llm.to_markdown()` and falls back to pdfplumber only when output is suspiciously short (fewer than 50 non-whitespace chars). For two-column academic PDFs, pymupdf4llm converts text columns to `**==> picture … <==**` markers, leaving `page.text` empty after `clean_page_text()` runs. The fallback fires too late and the `PageBlocks.blocks` list is empty, so `_chunk_from_blocks()` never runs and the line-based fallback produces lower-quality chunks.

**Solution:** Replace the pymupdf4llm call entirely. Use pdfplumber natively throughout `extract_page_blocks()`, porting the column-splitting and font-analysis logic from `pdf_extractor.py`.

### 1.1 Files changed

| File | Change |
|------|--------|
| `backend/src/multimodal/ingestion/utils.py` | Full rewrite of `extract_page_blocks()` |
| `backend/src/multimodal/types.py` | No structural change; `PageBlocks.blocks` can remain empty list |

### 1.2 Functions to port from `pdf_extractor.py`

**`get_body_font_size(page)`**
Estimates body-text font size using the statistical mode of all character sizes on a page, rounded to the nearest 0.5pt. The most frequent size is the body size.

```python
def get_body_font_size(page) -> float:
    sizes = [
        round(ch["size"] * 2) / 2
        for ch in page.chars
        if ch.get("size", 0) > 4
    ]
    if not sizes:
        return 10.0
    freq: dict[float, int] = {}
    for s in sizes:
        freq[s] = freq.get(s, 0) + 1
    return max(freq, key=freq.get)
```

**`chars_to_lines(chars)`**
Groups a flat list of char dicts into visual lines by vertical (top) coordinate. Characters within 2pt of each other share a line; within each line chars are sorted left-to-right.

**`detect_column_split(page)`**
Builds a 1pt histogram of word x0 positions restricted to the central 30% of page width. The longest zero-density run ≥ 10pt is the gutter. Returns `None` for single-column pages.

**`extract_page_text(page, body_size, column_aware)`**
Main dispatch. If `column_aware=True`, calls `detect_column_split()`, splits chars into left/right groups, processes each independently, and concatenates. Falls back to single-column if no gutter found.

### 1.3 New `extract_page_blocks()` implementation

```python
def extract_page_blocks(pdf_path: Path) -> list[PageBlocks]:
    pages = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            body_size = get_body_font_size(page)
            text = extract_page_text(page, body_size, column_aware=True).strip()
            text = re.sub(r"\(cid:\d+\)", "−", text)  # clean CID font artifacts
            pages.append(PageBlocks(
                page_number=page.page_number,
                text=text,
                blocks=[],          # line-based chunker in text.py does not need blocks
                width=page.width,
                height=page.height,
            ))
    return pages
```

### 1.4 Remove from `utils.py`

- `import pymupdf`
- `import pymupdf4llm`
- `_NOISE_RE` pattern (pymupdf4llm artefact cleaner)
- `clean_page_text()`
- `_MIN_USABLE_TEXT_CHARS` constant
- `_pdfplumber_page_text()` fallback function

### 1.5 Acceptance criteria

- `text_chunks > 0` for every two-column ACS/RSC journal PDF tested
- `page.text` non-empty for all pages (no picture-marker bleed-through)
- Column order preserved: left column text appears before right column text in the chunk

---

## Phase 2 — Upgrade Heading Detection

**Branch:** `feat/font-size-heading-detection` (can be stacked on Phase 1 branch)

**Problem:** `detect_heading()` in `section.py` relies on regex patterns and a hard-coded list of known section names (`abstract`, `introduction`, etc.). This misses numbered sub-headings (`2.1 Mechanism`), bold body-size headings common in journals, and non-English section names.

**Solution:** Replace the regex classifier with the font-size delta approach from `pdf_extractor.py`, using the `body_size` computed in Phase 1.

### 2.1 Thresholds (configurable at module level)

```python
H1_THRESHOLD = 4.0   # paper title, figure titles
H2_THRESHOLD = 2.0   # section headings  e.g. "1. INTRODUCTION"
H3_THRESHOLD = 0.8   # sub-section headings  e.g. "2.1 Mechanism"
DETECT_BOLD_HEADINGS = True  # bold body-size lines → treat as sub-heading
```

### 2.2 New `classify_line()` function

```python
def classify_line(line_chars: list[dict], body_size: float) -> str:
    """Return heading prefix ('### ', '## ', '# ', '') for one text line."""
    if not line_chars:
        return ""
    sizes = [ch["size"] for ch in line_chars if ch.get("size", 0) > 4]
    fontnames = [ch.get("fontname", "") for ch in line_chars]
    if not sizes:
        return ""
    avg_size = statistics.mean(sizes)
    delta = avg_size - body_size
    if delta >= H1_THRESHOLD:
        return "### "
    if delta >= H2_THRESHOLD:
        return "## "
    if delta >= H3_THRESHOLD:
        return "# "
    if DETECT_BOLD_HEADINGS:
        is_bold = any(
            "Bold" in f or "bold" in f or "Semibold" in f
            for f in fontnames
        )
        if is_bold and abs(delta) < 0.5:
            return "# "
    return ""
```

### 2.3 Backward compatibility

Keep `detect_heading(line: str) -> str | None` as the public API. Internally it can fall back to the regex approach when no char data is available (e.g. during tests). Add an optional `chars` parameter:

```python
def detect_heading(line: str, chars: list[dict] | None = None, body_size: float = 10.0) -> str | None:
    if chars is not None:
        prefix = classify_line(chars, body_size).strip()
        return line.strip() if prefix else None
    # existing regex fallback unchanged
    ...
```

### 2.4 Acceptance criteria

- Section headings correctly identified for numbered (`2.1 Results`) and unnumbered (`RESULTS`) formats
- No body-text sentences misclassified as headings
- `section` field populated on > 90% of chunks for tested PDFs

---

## Phase 3 — Verify and Patch `table.py` Parity

**Branch:** `chore/table-parity-validation`

**Problem:** `table.py` in the repo was ported from `table_extractor.py` but the port may have missed subtle predicate details. Silent row-drop bugs are the most common failure mode.

### 3.1 Checklist — confirm all fixes are present

| Fix | Description | Location |
|-----|-------------|----------|
| FIX 1 | `CAPTION_PATTERN` has `\|$` suffix so `"Table 1"` at end-of-line matches | `table.py` line ~30 |
| FIX 2 | `build_table_text_exclusion()` uses **only** lines strategy | `table.py` |
| FIX 3 | `SECTION_HEADER_PATTERN` present to detect section headers mid-table | `table.py` |
| FIX 4a | Footnote line stops body collection immediately | `_collect_body()` |
| FIX 4b | Long narrative prose (>8 words, no digits) stops body collection | `_collect_body()` |
| FIX 4c | Row x0 far left of table + long prose = two-column bleed stop | `_collect_body()` |
| FIX 5 | `_merge_wrapped_rows()` appends continuation rows (lowercase start) | `table.py` |

### 3.2 Validation procedure

```bash
# Run standalone script, export CSVs
python table_extractor.py

# Run repo extraction on same PDFs, export CSVs
python -c "
from multimodal.ingestion.table import extract_tables
from multimodal.ingestion.utils import extract_page_blocks
from multimodal.ingestion.section import build_section_spans_from_blocks
from pathlib import Path
import csv

for pdf in ['AO_5c05577.pdf', 'BJ_100833.pdf']:
    pages = extract_page_blocks(Path(pdf))
    spans = build_section_spans_from_blocks(pages)
    tables = extract_tables(pdf_path=Path(pdf), pages=pages, spans=spans,
                            journal_id='X', article_id='Y', source_path=pdf)
    for i, t in enumerate(tables):
        with open(f'repo_{Path(pdf).stem}_table{i}.csv', 'w') as f:
            csv.writer(f).writerows([row.split(',') for row in t.csv_data.splitlines()])
"

# Diff row counts
diff <(wc -l standalone_*.csv) <(wc -l repo_*.csv)
```

Any row-count discrepancy → inspect `_is_narrative()` and `_is_column_header_row()` predicates first.

### 3.3 Acceptance criteria

- Row counts match between standalone script and repo for all test PDFs
- No tables silently dropped (zero-row outputs)

---

## Phase 4 — Verify and Patch `equation.py` Parity

**Branch:** `feat/equation-fraction-recon`

**Problem:** The repo's `equation.py` was rewritten to use the pdfplumber approach from `equation_extractor.py`, but three critical post-processing steps are missing from the public `extract_equations()` entry point.

### 4.1 Missing pieces

**`reconstruct_fractions()`** — Rewrites three specific garbled formula patterns that occur when a two-column PDF splits display fractions across lines:

| Pattern | Input (garbled) | Output (clean) |
|---------|----------------|----------------|
| Cell viability | `Cell viability(%) = × 100 treated cells untreated cells` | `Cell viability (%) = (treated cells / untreated cells) × 100` |
| % Hemolysis | `% of hemolysis at 540 nm OD of sample OD of −ve control = × 100` | `% Hemolysis at 540 nm = (OD_sample − OD_−ve) / (OD_+ve − OD_−ve) × 100` |
| Biofilm eradication | `Eradication of biofilm (%) OD in control OD in treatment = OD in control` | `Eradication of biofilm (%) = (OD_control − OD_treatment) / OD_control` |

**`clean_pdf_text()`** — Fixes run-together words from PDF character-stream merging:

```python
KNOWN_PAIRS = [
    ('treatedcells',     'treated cells'),
    ('untreatedcells',   'untreated cells'),
    ('cellviability',    'Cell viability'),
    ('eradicationofbiofilm', 'Eradication of biofilm'),
    ...
]
```

**`dedup()`** — Fingerprint prefix-matching deduplication that keeps the longest (most complete) version of near-duplicate equations, not just exact-match dedup.

### 4.2 Where to add these

In `equation.py`, inside `extract_equations()`, after the `cleaned` filter loop:

```python
# Current code ends here:
cleaned = []
for eq in unique:
    eq['raw_text'] = clean_pdf_text(eq['raw_text'])   # ADD clean_pdf_text
    if is_real_equation(eq['raw_text']):
        cleaned.append(eq)
unique = cleaned

unique = reconstruct_fractions(unique)   # ADD this line — was missing
```

### 4.3 Acceptance criteria

- Cell viability, hemolysis, and biofilm eradication equations appear as clean single-line formulas
- No duplicate equations in output (dedup working)
- `total_equations_found` matches standalone script count ± 1

---

## Phase 5 — Enrich Table Caption Embedding

**Branch:** `feat/table-caption-embedding` (can be combined with Phase 6)

**Problem:** `TableChunk.embed_text` includes the caption but buries it after section and header fields. The caption is the primary semantic signal for retrieval — a query like *"cell viability calculation"* should hit the caption first.

### 5.1 Change: `types.py` — `TableChunk.embed_text`

```python
# Before
@property
def embed_text(self) -> str:
    col_hint = f"Column {self.column + 1}" if self.column is not None else None
    parts = [self.section, col_hint, self.caption, self.header, self.csv_data]
    return "\n\n".join(part for part in parts if part)

# After
@property
def embed_text(self) -> str:
    parts = []
    if self.caption:
        parts.append(self.caption)           # anchor — caption first
    if self.section:
        parts.append(f"Section: {self.section}")
    if self.column is not None:
        parts.append(f"Column {self.column + 1}")
    if self.header:
        parts.append(self.header)
    if self.csv_data:
        parts.append(self.csv_data)
    if self.caption:
        parts.append(self.caption)           # bookend — caption last
    return "\n\n".join(parts)
```

Repeating the caption at both ends of the string gives it disproportionate weight in the embedding without needing any changes to the embedding model call.

### 5.2 No other files change

`service.py` already calls `chunk.embed_text` for tables. `table.py` already collects the full multi-line caption in `_collect_caption()`. `storage.py` serializes metadata via `asdict()` which includes `caption` automatically.

### 5.3 Acceptance criteria

- Semantic search for caption keywords returns the correct table as top-1 result
- `TableChunk.embed_text` starts and ends with caption text when caption is non-null

---

## Phase 6 — Enrich Image Caption Extraction and Embedding

**Branch:** `feat/image-caption-embedding`

**Problem (extraction):** `_extract_figure_caption()` in `image.py` uses line-scan heuristics on `page.text` (plain string). It stops after 5 lines, misses multi-sentence captions, and has no access to font size to detect caption boundaries.

**Problem (embedding):** `service.py` passes `image.caption` (raw string) to `embed_text()`. This omits section heading and surrounding body context. The image-to-text blend weight (70/30) underweights text because the current caption is often just `"Extracted image from page 4"`.

### 6.1 Change: `types.py` — add `context` field and `embed_text` to `ExtractedImage`

```python
@dataclass(slots=True)
class ExtractedImage:
    image_id: str
    journal_id: str
    article_id: str
    source_path: str
    file_path: str
    page_number: int | None
    mime_type: str
    width: int | None = None
    height: int | None = None
    caption: str | None = None
    image_url: str | None = None
    section: str | None = None
    context: str | None = None      # NEW — 2 sentences of body text before figure
    content_type: ContentType = "image"

    @property
    def embed_text(self) -> str:    # NEW — was missing entirely
        parts = []
        if self.caption:
            parts.append(self.caption)
        if self.section:
            parts.append(f"Section: {self.section}")
        if self.context:
            parts.append(self.context)
        return "\n\n".join(parts)

    def metadata(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["kind"] = self.content_type
        payload["imageUrl"] = self.image_url
        payload["mimeType"] = self.mime_type
        return payload
```

### 6.2 Change: `image.py` — pdfplumber-based caption extractor

Replace `_extract_figure_caption()` with a pdfplumber word-stream reader:

```python
def _extract_figure_caption_pdfplumber(
    pdf_path: str,
    page_number: int,
    image_index: int,
) -> tuple[str | None, str | None]:
    """
    Returns (caption, context).

    caption  — full figure caption, multi-sentence, ends at natural boundary
    context  — up to 2 sentences of body text immediately before the figure
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if not (0 < page_number <= len(pdf.pages)):
                return None, None
            page = pdf.pages[page_number - 1]
            words = page.extract_words(x_tolerance=3, y_tolerance=3)
    except Exception:
        return None, None

    if not words:
        return None, None

    # Group words into lines by y-coordinate proximity
    lines: list[list[dict]] = []
    for word in sorted(words, key=lambda w: (round(w["top"] / 3) * 3, w["x0"])):
        if lines and abs(word["top"] - lines[-1][0]["top"]) <= 4:
            lines[-1].append(word)
        else:
            lines.append([word])

    line_texts = [" ".join(w["text"] for w in ln) for ln in lines]

    # Find caption start by matching "Figure N" or "Fig. N"
    fig_exact = re.compile(rf"\b(?:figure|fig\.?)\s*{image_index}\b", re.IGNORECASE)
    fig_any   = re.compile(r"^\s*(?:figure|fig\.?)\s*\d+", re.IGNORECASE)

    caption_start = next(
        (i for i, t in enumerate(line_texts) if fig_exact.search(t)), -1
    )
    if caption_start == -1:
        caption_start = next(
            (i for i, t in enumerate(line_texts) if fig_any.search(t)), -1
        )
    if caption_start == -1:
        return None, None

    # Collect caption lines
    caption_parts = [line_texts[caption_start]]
    for i in range(caption_start + 1, min(len(line_texts), caption_start + 12)):
        text = line_texts[i].strip()
        if not text:
            break
        if re.match(r"^\s*(?:figure|fig\.?|table|scheme)\s*\d+", text, re.IGNORECASE):
            break
        if text.isupper() and len(text.split()) <= 5:   # section heading
            break
        if re.match(r"^\d+[\.\s]", text):               # numbered section
            break
        caption_parts.append(text)
        if text.endswith("."):
            break

    caption = " ".join(caption_parts).strip() or None

    # Collect context: 2 sentences before the caption
    raw_context = " ".join(
        ln.strip() for ln in line_texts[max(0, caption_start - 6): caption_start]
        if ln.strip()
    )
    sentences = re.split(r"(?<=[.!?])\s+", raw_context)
    context = " ".join(sentences[-2:]).strip() if sentences else None

    return caption, context
```

Update `extract_images()` to call the new extractor and store `context`:

```python
caption, context = _extract_figure_caption_pdfplumber(
    str(pdf_path), page_number, image_index
)
# Fallback to line-scan if pdfplumber found nothing
if caption is None:
    page_obj = page_by_num.get(page_number)
    lines_on_page = page_obj.text.splitlines() if page_obj and page_obj.text else []
    caption_line_index = _find_caption_line(image_index, lines_on_page)
    caption = _extract_figure_caption(caption_line_index, lines_on_page)
    context = None

images.append(ExtractedImage(
    ...
    caption=caption or _build_image_caption(page_number, section),
    context=context,    # new field
    section=section,
))
```

### 6.3 Change: `service.py` — use `embed_text`, adjust blend weight

```python
# In index_pdf(), replace the image embedding block:
for image in document.images:
    try:
        visual_embedding = self.embedding_client.embed_file(image.file_path)

        if image.embed_text.strip():
            text_embedding = self.embedding_client.embed_text(image.embed_text)
            embedding = _blend_embeddings(
                visual_embedding,
                text_embedding,
                image_weight=0.55,    # was 0.70
                caption_weight=0.45,  # was 0.30 — richer text earns more weight
            )
        else:
            embedding = visual_embedding

    except Exception as exc:
        print(f"[WARN] Could not embed image {image.file_path}: {exc}")
        continue
    image_embeddings.append(embedding)
    images_to_store.append(image)
```

### 6.4 Acceptance criteria

- Multi-sentence figure captions extracted correctly for test PDFs
- `image.context` non-null for figures with preceding body text
- Semantic query for figure topic returns correct image as top-1 result
- Blend weight shift does not degrade image-only queries (test with visual-only queries like "microscopy image")

---

## Phase 7 — Dependency Cleanup

**Branch:** `feat/dep-cleanup` (open after Phase 1 is green on CI)

### 7.1 Remove from `requirements.txt` / `pyproject.toml`

```
pymupdf
pymupdf4llm
```

### 7.2 Remove from codebase

| Item | Location |
|------|----------|
| `import pymupdf` | `utils.py` |
| `import pymupdf4llm` | `utils.py` |
| `_NOISE_RE` | `utils.py` |
| `clean_page_text()` | `utils.py` |
| `_MIN_USABLE_TEXT_CHARS` | `utils.py` |
| `_pdfplumber_page_text()` | `utils.py` |
| `equation.py.bak` | delete file |
| `table.py.bak` | delete file |

### 7.3 Final dependency set

| Package | Purpose |
|---------|---------|
| `pdfplumber` | All PDF text, char, and table extraction |
| `fastapi` + `uvicorn` | API server |
| `chromadb` | Vector store |
| `google-genai` | Gemini embedding model |
| `Pillow` | Image dimension reading |
| `pdfimages` (Poppler, system) | Image extraction from PDFs |
| `python-dotenv` | Environment config |

### 7.4 Acceptance criteria

- `pip install -r requirements.txt` completes without installing pymupdf or pymupdf4llm
- All existing tests pass
- Docker image size reduced (pymupdf4llm is ~150MB with its native libs)

---

## Phase 8 — Re-index and Validate

**Branch:** not a code branch — operational step

**Why re-indexing is required:** Embeddings already stored in ChromaDB were computed from pymupdf4llm-extracted text. After Phase 1, the same PDFs will produce pdfplumber-extracted text which sits in a slightly different region of Gemini's embedding space. Mixing old and new vectors in the same collection will produce inconsistent search rankings.

### 8.1 Wipe and re-index procedure

```bash
# 1. Stop the server
# 2. Delete existing vectors
rm -rf backend/data/chroma/

# 3. Restart server
PYTHONPATH=backend/src uvicorn main:app --app-dir backend/src --reload

# 4. Re-index all PDFs
for pdf in path/to/your/pdfs/*.pdf; do
    curl -s -X POST http://localhost:8000/index \
        -H "Content-Type: application/json" \
        -d "{\"pdf_path\": \"$pdf\"}" | jq '.text_chunks, .equation_chunks, .table_chunks, .images'
done
```

### 8.2 Validation queries

Run these queries and verify the top result is semantically correct:

| Query | Expected top result type | Expected content |
|-------|--------------------------|------------------|
| `cell viability calculation formula` | equation | Cell viability (%) = (treated cells / untreated cells) × 100 |
| `biofilm eradication percentage` | equation | Eradication of biofilm (%) formula |
| `MIC values table` | table | Table with minimum inhibitory concentration data |
| `fluorescence microscopy biofilm image` | image | Confocal or fluorescence microscopy figure |
| `hemolysis assay results` | table or text | Hemolysis data section |

### 8.3 Chunk count sanity check

After re-indexing, compare chunk counts before and after. Expected changes:

| Chunk type | Expected direction | Reason |
|------------|-------------------|--------|
| `text_chunks` | Increase | pdfplumber captures more text from two-column layouts |
| `equation_chunks` | Stable or slight increase | Better dedup, fraction reconstruction |
| `table_chunks` | Stable | table.py logic was already correct |
| `images` | Stable | image extraction uses pdfimages, unchanged |

---

## Branch Order and Dependencies

```
Phase 1  feat/pdfplumber-text-pipeline
    └── Phase 2  feat/font-size-heading-detection   (stack on Phase 1)
        └── Phase 7  feat/dep-cleanup               (open after Phase 1 green on CI)

Phase 3  chore/table-parity-validation              (independent, validation only)

Phase 4  feat/equation-fraction-recon               (independent, low risk)

Phase 5  feat/table-caption-embedding  ─┐
Phase 6  feat/image-caption-embedding  ─┴─ can be one PR: feat/caption-embedding

Phase 8  (operational — run after all code branches merged)
```

Phases 1 and 2 are the critical path. Everything else is independent and can be merged in any order once Phase 1 is green.

---

## Summary of Retrieval Quality Improvements

| Area | Before | After |
|------|--------|-------|
| Two-column PDF text | Often empty due to pymupdf4llm picture markers | Fully extracted via pdfplumber column split |
| Section heading detection | Regex + known names only | Font-size delta + bold detection |
| Equation fractions | Garbled split across lines | Reconstructed as clean single-line formulas |
| Table embedding | Caption buried mid-string | Caption bookends the embed string |
| Image caption | Line-scan, 5-line max, regex only | pdfplumber word-stream, multi-sentence, with body context |
| Image embedding text signal | Raw caption string only | Caption + section + surrounding context |
| Image/text blend weight | 70% visual / 30% text | 55% visual / 45% text |