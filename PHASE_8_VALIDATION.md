# Phase 8: Re-index and Validate

## Overview

After implementing Phases 1-7, all PDFs must be re-indexed due to changes in text extraction that affect embedding vector positions in Gemini's embedding space. Old and new vectors are incompatible and will produce inconsistent search rankings.

## Why Re-indexing is Required

Embeddings already stored in ChromaDB were computed from pymupdf4llm-extracted text. After Phase 1, the same PDFs will produce pdfplumber-extracted text which sits in a slightly different region of Gemini's embedding space. Mixing old and new vectors in the same collection will produce inconsistent search rankings.

## Procedure

### Step 1: Stop the Server

Stop any running development server:
```bash
# If running in terminal, press Ctrl+C
```

### Step 2: Wipe ChromaDB and Images

```bash
# Remove existing vector store and extracted images
rm -rf backend/data/chroma/
rm -rf backend/data/images/
mkdir -p backend/data/images/
```

### Step 3: Verify Environment

```bash
# Activate Python environment
source .venv/bin/activate

# Verify dependencies installed
pip list | grep -E "pdfplumber|chromadb|google-genai|fastapi"
```

### Step 4: Start the Server

```bash
# Start development server with reload enabled
PYTHONPATH=backend/src GEMINI_API_KEY=<your-key> uvicorn main:app \
  --app-dir backend/src --reload --port 8000
```

The server should start with no errors. Open http://localhost:8000/health to verify it's running.

### Step 5: Re-index All PDFs

For each PDF in your `pdfs/` directory, call the /index endpoint:

```bash
# Example: Index a single PDF
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"pdf_path": "pdfs/BJ_100833.pdf"}' | jq '.'

# Or script to index all PDFs:
for pdf in pdfs/*.pdf; do
    echo "Indexing: $pdf"
    curl -s -X POST http://localhost:8000/index \
      -H "Content-Type: application/json" \
      -d "{\"pdf_path\": \"$pdf\"}" | jq '.
text_chunks, .equation_chunks, .table_chunks, .images'
done
```

### Expected Output Format

Each /index response should return chunk counts by type:
```json
{
  "document_id": "BJ_100833",
  "journal_id": "BJ",
  "article_id": "100833",
  "text_chunks": 145,
  "equation_chunks": 12,
  "table_chunks": 8,
  "images": 3
}
```

**Expected changes from Phase 1 (pdfplumber text extraction):**
- `text_chunks`: Increase due to improved two-column layout handling
- `equation_chunks`: Stable or slight increase (better dedup + fraction reconstruction)
- `table_chunks`: Stable (table.py logic unchanged)
- `images`: Stable (pdfimages extraction unchanged)

### Step 6: Chunk Count Validation

After indexing, verify chunk counts are consistent and reasonable. Document the results:

```
PDF: BJ_100833.pdf
- text_chunks:     145 (▲ from 120 expected in two-column layout)
- equation_chunks: 12  (stable)
- table_chunks:    8   (stable)
- images:          3   (stable)
```

### Step 7: Semantic Search Validation

Run these test queries from the Python REPL or via `/search` endpoint. Verify top-1 results are semantically correct:

#### Test Query 1: Cell viability calculation formula

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "cell viability calculation formula", "limit": 3}'
```

**Expected:** Top result is an equation chunk with the reconstructed formula:
```
Cell viability (%) = (treated cells / untreated cells) × 100
```

#### Test Query 2: Biofilm eradication percentage

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "biofilm eradication percentage", "limit": 3}'
```

**Expected:** Top result is an equation chunk for biofilm eradication formula.

#### Test Query 3: MIC values table

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "MIC values table", "limit": 3}'
```

**Expected:** Top result is a table chunk with MIC (minimum inhibitory concentration) data.

#### Test Query 4: Hemolysis assay results

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "hemolysis assay results", "limit": 3}'
```

**Expected:** Top result is a table or text chunk related to hemolysis data.

#### Test Query 5: Fluorescence microscopy biofilm image

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "fluorescence microscopy biofilm image", "limit": 3}'
```

**Expected:** Top result is an image chunk with caption containing relevant keywords.

### Step 8: Content Type Filtering Validation

Verify filtering by content_types parameter works correctly:

```bash
# Equation-only search
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "differential equation", "limit": 4, "content_types": ["equation"]}'

# Table-only search
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "MIC values", "limit": 4, "content_types": ["table"]}'

# Image-only search (caption + context blend)
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "microscopy", "limit": 4, "content_types": ["image"]}'
```

All should return relevant results of the specified type only.

### Step 9: Sign-off

If all validation queries pass with semantically correct top-1 results, Phase 8 is complete:

```
✅ ChromaDB re-indexed with pdfplumber text extraction
✅ Equation post-processing (fraction reconstruction, dedup) functional
✅ Table caption embedding (bookended) functional
✅ Image caption + context extraction functional
✅ Content-type filtering working
✅ All test queries return correct semantic results
```

## Troubleshooting

### "No results from /search"

1. Verify `/health` endpoint returns `{"status": "healthy"}`
2. Check ChromaDB collection count: Should be non-zero after indexing
3. Verify all PDFs were successfully indexed (check terminal output for errors)

### "All chunks are equations" or other classification issues

1. Check if Phase 4 post-processing is running by adding debug output to `clean_pdf_text()`
2. Verify `reconstruct_fractions()` and `dedup()` are being called in `extract_equations()`

### "Image embedding failed"

1. Verify GEMINI_API_KEY is set and valid
2. Check pdfplumber import is working: `python -c "import pdfplumber; print('OK')"`
3. Check that PIL is installed for image dimension extraction

### "Table captions not properly boosted"

1. Verify `TableChunk.embed_text` property is returning caption at start and end
2. Check that caption is non-empty: `grep -r "self.caption" backend/src/`

## Performance Notes

- Re-indexing typically takes 2-5 minutes per PDF depending on size and content richness
- Image dimension extraction adds 5-10% overhead but provides better retrieval context
- Equation post-processing (fraction reconstruction + dedup) is negligible overhead

## Reverting Changes

If issues arise, you can revert by:

1. Restore the previous ChromaDB backup (if available)
2. Or keep Phase 1-7 changes but revert to old embedding weights in service.py:
   - `image_weight=0.70` (was 0.55)
   - `caption_weight=0.30` (was 0.45)

---

**Done with Phase 8?** Commit your changes:
```bash
git add -A
git commit -m "Phase 1-7: Pdfplumber text extraction, enhanced headings, enriched captions, fraction reconstruction"
```
