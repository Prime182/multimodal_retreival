# Multimodal PDF Image Retrieval System

This repo now contains a working split between a Python backend and a React frontend for multimodal PDF search:

- `backend/src/multimodal/ingestion.py`
  Parses PDFs by extracting page text and embedded images separately, which avoids the Gemini six-page PDF limit.
- `backend/src/multimodal/embeddings.py`
  Embeds text, images, and PDFs with `embedding-001` through a shared helper layer.
- `backend/src/multimodal/storage.py`
  Persists embeddings plus UI-facing metadata in a local Chroma collection named `publisher_articles`.
- `backend/src/multimodal/service.py`
  Orchestrates ingest, embed, store, and search.
- `backend/src/multimodal/api.py`
  Exposes `/index`, `/search`, and `/assets/*` for the UI.
- `frontend/src/components/SearchResults.tsx`
  Renders ranked image results using `imageUrl` or `file_path`.

## Backend flow

1. `POST /index`
   Accepts a PDF path, extracts text and images, chunks text up to `8192` approximate tokens, embeds each chunk and image, and stores them in Chroma.
2. `POST /search`
   Embeds the text query with the same Gemini model, queries Chroma, filters to image-backed matches, and returns distances plus metadata.
3. `/assets/*`
   Serves extracted images so the browser can render them with stable URLs.

## Metadata contract

Embeddings are only numeric vectors. The UI depends on metadata for rendering, so image records store:

- `imageUrl`
- `file_path`
- `caption`
- `source_path`
- `page_number`

## Prerequisites

Python packages:

```bash
pip install -r requirements.txt
```

Frontend packages:

```bash
cd frontend
npm install
```

Environment:

```bash
cp .env.example .env
cp frontend/.env.example frontend/.env
```

System tools required for PDF ingestion:

- `pdftotext`
- `pdfimages`

These come from Poppler on most Linux distributions.

## Run the backend

```bash
PYTHONPATH=backend/src uvicorn main:app --app-dir backend/src --reload
```

## Run the frontend

```bash
cd frontend
npm run dev
```

## Example API calls

Index a PDF:

```bash
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"pdf_path":"path/to/article.pdf"}'
```

Search for images:

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"solar adoption chart","limit":5}'
```

## Notes

- The ingestion token count is approximate, not Gemini-token exact.
- Image extraction is local-first and writes files under `backend/data/images/`.
- Chroma persistence is local-first and writes under `backend/data/chroma/`.
