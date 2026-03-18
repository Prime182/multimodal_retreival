from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional convenience dependency
    def load_dotenv() -> bool:
        return False

from .embeddings import GeminiEmbeddingClient
from .service import MultimodalRetrievalService
from .storage import PublisherArticleStore

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "backend" / "data"
IMAGE_DIR = DATA_DIR / "images"
CHROMA_DIR = DATA_DIR / "chroma"


class IndexRequest(BaseModel):
    pdf_path: str = Field(..., description="Absolute or repo-relative path to the PDF file.")


class SearchRequest(BaseModel):
    query: str
    limit: int = Field(default=5, ge=1, le=25)


class ServiceContainer:
    def __init__(self) -> None:
        self.service: MultimodalRetrievalService | None = None

    def get_service(self) -> MultimodalRetrievalService:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="Set GEMINI_API_KEY before indexing or searching.",
            )
        if self.service is None:
            self.service = MultimodalRetrievalService(
                embedding_client=GeminiEmbeddingClient(api_key=api_key),
                store=PublisherArticleStore(persist_directory=CHROMA_DIR),
                asset_root=IMAGE_DIR,
                asset_url_prefix="/assets",
            )
        return self.service


def create_app() -> FastAPI:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    container = ServiceContainer()

    app = FastAPI(title="Multimodal PDF Image Retrieval System")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.mount("/assets", StaticFiles(directory=str(IMAGE_DIR)), name="assets")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/index")
    def index_pdf(request: IndexRequest) -> dict[str, object]:
        pdf_path = Path(request.pdf_path)
        if not pdf_path.is_absolute():
            pdf_path = BASE_DIR / pdf_path
        if not pdf_path.exists():
            raise HTTPException(status_code=404, detail="PDF not found.")

        service = container.get_service()
        document = service.index_pdf(pdf_path, IMAGE_DIR)
        return {
            "document_id": document.document_id,
            "source_path": document.source_path,
            "text_chunks": len(document.text_chunks),
            "images": len(document.images),
        }

    @app.post("/search")
    def search(request: SearchRequest) -> dict[str, object]:
        service = container.get_service()
        results = service.search_images(request.query, limit=request.limit)
        return {
            "query": request.query,
            "results": [
                {
                    "id": result.item_id,
                    "distance": result.distance,
                    "metadata": result.metadata,
                }
                for result in results
            ],
        }

    return app


app = create_app()
