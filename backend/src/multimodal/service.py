from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Sequence

from .embeddings import GeminiEmbeddingClient
from .ingestion import PDFIngestionAgent
from .storage import PublisherArticleStore
from .types import ExtractedImage, IngestedDocument, SearchResult


class MultimodalRetrievalService:
    def __init__(
        self,
        embedding_client: GeminiEmbeddingClient,
        store: PublisherArticleStore,
        ingestion_agent: PDFIngestionAgent | None = None,
        asset_root: str | Path | None = None,
        asset_url_prefix: str = "/assets",
    ) -> None:
        self.embedding_client = embedding_client
        self.store = store
        self.ingestion_agent = ingestion_agent or PDFIngestionAgent()
        self.asset_root = Path(asset_root).resolve() if asset_root else None
        self.asset_url_prefix = asset_url_prefix.rstrip("/") or "/assets"

    def index_pdf(self, pdf_path: str | Path, image_output_dir: str | Path) -> IngestedDocument:
        document = self.ingestion_agent.process_pdf(pdf_path, image_output_dir)

        patched_images: list[ExtractedImage] = []
        for image in document.images:
            image_url: str | None = None
            if self.asset_root is not None:
                image_path = Path(image.file_path).resolve()
                try:
                    relative_path = image_path.relative_to(self.asset_root)
                except ValueError:
                    pass
                else:
                    image_url = f"{self.asset_url_prefix}/{relative_path.as_posix()}"
            patched_images.append(replace(image, image_url=image_url))

        if patched_images:
            document.images = patched_images

        if document.text_chunks:
            self.store.add_text_chunks(
                document.text_chunks,
                self.embedding_client.embed_texts([chunk.embed_text for chunk in document.text_chunks]),
            )

        if document.equation_chunks:
            self.store.add_equations(
                document.equation_chunks,
                self.embedding_client.embed_texts(
                    [chunk.embed_text for chunk in document.equation_chunks]
                ),
            )

        if document.table_chunks:
            self.store.add_tables(
                document.table_chunks,
                self.embedding_client.embed_texts([chunk.embed_text for chunk in document.table_chunks]),
            )

        if document.images:
            image_embeddings: list[list[float]] = []
            images_to_store: list[ExtractedImage] = []
            for image in document.images:
                try:
                    embedding = self.embedding_client.embed_file(image.file_path)
                except Exception as exc:
                    print(f"[WARN] Could not embed image {image.file_path}: {exc}")
                    continue
                image_embeddings.append(embedding)
                images_to_store.append(image)

            if images_to_store:
                self.store.add_images(images_to_store, image_embeddings)

        return document

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        content_types: Sequence[str] | None = None,
    ) -> list[SearchResult]:
        query_embedding = self.embedding_client.embed_text(query)
        return self.store.search(query_embedding, limit=limit, content_types=content_types)

    def search_images(self, query: str, limit: int = 5) -> list[SearchResult]:
        return self.search(query, limit=limit, content_types=["image"])
