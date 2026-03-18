"""ChromaDB persistence helpers for multimodal publisher articles."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import chromadb
except ImportError:  # pragma: no cover - dependency may be installed later.
    chromadb = None  # type: ignore[assignment]

from .types import EquationChunk, ExtractedImage, SearchResult, TableChunk, TextChunk


DEFAULT_COLLECTION_NAME = "publisher_articles"


@dataclass(frozen=True)
class PublisherArticleRecord:
    id: str
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)
    document: str | None = None


def _clean_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    if not metadata:
        return {}

    cleaned: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, Path):
            cleaned[key] = str(value)
        elif isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        else:
            cleaned[key] = str(value)
    return cleaned


class PublisherArticleStore:
    def __init__(
        self,
        *,
        persist_directory: str | Path,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        client: Any | None = None,
    ) -> None:
        if chromadb is None and client is None:
            raise ImportError(
                "chromadb is required to create PublisherArticleStore; install it or pass an existing client."
            )
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.client = client or chromadb.PersistentClient(path=str(self.persist_directory))
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_many(self, records: Iterable[PublisherArticleRecord]) -> None:
        ids: list[str] = []
        embeddings: list[list[float]] = []
        metadatas: list[dict[str, Any]] = []
        documents: list[str] = []

        for record in records:
            ids.append(record.id)
            embeddings.append(record.embedding)
            metadatas.append(_clean_metadata(record.metadata))
            documents.append(record.document or "")

        if not ids:
            return

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )

    def add_text_chunks(
        self,
        chunks: Iterable[TextChunk],
        embeddings: Iterable[list[float]],
    ) -> None:
        records = [
            PublisherArticleRecord(
                id=chunk.chunk_id,
                embedding=embedding,
                metadata=chunk.metadata(),
                document=chunk.embed_text,
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        self.add_many(records)

    def add_equations(
        self,
        equations: Iterable[EquationChunk],
        embeddings: Iterable[list[float]],
    ) -> None:
        records = [
            PublisherArticleRecord(
                id=equation.chunk_id,
                embedding=embedding,
                metadata=equation.metadata(),
                document=equation.embed_text,
            )
            for equation, embedding in zip(equations, embeddings, strict=True)
        ]
        self.add_many(records)

    def add_tables(
        self,
        tables: Iterable[TableChunk],
        embeddings: Iterable[list[float]],
    ) -> None:
        records = [
            PublisherArticleRecord(
                id=table.chunk_id,
                embedding=embedding,
                metadata=table.metadata(),
                document=table.embed_text,
            )
            for table, embedding in zip(tables, embeddings, strict=True)
        ]
        self.add_many(records)

    def add_images(
        self,
        images: Iterable[ExtractedImage],
        embeddings: Iterable[list[float]],
    ) -> None:
        records = [
            PublisherArticleRecord(
                id=image.image_id,
                embedding=embedding,
                metadata=image.metadata(),
                document=image.caption,
            )
            for image, embedding in zip(images, embeddings, strict=True)
        ]
        self.add_many(records)

    def search(
        self,
        query_embedding: list[float],
        *,
        limit: int = 5,
        content_types: Sequence[str] | None = None,
    ) -> list[SearchResult]:
        include = ["metadatas", "distances"]
        where: dict[str, Any] | None = None
        if content_types:
            where = {"kind": {"$in": list(content_types)}}

        response = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=where,
            include=include,
        )

        ids = response.get("ids", [[]])
        metadatas = response.get("metadatas", [[]])
        distances = response.get("distances", [[]])

        results: list[SearchResult] = []
        for item_id, metadata, distance in zip(
            ids[0] if ids else [],
            metadatas[0] if metadatas else [],
            distances[0] if distances else [],
        ):
            if not metadata:
                continue
            results.append(
                SearchResult(
                    item_id=item_id,
                    distance=float(distance),
                    metadata=dict(metadata),
                )
            )
        return results

    def search_images(
        self,
        query_embedding: list[float],
        *,
        limit: int = 5,
    ) -> list[SearchResult]:
        return self.search(query_embedding, limit=limit, content_types=["image"])

    def collection_count(self) -> int:
        return self.collection.count()


def create_store(
    persist_directory: str | Path,
    *,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> PublisherArticleStore:
    return PublisherArticleStore(
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
