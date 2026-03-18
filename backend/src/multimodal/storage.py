"""ChromaDB persistence helpers for multimodal publisher articles."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    import chromadb
except ImportError:  # pragma: no cover - dependency may be installed later
    chromadb = None  # type: ignore[assignment]

from .types import ExtractedImage, SearchResult, TextChunk


DEFAULT_COLLECTION_NAME = "publisher_articles"


@dataclass(frozen=True)
class PublisherArticleRecord:
    """Normalized payload stored in ChromaDB."""

    id: str
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)
    document: str | None = None


def _clean_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """Keep metadata JSON-serializable and Chroma-friendly."""

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
    """Persistence wrapper around a Chroma collection.

    Embeddings are only numeric vectors; the UI depends on metadata like
    `imageUrl`, `file_path`, and `caption` to render the actual asset later.
    """

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

    def add(
        self,
        *,
        id: str,
        embedding: list[float],
        metadata: Mapping[str, Any] | None = None,
        document: str | None = None,
    ) -> None:
        """Add a single record to the collection."""

        record = PublisherArticleRecord(
            id=id,
            embedding=embedding,
            metadata=_clean_metadata(metadata),
            document=document,
        )
        self.collection.add(
            ids=[record.id],
            embeddings=[record.embedding],
            **(
                {"metadatas": [record.metadata]}
                if record.metadata
                else {}
            ),
            **(
                {"documents": [record.document]}
                if record.document is not None
                else {}
            ),
        )

    def add_many(self, records: Iterable[PublisherArticleRecord]) -> None:
        """Add many records in one call."""

        ids: list[str] = []
        embeddings: list[list[float]] = []
        metadatas: list[dict[str, Any]] = []
        documents: list[str] = []

        for record in records:
            ids.append(record.id)
            embeddings.append(record.embedding)
            metadatas.append(_clean_metadata(record.metadata))
            if record.document is not None:
                documents.append(record.document)

        payload: dict[str, Any] = {"ids": ids, "embeddings": embeddings}
        if any(metadatas):
            payload["metadatas"] = metadatas
        if documents and len(documents) == len(ids):
            payload["documents"] = documents

        self.collection.add(**payload)

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
                document=chunk.text,
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        if records:
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
        if records:
            self.add_many(records)

    def search_images(
        self,
        query_embedding: list[float],
        *,
        limit: int = 5,
    ) -> list[SearchResult]:
        response = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where={"kind": "image"},
            include=["metadatas", "distances"],
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
            if not metadata.get("imageUrl") and not metadata.get("file_path"):
                continue
            results.append(
                SearchResult(
                    item_id=item_id,
                    distance=float(distance),
                    metadata=dict(metadata),
                )
            )
        return results

    def collection_count(self) -> int:
        return self.collection.count()


def create_store(
    persist_directory: str | Path,
    *,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> PublisherArticleStore:
    """Factory for the canonical persistent store."""

    return PublisherArticleStore(
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
