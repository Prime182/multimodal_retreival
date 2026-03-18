from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TextChunk:
    chunk_id: str
    source_path: str
    text: str
    token_count: int
    page_start: int
    page_end: int
    caption: str | None = None

    def metadata(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["kind"] = "text"
        return payload


@dataclass(slots=True)
class ExtractedImage:
    image_id: str
    source_path: str
    file_path: str
    page_number: int
    mime_type: str
    width: int | None = None
    height: int | None = None
    caption: str | None = None
    image_url: str | None = None

    def metadata(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["kind"] = "image"
        payload["imageUrl"] = self.image_url
        return payload


@dataclass(slots=True)
class IngestedDocument:
    document_id: str
    source_path: str
    text_chunks: list[TextChunk] = field(default_factory=list)
    images: list[ExtractedImage] = field(default_factory=list)


@dataclass(slots=True)
class SearchResult:
    item_id: str
    distance: float
    metadata: dict[str, Any]


def build_document_id(path: str | Path) -> str:
    return Path(path).stem
