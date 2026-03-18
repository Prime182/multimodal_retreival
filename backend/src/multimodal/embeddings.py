"""Helpers for multimodal Gemini embeddings.

The `gemini-embedding-2-preview` model maps text, images, and PDFs into the
same vector space, which is what makes cross-media retrieval possible.
"""

from __future__ import annotations

import mimetypes
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover - dependency may be installed later
    genai = None  # type: ignore[assignment]
    genai_types = None  # type: ignore[assignment]


EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2-preview")


@dataclass(frozen=True)
class EmbeddedFile:
    """Normalized payload for embedding a local file."""

    path: Path
    mime_type: str
    data: bytes


def detect_mime_type(path: str | Path) -> str:
    """Best-effort MIME detection with sensible fallbacks."""

    file_path = Path(path)
    mime_type, _ = mimetypes.guess_type(str(file_path))
    if mime_type:
        return mime_type

    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return "application/pdf"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".txt":
        return "text/plain"
    return "application/octet-stream"


def read_binary_file(path: str | Path) -> EmbeddedFile:
    """Read a file in binary mode and attach its detected MIME type."""

    file_path = Path(path)
    with file_path.open("rb") as handle:
        data = handle.read()
    return EmbeddedFile(path=file_path, mime_type=detect_mime_type(file_path), data=data)


def _extract_embedding_vector(response: Any) -> list[float]:
    """Extract a flat embedding vector from common google-genai response shapes."""

    if isinstance(response, dict):
        content_embedding = response.get("content_embedding")
        if isinstance(content_embedding, dict):
            values = content_embedding.get("values")
            if values is not None:
                return list(values)
        if "embedding" in response and isinstance(response["embedding"], dict):
            values = response["embedding"].get("values")
            if values is not None:
                return list(values)
        if "embeddings" in response and response["embeddings"]:
            first = response["embeddings"][0]
            if isinstance(first, dict):
                values = first.get("values")
                if values is not None:
                    return list(values)
        if "values" in response:
            return list(response["values"])

    content_embedding = getattr(response, "content_embedding", None)
    if content_embedding is not None:
        values = getattr(content_embedding, "values", None)
        if values is not None:
            return list(values)

    embedding = getattr(response, "embedding", None)
    if embedding is not None:
        values = getattr(embedding, "values", None)
        if values is not None:
            return list(values)

    embeddings = getattr(response, "embeddings", None)
    if embeddings:
        first = embeddings[0]
        values = getattr(first, "values", None)
        if values is not None:
            return list(values)

    values = getattr(response, "values", None)
    if values is not None:
        return list(values)

    raise ValueError("Could not extract embedding vector from response")


def _build_part(data: bytes, mime_type: str) -> Any:
    """Try supported google-genai Part constructors in order of preference."""

    if genai_types is None:
        raise ImportError("google-genai is required for file embedding.")

    if hasattr(genai_types.Part, "from_bytes"):
        try:
            return genai_types.Part.from_bytes(data=data, mime_type=mime_type)
        except Exception:
            pass

    if hasattr(genai_types, "Blob"):
        try:
            return genai_types.Part(
                inline_data=genai_types.Blob(data=data, mime_type=mime_type)
            )
        except Exception:
            pass

    import base64

    return {
        "inline_data": {
            "mime_type": mime_type,
            "data": base64.b64encode(data).decode("utf-8"),
        }
    }


def _with_retry(fn: Any, *, max_attempts: int = 5, base_delay: float = 1.0) -> Any:
    """Retry transient Gemini errors with exponential backoff."""

    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as exc:
            message = str(exc).lower()
            is_retryable = (
                "429" in message
                or "quota" in message
                or "rate" in message
                or "timeout" in message
            )
            if not is_retryable or attempt == max_attempts - 1:
                raise
            delay = base_delay * (2 ** attempt)
            print(
                f"[WARN] Gemini rate-limit hit; retrying in {delay:.1f}s "
                f"(attempt {attempt + 1})"
            )
            time.sleep(delay)

    raise RuntimeError("Unreachable")


def embed_file(
    client: Any,
    path: str | Path,
    *,
    model: str = EMBEDDING_MODEL,
) -> list[float]:
    """Embed a local file using Gemini multimodal embeddings.

    The file is read in binary mode and the MIME type is supplied so Gemini can
    interpret text, images, and PDFs through the same embedding space.
    """

    embedded_file = read_binary_file(path)
    part = _build_part(embedded_file.data, embedded_file.mime_type)

    if genai_types is not None and hasattr(genai_types, "Content"):
        contents: Any = genai_types.Content(parts=[part])
    else:
        import base64

        contents = {
            "parts": [
                {
                    "inline_data": {
                        "mime_type": embedded_file.mime_type,
                        "data": base64.b64encode(embedded_file.data).decode("utf-8"),
                    }
                }
            ]
        }

    response = client.models.embed_content(model=model, contents=contents)
    return _extract_embedding_vector(response)


def embed_text(
    client: Any,
    text: str,
    *,
    model: str = EMBEDDING_MODEL,
) -> list[float]:
    """Embed plain text with the same multimodal model."""

    response = client.models.embed_content(model=model, contents=text)
    return _extract_embedding_vector(response)


def embed_many_files(
    client: Any,
    paths: Sequence[str | Path],
    *,
    model: str = EMBEDDING_MODEL,
) -> list[list[float]]:
    """Convenience helper for batch embedding local files."""

    return [embed_file(client, path, model=model) for path in paths]


class GeminiEmbeddingClient:
    """Thin wrapper around the Google GenAI client for multimodal embeddings."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        client: Any | None = None,
        model: str = EMBEDDING_MODEL,
    ) -> None:
        if client is None:
            if genai is None:
                raise ImportError(
                    "google-genai is required to create GeminiEmbeddingClient."
                )
            if not api_key:
                raise ValueError("api_key is required when no client is provided.")
            client = genai.Client(api_key=api_key)

        self.client = client
        self.model = model

    def embed_file(self, path: str | Path) -> list[float]:
        return _with_retry(lambda: embed_file(self.client, path, model=self.model))

    def embed_text(self, text: str) -> list[float]:
        return _with_retry(lambda: embed_text(self.client, text, model=self.model))

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]

    def embed_files(self, paths: Sequence[str | Path]) -> list[list[float]]:
        return embed_many_files(self.client, paths, model=self.model)
