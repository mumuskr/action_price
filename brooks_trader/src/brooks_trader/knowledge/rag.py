"""Local deterministic embeddings, FAISS indexing, and source-backed retrieval."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import faiss
import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from brooks_trader.models import BookParagraph, KnowledgeChunk

_WORD = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*|[\u3400-\u9fff]")
_INDEX_FILE = "knowledge.faiss"
_CHUNKS_FILE = "chunks.jsonl"
_MANIFEST_FILE = "manifest.json"


class ChunkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    target_characters: int = Field(ge=100)
    maximum_characters: int = Field(ge=100)
    overlap_paragraphs: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_sizes(self) -> ChunkConfig:
        if self.maximum_characters < self.target_characters:
            raise ValueError("maximum_characters cannot be below target_characters")
        return self


class EmbeddingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dimension: int = Field(ge=128)
    word_ngram_max: int = Field(ge=1, le=3)
    character_ngram: int = Field(ge=2, le=5)


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    default_top_k: int = Field(ge=1)


class KnowledgeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk: ChunkConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig


class RetrievalResult(BaseModel):
    """One matched chunk with a cosine-like inner-product score."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    score: float
    chunk: KnowledgeChunk


def load_knowledge_config(path: str | Path) -> KnowledgeConfig:
    source = Path(path).expanduser()
    with source.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError(f"knowledge configuration must be a mapping: {source}")
    return KnowledgeConfig.model_validate(raw)


def chunk_paragraphs(
    paragraphs: Sequence[BookParagraph],
    *,
    config: ChunkConfig,
) -> list[KnowledgeChunk]:
    """Build bounded chunks without crossing a book/chapter/section boundary."""
    if not paragraphs:
        return []
    chunks: list[KnowledgeChunk] = []
    group: list[BookParagraph] = []
    current_key: tuple[str, str, str | None, str] | None = None
    for paragraph in paragraphs:
        key = (
            paragraph.book,
            paragraph.chapter,
            paragraph.section,
            paragraph.source_file,
        )
        if current_key is not None and key != current_key:
            chunks.extend(_chunk_group(group, config=config))
            group = []
        group.append(paragraph)
        current_key = key
    chunks.extend(_chunk_group(group, config=config))
    return chunks


class HashingEmbedder:
    """Reproducible local feature hashing for lexical and multilingual retrieval.

    This is a retrieval embedding implementation, not a trained semantic model. It keeps
    licensed book text local and makes the Phase 8 index fully reproducible.
    """

    name = "brooks-hashing-v1"

    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.config.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            counts = Counter(_features(text, self.config))
            for feature, count in counts.items():
                digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
                index = int.from_bytes(digest, "little") % self.config.dimension
                matrix[row, index] += 1.0 + np.log1p(count)
        faiss.normalize_L2(matrix)
        return matrix


class FaissKnowledgeBase:
    """Persistent local vector store with metadata kept in validated JSONL."""

    def __init__(
        self,
        *,
        index: faiss.Index,
        chunks: Sequence[KnowledgeChunk],
        embedder: HashingEmbedder,
        default_top_k: int,
    ) -> None:
        if index.ntotal != len(chunks):
            raise ValueError("FAISS index and chunk metadata counts differ")
        if default_top_k < 1:
            raise ValueError("default_top_k must be at least one")
        self.index = index
        self.chunks = tuple(chunks)
        self.embedder = embedder
        self.default_top_k = default_top_k

    @classmethod
    def build(
        cls,
        chunks: Sequence[KnowledgeChunk],
        *,
        config: KnowledgeConfig,
    ) -> FaissKnowledgeBase:
        if not chunks:
            raise ValueError("at least one knowledge chunk is required")
        embedder = HashingEmbedder(config.embedding)
        vectors = embedder.embed([chunk.text for chunk in chunks])
        index = faiss.IndexFlatIP(config.embedding.dimension)
        index.add(vectors)
        return cls(
            index=index,
            chunks=chunks,
            embedder=embedder,
            default_top_k=config.retrieval.default_top_k,
        )

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        book: str | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve ranked source chunks with an optional exact book filter."""
        normalized_query = " ".join(query.split())
        if not normalized_query:
            raise ValueError("query cannot be empty")
        limit = top_k or self.default_top_k
        if limit < 1:
            raise ValueError("top_k must be at least one")
        candidates = len(self.chunks) if book is not None else min(limit, len(self.chunks))
        scores, indices = self.index.search(self.embedder.embed([normalized_query]), candidates)
        results: list[RetrievalResult] = []
        for score, index in zip(scores[0], indices[0], strict=True):
            if index < 0:
                continue
            chunk = self.chunks[int(index)]
            if book is not None and chunk.book != book:
                continue
            results.append(RetrievalResult(score=float(score), chunk=chunk))
            if len(results) == limit:
                break
        return results

    def save(self, directory: str | Path) -> Path:
        """Persist index, chunks, and a compatibility manifest; manifest commits last."""
        destination = Path(directory).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        index_temp = destination / f"{_INDEX_FILE}.tmp"
        chunks_temp = destination / f"{_CHUNKS_FILE}.tmp"
        manifest_temp = destination / f"{_MANIFEST_FILE}.tmp"
        faiss.write_index(self.index, str(index_temp))
        with chunks_temp.open("w", encoding="utf-8") as stream:
            for chunk in self.chunks:
                stream.write(json.dumps(chunk.model_dump(mode="json"), ensure_ascii=False))
                stream.write("\n")
        manifest = {
            "schema_version": "knowledge-index-v1",
            "embedder": self.embedder.name,
            "embedding": self.embedder.config.model_dump(mode="json"),
            "chunk_count": len(self.chunks),
            "default_top_k": self.default_top_k,
            "chunk_digest": _chunk_digest(self.chunks),
            "index_file_sha256": _file_digest(index_temp),
            "chunks_file_sha256": _file_digest(chunks_temp),
        }
        manifest_temp.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        index_temp.replace(destination / _INDEX_FILE)
        chunks_temp.replace(destination / _CHUNKS_FILE)
        manifest_temp.replace(destination / _MANIFEST_FILE)
        return destination

    @classmethod
    def load(cls, directory: str | Path) -> FaissKnowledgeBase:
        """Load and validate a complete on-disk knowledge index."""
        source = Path(directory).expanduser()
        manifest_path = source / _MANIFEST_FILE
        if not manifest_path.is_file():
            raise FileNotFoundError(f"knowledge index manifest does not exist: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != "knowledge-index-v1":
            raise ValueError("unsupported knowledge index schema")
        if manifest.get("embedder") != HashingEmbedder.name:
            raise ValueError("unsupported knowledge index embedder")
        index_path = source / _INDEX_FILE
        chunks_path = source / _CHUNKS_FILE
        if _file_digest(index_path) != manifest.get("index_file_sha256"):
            raise ValueError("FAISS index failed integrity check")
        if _file_digest(chunks_path) != manifest.get("chunks_file_sha256"):
            raise ValueError("knowledge chunk file failed integrity check")
        chunks = _read_chunks(chunks_path)
        if len(chunks) != manifest.get("chunk_count"):
            raise ValueError("knowledge index chunk count does not match manifest")
        if _chunk_digest(chunks) != manifest.get("chunk_digest"):
            raise ValueError("knowledge index chunk metadata failed integrity check")
        embedding = EmbeddingConfig.model_validate(manifest.get("embedding"))
        index = faiss.read_index(str(index_path))
        if index.d != embedding.dimension:
            raise ValueError("FAISS dimension does not match embedding manifest")
        return cls(
            index=index,
            chunks=chunks,
            embedder=HashingEmbedder(embedding),
            default_top_k=int(manifest["default_top_k"]),
        )


def _chunk_group(
    paragraphs: Sequence[BookParagraph],
    *,
    config: ChunkConfig,
) -> list[KnowledgeChunk]:
    expanded: list[BookParagraph] = []
    for paragraph in paragraphs:
        expanded.extend(
            paragraph.model_copy(update={"text": segment})
            for segment in _split_text(paragraph.text, config.maximum_characters)
        )
    chunks: list[KnowledgeChunk] = []
    start = 0
    while start < len(expanded):
        selected: list[BookParagraph] = []
        length = 0
        cursor = start
        while cursor < len(expanded):
            paragraph = expanded[cursor]
            additional = len(paragraph.text) + (2 if selected else 0)
            if selected and length + additional > config.maximum_characters:
                break
            selected.append(paragraph)
            length += additional
            cursor += 1
            if length >= config.target_characters:
                break
        chunks.append(_build_chunk(selected))
        if cursor >= len(expanded):
            break
        next_start = cursor - min(config.overlap_paragraphs, max(0, len(selected) - 1))
        start = max(start + 1, next_start)
    return chunks


def _split_text(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    words = text.split()
    if not words:
        return [text[index : index + limit] for index in range(0, len(text), limit)]
    segments: list[str] = []
    current: list[str] = []
    length = 0
    for word in words:
        if len(word) > limit:
            if current:
                segments.append(" ".join(current))
                current = []
                length = 0
            segments.extend(word[index : index + limit] for index in range(0, len(word), limit))
            continue
        additional = len(word) + (1 if current else 0)
        if current and length + additional > limit:
            segments.append(" ".join(current))
            current = [word]
            length = len(word)
        else:
            current.append(word)
            length += additional
    if current:
        segments.append(" ".join(current))
    return segments


def _build_chunk(paragraphs: Sequence[BookParagraph]) -> KnowledgeChunk:
    first = paragraphs[0]
    last = paragraphs[-1]
    text = "\n\n".join(paragraph.text for paragraph in paragraphs)
    source_documents = list(dict.fromkeys(paragraph.source_document for paragraph in paragraphs))
    reference = (
        f"{first.book} | {first.chapter}"
        + (f" | {first.section}" if first.section else "")
        + f" | paragraphs {first.paragraph}-{last.paragraph}"
    )
    identity = "\x1f".join(
        [
            first.book,
            first.chapter,
            first.section or "",
            str(first.paragraph),
            str(last.paragraph),
            text,
        ]
    )
    return KnowledgeChunk(
        chunk_id=hashlib.sha256(identity.encode("utf-8")).hexdigest(),
        book=first.book,
        chapter=first.chapter,
        section=first.section,
        paragraph_start=first.paragraph,
        paragraph_end=last.paragraph,
        text=text,
        source_file=first.source_file,
        source_documents=source_documents,
        source_reference=reference,
    )


def _features(text: str, config: EmbeddingConfig) -> list[str]:
    normalized = " ".join(text.casefold().split())
    words = _WORD.findall(normalized)
    features: list[str] = []
    for size in range(1, config.word_ngram_max + 1):
        features.extend(
            f"w{size}:{' '.join(words[index:index + size])}"
            for index in range(len(words) - size + 1)
        )
    compact = "".join(character for character in normalized if not character.isspace())
    size = config.character_ngram
    features.extend(
        f"c{size}:{compact[index:index + size]}" for index in range(max(0, len(compact) - size + 1))
    )
    return features


def _read_chunks(path: Path) -> list[KnowledgeChunk]:
    chunks: list[KnowledgeChunk] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                chunks.append(KnowledgeChunk.model_validate_json(line))
            except ValueError as error:
                raise ValueError(f"invalid knowledge chunk at line {line_number}") from error
    return chunks


def _chunk_digest(chunks: Sequence[KnowledgeChunk]) -> str:
    value = "\n".join(chunk.chunk_id for chunk in chunks)
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
