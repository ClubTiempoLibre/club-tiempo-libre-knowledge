"""Build a Markdown knowledge index and replace a Qdrant collection."""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import uuid
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

COLLECTION_NAME = "club-tiempo-libre-knowledge"
REPOSITORY_NAME = "club-tiempo-libre-knowledge"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_MAX_CHUNK_WORDS = 1800
DEFAULT_BATCH_SIZE = 2
DEFAULT_UPLOAD_BATCH_SIZE = 64
DEFAULT_QDRANT_TIMEOUT_SECONDS = 60.0
EXPECTED_BGE_M3_DIMENSION = 1024
MIN_CONTENT_CHARACTERS = 20

EXCLUDED_DIRECTORY_NAMES = {
    ".git",
    ".github",
    ".venv",
    "__pycache__",
    "_site",
    "build",
    "dist",
    "node_modules",
    "site",
    "venv",
}
EXCLUDED_FILE_PATHS = {
    PurePosixPath("README.md"),
    PurePosixPath("docs/admin/rag-indexing.md"),
}
MARKDOWN_SUFFIXES = {".md", ".mdx"}

HEADING_RE = re.compile(r"^(#{1,3})[ \t]+(.+?)[ \t]*$")
FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
WORD_RE = re.compile(r"\b[\wÀ-ÿ]+\b", re.UNICODE)
TRAILING_ATTRIBUTE_RE = re.compile(r"\s+\{[^{}]+\}\s*$")
TRAILING_HASH_RE = re.compile(r"\s+#+\s*$")

LOGGER = logging.getLogger("qdrant-indexer")


@dataclass(frozen=True)
class MarkdownSection:
    document_title: str
    heading_one: str
    section: str
    subsection: str
    heading_path: str
    occurrence: int
    body: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    content: str
    payload: dict[str, object]


def discover_markdown_files(repo_root: Path) -> list[Path]:
    """Return indexable Markdown files in deterministic order."""
    files: list[Path] = []
    for current_root, directory_names, file_names in os.walk(
        repo_root, topdown=True, followlinks=False
    ):
        directory_names[:] = sorted(
            directory_name
            for directory_name in directory_names
            if not directory_name.startswith(".")
            and directory_name.casefold() not in EXCLUDED_DIRECTORY_NAMES
        )
        current_path = Path(current_root)
        for file_name in sorted(file_names):
            if file_name.startswith("."):
                continue
            path = current_path / file_name
            if path.is_symlink() or path.suffix.casefold() not in MARKDOWN_SUFFIXES:
                continue
            relative_path = path.relative_to(repo_root)
            if PurePosixPath(relative_path.as_posix()) in EXCLUDED_FILE_PATHS:
                continue
            files.append(path)

    return sorted(files, key=lambda item: item.relative_to(repo_root).as_posix())


def strip_front_matter(lines: list[str]) -> list[str]:
    if not lines or lines[0].strip() != "---":
        return lines

    for index, line in enumerate(lines[1:], start=1):
        if line.strip() in {"---", "..."}:
            return lines[index + 1 :]
    return lines


def clean_heading(raw_heading: str) -> str:
    heading = TRAILING_HASH_RE.sub("", raw_heading).strip()
    return TRAILING_ATTRIBUTE_RE.sub("", heading).strip()


def first_level_one_heading(lines: Sequence[str]) -> str | None:
    fence_character = ""
    fence_length = 0

    for line in lines:
        fence_match = FENCE_RE.match(line)
        if fence_match:
            marker = fence_match.group(1)
            if not fence_character:
                fence_character = marker[0]
                fence_length = len(marker)
            elif marker[0] == fence_character and len(marker) >= fence_length:
                fence_character = ""
                fence_length = 0
            continue

        if fence_character:
            continue
        heading_match = HEADING_RE.match(line)
        if heading_match and len(heading_match.group(1)) == 1:
            return clean_heading(heading_match.group(2))

    return None


def meaningful_content(body: str) -> bool:
    visible_text = re.sub(r"[`*_>#|\[\](){}!\-]", "", body)
    return len(visible_text.strip()) >= MIN_CONTENT_CHARACTERS


def parse_markdown_sections(path: Path) -> list[MarkdownSection]:
    lines = strip_front_matter(path.read_text(encoding="utf-8").splitlines())
    first_heading = first_level_one_heading(lines)
    document_title = first_heading or path.stem.replace("-", " ").strip().title()

    if first_heading is None:
        LOGGER.warning(
            "No level-one heading found in %s; using file name as title", path
        )

    current_heading_one = document_title
    current_section = ""
    current_subsection = ""
    body_lines: list[str] = []
    parsed_sections: list[MarkdownSection] = []
    occurrences: defaultdict[str, int] = defaultdict(int)
    fence_character = ""
    fence_length = 0

    def flush_body() -> None:
        body = "\n".join(body_lines).strip()
        body_lines.clear()
        if not meaningful_content(body):
            return

        hierarchy = [
            heading
            for heading in (current_heading_one, current_section, current_subsection)
            if heading
        ]
        heading_path = " > ".join(hierarchy)
        occurrence = occurrences[heading_path]
        occurrences[heading_path] += 1
        parsed_sections.append(
            MarkdownSection(
                document_title=document_title,
                heading_one=current_heading_one,
                section=current_section,
                subsection=current_subsection,
                heading_path=heading_path,
                occurrence=occurrence,
                body=body,
            )
        )

    for line in lines:
        fence_match = FENCE_RE.match(line)
        if fence_match:
            marker = fence_match.group(1)
            if not fence_character:
                fence_character = marker[0]
                fence_length = len(marker)
            elif marker[0] == fence_character and len(marker) >= fence_length:
                fence_character = ""
                fence_length = 0
            body_lines.append(line)
            continue

        if fence_character:
            body_lines.append(line)
            continue

        heading_match = HEADING_RE.match(line)
        if not heading_match:
            body_lines.append(line)
            continue

        flush_body()
        level = len(heading_match.group(1))
        heading = clean_heading(heading_match.group(2))
        if level == 1:
            current_heading_one = heading
            current_section = ""
            current_subsection = ""
        elif level == 2:
            current_section = heading
            current_subsection = ""
        else:
            current_subsection = heading

    flush_body()
    return parsed_sections


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def split_words(text: str, max_words: int) -> list[str]:
    words = text.split()
    return [
        " ".join(words[index : index + max_words])
        for index in range(0, len(words), max_words)
    ]


def split_oversized_block(block: str, max_words: int) -> list[str]:
    lines = block.splitlines()
    if len(lines) == 1:
        sentences = re.split(r"(?<=[.!?])\s+", block)
        if len(sentences) == 1:
            return split_words(block, max_words)
        units = sentences
        separator = " "
    else:
        units = lines
        separator = "\n"

    parts: list[str] = []
    current: list[str] = []
    current_words = 0
    for unit in units:
        unit_words = word_count(unit)
        if unit_words > max_words:
            if current:
                parts.append(separator.join(current).strip())
                current = []
                current_words = 0
            parts.extend(split_words(unit, max_words))
            continue
        if current and current_words + unit_words > max_words:
            parts.append(separator.join(current).strip())
            current = []
            current_words = 0
        current.append(unit)
        current_words += unit_words

    if current:
        parts.append(separator.join(current).strip())
    return [part for part in parts if part]


def split_section_body(body: str, max_words: int) -> list[str]:
    if word_count(body) <= max_words:
        return [body.strip()]

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n[ \t]*\n", body)
        if paragraph.strip()
    ]
    parts: list[str] = []
    current: list[str] = []
    current_words = 0

    for paragraph in paragraphs:
        paragraph_words = word_count(paragraph)
        if paragraph_words > max_words:
            if current:
                parts.append("\n\n".join(current).strip())
                current = []
                current_words = 0
            parts.extend(split_oversized_block(paragraph, max_words))
            continue
        if current and current_words + paragraph_words > max_words:
            parts.append("\n\n".join(current).strip())
            current = []
            current_words = 0
        current.append(paragraph)
        current_words += paragraph_words

    if current:
        parts.append("\n\n".join(current).strip())
    return [part for part in parts if part]


def render_context(section: MarkdownSection) -> str:
    headings = [(1, section.heading_one), (2, section.section), (3, section.subsection)]
    return "\n\n".join(
        f"{'#' * level} {heading}" for level, heading in headings if heading
    )


def build_chunks(
    markdown_files: Sequence[Path],
    repo_root: Path,
    indexed_at: str,
    max_words: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []

    for path in markdown_files:
        source_path = path.relative_to(repo_root).as_posix()
        for section in parse_markdown_sections(path):
            for part_index, body_part in enumerate(
                split_section_body(section.body, max_words)
            ):
                stable_key = "\n".join(
                    (
                        source_path,
                        section.heading_path,
                        str(section.occurrence),
                        str(part_index),
                    )
                )
                chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, stable_key))
                content = f"{render_context(section)}\n\n{body_part}".strip()
                payload: dict[str, object] = {
                    "content": content,
                    "source_path": source_path,
                    "file_name": path.name,
                    "title": section.document_title,
                    "section": section.section,
                    "subsection": section.subsection,
                    "heading_path": section.heading_path,
                    "repo": REPOSITORY_NAME,
                    "indexed_at": indexed_at,
                    "chunk_id": chunk_id,
                }
                chunks.append(
                    Chunk(chunk_id=chunk_id, content=content, payload=payload)
                )

    return chunks


def required_environment(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Required environment variable {name} is not defined")
    return value


def positive_integer_environment(name: str, default: int) -> int:
    raw_value = os.environ.get(name, str(default))
    try:
        value = int(raw_value)
    except ValueError as error:
        raise RuntimeError(f"{name} must be an integer") from error
    if value <= 0:
        raise RuntimeError(f"{name} must be greater than zero")
    return value


def positive_float_environment(name: str, default: float) -> float:
    raw_value = os.environ.get(name, str(default))
    try:
        value = float(raw_value)
    except ValueError as error:
        raise RuntimeError(f"{name} must be a number") from error
    if value <= 0:
        raise RuntimeError(f"{name} must be greater than zero")
    return value


def validate_qdrant_url(url: str) -> None:
    parsed_url = urlparse(url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise RuntimeError("QDRANT_URL must be an absolute HTTP or HTTPS URL")
    if parsed_url.username or parsed_url.password:
        raise RuntimeError("QDRANT_URL must not contain credentials")


def qdrant_rest_port(url: str) -> int:
    parsed_url = urlparse(url)
    if parsed_url.port is not None:
        return parsed_url.port
    return 443 if parsed_url.scheme == "https" else 80


def connect_to_qdrant(url: str, api_key: str, timeout: float):
    try:
        from qdrant_client import QdrantClient
    except ImportError as error:
        raise RuntimeError(
            "qdrant-client is not installed; run pip install -r requirements.txt"
        ) from error

    client = QdrantClient(
        url=url,
        port=qdrant_rest_port(url),
        api_key=api_key,
        timeout=timeout,
        check_compatibility=False,
    )
    try:
        client.get_collections()
    except Exception as error:
        raise RuntimeError(
            "Qdrant is not reachable using the configured QDRANT_URL and API key"
        ) from error
    return client


def generate_embeddings(
    chunks: Sequence[Chunk], model_name: str, batch_size: int
) -> tuple[list[list[float]], int]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "sentence-transformers is not installed; run pip install -r requirements.txt"
        ) from error

    LOGGER.info("Loading embedding model: %s", model_name)
    model = SentenceTransformer(model_name)
    vector_size = model.get_sentence_embedding_dimension()
    if not vector_size:
        raise RuntimeError(
            f"Could not determine vector size for embedding model {model_name}"
        )
    if (
        model_name == DEFAULT_EMBEDDING_MODEL
        and vector_size != EXPECTED_BGE_M3_DIMENSION
    ):
        raise RuntimeError(
            f"Unexpected vector size for {DEFAULT_EMBEDDING_MODEL}: "
            f"expected {EXPECTED_BGE_M3_DIMENSION}, got {vector_size}"
        )

    encoded = model.encode(
        [chunk.content for chunk in chunks],
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return [vector.tolist() for vector in encoded], int(vector_size)


def replace_collection(
    client,
    chunks: Sequence[Chunk],
    vectors: Sequence[Sequence[float]],
    vector_size: int,
) -> int:
    from qdrant_client.models import Distance, PointStruct, VectorParams

    if client.collection_exists(COLLECTION_NAME):
        LOGGER.info("Deleting existing collection: %s", COLLECTION_NAME)
        client.delete_collection(collection_name=COLLECTION_NAME)

    LOGGER.info(
        "Creating collection %s with %d-dimensional cosine vectors",
        COLLECTION_NAME,
        vector_size,
    )
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    upload_batch_size = positive_integer_environment(
        "QDRANT_UPLOAD_BATCH_SIZE", DEFAULT_UPLOAD_BATCH_SIZE
    )
    uploaded = 0
    for batch_start in range(0, len(chunks), upload_batch_size):
        batch_chunks = chunks[batch_start : batch_start + upload_batch_size]
        batch_vectors = vectors[batch_start : batch_start + upload_batch_size]
        points = [
            PointStruct(id=chunk.chunk_id, vector=vector, payload=chunk.payload)
            for chunk, vector in zip(batch_chunks, batch_vectors, strict=True)
        ]
        client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
        uploaded += len(points)
        LOGGER.info("Uploaded %d/%d vectors", uploaded, len(chunks))

    stored_count = client.count(collection_name=COLLECTION_NAME, exact=True).count
    if stored_count != len(chunks):
        raise RuntimeError(
            f"Qdrant verification failed: expected {len(chunks)} vectors, found {stored_count}"
        )
    return uploaded


def positive_integer(value: str) -> int:
    parsed_value = int(value)
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed_value


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    default_repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=default_repo_root)
    parser.add_argument(
        "--max-words", type=positive_integer, default=DEFAULT_MAX_CHUNK_WORDS
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover and chunk Markdown without loading the model or contacting Qdrant",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    arguments = parse_arguments(argv)
    repo_root = arguments.repo_root.resolve()

    try:
        if not repo_root.is_dir():
            raise RuntimeError(f"Repository root does not exist: {repo_root}")

        markdown_files = discover_markdown_files(repo_root)
        LOGGER.info("Markdown documents found: %d", len(markdown_files))
        if not markdown_files:
            raise RuntimeError(
                "No valid Markdown documents were found; refusing to create an empty index"
            )

        indexed_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        chunks = build_chunks(
            markdown_files, repo_root, indexed_at, arguments.max_words
        )
        LOGGER.info("Chunks generated: %d", len(chunks))
        if not chunks:
            raise RuntimeError(
                "Markdown documents were found, but none contained indexable content"
            )

        if arguments.dry_run:
            LOGGER.info("Dry run complete; Qdrant was not modified")
            return 0

        qdrant_url = required_environment("QDRANT_URL")
        qdrant_api_key = required_environment("QDRANT_API_KEY")
        validate_qdrant_url(qdrant_url)
        timeout = positive_float_environment(
            "QDRANT_TIMEOUT_SECONDS", DEFAULT_QDRANT_TIMEOUT_SECONDS
        )
        client = connect_to_qdrant(qdrant_url, qdrant_api_key, timeout)
        LOGGER.info("Qdrant connectivity check succeeded")

        model_name = (
            os.environ.get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL).strip()
            or DEFAULT_EMBEDDING_MODEL
        )
        batch_size = positive_integer_environment(
            "EMBEDDING_BATCH_SIZE", DEFAULT_BATCH_SIZE
        )
        vectors, vector_size = generate_embeddings(chunks, model_name, batch_size)
        LOGGER.info(
            "Embeddings generated: %d vectors of size %d", len(vectors), vector_size
        )

        uploaded = replace_collection(client, chunks, vectors, vector_size)
        LOGGER.info(
            "Indexing complete: %d vectors uploaded to %s", uploaded, COLLECTION_NAME
        )
        client.close()
        return 0
    except Exception as error:
        LOGGER.error("Indexing failed: %s", error)
        return 1


if __name__ == "__main__":
    sys.exit(main())
