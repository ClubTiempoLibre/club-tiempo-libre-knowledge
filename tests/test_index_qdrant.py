from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from scripts.index_qdrant import (
    DEFAULT_EMBEDDING_MODEL,
    Chunk,
    build_chunks,
    connect_to_qdrant,
    discover_markdown_files,
    generate_embeddings,
    main,
    parse_markdown_sections,
    qdrant_rest_port,
    required_environment,
)

INDEXED_AT = (
    datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
)


class MarkdownDiscoveryTests(unittest.TestCase):
    def test_excludes_generated_hidden_and_technical_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            included = root / "docs" / "knowledge.md"
            excluded = [
                root / "README.md",
                root / ".github" / "notes.md",
                root / "site" / "generated.md",
                root / "docs" / ".hidden.md",
            ]
            included.parent.mkdir(parents=True)
            included.write_text(
                "# Knowledge\n\nUseful content lives here.", encoding="utf-8"
            )
            for path in excluded:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# Excluded\n\nTechnical content.", encoding="utf-8")

            self.assertEqual(discover_markdown_files(root), [included])


class QdrantConfigurationTests(unittest.TestCase):
    def test_derives_standard_rest_port_from_url(self) -> None:
        self.assertEqual(qdrant_rest_port("https://qdrant.example.com"), 443)
        self.assertEqual(qdrant_rest_port("http://192.0.2.1"), 80)
        self.assertEqual(qdrant_rest_port("http://192.0.2.1:6333"), 6333)

    def test_rejects_missing_required_environment(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            self.assertRaisesRegex(RuntimeError, "QDRANT_URL is not defined"),
        ):
            required_environment("QDRANT_URL")

    def test_reports_unreachable_qdrant(self) -> None:
        class UnreachableQdrantClient:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs

            def get_collections(self) -> None:
                raise TimeoutError("connection timed out")

        fake_module = types.ModuleType("qdrant_client")
        fake_module.__dict__["QdrantClient"] = UnreachableQdrantClient
        with (
            patch.dict(sys.modules, {"qdrant_client": fake_module}),
            self.assertRaisesRegex(RuntimeError, "Qdrant is not reachable"),
        ):
            connect_to_qdrant("https://qdrant.example.com", "test-api-key", 1.0)


class EmbeddingTests(unittest.TestCase):
    def test_generates_normalized_bge_m3_vectors_with_expected_dimension(self) -> None:
        class FakeVector(list[float]):
            def tolist(self) -> list[float]:
                return list(self)

        class FakeSentenceTransformer:
            def __init__(self, model_name: str) -> None:
                self.model_name = model_name

            def get_sentence_embedding_dimension(self) -> int:
                return 1024

            def encode(self, texts: list[str], **kwargs: object) -> list[FakeVector]:
                self.encode_arguments = kwargs
                if kwargs.get("normalize_embeddings") is not True:
                    raise AssertionError("embeddings must be normalized")
                return [FakeVector([1.0] + [0.0] * 1023) for _ in texts]

        fake_module = types.ModuleType("sentence_transformers")
        fake_module.__dict__["SentenceTransformer"] = FakeSentenceTransformer
        chunks = [Chunk(chunk_id="id", content="Texto de prueba", payload={})]

        with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
            vectors, vector_size = generate_embeddings(
                chunks, DEFAULT_EMBEDDING_MODEL, batch_size=2
            )

        self.assertEqual(vector_size, 1024)
        self.assertEqual(len(vectors), 1)
        self.assertEqual(len(vectors[0]), 1024)


class IndexerFailureTests(unittest.TestCase):
    def test_fails_when_no_markdown_is_found(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary_directory,
            self.assertLogs("qdrant-indexer", level="ERROR") as captured_logs,
        ):
            exit_code = main(["--repo-root", temporary_directory, "--dry-run"])

        self.assertEqual(exit_code, 1)
        self.assertIn(
            "No valid Markdown documents were found", "\n".join(captured_logs.output)
        )


class MarkdownChunkingTests(unittest.TestCase):
    def test_keeps_heading_hierarchy_and_ignores_headings_in_code_fences(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            document = root / "docs" / "guide.md"
            document.parent.mkdir(parents=True)
            document.write_text(
                """# Campamentos

## Inscripciones

Información general suficientemente extensa para indexar.

```markdown
### Esto no es una subsección
```

### Pagos

Los pagos se completan antes de la fecha indicada.
""",
                encoding="utf-8",
            )

            sections = parse_markdown_sections(document)
            chunks = build_chunks([document], root, INDEXED_AT, max_words=1800)

            self.assertEqual(
                [section.heading_path for section in sections],
                [
                    "Campamentos > Inscripciones",
                    "Campamentos > Inscripciones > Pagos",
                ],
            )
            self.assertEqual(chunks[1].payload["section"], "Inscripciones")
            self.assertEqual(chunks[1].payload["subsection"], "Pagos")
            self.assertTrue(
                chunks[1].content.startswith(
                    "# Campamentos\n\n## Inscripciones\n\n### Pagos"
                )
            )
            self.assertEqual(
                set(chunks[1].payload),
                {
                    "content",
                    "source_path",
                    "file_name",
                    "title",
                    "section",
                    "subsection",
                    "heading_path",
                    "repo",
                    "indexed_at",
                    "chunk_id",
                },
            )
            self.assertEqual(chunks[1].payload["chunk_id"], chunks[1].chunk_id)

    def test_splits_large_sections_and_keeps_stable_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            document = root / "docs" / "large.md"
            document.parent.mkdir(parents=True)
            paragraphs = [
                " ".join(f"palabra{index}_{word}" for word in range(40))
                for index in range(5)
            ]
            document.write_text(
                "# Documento\n\n## Sección\n\n" + "\n\n".join(paragraphs),
                encoding="utf-8",
            )

            first_run = build_chunks([document], root, INDEXED_AT, max_words=100)
            second_run = build_chunks(
                [document], root, "2026-02-01T00:00:00Z", max_words=100
            )

            self.assertGreater(len(first_run), 1)
            self.assertEqual(
                [chunk.chunk_id for chunk in first_run],
                [chunk.chunk_id for chunk in second_run],
            )
            self.assertTrue(
                all(
                    chunk.payload["heading_path"] == "Documento > Sección"
                    for chunk in first_run
                )
            )
            self.assertTrue(
                all(
                    chunk.content.startswith("# Documento\n\n## Sección")
                    for chunk in first_run
                )
            )


if __name__ == "__main__":
    unittest.main()
