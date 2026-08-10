from __future__ import annotations

import importlib.util
import unittest

from scripts.index_qdrant import (
    COLLECTION_NAME,
    Chunk,
    replace_collection,
)

QDRANT_CLIENT_AVAILABLE = importlib.util.find_spec("qdrant_client") is not None


@unittest.skipUnless(QDRANT_CLIENT_AVAILABLE, "qdrant-client is not installed")
class QdrantUploadTests(unittest.TestCase):
    def test_replaces_collection_and_verifies_count(self) -> None:
        from qdrant_client import QdrantClient

        client = QdrantClient(location=":memory:")
        chunks = [
            Chunk(
                chunk_id="01c6a30a-505c-5c37-a8bc-7d2bbce09bc7",
                content="First chunk",
                payload={
                    "content": "First chunk",
                    "chunk_id": "01c6a30a-505c-5c37-a8bc-7d2bbce09bc7",
                },
            ),
            Chunk(
                chunk_id="6c5f43d1-47bf-562a-8600-f2de1bc4af9a",
                content="Second chunk",
                payload={
                    "content": "Second chunk",
                    "chunk_id": "6c5f43d1-47bf-562a-8600-f2de1bc4af9a",
                },
            ),
        ]
        vectors = [[1.0, 0.0], [0.0, 1.0]]

        first_upload = replace_collection(client, chunks, vectors, vector_size=2)
        second_upload = replace_collection(
            client, chunks[:1], vectors[:1], vector_size=2
        )

        self.assertEqual(first_upload, 2)
        self.assertEqual(second_upload, 1)
        self.assertEqual(
            client.count(collection_name=COLLECTION_NAME, exact=True).count, 1
        )
        stored = client.retrieve(
            collection_name=COLLECTION_NAME, ids=[chunks[0].chunk_id], with_payload=True
        )
        self.assertEqual(stored[0].payload["content"], "First chunk")
        client.close()


if __name__ == "__main__":
    unittest.main()
