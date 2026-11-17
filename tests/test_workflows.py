"""Focused checks for semantic search and grounded RAG."""

import types
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from source import api, evaluation, rag, semantic


SEARCH_RESULT = {
    "text": "AWS DMS can migrate databases.",
    "metadata": {"source": "test_doc2.txt", "chunk_number": 0},
}


class WorkflowTests(unittest.TestCase):
    def test_chunking_and_metadata(self):
        self.assertEqual(semantic.chunk_text("a b c d", 3, 1), ["a b c", "c d"])
        chunks = semantic.chunk_documents([{"source": "a.txt", "text": "hello"}])
        self.assertEqual(chunks[0]["chunk_id"], "a.txt-0")

    def test_semantic_search(self):
        model = types.SimpleNamespace(
            encode_query=lambda *_args, **_kwargs: types.SimpleNamespace(
                tolist=lambda: [1.0]
            )
        )
        collection = types.SimpleNamespace(
            count=lambda: 1,
            query=lambda **_kwargs: {
                "ids": [["test_doc2.txt-0"]],
                "documents": [[SEARCH_RESULT["text"]]],
                "metadatas": [[SEARCH_RESULT["metadata"]]],
                "distances": [[0.1]],
            },
        )
        with (
            patch.object(semantic, "get_collection", return_value=collection),
            patch.object(semantic, "SentenceTransformer", return_value=model),
        ):
            self.assertEqual(semantic.semantic_search("migration"), [SEARCH_RESULT])

    def test_index_documents_removes_stale_chunks(self):
        model = types.SimpleNamespace(
            encode_document=lambda *_args, **_kwargs: [
                types.SimpleNamespace(tolist=lambda: [1.0])
            ]
        )
        collection = types.SimpleNamespace(
            get=lambda: {"ids": ["old.txt-0"]},
            upsert=lambda **_kwargs: None,
            delete=lambda **kwargs: setattr(collection, "deleted", kwargs["ids"]),
            count=lambda: 1,
        )
        with (
            patch.object(
                semantic,
                "load_documents",
                return_value=[{"source": "new.txt", "text": "new contents"}],
            ),
            patch.object(semantic, "SentenceTransformer", return_value=model),
            patch.object(semantic, "get_collection", return_value=collection),
        ):
            self.assertEqual(semantic.index_documents(), 1)
        self.assertEqual(collection.deleted, ["old.txt-0"])

    def test_grounded_rag_answer(self):
        response = types.SimpleNamespace(
            message=types.SimpleNamespace(content="AWS DMS. [SOURCE 1]")
        )
        with (
            patch.object(rag, "semantic_search", return_value=[SEARCH_RESULT]),
            patch.object(rag, "chat", return_value=response),
        ):
            result = rag.rag_search("Which service migrates databases?")
        self.assertEqual(result["sources"][0]["citation"], "SOURCE 1")
        self.assertIn("[SOURCE 1]", result["answer"])

    def test_rag_handles_empty_model_response(self):
        response = types.SimpleNamespace(
            message=types.SimpleNamespace(content=None)
        )
        with (
            patch.object(rag, "semantic_search", return_value=[SEARCH_RESULT]),
            patch.object(rag, "chat", return_value=response),
        ):
            result = rag.rag_search("Which service migrates databases?")
        self.assertEqual(
            result["answer"],
            "The local language model returned an empty response.",
        )

    def test_api_exposes_two_workflows_and_health(self):
        with (
            patch.object(api, "index_documents", return_value=1) as index,
            patch.object(api, "semantic_search", return_value=[SEARCH_RESULT]),
            patch.object(
                api,
                "rag_search",
                return_value={
                    "answer": "AWS DMS. [SOURCE 1]",
                    "sources": [
                        {"source": "test_doc2.txt"},
                        {"source": "test_doc2.txt"},
                    ],
                },
            ),
            TestClient(api.app) as client,
        ):
            self.assertEqual(
                client.get("/search?query=migration&chunks=3").status_code,
                200,
            )
            self.assertEqual(
                client.get("/search?query=migration&chunks=3").json(),
                {
                    "query": "migration",
                    "results": [
                        {
                            "text": "AWS DMS can migrate databases.",
                            "source": "test_doc2.txt",
                        }
                    ],
                },
            )
            response = client.post("/ask", json={"question": "migration"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                response.json(),
                {
                    "query": "migration",
                    "answer": "AWS DMS. [SOURCE 1]",
                    "sources": ["test_doc2.txt"],
                },
            )
            self.assertEqual(client.get("/health").json(), {"status": "ok"})
        index.assert_called_once_with()

    def test_evaluation_dataset_matches_current_document_names(self):
        expected = {
            source
            for case in evaluation.load_dataset()
            for source in case["expected_sources"]
        }
        actual = {document["source"] for document in semantic.load_documents()}
        self.assertTrue(expected.issubset(actual))


if __name__ == "__main__":
    unittest.main()
