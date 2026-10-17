"""Focused checks for the four documented workflows."""

import json
import types
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from source import api, assistant, evaluation, rag, semantic


SEARCH_RESULT = {
    "chunk_id": "test_doc2.txt-0",
    "text": "AWS DMS can migrate databases.",
    "metadata": {"source": "test_doc2.txt", "chunk_number": 0},
    "distance": 0.1,
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
                "ids": [[SEARCH_RESULT["chunk_id"]]],
                "documents": [[SEARCH_RESULT["text"]]],
                "metadatas": [[SEARCH_RESULT["metadata"]]],
                "distances": [[SEARCH_RESULT["distance"]]],
            },
        )
        with (
            patch.object(semantic, "get_collection", return_value=collection),
            patch.object(semantic, "get_embedding_model", return_value=model),
        ):
            self.assertEqual(semantic.semantic_search("migration"), [SEARCH_RESULT])

    def test_grounded_rag_answer(self):
        response = types.SimpleNamespace(
            message=types.SimpleNamespace(content="AWS DMS. [SOURCE 1]")
        )
        with (
            patch.object(rag, "semantic_search", return_value=[SEARCH_RESULT]),
            patch.object(rag, "chat", return_value=response),
        ):
            result = rag.answer_question("Which service migrates databases?")
        self.assertEqual(result["sources"][0]["citation"], "SOURCE 1")
        self.assertIn("[SOURCE 1]", result["answer"])

    def test_list_and_read_document_tools(self):
        documents = [{"source": "notes.txt", "text": "Complete contents"}]
        with patch.object(assistant, "load_documents", return_value=documents):
            self.assertEqual(json.loads(assistant.list_documents()), ["notes.txt"])
            self.assertEqual(assistant.read_document("notes.txt"), "Complete contents")
            self.assertIn("not found", assistant.read_document("missing.txt"))

    def test_assistant_returns_grounded_tool_answer_directly(self):
        call = types.SimpleNamespace(
            function=types.SimpleNamespace(
                name="grounded_answer",
                arguments={"question": "migration"},
            )
        )
        tool_request = types.SimpleNamespace(
            message=types.SimpleNamespace(content="", tool_calls=[call])
        )
        with (
            patch.object(assistant, "chat", return_value=tool_request) as model,
            patch.object(
                assistant,
                "grounded_answer",
                return_value="AWS DMS. [SOURCE 1]",
            ) as tool,
        ):
            original = assistant.TOOL_MAP["grounded_answer"]
            assistant.TOOL_MAP["grounded_answer"] = assistant.grounded_answer
            try:
                answer = assistant.assistant_search("Which service migrates databases?")
            finally:
                assistant.TOOL_MAP["grounded_answer"] = original
        tool.assert_called_once_with(question="migration")
        model.assert_called_once()
        self.assertEqual(answer, "AWS DMS. [SOURCE 1]")

    def test_api_exposes_two_workflows_and_health(self):
        client = TestClient(api.app)
        with patch.object(api, "semantic_search", return_value=[SEARCH_RESULT]):
            self.assertEqual(client.get("/search?q=migration").status_code, 200)
        with patch.object(api, "run_assistant", return_value="AWS DMS. [SOURCE 1]"):
            response = client.post("/ask", json={"question": "migration"})
            self.assertEqual(response.status_code, 200)
            self.assertIn("SOURCE 1", response.json()["answer"])
        self.assertEqual(client.get("/health").json(), {"status": "ok"})

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
