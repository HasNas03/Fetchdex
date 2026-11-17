"""Focused checks for semantic search and grounded RAG."""

import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from source import cli, evaluation, rag, semantic


SEARCH_RESULT = {
    "text": "AWS DMS can migrate databases.",
    "metadata": {"source": "test_doc2.txt", "chunk_number": 0},
}


class WorkflowTests(unittest.TestCase):
    def test_cli_index(self):
        with patch.object(cli, "index_documents", return_value=6) as index:
            self.assertEqual(cli.run_index("/notes"), "Indexed 6 document chunks.")
        index.assert_called_once_with("/notes")

    def test_load_documents_from_client_folder(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "notes.md").write_text("main notes", encoding="utf-8")
            nested = root / "nested"
            nested.mkdir()
            (nested / "more.txt").write_text("more notes", encoding="utf-8")
            (root / "ignored.pdf").write_text("ignored", encoding="utf-8")

            documents = semantic.load_documents(root)

        self.assertEqual(
            [document["source"] for document in documents],
            ["nested/more.txt", "notes.md"],
        )

    def test_load_documents_from_client_file(self):
        with TemporaryDirectory() as folder:
            file_path = Path(folder) / "notes.md"
            file_path.write_text("one document", encoding="utf-8")
            documents = semantic.load_documents(file_path)

        self.assertEqual(
            documents,
            [{"source": "notes.md", "text": "one document"}],
        )

    def test_cli_search_reuses_existing_index(self):
        collection = types.SimpleNamespace(count=lambda: 1)
        with (
            patch("sys.argv", ["fetchdex", "search", "migration"]),
            patch.object(cli, "get_collection", return_value=collection),
            patch.object(cli, "run_search", return_value="search result"),
            patch.object(cli, "index_documents") as index,
            patch("builtins.print"),
        ):
            cli.main()
        index.assert_not_called()

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
            self.assertEqual(semantic.index_documents("/notes"), 1)
        self.assertEqual(collection.deleted, ["old.txt-0"])

    def test_grounded_rag_answer(self):
        response = types.SimpleNamespace(
            message=types.SimpleNamespace(content="AWS DMS.")
        )
        with (
            patch.object(rag, "semantic_search", return_value=[SEARCH_RESULT]),
            patch.object(rag, "chat", return_value=response) as model,
        ):
            result = rag.rag_search("Which service migrates databases?")
        self.assertEqual(result["sources"], ["test_doc2.txt"])
        self.assertEqual(result["answer"], "AWS DMS.")
        prompt = model.call_args.kwargs["messages"][0]["content"]
        self.assertNotIn("[SOURCE", prompt)

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

    def test_cli_exposes_two_workflows(self):
        with (
            patch.object(cli, "semantic_search", return_value=[SEARCH_RESULT]),
            patch.object(
                cli,
                "rag_search",
                return_value={
                    "answer": "AWS DMS.",
                    "sources": ["test_doc2.txt"],
                },
            ),
        ):
            search_output = cli.run_search("migration", 3)
            ask_output = cli.run_ask("migration", 3)

        self.assertIn("AWS DMS can migrate databases.", search_output)
        self.assertIn("Source: test_doc2.txt", search_output)
        self.assertIn("AWS DMS.", ask_output)
        self.assertIn("- test_doc2.txt", ask_output)

    def test_evaluation_dataset_matches_current_document_names(self):
        expected = {
            source
            for case in evaluation.load_dataset()
            for source in case["expected_sources"]
        }
        actual = {
            document["source"] for document in semantic.load_documents("documents")
        }
        self.assertTrue(expected.issubset(actual))


if __name__ == "__main__":
    unittest.main()
