"""run Fetchdex semantic search and grounded RAG from the command line"""

import argparse

from source.rag import rag_search
from source.semantic import get_collection, index_documents, semantic_search


# rebuild the Chroma index only when the user requests it
def run_index(document_path: str) -> str:
    """synchronize a client-provided file or folder with Chroma"""
    chunk_count = index_documents(document_path)
    return f"Indexed {chunk_count} document chunks."


# run semantic search and format its matching chunks for the terminal
def run_search(query: str, chunks: int = 3) -> str:
    """return matching document text and source filenames"""
    results = semantic_search(query, chunks)
    if not results:
        return "No matching document chunks were found."

    output = [f"Query: {query}"]
    for number, result in enumerate(results, start=1):
        output.extend(
            [
                "",
                f"Result {number}",
                f"Source: {result['metadata']['source']}",
                f"Text: {result['text']}",
            ]
        )
    return "\n".join(output)


# run grounded RAG and show the source filenames below the answer
def run_ask(question: str, chunks: int = 3) -> str:
    """return a grounded answer followed by its source filenames"""
    response = rag_search(question, chunks)
    output = [f"Question: {question}", "", "Answer:", response["answer"]]

    if response["sources"]:
        output.extend(["", "Sources:"])
        output.extend(f"- {source}" for source in response["sources"])
    return "\n".join(output)


# read the user's command and run the selected flow
def main() -> None:
    """run indexing, semantic search or grounded RAG"""
    parser = argparse.ArgumentParser(description="Search your Fetchdex documents.")
    commands = parser.add_subparsers(dest="command", required=True)

    index_parser = commands.add_parser(
        "index", help="synchronize documents with the Chroma index"
    )
    index_parser.add_argument("path", help="path to a .txt/.md file or folder")

    search_parser = commands.add_parser(
        "search", help="return matching document chunks without an LLM"
    )
    search_parser.add_argument("query", help="text to search for")
    search_parser.add_argument("--chunks", type=int, default=3, choices=range(1, 11))

    ask_parser = commands.add_parser(
        "ask", help="generate a grounded answer with Qwen"
    )
    ask_parser.add_argument("question", help="question to answer from the documents")
    ask_parser.add_argument("--chunks", type=int, default=3, choices=range(1, 11))

    args = parser.parse_args()

    try:
        if args.command == "index":
            result = run_index(args.path)
        elif get_collection().count() == 0:
            raise RuntimeError(
                'No documents are indexed. Run: python fetchdex index "/path/to/documents"'
            )
        elif args.command == "search":
            result = run_search(args.query, args.chunks)
        else:
            result = run_ask(args.question, args.chunks)
    except (ValueError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")

    print(result)


if __name__ == "__main__":
    main()
