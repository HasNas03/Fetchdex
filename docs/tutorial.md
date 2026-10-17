# Fetchdex Simplified: Four Flows, Two Entry Points

This tutorial explains the simplified Fetchdex project from the outside in. It follows the current code in this folder.

Users see two entry points:

```text
Fetchdex
├── Semantic Search
│   └── embeddings + Chroma
└── AI Assistant
    ├── grounded RAG answer
    ├── list documents
    └── read a document
```

That produces four flows to understand:

1. Semantic Search.
2. Grounded RAG Answer.
3. List Documents.
4. Read a Document.

The former standalone RAG option was not deleted. It became the AI Assistant's `grounded_answer` tool.

## 1. Big Picture

Fetchdex turns local text documents into searchable data and lets a local language model work with those documents.

```text
documents/*.txt
      ↓
load and split into chunks
      ↓
MiniLM embedding vectors
      ↓
Chroma database on disk
      ↓
semantic results or grounded AI answers
```

Two different models have different jobs:

| Component | Responsibility |
|---|---|
| `all-MiniLM-L6-v2` | Convert passages and questions into vectors for similarity search. |
| `qwen3:1.7b` | Select assistant tools and generate natural-language answers. |

```text
Embedding model != language model
Chroma != Ollama
FastAPI != Uvicorn
Semantic search != RAG
RAG != agent routing
```

The project does not train either model. It calls existing models locally.

## 2. Project Structure

```text
Fetchdex-Simplified/
├── source/
│   ├── __init__.py      marks source as a Python package
│   ├── knowledge.py     documents, chunks, embeddings, Chroma, search
│   ├── rag.py           retrieval-grounded generation and citations
│   ├── assistant.py     tool definitions and agent loop
│   ├── api.py           HTTP routes for search and assistant
│   └── evaluation.py    retrieval and RAG checks
├── documents/           input .txt files
├── evaluation/
│   └── golden_dataset.json
├── docs/
│   ├── tutorial.md
│   └── tutorial.html
├── tests/
│   └── test_workflows.py
├── requirements.txt
└── README.md
```

There is no single file that executes everything. Each command starts at a different entry point:

| Command or request | Entry point |
|---|---|
| `python -m source.knowledge index` | `knowledge.main()` |
| `GET /search` | `api.search()` |
| `POST /ask` | `api.ask()` |
| `python -m source.assistant "..."` | `assistant.main()` |
| `python -m source.evaluation retrieval` | `evaluation.main()` |

## 3. Preparation: Build the Search Index

All four flows depend on the source documents. Semantic search and RAG also depend on the Chroma index.

Run:

```bash
python -m source.knowledge preview
python -m source.knowledge index
```

The indexing sequence is:

```text
load_documents()
      ↓
chunk_documents()
      ↓
get_embedding_model()
      ↓
encode_document(...)
      ↓
get_collection()
      ↓
collection.upsert(...)
```

### File: `source/knowledge.py`

This file owns the complete knowledge-base lifecycle.

| Function | What it does |
|---|---|
| `load_documents()` | Reads sorted `.txt` files and returns filename/text dictionaries. |
| `chunk_text()` | Splits one string into overlapping 40-word windows with 10-word overlap. |
| `chunk_documents()` | Applies chunking to every document and adds source metadata. |
| `get_embedding_model()` | Lazily loads MiniLM once per Python process. |
| `get_collection()` | Lazily opens the persistent Chroma collection once per process. |
| `index_documents()` | Coordinates chunking, embedding, and Chroma upsert. |
| `semantic_search()` | Embeds a question and returns nearest stored chunks. |
| `preview_documents()` | Prints chunks without writing the database. |
| `main()` | Selects `preview` or `index` from the command line. |

`@lru_cache(maxsize=1)` is important. The first call creates the model or collection object. Later calls in that process reuse it.

Chunk overlap preserves context at boundaries:

```text
chunk 0: words  0–39
chunk 1: words 30–69
chunk 2: words 60–99
```

`upsert` inserts a new chunk ID or updates an existing one. It does not automatically delete stale IDs left by removed or shortened documents.

# Flow 1 — Semantic Search

## Purpose

Return relevant document passages without asking Qwen3 to produce an answer.

```text
GET /search?q=database+migration&k=3
      ↓
api.search()
      ↓
knowledge.semantic_search()
      ↓
MiniLM encodes the question
      ↓
Chroma compares it with stored chunk vectors
      ↓
raw matching chunks + filenames + distances
```

## Files and functions

### `source/api.py`

`search()` is the HTTP boundary. FastAPI validates:

- `q`: required text, 1–1000 characters.
- `k`: integer from 1–10, default 3.

It converts application errors into HTTP responses:

| Failure | HTTP status |
|---|---:|
| Invalid direct search input | 400 |
| Embedding/database search failure | 503 |
| Unexpected failure | 500 |

### `source/knowledge.py`

`semantic_search(query, k)` performs the actual workflow:

1. Reject blank queries and `k < 1`.
2. Open the cached Chroma collection.
3. Return `[]` if the collection is empty.
4. Encode the question with `encode_query()`.
5. Ask Chroma for the nearest vectors.
6. Convert Chroma's batched response into ordinary dictionaries.

The result looks like:

```json
{
  "query": "database migration",
  "results": [
    {
      "chunk_id": "test_doc2.txt-0",
      "text": "AWS DMS stands for Database Migration Service...",
      "metadata": {
        "source": "test_doc2.txt",
        "chunk_number": 0
      },
      "distance": 0.18
    }
  ]
}
```

The distance above is illustrative. With cosine distance, a lower value means a closer match.

## What this flow does not call

- It does not call Ollama.
- It does not load Qwen3.
- It does not generate an answer.
- It does not guarantee that a result is relevant because there is no distance threshold.

# Flow 2 — Grounded RAG Answer

## Purpose

Answer a question using retrieved evidence and citations. This is the former standalone RAG workflow, now available as an AI Assistant tool.

```text
POST /ask
      ↓
api.ask()
      ↓
assistant.run_assistant()
      ↓
Qwen3 selects grounded_answer
      ↓
rag.answer_question()
      ↓
semantic_search() → evidence chunks
      ↓
build_context() + build_prompt()
      ↓
Qwen3 writes cited answer
      ↓
assistant returns that answer directly
```

This normally requires two Qwen3 calls:

1. The assistant-routing call selects `grounded_answer`.
2. The RAG-generation call writes the evidence-based answer.

There is no third formatting call because `run_assistant()` recognizes that RAG already produced the final answer.

## Files and functions

### `source/api.py`

`AskRequest` validates the JSON body:

```json
{"question": "Which AWS service can migrate databases?"}
```

`ask()` rejects whitespace-only questions, calls `run_assistant()`, and returns:

```json
{
  "question": "Which AWS service can migrate databases?",
  "answer": "AWS DMS can migrate databases. [SOURCE 1]"
}
```

### `source/assistant.py`

`run_assistant()` sends the user message, system prompt, and three tool definitions to Qwen3. For a document-content question, the prompt requires `grounded_answer()`.

`grounded_answer(question, k)` is a small adapter. It calls `rag.answer_question()` and returns its final answer text.

### `source/rag.py`

| Function | What it does |
|---|---|
| `build_context()` | Converts ranked chunks into `[SOURCE 1]`, `[SOURCE 2]`, and similar evidence blocks. |
| `build_prompt()` | Adds grounding, refusal, and citation rules around the evidence and question. |
| `source_metadata()` | Creates citation records from retrieved chunks. |
| `answer_question()` | Coordinates retrieval, refusal, generation, source metadata, and failure handling. |

The context sent to Qwen3 resembles:

```text
[SOURCE 1]
Document: test_doc2.txt
Chunk: 0
Content:
AWS DMS stands for Database Migration Service...
```

`answer_question()` returns a richer internal dictionary containing the answer, source metadata, and complete retrieved chunks. `grounded_answer()` exposes only the final answer string to the outer assistant.

## Empty evidence and model failure

If semantic search returns nothing, RAG returns:

```text
I don't know based on the provided documents.
```

It does not call Qwen3 in that case.

If Ollama fails after retrieval succeeds, RAG returns an unavailable-model message while retaining source information in its internal response.

## What grounding does and does not guarantee

The prompt instructs Qwen3 to use only the supplied evidence and cite it. A language model can still disobey instructions or make a mistake. That is why `source/evaluation.py` checks expected facts, refusals, and citation numbers.

# Flow 3 — List Documents

## Purpose

Answer questions such as “What documents are available?” without semantic search or RAG.

```text
POST /ask
      ↓
api.ask()
      ↓
assistant.run_assistant()
      ↓
Qwen3 selects list_documents
      ↓
knowledge.load_documents()
      ↓
filenames return to Qwen3
      ↓
Qwen3 presents the list
```

## Files and functions

### `source/assistant.py`

`list_documents()` calls `load_documents()`, selects each `source` filename, and serializes the list as JSON text. Tool messages must contain text, so JSON is a convenient structured representation.

`run_assistant()` appends that tool result to the conversation and asks Qwen3 to produce the user-facing response.

### `source/knowledge.py`

`load_documents()` reads current files directly from `documents/`. Therefore, newly added filenames can appear in this flow before they are indexed in Chroma.

## What this flow does not call

- It does not create embeddings.
- It does not search Chroma.
- It does not run the RAG prompt.
- Qwen3 is still used to choose the tool and phrase the final response.

# Flow 4 — Read a Document

## Purpose

Return one complete document when the user supplies its exact filename.

```text
POST /ask
      ↓
api.ask()
      ↓
assistant.run_assistant()
      ↓
Qwen3 selects read_document(name)
      ↓
knowledge.load_documents()
      ↓
exact filename match
      ↓
complete text returns to Qwen3
      ↓
Qwen3 responds to the user
```

## Files and functions

### `source/assistant.py`

`read_document()` receives `name` and:

1. Rejects an empty filename.
2. Loads current document records.
3. Compares each `source` with `name` using exact equality.
4. Returns the complete text on a match.
5. Returns a clear “not found” message otherwise.

`run_assistant()` executes the function with arguments selected by Qwen3:

```python
function(**call.function.arguments)
```

If the arguments are `{"name": "test_doc2.txt"}`, this is equivalent to:

```python
read_document(name="test_doc2.txt")
```

### `source/knowledge.py`

`load_documents()` supplies the complete text. This flow reads from disk and does not reconstruct the document from indexed chunks.

## What this flow does not call

- It does not require the Chroma index.
- It does not create or compare embeddings.
- It does not use the RAG prompt unless the assistant later selects the RAG tool in another step.

# 8. How the AI Assistant Chooses Among Three Tools

`assistant.py` provides actual Python function objects to Ollama:

```python
TOOLS = [grounded_answer, list_documents, read_document]
```

The Ollama SDK derives tool schemas from function names, type annotations, and docstrings. The model may respond with a structured request such as:

```json
{
  "name": "read_document",
  "arguments": {"name": "test_doc2.txt"}
}
```

Python controls execution through an allowlist:

```python
TOOL_MAP = {tool.__name__: tool for tool in TOOLS}
```

The model cannot execute an arbitrary function name through this lookup. Unknown names return explanatory text.

The loop allows several tool rounds. Tool failures become tool-result text, allowing the model to recover. If no final answer appears after five rounds, the function returns a step-limit message.

The system prompt allows direct answers only for explaining Fetchdex itself. Document-content questions are instructed to use `grounded_answer`. This is strong guidance, although model instructions are not a mathematical guarantee.

# 9. API and Server Roles

Run:

```bash
python -m uvicorn source.api:app --reload
```

```text
HTTP client
    ↓
Uvicorn server
    ↓
FastAPI validation and routing
    ↓
source/api.py endpoint
    ↓
application functions
    ↓
FastAPI converts returned dictionaries to JSON
```

```text
FastAPI = application framework
Uvicorn = process that listens for HTTP requests
Pydantic = validates POST body fields
```

The API intentionally exposes only two functional workflows:

| Route | Workflow |
|---|---|
| `GET /search` | Semantic Search |
| `POST /ask` | AI Assistant, which selects one of three tools |

`GET /health` is operational status, not a user workflow.

The functions in `source/api.py` are:

| Function | What it does |
|---|---|
| `AskRequest` | Defines and validates the `POST /ask` request body. |
| `search()` | Runs semantic search and translates failures into HTTP errors. |
| `ask()` | Runs the AI Assistant and returns its final answer. |
| `health()` | Returns the lightweight `{"status": "ok"}` liveness response. |

# 10. Evaluation

The job-oriented project keeps retrieval and generated-answer evaluation even though RAG is no longer a separate user option.

### File: `source/evaluation.py`

| Function | What it does |
|---|---|
| `load_dataset()` | Reads golden questions and expected outcomes. |
| `evaluate_retrieval()` | Checks Hit@1, Hit@K, and reciprocal rank for one case. |
| `evaluate_rag()` | Checks expected terms, refusal behavior, and citation-number behavior. |
| `run_evaluation()` | Runs the requested evaluator across applicable cases. |
| `main()` | Parses `retrieval` or `rag` and prints JSON. |

Run:

```bash
python -m source.evaluation retrieval
python -m source.evaluation rag
```

The golden dataset filenames now match the supplied files:

```text
test_doc1.txt → FastAPI facts
test_doc2.txt → AWS facts
```

The evaluation is deterministic in how it scores text, but generated answers can vary. Expected-term checks are intentionally simple: a semantically correct synonym can fail if it omits the expected substring.

# 11. Complete Function Map

## `source/knowledge.py`

```text
load_documents
  ├── used by indexing
  ├── used by list_documents
  └── used by read_document

chunk_text
  └── chunk_documents
        └── index_documents

get_embedding_model ─┐
get_collection ──────┼── semantic_search
                     └── index_documents
```

## `source/rag.py`

```text
semantic_search
      ↓
answer_question
      ├── build_context
      ├── build_prompt
      └── source_metadata
```

## `source/assistant.py`

```text
run_assistant
      ├── grounded_answer → rag.answer_question
      ├── list_documents → knowledge.load_documents
      └── read_document  → knowledge.load_documents
```

## `source/api.py`

```text
search → knowledge.semantic_search
ask    → assistant.run_assistant
health → static liveness dictionary
```

## `source/evaluation.py`

```text
load_dataset
      ├── run_evaluation(retrieval) → evaluate_retrieval
      └── run_evaluation(rag)       → evaluate_rag
```

# 12. Commands

From the project root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
ollama pull qwen3:1.7b
python -m source.knowledge preview
python -m source.knowledge index
python -m uvicorn source.api:app --reload
```

Direct assistant call:

```bash
python -m source.assistant "What documents do I have?"
```

HTTP examples:

```bash
curl --get http://127.0.0.1:8000/search \
  --data-urlencode 'q=database migration' \
  --data-urlencode 'k=3'
```

```bash
curl http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"Which AWS service can migrate databases?"}'
```

# 13. Final Mental Model

```text
                         FETCHDEX

documents/*.txt
      │
      ├── load → chunk → MiniLM → Chroma index
      │                              │
      │                              ▼
      │                       Semantic Search
      │                       raw matching chunks
      │
      └───────────────────────────────┐
                                      ▼
                                AI Assistant
                   ┌──────────────────┼──────────────────┐
                   ▼                  ▼                  ▼
            Grounded RAG       List Documents      Read Document
                   │                  │                  │
            search + Qwen3       filenames          complete text
            cited answer             │                  │
                   └──────────────────┴──────────────────┘
                                      │
                                user response
```

The architecture is compact while still demonstrating embeddings, vector retrieval, RAG, citations, tool calling, agent orchestration, APIs, validation, error handling, and evaluation.
