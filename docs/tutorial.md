# Fetchdex: Semantic Search and Grounded RAG

Fetchdex has two API workflows. Both use the same local document index.

```text
documents/*.txt or *.md
        ↓
API startup indexes the documents
        ↓
Chroma vector collection
        ├── GET /search → matching text and source files
        └── POST /ask   → Qwen answer and source files
```

## 1. Project Structure

```text
Fetchdex/
├── source/
│   ├── __init__.py
│   ├── semantic.py
│   ├── rag.py
│   ├── api.py
│   └── evaluation.py
├── documents/
├── evaluation/
│   └── golden_dataset.json
├── tests/
│   └── test_workflows.py
├── requirements.txt
└── README.md
```

| File | Responsibility |
|---|---|
| `semantic.py` | Load, chunk, embed, index, and retrieve documents. |
| `rag.py` | Turn retrieved evidence into a grounded Qwen answer. |
| `api.py` | Index at startup and expose `/search`, `/ask`, and `/health`. |
| `evaluation.py` | Evaluate retrieval and RAG against the golden dataset. |

`assistant.py` is no longer needed. `/ask` always runs RAG directly.

## 2. Startup Indexing

Starting the API calls `index_documents()` once:

```text
api.lifespan()
      ↓
semantic.index_documents()
      ↓
load documents
      ↓
create overlapping chunks
      ↓
create embeddings
      ↓
upsert current chunks into Chroma
      ↓
delete stale chunks
```

Restart the API after adding, changing, or removing documents.

## 3. Flow One: Semantic Search

Request:

```http
GET /search?query=database%20migration&chunks=3
```

Flow:

```text
api.search()
      ↓
semantic_search(query, chunks)
      ↓
embed the query
      ↓
retrieve the closest Chroma chunks
      ↓
return text and source filenames
```

Example response:

```json
{
  "query": "database migration",
  "results": [
    {
      "text": "AWS DMS can migrate databases.",
      "source": "test_doc2.txt"
    }
  ]
}
```

This flow does not call Qwen.

## 4. Flow Two: Grounded RAG

Request:

```http
POST /ask
Content-Type: application/json
```

```json
{
  "question": "Which AWS service can migrate databases?"
}
```

Flow:

```text
api.ask()
      ↓
rag.rag_search()
      ↓
semantic_search()
      ↓
build numbered source context
      ↓
send evidence and question to Qwen through Ollama
      ↓
return the answer and unique source filenames
```

Example response:

```json
{
  "query": "Which AWS service can migrate databases?",
  "answer": "AWS DMS can migrate databases. [SOURCE 1]",
  "sources": ["test_doc2.txt"]
}
```

### RAG helpers

| Function | Purpose |
|---|---|
| `build_context()` | Formats retrieved chunks as numbered evidence blocks. |
| `build_prompt()` | Adds grounding, citation, and refusal instructions. |
| Source list inside `rag_search()` | Numbers each citation and records its filename. |
| `rag_search()` | Runs retrieval and generation once. |

If search finds no evidence, RAG returns the standard refusal. If Ollama is unavailable, `rag.py` raises `RuntimeError` and the API returns HTTP 503.

## 5. Evaluation

`evaluation.py` evaluates the two workflows separately:

```text
retrieval mode → expected source ranking
rag mode       → expected facts, refusals, and citation numbers
```

Commands:

```bash
python -m source.evaluation retrieval
python -m source.evaluation rag
```

## 6. Setup and Commands

```bash
cd ~/Desktop/Fetchdex
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
ollama pull qwen3:1.7b
python -m uvicorn source.api:app --reload
```

Semantic search:

```bash
curl --get http://127.0.0.1:8000/search \
  --data-urlencode 'query=database migration' \
  --data-urlencode 'chunks=3'
```

Grounded RAG:

```bash
curl http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"Which AWS service can migrate databases?"}'
```

Tests:

```bash
python -m unittest discover -s tests -v
```

## Final Mental Model

```text
                    Fetchdex API
                         │
            ┌────────────┴────────────┐
            │                         │
      GET /search                 POST /ask
            │                         │
            ▼                         ▼
      semantic.py               rag.py
            │                         │
            └──── Chroma evidence ────┘
                                      │
                                      ▼
                                Ollama / Qwen
                                      │
                                      ▼
                              Answer + sources
```
