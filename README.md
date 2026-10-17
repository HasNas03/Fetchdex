# Fetchdex (in-progress)

Fetchdex has two user-facing workflows:

1. **Semantic Search** returns matching document chunks without an LLM, uses embeddings and Chroma
2. **AI Assistant** selects one of three tools: grounded RAG answer, list documents, or read a document.

## Setup

Run from this project directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
ollama pull qwen3:1.7b
```

Ollama must be running for the AI Assistant. Semantic search does not use Qwen3.

## Prepare the knowledge base

Put `.txt` files in `documents/`, then inspect and index them:

```bash
python -m source.knowledge preview
python -m source.knowledge index
```

Indexing writes generated data to `data/chroma/`.

## Run the API

```bash
python -m uvicorn source.api:app --reload
```

Open <http://127.0.0.1:8000/docs>.

- `GET /search?q=...&k=3` runs non-generative semantic search.
- `POST /ask` sends `{"question": "..."}` to the AI Assistant.
- `GET /health` returns a liveness response.

## Run the assistant directly

```bash
python -m source.assistant "Which AWS service can migrate databases?"
```

## Evaluate retrieval and RAG

```bash
python -m source.evaluation retrieval
python -m source.evaluation rag
```

The evaluation dataset is `evaluation/golden_dataset.json`.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Documentation

Open `docs/tutorial.html` in a browser for the complete four-flow walkthrough. The editable source is `docs/tutorial.md`.
