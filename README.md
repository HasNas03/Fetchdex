# Fetchdex

Fetchdex is an AI-based tool that uses your study notes as context so you can find and review what matters in seconds.

The Fetchdex API exposes 2 features: 

1. **Semantic Search** returns matching document chunks using chunk-based embeddings, generating answers without an LLM.
2. **Assistant Search** uses Ollama Qwen to retrieves relevant chunks as context and generate a cited answer.

## Setup

Run from this project directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
ollama pull qwen3:1.7b
```

Ollama must be running for the AI Assistant. Semantic search does not use Qwen3.

## Documents and indexing

Put `.txt` files in `documents/`. Starting the API automatically loads,
chunks, embeds, and synchronizes those documents with `data/chroma/`.
Restart the API after changing the documents.

## Run the API

```bash
python -m uvicorn source.api:app --reload
```

Open <http://127.0.0.1:8000/docs>.

- `GET /search?query=...&chunks=3` runs non-generative semantic search.
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
