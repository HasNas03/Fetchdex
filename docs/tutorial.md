# Fetchdex: Semantic Search and Grounded RAG

Fetchdex has two command-line workflows. Both use the same local document index.

```text
client-provided .txt/.md file or folder
        ↓
index command synchronizes the documents
        ↓
Chroma vector collection
        ├── search → matching text and source files
        └── ask    → Qwen answer and source files
```

## 1. Project Structure

```text
Fetchdex/
├── source/
│   ├── __init__.py
│   ├── semantic.py
│   ├── rag.py
│   ├── cli.py
│   └── evaluation.py
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
| `cli.py` | Run indexing, search, or RAG commands and display results. |
| `evaluation.py` | Evaluate retrieval and RAG against the golden dataset. |

## 2. Indexing

Provide a file or folder path when building the index:

```text
python fetchdex index "/path/to/notes"
        ↓
cli.main()
        ↓
semantic.index_documents()
        ↓
load → chunk → embed → update Chroma → delete stale chunks
```

Folder paths are scanned recursively for `.txt` and `.md` files. Run `index`
again after changing files at that path. The new index replaces the previously
indexed document set. `search` and `ask` reuse it without needing the path.

## 3. Flow One: Semantic Search

```bash
python fetchdex index "/path/to/notes"
python fetchdex search "database migration" --chunks 3
```

```text
cli.run_search()
        ↓
semantic_search(query, chunks)
        ↓
embed query → retrieve closest Chroma chunks
        ↓
print matching text and source filenames
```

This workflow does not call Qwen.

## 4. Flow Two: Grounded RAG

```bash
python fetchdex ask "Which AWS service can migrate databases?" --chunks 3
```

```text
cli.run_ask()
        ↓
rag.rag_search()
        ↓
semantic_search()
        ↓
build document evidence and grounding instructions
        ↓
send evidence and question to Qwen through Ollama
        ↓
print answer and source filenames
```

Example:

```text
Answer:
AWS DMS can migrate databases.

Sources:
- test_doc2.txt
```

The `Sources` section shows which file supplied the retrieved evidence.

### RAG helpers

| Function | Purpose |
|---|---|
| `build_context()` | Formats retrieved chunks as document evidence blocks. |
| `build_prompt()` | Adds grounding and refusal instructions. |
| `rag_search()` | Retrieves evidence, calls Qwen, and returns answer and sources. |

## 5. Evaluation

```bash
python -m source.evaluation retrieval
python -m source.evaluation rag
```

- Retrieval mode checks whether expected files rank near the top.
- RAG mode checks expected facts, refusal behavior, and source filenames.

## 6. Setup and Commands

```bash
cd ~/Desktop/Fetchdex
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
ollama pull qwen3:1.7b
```

Ollama must be open or running before using `ask`:

```bash
ollama serve
```

Tests:

```bash
python -m unittest discover -s tests -v
```

## Final Mental Model

```text
                         Terminal
                            │
               ┌────────────┴────────────┐
               │                         │
            search                      ask
               │                         │
               ▼                         ▼
         semantic.py                  rag.py
               │                         │
               └──── Chroma evidence ────┘
                                         │
                                         ▼
                                   Ollama / Qwen
                                         │
                                         ▼
                                Answer + sources
```
