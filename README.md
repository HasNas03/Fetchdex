# Fetchdex

Fetchdex searches your local `.txt` and `.md` notes from the terminal.

- `search` returns relevant document passages without an LLM.
- `ask` uses those passages to generate a grounded Qwen answer.

## First-time setup

```bash
cd ~/Desktop/Fetchdex
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
ollama pull qwen3:1.7b
```

## Use Fetchdex

Starting from a newly opened terminal:

### 1. Enter the project and activate its environment

```bash
cd ~/Desktop/Fetchdex
source .venv/bin/activate
```

### 2. Index a file or folder

Provide the path to one `.txt`/`.md` file or a folder containing them:

```bash
python fetchdex index "/Users/yourname/Documents/notes"
```

Relative paths also work:

```bash
python fetchdex index "documents"
```

Run `index` again only after adding, editing, or deleting files at that path.

### 3.1. Semantic Search (without an LLM)

```bash
python fetchdex search "what is switch in Java" --chunks 3
```

### 3.2. RAG Search (Qwen)

Run `ollama serve` in a separate terminal. Then run:

```bash
python fetchdex ask "what is switch in Java" --chunks 3
```

`--chunks` is optional and defaults to `3`.
