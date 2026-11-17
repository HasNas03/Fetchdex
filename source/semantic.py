# responsible for indexing knowledge base and performing semantic search

from functools import lru_cache
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_PATH = "data/chroma"
COLLECTION_NAME = "personal_knowledge"

# -----------------------------------------------------------------------------------------------------------------------------
# HELPER FUNCTIONS
# single document chunker (general)
def chunk_text(text: str, chunk_size: int = 40, overlap: int = 10) -> list[str]:
    """
    split text into overlapping chunks
    chunk_size: maximum number of words per chunk
    overlap: number of repeated words between chunks
    """
    # validate chunk size and validate overlap
    if chunk_size < 1: raise ValueError("chunk_size must be at least 1")
    if overlap < 0 or overlap >= chunk_size: raise ValueError("overlap must be between 0 and chunk_size - 1")

    chunks = []
    words = text.split()

    # start at first word
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words)
        chunks.append(chunk)
        # move forward while preserving overlap
        start += chunk_size - overlap

    return chunks

# retrieve chromadb collection (general)
@lru_cache(maxsize=1)
def get_collection():
    """
    open the local chroma database and returns a handle to your stored vector database collection
    """
    # connect to persistent chroma database
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    # get collection or create it if needed
    collection = client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"},)
    return collection

# add embeds in a collection (general)
def store_embeds(collection, embedded_chunks: list[dict]) -> None:
    """add embedded chunks in chroma"""
    ids = []
    documents = []
    embeddings = []
    metadatas = []

    # separate chunk values into chroma fields
    for chunk in embedded_chunks:
        ids.append(chunk["chunk_id"])
        documents.append(chunk["text"])
        embeddings.append(chunk["embedding"])
        metadatas.append({
            "source": chunk["source"],
            "chunk_number": chunk["chunk_number"],
        })

    # insert new chunks or update existing ids
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

# -----------------------------------------------------------------------------------------------------------------------------

# load source documents from a client-provided file or folder
def load_documents(document_path: str | Path) -> list[dict]:
    """load one text file or every text file inside a folder"""
    path = Path(document_path).expanduser()
    if not path.exists():
        raise ValueError(f"Document path does not exist: {path}")

    # accept one supported file or recursively scan a folder
    if path.is_file():
        if path.suffix.lower() not in {".txt", ".md"}:
            raise ValueError("Document must be a .txt or .md file")
        file_paths = [path]
    else:
        file_paths = sorted(
            file_path
            for file_path in path.rglob("*")
            if file_path.is_file() and file_path.suffix.lower() in {".txt", ".md"}
        )

    if not file_paths:
        raise ValueError(f"No .txt or .md files found in: {path}")

    # holds every loaded document
    documents = []

    # read each file and retain a unique, readable source path
    for file_path in file_paths:
        text = file_path.read_text(encoding="utf-8")
        document = {
            "source": file_path.name if path.is_file() else str(file_path.relative_to(path)),
            "text": text,
        }
        documents.append(document)

    return documents

# chunk our source documents
def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    convert multiple documents into chunks while preserving their source metadata
    """
    # holds all chunks from all documents
    all_chunks = []

    # process every document
    for document in documents:
        # input into general chunk-maker to get chunks for document
        text_chunks = chunk_text(document["text"])

        # create chunk object and include metadata 
        for chunk_number, text in enumerate(text_chunks):
            chunk = {
                "chunk_id": f"{document['source']}-{chunk_number}",
                "source": document["source"], # filename
                "chunk_number": chunk_number,
                "text": text,
            }
            # add to master chunk list for all input documents, all in sequence
            all_chunks.append(chunk)

    return all_chunks

# embed chunks
def embed_documents(model, chunks: list[dict]) -> list[dict]:
    """
    convert every chunk into an embedding vector
    """
    # holds chunk object containing chunk and embedding
    embedded_chunks = []

    # extract text from each chunk into list of text
    texts = [chunk["text"] for chunk in chunks]
    # convert to embeddings (returns embeddings in same order as input texts
    embeddings = model.encode_document(texts, normalize_embeddings=True)
    # attach each embedding to its chunk
    for chunk, embedding in zip(chunks, embeddings):
        # creates a new dict made of the old chunk dict's keys with a new "embedding" key
        embedded_chunk = {
            **chunk, # unpack all key/value pairs from the chunk dictionary
            "embedding": embedding.tolist(),}

        embedded_chunks.append(embedded_chunk)

    return embedded_chunks

# Startup - load, chunk, embed, and store data into collection
def index_documents(document_path: str | Path) -> int:
    """load, chunk, embed and synchronize all documents with Chroma"""
    # 1. load all documents
    documents = load_documents(document_path)
    # 2. create chunks
    chunks = chunk_documents(documents)

    # 3. get chroma collection and its currently stored ids
    collection = get_collection()
    existing_ids = collection.get()["ids"]

    # clear the old index if no source content exists
    if not chunks:
        if existing_ids:
            collection.delete(ids=existing_ids)
        return 0

    # 4. get embedding model
    model = SentenceTransformer(EMBEDDING_MODEL)
    # 5. embed chunks
    embedded_chunks = embed_documents(model, chunks)
    # 6. insert new chunks and update chunks with existing ids
    store_embeds(collection, embedded_chunks)

    # 7. delete stored chunks that no longer exist in the source documents
    current_ids = {chunk["chunk_id"] for chunk in embedded_chunks}
    stale_ids = [chunk_id for chunk_id in existing_ids if chunk_id not in current_ids]
    if stale_ids:
        collection.delete(ids=stale_ids)

    # 8. return total stored chunk count
    return collection.count()
 
# (MAIN) semantic search function (returns most semantically-similar chunks)
def semantic_search(query: str, k: int = 3) -> list[dict]:
    """semantically retrieve top-k matching chunks from chroma without using an llm"""
    # validate query
    if not query.strip(): raise ValueError("Search query cannot be empty")
    # validate k (how many top search results you want back)
    if k < 1: raise ValueError("k must be at least 1")

    collection = get_collection()
    collection_count = collection.count()
    if collection_count == 0: return []

    try:
        model = SentenceTransformer(EMBEDDING_MODEL)
        # convert user query into embedding vector
        query_embedding = model.encode_query(query, normalize_embeddings=True).tolist()
        # search chroma for nearest embeddings
        results = collection.query(query_embeddings=[query_embedding], n_results=min(k, collection_count)) # number of results (do not request more results than exist)

    except Exception as error: raise RuntimeError(f"Vector search failed: {error}") from error

    # chroma returns nested lists because it supports multiple queries
    documents = results["documents"][0] if results["documents"] is not None else []
    metadatas = results["metadatas"][0] if results["metadatas"] is not None else []
    # combine each matching passage with its source metadata
    return [
        {
            "text": text,
            "metadata": metadata,
        }
        for text, metadata in zip(documents, metadatas)
    ]
