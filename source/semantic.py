# responsible for indexing knowledge base and performing semantic search
from functools import lru_cache
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

DOCUMENTS_FOLDER = Path("documents")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_PATH = "data/chroma"
COLLECTION_NAME = "personal_knowledge"

# -----------------------------------------------------------------------------------------------------------------------------
# HELPER FUNCTIONS
# single document chunker (general function)
def chunk_text(text: str, chunk_size: int = 40, overlap: int = 10) -> list[str]:
    """
    split text into overlapping chunks
    chunk_size: maximum number of words per chunk
    overlap: number of repeated words between chunks
    """
    # validate chunk size
    if chunk_size < 1: raise ValueError("chunk_size must be at least 1")
    # validate overlap
    if overlap < 0 or overlap >= chunk_size: raise ValueError("overlap must be between 0 and chunk_size - 1")

    # holds created chunks of the document
    chunks = []
    # split text into words
    words = text.split()

    # start at first word
    start = 0

    while start < len(words):
        # calculate chunk end
        end = start + chunk_size
        # get words for current chunk
        chunk_words = words[start:end]
        # join words into normal text
        chunk = " ".join(chunk_words)
        # add chunk
        chunks.append(chunk)
        # move forward while preserving overlap
        start += chunk_size - overlap
    # return list of chunks (str) for the input document
    return chunks

# retrieve embedding model
@lru_cache(maxsize=1)
def get_embedding_model():
    """
    load and return the embedding model once per python process
    """
    model = SentenceTransformer(EMBEDDING_MODEL)
    return model

# retrieve collection (general)
@lru_cache(maxsize=1)
def get_collection():
    """
    open the local chroma database and returns a handle to your stored vector database collection
    """
    # connect to persistent chroma database
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # get collection or create it if needed
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    return collection

# store embeds in a collection (general)
def store_embeds(collection, embedded_chunks: list[dict]) -> None:
    """store embedded chunks in chroma"""
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

# 1. load our source documents
def load_documents() -> list[dict]:
    """load every .txt file from the documents folder
    output: [{"source": "abc.txt", "text": "..."}, ...]
    """
    # holds every loaded document
    documents = []

    # find and process every .txt file
    for file_path in DOCUMENTS_FOLDER.glob("*.txt"):
        # read entire file
        text = file_path.read_text(encoding="utf-8")
        # store filename and contents
        document = {
            "source": file_path.name,
            "text": text,
        }
        # add document to list
        documents.append(document)

    return documents

# 2. chunk our source documents
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

# 3. embed chunks
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

# 4. store embedded chunk objects in collection
def index_documents() -> int:
    """load, chunk, embed and store all documents """
    # 1. load documents
    documents = load_documents()
    # 2. create chunks
    chunks = chunk_documents(documents)
    # return early if nothing exists
    if not chunks: return 0
    # 3. get embedding model
    model = get_embedding_model()
    # 4. embed chunks
    embedded_chunks = embed_documents(model, chunks)
    # 5. get chroma collection
    collection = get_collection()
    # 6. store embedded chunks
    store_embeds(collection, embedded_chunks)
    # 7. return total stored chunk count
    return collection.count()
 
# (MAIN) semantic search function (returns most semantically-similar chunks)
def semantic_search(query: str, k: int = 3) -> list[dict]:
    """semantically retrieve top-k matching chunks from chroma without using an llm"""
    # validate query
    if not query.strip(): raise ValueError("Search query cannot be empty")
    # validate k (how many top search results you want back)
    if k < 1: raise ValueError("k must be at least 1")

    # get chroma collection
    collection = get_collection()
    # get number of stored chunks
    collection_count = collection.count()
    # return empty result if nothing is indexed
    if collection_count == 0: return []

    try:
        # get embedding model
        model = get_embedding_model()
        # convert user query into embedding vector
        query_embedding = model.encode_query(query, normalize_embeddings=True).tolist()
        # search chroma for nearest embeddings
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, collection_count), # # of results (do not request more results than exist)
        )

    except Exception as error: raise RuntimeError(f"Vector search failed: {error}") from error

    # holds cleaned search results
    search_results = []
    # chroma returns nested lists because it supports multiple queries
    ids = results["ids"][0]
    documents = results["documents"][0] if results["documents"] is not None else []
    metadatas = results["metadatas"][0] if results["metadatas"] is not None else []
    distances = results["distances"][0] if results["distances"] is not None else []
    # combine matching result fields
    for i in range(len(ids)):
        result = {
            "chunk_id": ids[i],
            "text": documents[i],
            "metadata": metadatas[i],
            "distance": distances[i],
        }

        search_results.append(result)

    return search_results

# -----------------------------------------------------------------------------------------------------------------------------

# 9. preview chunks
# def preview_documents() -> None:
#     """
#     load and chunk documents then print them for inspection
#     """
#     # load documents
#     documents = load_documents()
#     print(f"Loaded {len(documents)} documents")

#     # chunk documents
#     chunks = chunk_documents(documents)
#     print(f"Created {len(chunks)} chunks\n")

#     # print every chunk
#     for chunk in chunks:
#         print("=" * 70)
#         print(f"CHUNK ID: {chunk['chunk_id']}")
#         print(f"SOURCE: {chunk['source']}")
#         print(f"CHUNK NUMBER: {chunk['chunk_number']}")
#         print("-" * 70)
#         print(chunk["text"])
#         print()

# ? 10. command line entry point
# def main() -> None:
#     """
#     run either:

#     python -m source.knowledge preview
#     python -m source.knowledge index
#     """
#     import argparse

#     # create command line parser
#     parser = argparse.ArgumentParser(description="Prepare the Fetchdex knowledge base")
#     # user must choose preview or index
#     parser.add_argument("mode", choices=["preview", "index"])
#     # read command line arguments
#     args = parser.parse_args()
#     # preview chunks
#     # if args.mode == "preview":
#     #     preview_documents()
#     # index chunks
#     # else:
#     #     stored_count = index_documents()
#     #     print(f"Stored {stored_count} chunks in Chroma")
#     stored_count = index_documents()
#     print(f"Stored {stored_count} chunks in Chroma")

# if __name__ == "__main__":
#     main()