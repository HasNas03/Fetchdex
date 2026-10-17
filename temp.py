
def load_documents() -> list[dict]:
    """
    Load every .txt file from the documents folder.

    Returns:
        [
            {
                "source": "file1.txt",
                "text": "..."
            },
            {
                "source": "file2.txt",
                "text": "..."
            }
        ]
    """

    # Holds every loaded document.
    documents = []

    # Find every .txt file in the documents folder.
    file_paths = DOCUMENTS_FOLDER.glob("*.txt")

    # Process each file one at a time.
    for file_path in file_paths:

        # Read the entire text file.
        text = file_path.read_text(
            encoding="utf-8"
        )

        # Represent the document as a dictionary.
        document = {
            "source": file_path.name,
            "text": text,
        }

        # Add the document to the final list.
        documents.append(document)

    # Return all loaded documents.
    return documents


# -----------------------------------------------------------------------------------------------------------------------------
# 2. Create chunks

def chunk_text(
    text: str,
    chunk_size: int = 40,
    overlap: int = 10,
) -> list[str]:
    """
    Split text into overlapping chunks of words.

    Example:
        chunk_size = 40
        overlap = 10

    Chunk 1:
        words 0-39

    Chunk 2:
        words 30-69

    Chunk 3:
        words 60-99

    The overlap allows neighboring chunks to share some context.
    """

    # chunk_size must be at least 1.
    if chunk_size < 1:
        raise ValueError(
            "chunk_size must be at least 1."
        )

    # overlap cannot be negative.
    if overlap < 0:
        raise ValueError(
            "overlap cannot be negative."
        )

    # overlap must be smaller than chunk_size.
    #
    # Example:
    # chunk_size = 40
    # overlap = 40
    #
    # would give:
    # step = 40 - 40 = 0
    #
    # which would cause the loop to never move forward.
    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be between 0 and chunk_size - 1."
        )

    # Convert the entire string into individual words.
    words = text.split()

    # This will hold every produced chunk.
    chunks = []

    # Calculate how far forward we move after creating each chunk.
    #
    # Example:
    # chunk_size = 40
    # overlap = 10
    #
    # step = 30
    #
    # So:
    # first chunk starts at word 0
    # second chunk starts at word 30
    # third chunk starts at word 60
    step = chunk_size - overlap

    # Start at the first word.
    start = 0

    # Continue until we pass the end of the document.
    while start < len(words):

        # Calculate where this chunk should end.
        end = start + chunk_size

        # Extract only the words belonging to this chunk.
        chunk_words = words[start:end]

        # Join the words back into normal text.
        chunk = " ".join(chunk_words)

        # Save the chunk.
        chunks.append(chunk)

        # Move forward while retaining the configured overlap.
        start += step

    # Return all chunks.
    return chunks


def chunk_documents(
    documents: list[dict]
) -> list[dict]:
    """
    Convert complete documents into smaller chunks while
    preserving document metadata.

    Input:
        [
            {
                "source": "notes.txt",
                "text": "..."
            }
        ]

    Output:
        [
            {
                "chunk_id": "notes.txt-0",
                "source": "notes.txt",
                "chunk_number": 0,
                "text": "..."
            },
            ...
        ]
    """

    # Holds chunks from every document.
    all_chunks = []

    # Process one document at a time.
    for document in documents:

        # Extract the document text.
        document_text = document["text"]

        # Split the complete document into smaller chunks.
        text_chunks = chunk_text(
            document_text
        )

        # enumerate() gives us both:
        #
        # chunk_number
        # text
        #
        # Example:
        # 0, "first chunk"
        # 1, "second chunk"
        # 2, "third chunk"
        for chunk_number, text in enumerate(text_chunks):

            # Create a dictionary representing this chunk.
            chunk = {
                # Unique Chroma ID for this chunk.
                "chunk_id": (
                    f"{document['source']}-{chunk_number}"
                ),

                # Original document filename.
                "source": document["source"],

                # Position of this chunk within its document.
                "chunk_number": chunk_number,

                # Actual chunk text.
                "text": text,
            }

            # Add this chunk to the complete chunk list.
            all_chunks.append(chunk)

    # Return chunks from all documents.
    return all_chunks


# -----------------------------------------------------------------------------------------------------------------------------
# 3. Load embedding model

@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """
    Load and return the SentenceTransformer embedding model.

    @lru_cache(maxsize=1) means:

    The first time this function runs:
        -> load the model from disk / Hugging Face cache

    Later calls:
        -> return the already-loaded model

    This prevents the application from repeatedly loading the
    same embedding model into memory.
    """

    # Create/load the embedding model.
    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    # Return the model object.
    return model


# -----------------------------------------------------------------------------------------------------------------------------
# 4. Connect to Chroma

@lru_cache(maxsize=1)
def get_collection():
    """
    Open the persistent Chroma database and return our collection.

    The database is stored locally at:

        data/chroma

    @lru_cache(maxsize=1) means we reuse the same collection
    object during the lifetime of this Python process.
    """

    # Connect to the persistent Chroma database on disk.
    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    # Get the existing collection if it exists.
    #
    # Otherwise create it.
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,

        # Tell Chroma that embedding distance should be
        # measured using cosine distance.
        metadata={
            "hnsw:space": "cosine"
        },
    )

    # Return the collection object.
    return collection


# -----------------------------------------------------------------------------------------------------------------------------
# 5. Index documents

def index_documents() -> int:
    """
    Complete indexing pipeline:

    1. load documents
    2. chunk documents
    3. load embedding model
    4. embed chunk text
    5. connect to Chroma
    6. upsert chunks + embeddings + metadata
    7. return number of stored chunks
    """

    # ---------------------------------------------------------------------
    # Step 1: Load documents

    documents = load_documents()

    # ---------------------------------------------------------------------
    # Step 2: Split documents into chunks

    chunks = chunk_documents(
        documents
    )

    # If there are no chunks, there is nothing to index.
    if not chunks:
        return 0

    # ---------------------------------------------------------------------
    # Step 3: Load the embedding model

    model = get_embedding_model()

    # ---------------------------------------------------------------------
    # Step 4: Extract only text from each chunk

    texts = []

    for chunk in chunks:
        texts.append(
            chunk["text"]
        )

    # ---------------------------------------------------------------------
    # Step 5: Convert chunk text into embedding vectors

    embeddings = model.encode_document(
        texts,
        normalize_embeddings=True,
    )

    # ---------------------------------------------------------------------
    # Step 6: Connect to Chroma

    collection = get_collection()

    # ---------------------------------------------------------------------
    # Step 7: Prepare each Chroma field

    ids = []

    documents_to_store = []

    embedding_vectors = []

    metadatas = []

    # Process each chunk together with its generated embedding.
    for chunk, embedding in zip(
        chunks,
        embeddings,
    ):

        # Chroma ID.
        ids.append(
            chunk["chunk_id"]
        )

        # Actual text stored in Chroma.
        documents_to_store.append(
            chunk["text"]
        )

        # Convert NumPy embedding vector into normal Python list.
        embedding_vectors.append(
            embedding.tolist()
        )

        # Metadata stored alongside the chunk.
        metadata = {
            "source": chunk["source"],
            "chunk_number": chunk["chunk_number"],
        }

        metadatas.append(
            metadata
        )

    # ---------------------------------------------------------------------
    # Step 8: Insert or update records in Chroma

    collection.upsert(
        ids=ids,
        documents=documents_to_store,
        embeddings=embedding_vectors,
        metadatas=metadatas,
    )

    # ---------------------------------------------------------------------
    # Step 9: Return total number of chunks in the collection

    stored_count = collection.count()

    return stored_count


# -----------------------------------------------------------------------------------------------------------------------------
# 6. Semantic search

def semantic_search(
    query: str,
    k: int = 3,
) -> list[dict]:
    """
    Search Chroma for the chunks whose embeddings are
    semantically closest to the user's query.

    This function does NOT call an LLM.

    It only performs vector retrieval.

    Example:

        query:
            "What did I write about AWS?"

        result:
            top matching chunks from Chroma
    """

    # Remove surrounding spaces and make sure the query
    # contains actual text.
    if not query.strip():
        raise ValueError(
            "Search query cannot be empty."
        )

    # k determines how many results we want.
    if k < 1:
        raise ValueError(
            "k must be at least 1."
        )

    # Get the Chroma collection.
    collection = get_collection()

    # Determine how many chunks currently exist.
    collection_count = collection.count()

    # If the database contains no chunks, there is
    # nothing to search.
    if collection_count == 0:
        return []

    try:

        # -----------------------------------------------------------------
        # Step 1: Load embedding model

        model = get_embedding_model()

        # -----------------------------------------------------------------
        # Step 2: Convert the user's query into an embedding

        query_vector = model.encode_query(
            query,
            normalize_embeddings=True,
        )

        # Convert NumPy vector into normal Python list.
        query_vector = query_vector.tolist()

        # -----------------------------------------------------------------
        # Step 3: Decide how many results Chroma should return

        # We cannot ask for more results than actually exist.
        #
        # Example:
        #
        # k = 10
        # collection contains 4 chunks
        #
        # n_results becomes 4
        result_count = min(
            k,
            collection_count,
        )

        # -----------------------------------------------------------------
        # Step 4: Search Chroma using the query embedding

        raw_results = collection.query(
            query_embeddings=[
                query_vector
            ],
            n_results=result_count,
        )

    except Exception as error:

        # Convert lower-level Chroma/model errors into a clearer
        # application-level error.
        raise RuntimeError(
            f"Vector search failed: {error}"
        ) from error

    # ---------------------------------------------------------------------
    # Step 5: Extract individual result fields

    # Chroma returns nested lists because one query call can
    # technically contain multiple query embeddings.
    #
    # We only sent ONE query, so we access index [0].

    ids = raw_results.get(
        "ids",
        [[]]
    )[0]

    texts = raw_results.get(
        "documents",
        [[]]
    )[0]

    metadatas = raw_results.get(
        "metadatas",
        [[]]
    )[0]

    distances = raw_results.get(
        "distances",
        [[]]
    )[0]

    # ---------------------------------------------------------------------
    # Step 6: Convert Chroma's response into cleaner dictionaries

    search_results = []

    # zip() groups corresponding values together.
    #
    # Example:
    #
    # id1, text1, metadata1, distance1
    # id2, text2, metadata2, distance2
    for (
        chunk_id,
        text,
        metadata,
        distance,
    ) in zip(
        ids,
        texts,
        metadatas,
        distances,
    ):

        # Build a cleaner result object.
        result = {
            "chunk_id": chunk_id,
            "text": text,
            "metadata": metadata,
            "distance": distance,
        }

        # Add it to the final result list.
        search_results.append(
            result
        )

    return search_results


# -----------------------------------------------------------------------------------------------------------------------------
# 7. Preview documents

def preview_documents() -> None:
    """
    Load and chunk documents, then print the resulting chunks.

    This does NOT:
        - generate embeddings
        - modify Chroma
        - call an LLM

    It is only a debugging / inspection helper.
    """

    # Load all documents.
    documents = load_documents()

    # Show how many files were found.
    print(
        f"Loaded {len(documents)} documents."
    )

    # Chunk all documents.
    chunks = chunk_documents(
        documents
    )

    # Show how many total chunks were created.
    print(
        f"Created {len(chunks)} chunks.\n"
    )

    # Print each chunk.
    for chunk in chunks:

        print(
            "=" * 70
        )

        print(
            f"CHUNK ID: {chunk['chunk_id']}"
        )

        print(
            f"SOURCE: {chunk['source']}"
        )

        print(
            f"CHUNK NUMBER: {chunk['chunk_number']}"
        )

        print(
            "-" * 70
        )

        print(
            chunk["text"]
        )

        print()


# -----------------------------------------------------------------------------------------------------------------------------
# 8. CLI entry point

def main() -> None:
    """
    Allow knowledge.py to be run directly from the terminal.

    Supported commands:

        python -m source.knowledge preview

        python -m source.knowledge index
    """

    # argparse is only needed when running this file
    # directly from the command line.
    import argparse

    # Create the command-line argument parser.
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the Fetchdex knowledge base."
        )
    )

    # Require one positional argument called "mode".
    #
    # It must be either:
    #
    # preview
    # index
    parser.add_argument(
        "mode",
        choices=[
            "preview",
            "index",
        ],
    )

    # Read the command-line arguments.
    args = parser.parse_args()

    # If user runs:
    #
    # python -m source.knowledge preview
    #
    # print the chunks.
    if args.mode == "preview":

        preview_documents()

    # Otherwise:
    #
    # python -m source.knowledge index
    #
    # build/update the Chroma index.
    else:

        stored_count = index_documents()

        print(
            f"Stored {stored_count} chunks in Chroma."
        )


# -----------------------------------------------------------------------------------------------------------------------------
# 9. Run main() only when this file is executed directly

if __name__ == "__main__":
    main()