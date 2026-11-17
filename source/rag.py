"""generate answers grounded in retrieved document evidence"""
from ollama import chat
from source.semantic import semantic_search

MODEL_NAME = "qwen3:1.7b"
REFUSAL = "I don't know based on the provided documents"

# -----------------------------------------------------------------------------------------------------------------------------
# HELPER FUNCTIONS

# format search results for the language model
def build_context(results: list[dict]) -> str:
    """convert retrieved chunks into numbered source blocks
       the source numbers let the model cite evidence as [SOURCE X]"""
    # format each result and join the source blocks with separators
    return "\n\n---\n\n".join(
        (
            # number the source and include its filename, chunk number and text
            f"[SOURCE {number}]\n"
            f"Document: {result['metadata']['source']}\n"
            f"Chunk: {result['metadata']['chunk_number']}\n"
            f"Content:\n{result['text']}"
        )
        # start numbering at 1 so citations are easy to read
        for number, result in enumerate(results, start=1)
    )

# create the complete instructions sent to the language model
def build_prompt(question: str, context: str) -> str:
    """combine the retrieved context and user question with grounding rules"""
    # tell the model to use only our documents, cite sources and refuse unsupported questions
    return f"""
You are a personal knowledge assistant. Answer using only the provided context. Do not invent facts or use outside knowledge.
If the context does not support an answer, say: "{REFUSAL}". Cite supporting evidence(s) with [SOURCE X]. Keep the answer concise.
                CONTEXT:
                {context}
                QUESTION:
                {question}
                ANSWER:
"""

# (MAIN) grounded RAG function
def rag_search(question: str, k: int = 3) -> dict:
    """retrieve relevant chunks and ask the local model to answer from them 
       returns the answer together with its sources"""

    # 1. retrieve the top-k document chunks related to the question
    results = semantic_search(question, k)

    # 2. refuse immediately when the search found no evidence
    if not results:
        return {
            "answer": REFUSAL,
            "sources": [],
        }

    # 3. prepare source information that will be returned with the answer
    sources = [
        {
            # match the source number used in the prompt and preserve its filename
            "citation": f"SOURCE {number}",
            "source": result["metadata"]["source"],
        }
        for number, result in enumerate(results, start=1)
    ]

    # 4. ask the local language model to answer (the retrieved evidence is context)
    try:
        response = chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    # build the numbered context first, then place it in the prompt
                    "content": build_prompt(question, build_context(results)),
                }],)

    # 5. let the API return HTTP 503 if ollama or the model is unavailable
    except Exception as error:
        raise RuntimeError(f"Local language model unavailable: {error}") from error

    # 6. return the model's answer and the evidence used to create it
    return {
        # ollama stores the generated text in response.message.content
        "answer": (response.message.content or "The local language model returned an empty response."),
        "sources": sources,
    }
