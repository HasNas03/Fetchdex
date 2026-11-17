"""generate answers grounded in retrieved document evidence"""
from ollama import chat
from source.semantic import semantic_search

MODEL_NAME = "qwen3:1.7b"
REFUSAL = "I don't know based on the provided documents"

# -----------------------------------------------------------------------------------------------------------------------------
# HELPER FUNCTIONS

# format search results for the language model
def build_context(results: list[dict]) -> str:
    """convert retrieved chunks into document evidence blocks"""
    # include each chunk's filename and text, separated from the next chunk
    return "\n\n---\n\n".join(
        (
            f"Document: {result['metadata']['source']}\n"
            f"Content:\n{result['text']}"
        )
        for result in results
    )

# create the complete instructions sent to the language model
def build_prompt(question: str, context: str) -> str:
    """combine the retrieved context and user question with grounding rules"""
    # tell the model to use only our documents and refuse unsupported questions
    return f"""
You are a personal knowledge assistant. Answer using only the provided context. Do not invent facts or use outside knowledge.
If the context does not support an answer, say: "{REFUSAL}". Keep the answer concise. Do not add citations to the answer.
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

    # 3. preserve each filename once for the final source list
    sources = list(dict.fromkeys(
        result["metadata"]["source"] for result in results
    ))

    # 4. ask the local language model to answer (the retrieved evidence is context)
    try:
        response = chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    # build the document context first, then place it in the prompt
                    "content": build_prompt(question, build_context(results)),
                }],)

    # 5. let the command-line program report if ollama or the model is unavailable
    except Exception as error:
        raise RuntimeError(f"Local language model unavailable: {error}") from error

    # 6. return the model's answer and the filenames used to create it
    answer = response.message.content or "The local language model returned an empty response."
    return {
        "answer": answer,
        # do not show unrelated retrieved files when the model refused to answer
        "sources": [] if REFUSAL.lower() in answer.lower() else sources,
    }
