"""Expose Fetchdex through HTTP."""
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from source.assistant import assistant_search
from source.semantic import semantic_search


app = FastAPI(
    title="Fetchdex",
    description="Local semantic search and document AI assistant.",
    version="2.0.0",
)

class AskRequest(BaseModel):
    """validate the JSON body for POST requests"""
    question: str = Field(..., min_length=1, max_length=1000)


@app.get("/search")
def search(q: str = Query(..., min_length=1, max_length=1000), k: int = Query(3, ge=1, le=10),):
    """return raw semantic matches without calling the LLM"""
    try:
        return {"query": q, 
                "answer": semantic_search(q, k)}
    
    except ValueError as error: raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error: raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error: raise HTTPException(status_code=500, detail=f"Search failed: {error}") from error


@app.post("/ask")
def ask(request: AskRequest):
    """route a request through the LLM assistant and its document tools"""
    if not request.question.strip(): raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        return {
            "query": request.question,
            "answer": assistant_search(request.question),}
    
    except ValueError as error: raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error: raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error: raise HTTPException(status_code=500, detail=f"Assistant failed: {error}") from error


@app.get("/health")
def health():
    """return a simple liveness response"""
    return {"status": "ok"}
