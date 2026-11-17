# expose Fetchdex API through http

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from source.assistant import assistant_search
from source.semantic import index_documents, semantic_search


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """build the Chroma index once whenever the API starts"""
    index_documents()
    yield


app = FastAPI(title="Fetchdex", description="AI Study Assistant.", version="2.0.0", lifespan=lifespan)

class AskRequest(BaseModel):
    """validate JSON body to ensure question field between 1 and 1000 length"""
    question: str = Field(..., min_length=1, max_length=1000)


@app.get("/search")
def search(query: str = Query(..., min_length=1, max_length=1000), chunks: int = Query(3, ge=1, le=10),):
    """
    return raw semantic matches without calling LLM
    - query: search query inputted by user
    - chunks: number of matching chunks to return
    """
    try:
        results = semantic_search(query, chunks)
        return {
            "query": query,
            "results": [
                {
                    "text": result["text"],
                    "source": result["metadata"]["source"],
                }
                for result in results
            ],
        }

    except ValueError as error: raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error: raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error: raise HTTPException(status_code=500, detail=f"Search failed: {error}") from error


@app.post("/ask")
def ask(request: AskRequest):
    """
    route a request through the LLM assistant and its tools
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Error: Question cannot be empty.")

    try:
        return {
            "query": request.question,
            "answer": assistant_search(request.question),}

    except ValueError as error: raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error: raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error: raise HTTPException(status_code=500, detail=f"Assistant failed: {error}") from error


@app.get("/health")
def health():
    """return simple liveness response"""
    return {"status": "ok"}
