from fastapi import FastAPI, HTTPException, Query, Depends
from pydantic import BaseModel, Field
from typing import List, Optional
import os
from dotenv import load_dotenv
import time

from .chunker import TextChunker
from .vectorstore import FAISSVectorStore

load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="RAG Service API",
    description="Vector store and retrieval API",
    version="1.0.0"
)


# Dependency injection for services
def get_chunker() -> TextChunker:
    """Get or create chunker instance."""
    if not hasattr(get_chunker, "_instance"):
        get_chunker._instance = TextChunker()
    return get_chunker._instance


def get_vector_store() -> FAISSVectorStore:
    """Get or create vector store instance."""
    if not hasattr(get_vector_store, "_instance"):
        get_vector_store._instance = FAISSVectorStore()
    return get_vector_store._instance


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    # Services will be initialized lazily on first use via dependencies
    print("✅ App startup complete")


# Pydantic models
class IngestRequest(BaseModel):
    """Request model for document ingestion."""
    text: str = Field(..., min_length=1, description="Document text to ingest")
    doc_id: str = Field(..., min_length=1, description="Unique document ID")


class SearchResult(BaseModel):
    """Result item from search."""
    rank: int
    id: str
    text: str
    score: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    documents_indexed: int


# Endpoints
@app.post("/ingest", responses={200: {"description": "Document ingested successfully"}})
async def ingest(request: IngestRequest, chunker: TextChunker = Depends(get_chunker), vector_store: FAISSVectorStore = Depends(get_vector_store)):
    """Ingest a document into the vector store.
    
    Args:
        request: Document text and ID
        
    Returns:
        Success status and number of chunks created
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    # Chunk the text
    chunks = chunker.chunk(request.text)
    
    if not chunks:
        raise HTTPException(status_code=400, detail="Text produced no chunks")
    
    # Create IDs for chunks
    chunk_ids = [f"{request.doc_id}_{i}" for i in range(len(chunks))]
    
    # Add to vector store
    vector_store.add_texts(chunks, chunk_ids)
    
    return {
        "status": "success",
        "chunks_created": len(chunks),
        "doc_id": request.doc_id
    }


@app.get("/search", response_model=List[SearchResult])
async def search(
    query: str = Query(..., min_length=1, max_length=1000, description="Search query"),
    top_k: int = Query(5, ge=1, le=20, description="Number of results"),
    chunker: TextChunker = Depends(get_chunker),
    vector_store: FAISSVectorStore = Depends(get_vector_store)
):
    """Search for similar documents.
    
    Args:
        query: Search query text
        top_k: Number of results to return
        
    Returns:
        List of search results ranked by relevance
    """
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    if top_k > 20:
        raise HTTPException(status_code=400, detail="top_k cannot exceed 20")
    
    # Search
    start_time = time.time()
    results = vector_store.search(query, top_k=top_k)
    latency_ms = (time.time() - start_time) * 1000
    
    # Convert to response model
    response = [SearchResult(**result) for result in results]
    
    return response


@app.get("/health", response_model=HealthResponse)
async def health(vector_store: FAISSVectorStore = Depends(get_vector_store)):
    """Health check endpoint.
    
    Returns:
        Service status and document count
    """
    return {
        "status": "healthy",
        "documents_indexed": vector_store.index.ntotal
    }


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)