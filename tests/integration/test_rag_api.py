import pytest
import sys
from unittest.mock import Mock, patch
import numpy as np

# Mock SentenceTransformer BEFORE importing FastAPI app
mock_model = Mock()
mock_model.encode.side_effect = lambda texts, **kwargs: np.random.randn(
    len(texts) if isinstance(texts, list) else 1, 384
).astype('float32')

sys.modules['sentence_transformers'] = Mock()
sys.modules['sentence_transformers'].SentenceTransformer = Mock(return_value=mock_model)

from fastapi.testclient import TestClient
from src.rag_service.app import app


@pytest.fixture
def client():
    """Provide test client."""
    return TestClient(app)


def test_health_endpoint(client):
    """Test health check endpoint."""
    response = client.get("/health")
    
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert "documents_indexed" in response.json()


def test_ingest_simple_document(client):
    """Test ingesting a simple document."""
    response = client.post("/ingest", json={
        "text": "This is a test document about FastAPI and FAISS.",
        "doc_id": "test1"
    })
    
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["chunks_created"] > 0


def test_search_returns_results(client):
    """Test search returns relevant results."""
    # First ingest
    client.post("/ingest", json={
        "text": "FastAPI is a modern Python web framework for building APIs.",
        "doc_id": "apidoc"
    })
    
    # Then search
    response = client.get("/search?query=Python%20framework&top_k=1")
    
    assert response.status_code == 200
    results = response.json()
    # With mocked embeddings, just verify we get results
    assert len(results) > 0
    # Verify result has required fields
    assert "rank" in results[0]
    assert "text" in results[0]
    assert "score" in results[0]
    assert "id" in results[0]


def test_search_empty_query(client):
    """Test search with empty query returns error."""
    response = client.get("/search?query=&top_k=5")
    
    assert response.status_code == 422  # Validation error


def test_search_invalid_top_k(client):
    """Test search with invalid top_k returns error."""
    response = client.get("/search?query=test&top_k=1000")
    
    assert response.status_code == 422  # Validation error


def test_ingest_empty_text(client):
    """Test ingesting empty text returns error."""
    response = client.post("/ingest", json={
        "text": "",
        "doc_id": "empty"
    })
    
    assert response.status_code == 422  # Pydantic min_length validation


def test_ingest_missing_text(client):
    """Test ingesting without text returns error."""
    response = client.post("/ingest", json={
        "doc_id": "nodoc"
    })
    
    assert response.status_code == 422  # Validation error