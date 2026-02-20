import pytest
import tempfile
import os
import numpy as np
import src.rag_service.vectorstore as vectorstore_module
from src.rag_service.vectorstore import FAISSVectorStore


@pytest.fixture(autouse=True)
def mock_sentence_transformer(monkeypatch):
    class DummyModel:
        def encode(self, texts, convert_to_numpy=True):
            return np.zeros((len(texts), 384), dtype=np.float32)

    monkeypatch.setattr(
        vectorstore_module,
        "SentenceTransformer",
        lambda *args, **kwargs: DummyModel(),
    )


def test_vectorstore_init():
    """Test vectorstore initialization."""
    store = FAISSVectorStore(model_name="all-MiniLM-L6-v2")
    
    assert store.index is not None
    assert store.dimension == 384
    assert store.index.ntotal == 0


def test_add_texts():
    """Test adding texts to store."""
    store = FAISSVectorStore()
    texts = [
        "Python is a programming language",
        "FastAPI is a web framework"
    ]
    ids = ["doc1", "doc2"]
    
    store.add_texts(texts, ids)
    
    assert store.index.ntotal == 2
    assert "doc1" in store.metadata
    assert "doc2" in store.metadata


def test_search_returns_results():
    """Test search returns relevant results."""
    store = FAISSVectorStore()
    store.add_texts(
        ["Python programming language"],
        ["doc1"]
    )
    
    results = store.search("Python code", top_k=1)
    
    assert len(results) == 1
    assert results[0]["id"] == "doc1"
    assert results[0]["score"] > 0.5  # Should have high similarity


def test_search_empty_store():
    """Test search on empty store returns empty list."""
    store = FAISSVectorStore()
    
    results = store.search("anything", top_k=5)
    
    assert results == []


def test_persistence():
    """Test saving and loading from disk."""
    store = FAISSVectorStore()
    store.add_texts(["Test document"], ["test1"])
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Persist
        store.persist(tmpdir)
        
        # Load into new store
        store2 = FAISSVectorStore()
        store2.load(tmpdir)
        
        # Verify
        assert store2.index.ntotal == 1
        assert "test1" in store2.metadata