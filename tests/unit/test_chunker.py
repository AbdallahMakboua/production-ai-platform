import pytest
from src.rag_service.chunker import TextChunker


def test_chunk_normal_text():
    """Test chunking of normal-length text."""
    chunker = TextChunker(chunk_size=100, overlap=10)
    text = "This is a test. " * 50  # ~800 words total
    
    chunks = chunker.chunk(text)
    
    assert len(chunks) > 1, "Should create multiple chunks"
    assert all(len(c) > 0 for c in chunks), "No empty chunks"
    assert len(chunks[0].split()) <= 100


def test_chunk_empty_text():
    """Test chunking empty text returns empty list."""
    chunker = TextChunker()
    
    assert chunker.chunk("") == []
    assert chunker.chunk("   ") == []


def test_chunk_small_text():
    """Test text smaller than chunk_size returns single chunk."""
    chunker = TextChunker(chunk_size=100)
    text = "Small text with few words"
    
    chunks = chunker.chunk(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_overlap_works():
    """Test that overlap preserves context between chunks."""
    chunker = TextChunker(chunk_size=20, overlap=5)
    text = "word " * 100
    
    chunks = chunker.chunk(text)
    
    # Verify chunks overlap
    if len(chunks) > 1:
        chunk1_words = set(chunks[0].split())
        chunk2_words = set(chunks[1].split())
        overlap_words = chunk1_words & chunk2_words
        assert len(overlap_words) > 0, "Should have overlapping words"