"""Pytest configuration and shared fixtures for all tests."""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import os

# Mock SentenceTransformer before any imports
@pytest.fixture(scope="session", autouse=True)
def mock_sentence_transformer():
    """Mock SentenceTransformer globally to prevent model downloads."""
    mock_model = Mock()
    mock_model.encode.side_effect = lambda texts, **kwargs: np.random.randn(
        len(texts) if isinstance(texts, list) else 1, 384
    ).astype('float32')
    
    with patch('sentence_transformers.SentenceTransformer', return_value=mock_model):
        yield


@pytest.fixture
def mock_faiss_model():
    """Provide a mock SentenceTransformer for FAISS tests."""
    mock_model = Mock()
    mock_model.encode.side_effect = lambda texts, **kwargs: np.random.randn(
        len(texts) if isinstance(texts, list) else 1, 384
    ).astype('float32')
    return mock_model
