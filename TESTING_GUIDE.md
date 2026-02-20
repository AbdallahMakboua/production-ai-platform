# Testing & Verification Guide

## 📋 Overview

Every day requires a **Testing Pyramid** approach:
1. **Unit Tests** (70%) - Fast, isolated, single function/class
2. **Integration Tests** (25%) - Component interaction, endpoints
3. **E2E Tests** (5%) - Full workflow (starting Day 7)
4. **Manual Verification** - curl, scripts, browser

---

## 🧪 Unit Testing Convention

### Pattern
```python
# tests/unit/test_module_name.py
import pytest
from src.module.file import FunctionOrClass

class TestFunctionName:
    """Test suite for specific function/class"""
    
    def setup_method(self):
        """Run before each test"""
        self.fixture = setup_test_data()
    
    def test_happy_path__(self):
        """Test normal, expected behavior"""
        result = FunctionOrClass.method(valid_input)
        assert result == expected_output
    
    def test_edge_case_empty_input(self):
        """Test boundary conditions"""
        result = FunctionOrClass.method("")
        assert result == []
    
    def test_error_invalid_input(self):
        """Test error handling"""
        with pytest.raises(ValueError):
            FunctionOrClass.method(invalid_data)
    
    def teardown_method(self):
        """Run after each test"""
        cleanup()

# Rule: For every 10 lines of code, write 15-20 lines of test code
# Rule: 3 tests minimum per class/function
```

### Running Unit Tests
```bash
# Single file
pytest tests/unit/test_module.py -v

# All unit tests with coverage
pytest tests/unit/ -v --cov=src

# Show which lines aren't covered
pytest tests/unit/ --cov=src --cov-report=html
open htmlcov/index.html
```

---

## 🔗 Integration Testing Convention

### Pattern
```python
# tests/integration/test_api_flow.py
import pytest
from fastapi.testclient import TestClient
from src.service.app import app

@pytest.fixture
def client():
    """Provide test client for each test"""
    return TestClient(app)

@pytest.fixture(autouse=True)
def setup_clean_state(client):
    """Ensure each test starts fresh"""
    # Clear database/cache
    yield
    # Cleanup after test

class TestEndpointFlow:
    """Test multi-step API workflows"""
    
    def test_ingest_then_search_flow(self, client):
        """Full: POST /ingest → GET /search"""
        # Step 1: Setup
        ingest_response = client.post("/ingest", json={...})
        assert ingest_response.status_code == 200
        doc_id = ingest_response.json()["doc_id"]
        
        # Step 2: Act
        search_response = client.get(f"/search?doc_id={doc_id}")
        
        # Step 3: Assert
        assert search_response.status_code == 200
        assert search_response.json()["found"] == True
    
    def test_concurrent_ingests(self, client):
        """Test parallel requests"""
        # Use pytest-asyncio or ThreadPoolExecutor
        pass
    
    def test_error_recovery(self, client):
        """Test API behavior on failure and recovery"""
        pass
```

### Running Integration Tests
```bash
# Start service first
uvicorn src.app:app --reload &

# Run integration tests
pytest tests/integration/ -v

# Stop service
pkill uvicorn
```

---

## ✅ Manual Verification Pattern

### Curl Scripts
```bash
#!/bin/bash
# scripts/test_day01_rag.sh

set -e  # Exit on error

echo "=== Testing RAG Service ==="

# Test 1: Health check
echo "Test 1: Health endpoint"
response=$(curl -s http://localhost:8000/health)
echo "Response: $response"
assert_contains "$response" "healthy"

# Test 2: Ingest document
echo "Test 2: Ingest document"
response=$(curl -s -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "text": "FastAPI is awesome",
    "doc_id": "test1"
  }')
echo "Response: $response"
assert_contains "$response" "success"

# Test 3: Search
echo "Test 3: Search for content"
response=$(curl -s "http://localhost:8000/search?query=fastapi&top_k=1")
echo "Response: $response"
assert_contains "$response" "FastAPI"

echo "✅ All manual tests passed!"
```

### Verification Checklist
```
□ Service starts without errors
□ Health endpoint responds 200
□ Ingest accepts valid documents
□ Ingest rejects invalid documents with 400
□ Search returns results for relevant queries
□ Search returns empty for no-match queries
□ Pagination works (limit/offset)
□ Error messages are descriptive
□ Latency acceptable (<500ms)
□ No memory leaks after 100 requests
```

---

## 📊 Coverage Goals By Day

| Day | Requirement | How to Check |
|-----|-------------|--------------|
| 1 | >85% | `pytest --cov=src.rag_service` |
| 2 | >87% | `pytest --cov=src.rag_service` |
| 3 | >88% | `pytest --cov=src.rag_service` |
| 4+ | >90% | Always achieve 90%+ |

### Coverage Types (Good Practice)
- **Line coverage**: Did the line execute? (minimum)
- **Branch coverage**: Did both if/else paths execute? (better)
- **Path coverage**: Did all combinations execute? (best)

### What NOT to Test (and Why)
```python
# ❌ Don't test third-party libraries
def test_sentence_transformer_works():
    model = SentenceTransformer(...)  # Don't test this
    # Just assume it works, add one sanity check if needed

# ❌ Don't test external APIs in unit tests
def test_openai_api():
    response = openai.ChatCompletion.create(...)  # Mock this instead

# ✅ Do test your code that USES these
def test_my_embedding_wrapper():
    embeddings = MyEmbeddingWrapper.embed("text")  # Test your adapter
    assert len(embeddings) == 384
```

---

## 🔧 Mocking & Fixtures

### Basic Fixtures
```python
import pytest

@pytest.fixture
def sample_documents():
    """Reusable test data"""
    return [
        {"id": "doc1", "text": "First document"},
        {"id": "doc2", "text": "Second document"},
    ]

@pytest.fixture
def vector_store(sample_documents):
    """Setup vector store with test data"""
    store = FAISSVectorStore()
    for doc in sample_documents:
        store.add_text(doc["text"], doc["id"])
    yield store  # Provide to test
    store.cleanup()  # Cleanup after

def test_something(vector_store, sample_documents):
    """Fixtures injected automatically"""
    results = vector_store.search("First")
    assert results[0]["id"] == "doc1"
```

### Mocking External Calls
```python
from unittest.mock import patch, MagicMock

def test_embedding_with_mock():
    """Mock external embedding service"""
    with patch('sentence_transformers.SentenceTransformer') as mock_model:
        mock_model.return_value.encode.return_value = [0.1, 0.2, 0.3]
        
        embeddings = get_embeddings("text")
        assert embeddings == [0.1, 0.2, 0.3]
        mock_model.return_value.encode.assert_called_once()
```

---

## 🚀 Test Execution Workflow

### Before Each Commit
```bash
# 1. Run all tests with coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# 2. Check coverage meets minimum (90%)
# If not: add more tests

# 3. Run linting
flake8 src/

# 4. Type checking (optional but recommended)
mypy src/ --ignore-missing-imports

# 5. Format code
black src/ tests/

# 6. Only then commit
git add .
git commit -m "feat: clear message"
```

### Sample Pre-Commit Script
```bash
#!/bin/bash
# .githooks/pre-commit

set -e

echo "🧪 Running tests..."
pytest tests/ --cov=src -q
coverage_percent=$(coverage report -m | grep TOTAL | awk '{print $(NF-1)}' | tr -d '%')

if [ "$coverage_percent" -lt 90 ]; then
    echo "❌ Coverage ${coverage_percent}% < 90%"
    exit 1
fi

echo "✅ Tests pass, coverage is ${coverage_percent}%"
```

---

## 📈 Pytest Configuration

### pytest.ini
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --strict-markers
    --tb=short
    --disable-warnings
markers =
    slow: marks tests as slow
    integration: marks tests as integration
    unit: marks tests as unit
```

### Run Specific Test Types
```bash
# Only unit tests
pytest tests/unit/ -m "unit"

# Only integration tests  
pytest tests/integration/ -m "integration"

# Skip slow tests
pytest -m "not slow"
```

---

## 🐛 Debugging Failed Tests

### Step 1: Run with verbose output
```bash
pytest tests/test_file.py::TestClass::test_method -vv
```

### Step 2: Print debug information
```python
def test_something(vector_store):
    result = vector_store.search("query")
    print(f"DEBUG: {result}")  # Visible with pytest -s
    assert result == expected

# Run with print visible
pytest -s tests/test_file.py
```

### Step 3: Use debugger
```python
def test_something():
    import pdb; pdb.set_trace()  # Breakpoint
    # or in Python 3.7+:
    breakpoint()  # Breakpoint
    
# Run with debugger
pytest --pdb tests/test_file.py
```

### Step 4: Check for flaky tests
```bash
# Run test 10 times (useful for race conditions)
pytest tests/test_file.py::test_method --count=10
```

---

## 📋 Critical Paths Testing

For core RAG functionality:

```python
# MUST HAVE 100% COVERAGE
def test_vector_search_correctness():
    """Cosine similarity > 0.7 for exact match"""
    store = FAISSVectorStore()
    store.add_texts(["Python programming"], ["doc1"])
    
    results = store.search("Python code")
    assert results[0]["score"] > 0.7
    assert results[0]["id"] == "doc1"

def test_chunking_no_data_loss():
    """All original text present after chunking"""
    chunker = TextChunker()
    original = "sentence1. sentence2. sentence3."
    
    chunks = chunker.chunk(original)
    reassembled = " ".join(chunks)
    
    # Check all original text is present
    for sentence in original.split("."):
        if sentence.strip():
            assert sentence.strip() in reassembled

def test_ingestion_then_retrieval():
    """End-to-end: ingest → embed → store → retrieve"""
    client = TestClient(app)
    
    # Ingest
    client.post("/ingest", json={"text": "...", "doc_id": "x"})
    
    # Retrieve
    response = client.get("/search?query=...")
    assert response.status_code == 200
    assert len(response.json()) > 0
```

---

## 🔄 Continuous Integration (CI) Pattern

```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      
      - name: Run tests
        run: |
          pytest tests/ -v --cov=src --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v2
        with:
          files: ./coverage.xml
```

---

## ✅ Day-by-Day Testing Expectations

### Day 1: Vector Store
- 15+ unit tests (chunking, vectorstore, FAISS ops)
- 6+ integration tests (API endpoints)
- Manual: curl ingest + search
- Coverage: 90%+

### Day 2: Reranking  
- 4+ reranker unit tests
- 2+ quality comparison tests
- Latency benchmark
- Coverage: 90%+

### Day 3-6: RAG Pipeline & Agent
- Agent reasoning tests
- Tool calling mocks
- Stream handling tests
- Async operation tests
- Coverage: 90%+

### Day 7: Observability
- Metrics collection tests
- Log parsing tests
- Alert threshold tests
- Coverage: 90%+

### Days 8-14: Execution Layer
- Security tests (auth, validation)
- Cache behavior tests
- Rate limiting tests
- Circuit breaker tests
- E2E flow tests (10+)
- Coverage: 90%+

### Days 15-21: Production
- Multi-agent coordination tests
- Database transaction tests
- K8s deployment tests
- Load testing (simulated)
- Chaos testing (failure scenarios)
- Security audit tests

---

## 📞 When Tests Fail

### Red ❌ → Green ✅ → Refactor 🔄 Cycle

1. **Red**: Write failing test
2. **Green**: Write minimum code to pass
3. **Refactor**: Clean up implementation

### Common Failures & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `AssertionError: X != Y` | Logic error | Debug with print/pdb |
| `ImportError: no module` | Dependency missing | `pip install X` |
| `Timeout` | Code too slow | Profile with `cProfile` |
| `Flaky test` | Race condition | Add synchronization |
| `Coverage gap` | Code path untested | Write specific test |

---

## 🎯 Code Coverage Deep Dive

### Check what's NOT covered
```bash
pytest tests/ --cov=src --cov-report=term-missing

# Shows line numbers not covered:
# src/module.py:45 not covered
# src/module.py:67-72 not covered (branch)
```

### HTML Coverage Report
```bash
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html
```

### Coverage per Module Goals
```
rag_service/chunker.py:     100%  (critical)
rag_service/vectorstore.py: 100%  (critical)
rag_service/app.py:         95%   (API critical)
rag_service/utils.py:       85%   (helpers can be lower)
```

---

## 🚨 Testing Failures in Phase 3 (Production)

### Load Testing
```python
import concurrent.futures

def test_concurrent_requests():
    """Test 50 concurrent searches"""
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        futures = [
            executor.submit(client.get, f"/search?query=test{i}")
            for i in range(50)
        ]
        results = [f.result() for f in futures]
        
    success_count = sum(1 for r in results if r.status_code == 200)
    assert success_count > 45  # At least 90% succeed
```

### Chaos Testing
```python
def test_service_with_database_down():
    """Test graceful degradation"""
    with patch('db.connection') as mock_db:
        mock_db.side_effect = ConnectionError()
        
        response = client.get("/search?query=test")
        # Should return cached result or error gracefully
        assert response.status_code in [200, 503]
```

This testing guide is your reference for every day. Return here when unsure how to test something!
