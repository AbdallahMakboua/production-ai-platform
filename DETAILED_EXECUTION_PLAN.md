# 21-Day Production AI Systems Engineer - Detailed Execution Plan

## 📋 How to Use This Document

### 1. **Daily Workflow**
   - Start morning: Read "Micro Tasks" for the day
   - Work through tasks in order (✓ check them off)
   - Run verification tests after each major section
   - Commit code when tests pass
   - End of day: Ask for documentation help with the template

### 2. **When to Commit**
   - After each micro-task passes verification tests
   - Use the provided commit message template
   - Tag with day indicator: `day-01`, `day-02`, etc.

### 3. **When to Ask for Documentation Help**
   - End of day (or when major task completes)
   - Message: "Documenting day X" + provide test results and key metrics
   - I'll help you fill the daily log markdown with proper formatting

### 4. **Test Structure**
   Each day has:
   - Unit tests (for functions/modules)
   - Integration tests (for service endpoints)
   - Verification curl/script (manual proof)

---

## PHASE 1: INTELLIGENCE LAYER (Days 1-7)

### 📅 DAY 1: Vector Store Foundation + FastAPI Service

**Objective:** Build local RAG API with FAISS for semantic search, proving retrieval quality.

#### 🎯 Micro Tasks

**TASK 1.1: Project Setup & Dependencies (45 min)**
- [X] Initialize Python project structure
  ```bash
  cd /Users/abdallahmakboua/Desktop/AI/production-ai-platform
  python -m venv venv
  source venv/bin/activate
  pip install fastapi uvicorn sentence-transformers faiss-cpu pydantic pytest pytest-cov python-dotenv
  ```
- [ ] Create `.env` file with service config
  ```
  EMBEDDING_MODEL=all-MiniLM-L6-v2
  VECTOR_DIM=384
  CHUNK_SIZE=512
  CHUNK_OVERLAP=50
  PORT=8000
  ```
- [ ] Create folder structure
  ```bash
  mkdir -p src/rag_service tests/unit tests/integration
  touch src/rag_service/__init__.py src/rag_service/app.py src/rag_service/vectorstore.py src/rag_service/chunker.py
  ```

**✅ Verification:**
```bash
# Check Python env
python --version  # Should be 3.10+
pip list | grep -E "fastapi|sentence-transformers|faiss"

# Check structure
ls -la src/rag_service/
```

**Commits:**
```
chore(setup): initialize python project with RAG dependencies

- Create virtual environment with fastapi, sentence-transformers, faiss
- Set up .env configuration template
- Initialize src/rag_service directory
```

---

**TASK 1.2: Text Chunking Module (1 hour)**
- [ ] Implement `src/rag_service/chunker.py`
  ```python
  - Class: TextChunker
    - __init__(chunk_size=512, overlap=50)
    - chunk(text: str) -> List[str]
    - Should handle edge cases: empty text, text < chunk_size
    - Strategy: word-based splitting (not char-based)
  ```
- [ ] Handle edge cases:
  - Empty text returns empty list
  - Text smaller than chunk_size returns single chunk
  - Overlap properly maintains context

**✅ Unit Tests:**
```python
# tests/unit/test_chunker.py
def test_chunk_normal_text():
    chunker = TextChunker(chunk_size=100, overlap=10)
    text = "This is a long text... " * 50  # ~2000 chars
    chunks = chunker.chunk(text)
    assert len(chunks) > 1
    assert len(chunks[0].split()) <= 100
    assert len(chunks[-1].split()) > 0

def test_chunk_empty_text():
    chunker = TextChunker()
    assert chunker.chunk("") == []

def test_chunk_small_text():
    chunker = TextChunker(chunk_size=100)
    text = "Small text"
    chunks = chunker.chunk(text)
    assert len(chunks) == 1
    assert chunks[0] == text

def test_overlap_works():
    chunker = TextChunker(chunk_size=20, overlap=5)
    text = "word " * 50
    chunks = chunker.chunk(text)
    # Check that consecutive chunks share words
    if len(chunks) > 1:
        overlap_words = set(chunks[0].split()) & set(chunks[1].split())
        assert len(overlap_words) > 0
```

**✅ Verification:**
```bash
pytest tests/unit/test_chunker.py -v --cov=src.rag_service.chunker
# Expect: 4 passed, 100% coverage
```

**Commits:**
```
feat(rag): implement text chunking with configurable overlap

- Add TextChunker class with word-based splitting
- Support 512-token chunks with 50-token overlap
- Handle edge cases: empty text, small text
- Unit tests: 4 test cases, 100% coverage
```

---

**TASK 1.3: FAISS Vector Store Layer (1.5 hours)**
- [ ] Implement `src/rag_service/vectorstore.py`
  ```python
  - Class: FAISSVectorStore
    - __init__(embedding_model, vector_dim=384)
    - add_texts(texts: List[str], ids: List[str]) -> None
    - search(query: str, top_k: int) -> List[Dict]
      Returns: [{"id": str, "text": str, "score": float}, ...]
    - persist(path: str) -> None
    - load(path: str) -> None
  ```
- [ ] Ensure embeddings are normalized
- [ ] Store metadata (id → original text mapping)
- [ ] Use FAISS IndexFlatL2 (exact search, not approximate)

**✅ Unit Tests:**
```python
# tests/unit/test_vectorstore.py
def test_vectorstore_init():
    store = FAISSVectorStore(model_name="all-MiniLM-L6-v2")
    assert store.index is not None
    assert store.dimension == 384

def test_add_texts():
    store = FAISSVectorStore()
    texts = ["Python is a programming language", "FastAPI is a web framework"]
    ids = ["doc1", "doc2"]
    store.add_texts(texts, ids)
    assert store.index.ntotal == 2

def test_search_returns_results():
    store = FAISSVectorStore()
    store.add_texts(["Python programming language"], ["doc1"])
    results = store.search("Python code", top_k=1)
    assert len(results) == 1
    assert results[0]["id"] == "doc1"
    assert results[0]["score"] > 0.5  # Cosine similarity threshold

def test_search_empty_store():
    store = FAISSVectorStore()
    results = store.search("anything", top_k=5)
    assert results == []

def test_persistence():
    import tempfile
    store = FAISSVectorStore()
    store.add_texts(["Test document"], ["test1"])
    
    with tempfile.TemporaryDirectory() as tmpdir:
        store.persist(tmpdir)
        store2 = FAISSVectorStore()
        store2.load(tmpdir)
        assert store2.index.ntotal == 1
```

**✅ Verification:**
```bash
pytest tests/unit/test_vectorstore.py -v --cov=src.rag_service.vectorstore
# Expect: 5 passed, 95%+ coverage
```

**Commits:**
```
feat(rag): implement FAISS-backed vector store with persistence

- Add FAISSVectorStore using sentence-transformers embeddings
- Support add_texts, search with configurable top_k
- Implement persist/load for model checkpointing
- Unit tests: 5 test cases, 95% coverage
```

---

**TASK 1.4: FastAPI Endpoints (1.5 hours)**
- [ ] Implement `src/rag_service/app.py`
  ```python
  - FastAPI app on port 8000
  - POST /ingest
    Input: {"text": str, "doc_id": str}
    Process: Chunk → Embed → Store
    Output: {"status": "success", "chunks_created": int}
  
  - GET /search
    Query: ?query=X&top_k=5
    Output: [{"id": str, "text": str, "score": float, "rank": int}, ...]
  
  - GET /health
    Output: {"status": "healthy", "documents_indexed": int}
  
  - Pydantic models for validation
  - Error handling with descriptive messages
  ```
- [ ] Add input validation (query length, top_k limits)
- [ ] Add startup event to initialize vectorstore

**✅ Integration Tests:**
```python
# tests/integration/test_rag_api.py
import pytest
from fastapi.testclient import TestClient
from src.rag_service.app import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_ingest_simple_document():
    response = client.post("/ingest", json={
        "text": "This is a test document about FastAPI and FAISS.",
        "doc_id": "test1"
    })
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["chunks_created"] > 0

def test_search_returns_results():
    # First ingest a document
    client.post("/ingest", json={
        "text": "FastAPI is a modern Python web framework for building APIs.",
        "doc_id": "apidoc"
    })
    
    # Search for related concept
    response = client.get("/search?query=Python%20framework&top_k=3")
    assert response.status_code == 200
    results = response.json()
    assert len(results) > 0
    assert results[0]["score"] > 0.6

def test_search_empty_query():
    response = client.get("/search?query=&top_k=5")
    assert response.status_code == 400  # Bad request

def test_search_invalid_top_k():
    response = client.get("/search?query=test&top_k=1000")
    assert response.status_code == 400  # Exceed max

def test_ingest_empty_text():
    response = client.post("/ingest", json={"text": "", "doc_id": "empty"})
    assert response.status_code == 400
```

**✅ Verification (Manual):**
```bash
# Start server
uvicorn src.rag_service.app:app --reload --port 8000 &

# Test ingest
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Vector databases are the memory layer for AI systems. FAISS enables fast similarity search over millions of vectors. Semantic search improves retrieval quality by 40% over keyword search.",
    "doc_id": "rag-101"
  }'

# Test search
curl "http://localhost:8000/search?query=vector%20similarity&top_k=3"
# Expect: Status 200, top result has score > 0.7

# Test health
curl http://localhost:8000/health
```

**Commits:**
```
feat(rag): implement FastAPI endpoints for ingest and search

- Add POST /ingest: chunk, embed, and store documents
- Add GET /search: retrieve top-k similar documents
- Add GET /health: service status and document count
- Pydantic validation for all inputs
- Integration tests: 6 test cases
```

---

**TASK 1.5: Docker Setup (30 min)**
- [ ] Create `docker-compose.yml` in root
  ```yaml
  version: '3.8'
  services:
    rag-api:
      build: .
      ports:
        - "8000:8000"
      environment:
        - EMBEDDING_MODEL=all-MiniLM-L6-v2
        - PORT=8000
      volumes:
        - ./data:/app/data
      command: uvicorn src.rag_service.app:app --host 0.0.0.0 --port 8000
  ```
- [ ] Create `Dockerfile` in root
  ```dockerfile
  FROM python:3.11-slim
  WORKDIR /app
  COPY requirements.txt .
  RUN pip install -r requirements.txt
  COPY src src
  ```
- [ ] Create `requirements.txt`
  ```
  fastapi==0.104.1
  uvicorn[standard]==0.24.0
  sentence-transformers==2.2.2
  faiss-cpu==1.7.4
  pydantic==2.5.0
  pytest==7.4.3
  pytest-cov==4.1.0
  python-dotenv==1.0.0
  ```

**✅ Verification:**
```bash
# Build and run
docker-compose up --build

# Test from another terminal
curl http://localhost:8000/health

# Tear down
docker-compose down
```

**Commits:**
```
chore(docker): add docker and docker-compose for RAG service

- Create Dockerfile for FastAPI application
- Add docker-compose.yml for local development
- Generate requirements.txt with pinned versions
```

---

**TASK 1.6: Documentation & Integration Test (30 min)**
- [ ] Run full test suite with coverage
  ```bash
  pytest tests/ -v --cov=src.rag_service --cov-report=html
  ```
- [ ] Check coverage is >90%
- [ ] Document architecture in `docs/architecture/day01-rag-foundation.md`
  - Component diagram (text format)
  - Data flow (ingest → chunk → embed → store → search)
  - Why FAISS (speed, accuracy tradeoffs)

**✅ Verification:**
```bash
pytest tests/ -v --cov=src.rag_service
# Expect: 15+ tests passing, >90% coverage

# Check generated HTML report
open htmlcov/index.html
```

**Commits:**
```
docs(rag): add architecture documentation for Day 1

- Document component overview and data flows
- Add rationale for FAISS vs alternatives
```

---

#### 📊 DAY 1 Summary Checklist

Before moving to Day 2, verify:
- [ ] All 15+ tests passing
- [ ] Coverage >90%
- [ ] Docker builds and runs successfully
- [ ] Can ingest documents and retrieve them
- [ ] Retrieved documents have similarity score >0.7
- [ ] Code is committed with proper messages
- [ ] Ready to ask for documentation help

**How to Ask for Documentation Help (End of Day 1):**
```
"Documenting Day 1. Completed:
- Vector Store (FAISS): ✅ 5 unit tests passed
- FastAPI Endpoints: ✅ 6 integration tests passed  
- Text Chunking: ✅ 4 unit tests passed
- Docker setup: ✅ running
- Overall test coverage: 92%
- Total commits: 4
- Lines of code: ~450

Please help me format this into the daily log using the template."
```

---

### 📅 DAY 2: Retrieval Quality + Reranking

**Objective:** Implement two-stage retrieval (recall → precision) using cross-encoders.

#### 🎯 Micro Tasks

**TASK 2.1: Cross-Encoder Integration (1 hour)**
- [ ] Implement `src/rag_service/reranker.py`
  ```python
  - Class: CrossEncoderReranker
    - __init__(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    - rerank(query: str, candidates: List[str]) -> List[Dict]
      Returns: [{"text": str, "score": float}, ...]
    - Score normalization to [0, 1] range
  ```
- [ ] Reranker should score query + document pairs together
- [ ] Return top results sorted by reranker score

**✅ Unit Tests:**
```python
def test_reranker_init():
    reranker = CrossEncoderReranker()
    assert reranker.model is not None

def test_rerank_candidates():
    reranker = CrossEncoderReranker()
    query = "What is FastAPI?"
    candidates = [
        "FastAPI is a modern Python web framework",
        "Python is a programming language",
        "Web frameworks help build APIs"
    ]
    results = reranker.rerank(query, candidates)
    assert len(results) <= len(candidates)
    assert results[0]["score"] > results[-1]["score"]  # Sorted

def test_rerank_empty_candidates():
    reranker = CrossEncoderReranker()
    results = reranker.rerank("query", [])
    assert results == []

def test_rerank_single_candidate():
    reranker = CrossEncoderReranker()
    results = reranker.rerank("query", ["single text"])
    assert len(results) == 1
```

**Commits:**
```
feat(rag): add cross-encoder reranking for precision improvement

- Implement CrossEncoderReranker with ms-marco model
- Score normalization to [0, 1]
- Unit tests: 4 test cases
```

**TASK 2.2: Enhanced Search Pipeline (45 min)**
- [ ] Update `src/rag_service/app.py`
  ```python
  - Modify /search endpoint:
    - Query params: ?query=X&top_k=5&rerank=true
    - If rerank=true:
      1. FAISS retrieves top_k=20 (first stage)
      2. Cross-encoder reranks to top_k=5 (second stage)
      3. Return reranked results
    - Add timing metrics in response
  ```
- [ ] Add `/metrics` endpoint to return:
  - Total documents
  - Average search latency
  - Cache hit rate (prep for Day 10)

**✅ Integration Tests:**
```python
def test_search_with_reranking():
    # Ingest diverse documents
    docs = [
        ("FastAPI web framework", "doc1"),
        ("Python programming language", "doc2"),
        ("API development best practices", "doc3"),
    ]
    for text, doc_id in docs:
        client.post("/ingest", json={"text": text, "doc_id": doc_id})
    
    # Search with reranking
    response = client.get("/search?query=fast%20api&rerank=true&top_k=3")
    assert response.status_code == 200
    results = response.json()
    # First result should be most relevant
    assert "FastAPI" in results[0]["text"]

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "documents_indexed" in response.json()
    assert "avg_search_latency_ms" in response.json()
```

**Commits:**
```
feat(rag): add reranking to search pipeline and metrics endpoint

- Modify /search to support two-stage retrieval
- Add /metrics endpoint for performance monitoring
- Include search latency in response
- Integration tests: 2 new test cases
```

**TASK 2.3: Performance & Comparison (45 min)**
- [ ] Create `tests/integration/test_retrieval_quality.py`
  ```python
  - Test: Search WITH vs WITHOUT reranking
    - Doc: "FastAPI is a modern Python framework"
    - Query: "python web framework"
    - Assert: WITH reranking has higher relevance score
  - Measure latency increase (should be <100ms)
  ```
- [ ] Create benchmark notebook: `docs/notebooks/retrieval_benchmark.ipynb`
  - Ingest 10 sample documents
  - Run 20 different queries
  - Compare FAISS-only vs FAISS+Reranker
  - Plot: latency vs accuracy tradeoff

**✅ Verification:**
```bash
pytest tests/integration/test_retrieval_quality.py -v

# Manual test
curl "http://localhost:8000/search?query=python%20framework&rerank=false&top_k=5"
curl "http://localhost:8000/search?query=python%20framework&rerank=true&top_k=5"
# Compare "score" fields - reranked should emphasize better matches
```

**Commits:**
```
feat(rag): add retrieval quality benchmarks and comparison tests

- Add retrieval quality comparison tests
- Create jupyter notebook for latency/accuracy analysis
- Document performance tradeoffs
```

**TASK 2.4: Integration Test (30 min)**
- [ ] Full end-to-end test combining Tasks 2.1-2.3
- [ ] Run: `pytest tests/ -v --cov=src.rag_service`
- [ ] Document learnings

#### 📊 DAY 2 Summary Checklist
- [ ] All tests passing (15+)
- [ ] Coverage >90%
- [ ] Reranking improves top-1 relevance by 10%+
- [ ] Latency increase <150ms
- [ ] 2 new endpoints working
- [ ] Code committed
- [ ] Ready for documentation

---

### 📅 DAY 3: Context Assembly + Prompt Engineering

**Objective:** Build prompt construction pipeline with retrieved context.

#### 🎯 Micro Tasks

**TASK 3.1: Context Assembly (1 hour)**
- [ ] Create `src/rag_service/context_assembler.py`
  ```python
  - Class: ContextAssembler
    - __init__(max_tokens=2000, separator="\n---\n")
    - assemble(query: str, retrieved_chunks: List[Dict]) -> str
      Returns: assembled context string with metadata
  - Format: [SOURCE: doc_id | CONFIDENCE: 0.85]\nchunk_text\n---\n...
  ```

**TASK 3.2: Prompt Templates (1 hour)**
- [ ] Create `src/rag_service/prompt_templates.py`
  ```python
  - Template: RAG_QA (question-answering)
  - Template: RAG_SUMMARIZE (summarization)
  - Template: RAG_ANALYZE (analysis)
  - All include: {context}, {query}, {chat_history}
  ```

**TASK 3.3: Integration Test (30 min)**
- [ ] Tests for context formatting
- [ ] Tests for prompt rendering

---

### 📅 DAY 4: LangGraph Agent - Basic Reasoning Loop

### 📅 DAY 5: Multi-Tool Agent + Function Calling

### 📅 DAY 6: Streaming Responses + Async Architecture

### 📅 DAY 7: Observability + Metrics + Phase 1 Integration

---

## PHASE 2: EXECUTION LAYER (Days 8-14)

### 📅 DAY 8: MCP Server Implementation

### 📅 DAY 9: Guardrails - Input/Output Validation

### 📅 DAY 10: Semantic Caching + Performance Optimization

### 📅 DAY 11: Error Handling + Retry Logic + Circuit Breakers

### 📅 DAY 12: Authentication + Rate Limiting + API Security

### 📅 DAY 13: Testing Strategy - Unit, Integration, E2E

### 📅 DAY 14: CI/CD Pipeline + GitHub Actions

---

## PHASE 3: SYSTEM DESIGN + PRODUCTION (Days 15-21)

### 📅 DAY 15: Multi-Agent System - Specialist Agents Pattern

### 📅 DAY 16: A2A Communication + MCP Integration

### 📅 DAY 17: Production Database + Data Persistence

### 📅 DAY 18: Kubernetes Deployment + Auto-scaling

### 📅 DAY 19: Monitoring, Alerting, and SLOs

### 📅 DAY 20: Security Hardening + Penetration Testing

### 📅 DAY 21: Final Integration + Production Readiness

---

## 🔄 Commit Strategy Summary

### Commit Frequency
- **Per Micro Task**: After tests pass
- **Typical Day**: 3-5 commits
- **Scope**: One logical feature per commit

### Commit Message Format
```
<type>(<scope>): <subject>

<body>
- New functionality added
- Tests added/modified: X new/X modified
- Coverage: X%
```

### Commit Types
- `feat`: New feature
- `fix`: Fix existing behavior
- `test`: Adding or updating tests
- `docs`: Adding documentation
- `chore`: Build, deps, tooling

### Tags
Use lightweight git tags for phase completion:
```bash
git tag day-01
git tag phase-1-complete
```

---

## 📝 Documentation Flow

### End-of-Day Documentation Prompt

Use this template when asking for documentation help:

```
"Documenting Day [X]: [Title]

✅ Completed:
- [Component]: [X tests], [coverage]%
- [Feature]: [description]

📊 Metrics:
- Tests: [total] passed
- Coverage: [%]
- Lines of code: ~[number]
- Time spent: [X] hours
- Commits: [number]

🔗 Related commits:
- [commit message]
- [commit message]

🧠 Key learnings:
- [learning 1]
- [learning 2]

🐛 Blockers faced:
- [blocker + solution]

Please help me format into daily log."
```

### Daily Log Structure
See [docs/daily-logs/DAY_TEMPLATE.md](docs/daily-logs/DAY_TEMPLATE.md)

---

## ✅ Testing Pyramid

```
      /\           E2E Tests
     /  \          (Few, slow)
    /____\
   /      \        Integration Tests
  /________\       (Medium, moderate)
 /          \
/__________  \ Unit Tests
              (Many, fast)
```

### Testing Per Day
- **Unit Tests**: 70% - module/function logic
- **Integration Tests**: 25% - component interactions
- **E2E Tests**: 5% - full workflow (starting Day 7)

### Coverage Goals
- Minimum: 85%
- Target: 90%+
- Critical paths: 100%

---

## 🚨 Common Pitfalls & Solutions

| Pitfall | Day | Solution |
|---------|-----|----------|
| Skipping tests | All | Run tests before commit |
| Insufficient error handling | 1+ | Add try/except with tests |
| No input validation | 1+ | Use Pydantic models |
| Blocking async code | 6+ | Use asyncio.create_task |
| No monitoring | All | Add metrics from Day 1 |
| Tight coupling | 8+ | Use dependency injection |
| No retry logic | 11+ | Add exponential backoff |
| Hardcoded config | All | Use .env or config files |

---

## 📞 Checkpoint Conversations

**After Day 1:** "Vector store is working. Can retrieve documents with >0.7 similarity."

**After Day 3:** "RAG pipeline complete. Can answer questions using retrieved context."

**After Day 7:** "Intelligence layer done. Starting execution layer tomorrow."

**After Day 14:** "Execution layer complete. Can deploy to production with CI/CD."

**After Day 21:** "Full system ready. Multi-agent orchestration working end-to-end."
