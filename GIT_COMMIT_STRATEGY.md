# Git Commit Strategy & Workflow

## 🎯 Commit Philosophy

**One logical change = One commit**

Each commit should:
- ✅ Pass all tests
- ✅ Have clear, descriptive message
- ✅ Be small enough to review in <5 minutes
- ✅ Not break anything else
- ❌ Never contain unrelated changes

---

## 📝 Commit Message Format

### Structure
```
<type>(<scope>): <subject>

<body>

<footer>
```

### Components

#### Type (required)
- `feat`: New feature
- `fix`: Bug fix
- `test`: Adding/updating tests
- `refactor`: Code restructure without behavior change
- `perf`: Performance improvement
- `docs`: Documentation
- `chore`: Build, deps, CI/CD
- `style`: Formatting (with automated tool)

#### Scope (recommended)
- `rag`: RAG service module
- `agent`: Agent/LangGraph module
- `mcp`: MCP server module
- `api`: FastAPI endpoints
- `test`: Test infrastructure

#### Subject (required)
- Imperative, present tense: "add" not "added"
- No period at end
- Max 50 characters
- Lowercase first letter

#### Body (optional but good practice)
- Explain WHAT changed and WHY (not HOW)
- Wrap at 72 characters
- Separate from subject with blank line
- Lists helpful with `-`

#### Footer (for tracking)
- `Closes #123` (GitHub issue)
- `Relates-to #456`
- `Breaking-change:` (for semver)

---

## 📋 Example Commits

### Example 1: Feature Commit
```
feat(rag): implement FAISS vector store with persistence

- Add FAISSVectorStore class for similarity search
- Support add_texts() and search(top_k) operations
- Implement persist/load for model checkpointing
- Use sentence-transformers all-MiniLM-L6-v2 embeddings

All operations vectorized for performance. Tested with unit and
integration tests achieving 100% coverage for critical paths.

Tests: 5 unit tests, 2 integration tests
Coverage: 95%
Lines: ~180

Closes #42
```

### Example 2: Test Commit
```
test(rag): add comprehensive chunking tests

- Add test_chunk_normal_text() for standard usage
- Add test_chunk_edge_cases() for empty/small texts
- Add test_overlap_works() for context continuity
- All edge cases covered

Ensures chunking robustness before reranking implementation.

Tests added: 4
Coverage: 100% for chunker module
```

### Example 3: Fix Commit
```
fix(api): handle empty search results gracefully

Empty query results previously threw exception. Now returns
empty list with proper HTTP 200 status code.

Before: GET /search?query=gibberish → 500 Error
After:  GET /search?query=gibberish → 200 []

Tests: Added test_search_no_results()
Relates-to #89
```

### Example 4: Doc Commit
```
docs(architecture): add Day 1 RAG vector store diagram

- Component diagram (Text format in markdown)
- Data flow: ingest → chunk → embed → store → search
- Decision rationale for FAISS vs Pinecone
- Performance characteristics documented

Related to day-01 completion
```

---

## 🔄 Daily Commit Pattern

### Typical Day Commits (4-5 commits)

**MORNING: Setup Commit**
```
chore(setup): prepare Day X environment

- Create src/module/file.py scaffolding
- Add pytest fixtures for testing
- Update .env with Day X parameters
```

**MID-MORNING: Component Commit**
```
feat(module): implement [primary feature]

- Add primary class/function
- Include comprehensive docstrings
- Preliminary tests

Tests: X unit tests
Coverage: Y%
```

**AFTERNOON: Testing Commit**
```
test(module): add integration tests and fixtures

- Add integration test suite (X tests)
- Create test fixtures for Day X+1
- Verify manual endpoints with curl

Coverage: Y% (up from Z%)
All tests passing: ✅
```

**LATE AFTERNOON: Enhancement Commit**
```
feat(module): add [secondary feature/improvement]

- Add optional parameter X
- Improve error messages
- Add performance metrics

Tests: X more tests
```

**END OF DAY: Documentation Commit**
```
docs(day-X): add architecture and technical notes

- Document component interaction flows
- Add rationale for design decisions
- Note for next day's dependencies

Ready for Day X+1 startup
```

---

## ✅ Pre-Commit Checklist

Before running `git commit`, verify:

- [ ] All tests pass: `pytest tests/ -q`
- [ ] Coverage >90%: `pytest --cov=src`
- [ ] No uncommitted changes in other modules: `git status`
- [ ] Code formatted: `black src/` (if using)
- [ ] Meaningful commit message written
- [ ] Related GitHub issue noted in footer
- [ ] New code has docstrings

```bash
# Automated pre-commit check
./scripts/pre-commit-check.sh
```

---

## 🔗 Commit Workflow for This Project

### Step 1: Create Feature Branch (Optional but Recommended)
```bash
git checkout -b day-01-vector-store
```

### Step 2: Build and Test
```bash
# Write code in small increments
# Test frequently as you go

# When mini-task complete:
pytest tests/unit/test_chunker.py -v --cov=src.rag_service.chunker
```

### Step 3: Stage Changes
```bash
# Option A: Stage specific file
git add src/rag_service/chunker.py

# Option B: Interactive staging (pick which changes to add)
git add -p

# Check what will be committed
git diff --cached
```

### Step 4: Commit
```bash
git commit -m "feat(rag): implement text chunking"
# Or use editor for multi-line message:
git commit
```

### Step 5: Repeat Steps 2-4 for next micro-task

### Step 6: Push at End of Day
```bash
# Local commits only until day is complete
git push origin day-01-vector-store
```

### Step 7: Create Pull Request (Optional)
```bash
# GitHub: Create PR from day-01-vector-store → main
# Request review if working with team
# Merge after approval
```

---

## 🏷 Tagging Strategy

### Tag Name Format
```
<phase>-<day>-<status>
day-01-complete
day-02-in-progress
phase-1-complete
```

### Commands
```bash
# Create lightweight tag
git tag day-01-complete

# Create annotated tag (recommended)
git tag -a day-01-complete \
  -m "Day 1: Vector Store Foundation - All tests passing, 90% coverage"

# List tags
git tag -l

# Push tags to remote
git push origin --tags

# Delete tag if mistake
git tag -d day-01-complete
git push origin --delete day-01-complete
```

---

## 🔍 Checking Commit History

### View Recent Commits
```bash
# Brief log
git log --oneline -10

# Detailed log
git log -10

# With graph
git log --graph --oneline --all

# For specific file
git log --oneline src/rag_service/app.py
```

### Example Output
```
d3e4f2a (HEAD -> main) docs(day-01): add vector store architecture
b2c1a09 test(rag): add comprehensive endpoint tests
a1f0e08 feat(rag): implement FastAPI endpoints
9e8d7c6 feat(rag): add FAISS vector store
8d7c6b5 feat(rag): implement text chunking
```

### Navigating History
```bash
# Show specific commit
git show d3e4f2a

# Show what changed in file over commits
git log -p src/rag_service/chunker.py

# Find who changed what
git blame src/rag_service/chunker.py
```

---

## 🐛 Fixing Mistakes

### Mistake 1: Wrong Message
```bash
# Fix last commit message
git commit --amend -m "new message"

# Only works if NOT yet pushed
# If already pushed: create new commit with fix
git revert COMMIT_HASH
```

### Mistake 2: Committed Wrong Files
```bash
# Undo last commit, keep changes unstaged
git reset --soft HEAD~1

# Cherry-pick what you want to re-commit
git add correct_file.py
git commit -m "fix: only correct file"
```

### Mistake 3: Pushed Bad Commit
```bash
# If only you have it locally:
git reset --hard HEAD~1
git push -f origin day-01  # ONLY if you're sure

# If others grabbed it: revert instead
git revert BAD_COMMIT_HASH
git push origin main
```

### Mistake 4: Merged Wrong Branch
```bash
git reset --hard HEAD~1  # Undo merge
```

---

## 🔄 Squashing Commits (Before PR Merge)

If you have many commits that should be one:

```bash
# Interactive rebase last 5 commits
git rebase -i HEAD~5

# In editor:
# pick first_commit
# squash second_commit
# squash third_commit
# pick important_commit

# Save and resolve any conflicts

git push -f origin branch_name  # Only if not merged yet
```

---

## 📊 Daily Commit Summary Template

End of day, generate summary:

```bash
#!/bin/bash
# Save this as scripts/daily-summary.sh

COMMIT_COUNT=$(git rev-list --all --count)
TODAY=$(date +%Y-%m-%d)

echo "=== Daily Commit Summary: $TODAY ==="
echo ""
echo "Today's commits:"
git log --oneline --since="$TODAY 00:00" --until="$TODAY 23:59"
echo ""
echo "Total commits this session:"
git rev-list --all --count
echo ""
echo "Files changed today:"
git diff --name-only HEAD~5..HEAD 2>/dev/null | sort -u
```

---

## 🚀 Multi-Person Workflow (If Working with Team)

### Pull Before Push
```bash
git pull origin main  # Get latest changes

# If conflicts:
# Resolve conflicts manually
# git add resolved_file.py
# git commit -m "Merge: resolve conflicts from main"

git push origin day-01-vector-store
```

### Creating Pull Request
```bash
# Push your branch
git push origin day-01-vector-store

# On GitHub:
# 1. Create Pull Request (PR)
# 2. Add description from your daily log
# 3. Tag reviewers
# 4. Wait for approval
# 5. Merge to main
```

### Code Review Checklist
- [ ] Tests all pass
- [ ] Coverage >90%
- [ ] Code follows conventions
- [ ] Commit messages clear
- [ ] Documentation updated
- [ ] No hard-coded secrets

---

## 📋 Commits Per Day Target

| Phase | Day | Target Commits | Expected LOC | Files Changed |
|-------|-----|----------------|--------------|---------------|
| 1 | 1 | 4-5 | 400-600 | 8-10 |
| 1 | 2 | 3-4 | 200-300 | 5-7 |
| 1 | 3 | 4-5 | 300-500 | 6-8 |
| 1 | 4-7 | 5-6 | 300-800 | 8-15 |
| 2 | 8-14 | 5-6 | 300-600 | 8-12 |
| 3 | 15-21 | 6-8 | 500-1000 | 15-25 |

---

## 🎓 Best Practices Summary

### DO ✅
- ✅ Commit frequently (multiple times per day)
- ✅ Write clear, descriptive messages
- ✅ Commit one logical change at a time
- ✅ Keep commits small (<200 lines added)
- ✅ Always run tests before committing
- ✅ Use feature branches for bigger features
- ✅ Reference GitHub issues in commits
- ✅ Tag important milestones

### DON'T ❌
- ❌ Commit failing tests
- ❌ Commit debug code or print statements
- ❌ Commit .env files with secrets
- ❌ Mix formatting and logic changes
- ❌ Write vague messages ("fix stuff")
- ❌ Commit large generated files
- ❌ Force push to shared branches
- ❌ Commit commented-out code

---

## 📞 Example Daily Workflow

```bash
# MORNING - Day 01 Start
git checkout -b day-01-vector-store
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Implement chunking
# ... write code ...
pytest tests/unit/test_chunker.py -v
git add src/rag_service/chunker.py
git commit -m "feat(rag): implement text chunking"

# Implement FAISS wrapper
# ... write code ...
pytest tests/unit/test_vectorstore.py -v
git add src/rag_service/vectorstore.py
git commit -m "feat(rag): add FAISS vector store"

# Implement API
# ... write code ...
pytest tests/integration/test_api.py -v
git add src/rag_service/app.py
git commit -m "feat(rag): implement FastAPI endpoints"

# Docker setup
# ... create files ...
docker-compose up --build
git add docker-compose.yml Dockerfile requirements.txt
git commit -m "chore(docker): add compose and dockerfile"

# Tests and docs
pytest tests/ -v --cov=src
git add docs/
git commit -m "docs(day-01): add architecture documentation"

# END OF DAY
git log --oneline -5
git push origin day-01-vector-store
```

---

## 🔗 Integration with Daily Logs

When documenting at end of day:

```markdown
## 🔗 Git Commits

$(git log --oneline -5)

### Commit Details

**Commit 1:** feat(rag): implement text chunking
- Added TextChunker class with word-based splitting
- Support configurable chunk size and overlap
- 4 unit tests, 100% coverage

**Commit 2:** feat(rag): add FAISS vector store
- FAISSVectorStore with add_texts and search
- Persistence support for checkpointing
- 5 unit tests, 95% coverage

... etc ...
```

This is your git commit reference guide!
