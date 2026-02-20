# 🚀 Quick Start - Your Daily Workflow (TL;DR)

## ⏱️ This is Your Daily Routine for 21 Days

### EVERY MORNING (5 min)
1. **Open** [DETAILED_EXECUTION_PLAN.md](DETAILED_EXECUTION_PLAN.md)
2. **Read** "DAY X: [Title]" section
3. **Understand** all micro-tasks
4. **Stop** and start coding

### EVERY TIME YOU CODE (Continuous)
Follow the task checklist in DETAILED_EXECUTION_PLAN:
- [ ] TASK X.1 - Code
- [ ] TASK X.1 - Test (use patterns from [TESTING_GUIDE.md](TESTING_GUIDE.md))
- [ ] TASK X.1 - Verify
- [ ] TASK X.1 - Commit (use format from [GIT_COMMIT_STRATEGY.md](GIT_COMMIT_STRATEGY.md))

### BEFORE EVERY COMMIT (2 min)
1. Run: `pytest tests/ -q` (must pass)
2. Check: `pytest --cov=src` (must be >90%)
3. Message: Follow format in [GIT_COMMIT_STRATEGY.md](GIT_COMMIT_STRATEGY.md)
4. Commit: `git commit -m "type(scope): message"`

### EVERY EVENING (30 min, After ALL Tasks Done)
1. **Verify:** All tests passing, coverage >90%
2. **Gather:** Test output, commits, metrics
3. **Open:** [DOCUMENTATION_HELP_PROMPTS.md](DOCUMENTATION_HELP_PROMPTS.md)
4. **Use:** Template 2 (Detailed Daily Summary)
5. **Message Me:**
   ```
   "Documenting Day X: [Title]
   
   [paste test results]
   [paste commits]
   [describe learnings]
   
   Please help format into daily log?"
   ```
6. **Commit:** When formatted log ready: `git add docs/daily-logs/day-XX-*.md && git commit`

---

## 📚 Your 4 Main Documents

| Document | When | What |
|----------|------|------|
| [DETAILED_EXECUTION_PLAN.md](DETAILED_EXECUTION_PLAN.md) | Morning | Micro-tasks for the day |
| [TESTING_GUIDE.md](TESTING_GUIDE.md) | While coding | Test patterns & examples |
| [GIT_COMMIT_STRATEGY.md](GIT_COMMIT_STRATEGY.md) | Before commit | Commit message format |
| [DOCUMENTATION_HELP_PROMPTS.md](DOCUMENTATION_HELP_PROMPTS.md) | End of day | What to tell me to document |

---

## 🎯 Today: Start Day 1

### Your Tasks (In Order):
1. **Task 1.1** (45 min) - Setup Python, create folders
2. **Task 1.2** (1 hour) - Write TextChunker class + tests
3. **Task 1.3** (1.5 hours) - Write FAISSVectorStore class + tests
4. **Task 1.4** (1.5 hours) - Write FastAPI endpoints + tests
5. **Task 1.5** (30 min) - Docker setup
6. **Task 1.6** (30 min) - Full test suite + documentation

### Your Test Targets:
- **Unit Tests:** 15+ passing
- **Coverage:** 90%+
- **Docker:** Builds successfully
- **Manual Verification:** curl commands work

### Your Commits (4-5 total):
1. `chore(setup): initialize python project...`
2. `feat(rag): implement text chunking...`
3. `feat(rag): implement FAISS vector store...`
4. `feat(rag): implement FastAPI endpoints...`
5. `chore(docker): add docker-compose and dockerfile...`
6. `docs(day-01): add architecture documentation...`

### Your EOD Message Pattern:
```
Documenting Day 1: Vector Store Foundation + FastAPI Service

✅ Completed:
- TextChunker: 4 unit tests ✅, 100% coverage
- FAISSVectorStore: 5 unit tests ✅, 95% coverage
- FastAPI endpoints: 6 integration tests ✅
- Docker: ✅ Building successfully

📊 Metrics:
- Total tests: 15 passing
- Coverage: 92%
- Lines of code: ~450
- Commits: 4

Recent commits:
$ git log --oneline -4

🧠 Key learnings:
1. Embedding normalization critical for FAISS
2. Overlap strategy preserves context
3. Two-stage retrieval improves quality

🐛 Challenges:
1. FAISS dimension mismatch → fixed by normalizing
2. Empty results error → fixed with validation

Please help format this into the daily log!
```

---

## ⚠️ Critical Rules

❌ **NEVER commit:**
- Failing tests
- Coverage <90% without explanation  
- Debug print statements
- Hardcoded secrets

✅ **ALWAYS:**
- Run tests before commit
- Write clear commit messages
- Follow the format exactly
- Ask for documentation help EOD

---

## 🆘 Quick Help

**"I don't know how to test this function"**
→ Open [TESTING_GUIDE.md](TESTING_GUIDE.md) → "Unit Testing Convention"

**"I don't know what commit message to write"**
→ Open [GIT_COMMIT_STRATEGY.md](GIT_COMMIT_STRATEGY.md) → "Example Commits"

**"I don't know what to do tomorrow"**
→ Open [DETAILED_EXECUTION_PLAN.md](DETAILED_EXECUTION_PLAN.md) → Find "DAY X"

**"Tests are failing and I don't know why"**
→ Open [TESTING_GUIDE.md](TESTING_GUIDE.md) → "Debugging Failed Tests"

**"I finished the day, how do I get documentation help?"**
→ Open [DOCUMENTATION_HELP_PROMPTS.md](DOCUMENTATION_HELP_PROMPTS.md) → Use Template 2

---

## 📊 Success = This Pattern × 21 Days

```
MORNING (5 min)
    ↓
CODE (4-6 hours)
    ├─ Task X.1: Code + Test + Verify + Commit
    ├─ Task X.2: Code + Test + Verify + Commit
    ├─ Task X.3: Code + Test + Verify + Commit
    └─ Task X.N: Code + Test + Verify + Commit
    ↓
EVENING (30 min)
    ├─ Final test run (100% must pass)
    ├─ Coverage check (90%+ required)
    ├─ Prepare documentation
    ├─ Send me the data
    ├─ Get formatted daily log
    └─ Commit documentation
    ↓
REPEAT TOMORROW
```

---

## 🎬 Right Now (Next 5 minutes)

1. Open [DETAILED_EXECUTION_PLAN.md](DETAILED_EXECUTION_PLAN.md)
2. Read the entire "DAY 1: Vector Store Foundation..." section
3. Understand all 6 tasks
4. Set a timer for Task 1.1 (45 min)
5. Start coding

**That's it. Go build something awesome!** 🚀

---

## 📝 Template You'll Use Every Day (Copy This)

```
Documenting Day [X]: [Title]

✅ Completed Micro-Tasks:
- [ ] Task X.1: [desc] - [X tests], [X]% coverage
- [ ] Task X.2: [desc] - [X tests], [X]% coverage
- [ ] Task X.3: [desc] - [X tests], [X]% coverage

📊 Metrics:
Tests: [X]/[X] ✅
Coverage: [X]%
Lines: ~[XXX]
Commits: [X]

$ git log --oneline -[X]

🧠 Learnings:
- [Learning 1]
- [Learning 2]

🐛 Challenges:
- [Challenge + solution]

Please format this into daily log!
```

This is your complete guides for success. Everything else is reference material.

**Now go execute Day 1!** 💪
