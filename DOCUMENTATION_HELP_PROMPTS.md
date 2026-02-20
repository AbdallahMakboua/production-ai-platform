# Documentation Help Prompts & Templates

## 📞 When to Ask for Documentation Help

You should ask for documentation help when:

1. **End of each day** - After completing all micro-tasks
2. **Phase completion** - Every 7 days
3. **Blocker resolved** - If you solved a significant problem
4. **Architecture decision** - When making important decisions
5. **Test failure patterns** - When discovering interesting test insights

---

## 📝 Daily Documentation Prompt Template

Use this template at the **end of each day** after all tests pass:

### Template 1: Basic Daily Summary (5 mins to prepare)

```
Subject: Documenting Day X: [Title from plan]

Completed:
- [Component 1]: [description] ✅
- [Component 2]: [description] ✅

Tests:
$ pytest tests/ -v --cov=src

[paste last 20 lines of pytest output]

Commits:
$ git log --oneline -5

[paste output]

Lines of code: ~XXX
Metrics: [Coverage Y%, X tests passed]
Time spent: [X] hours

Please help me format this into the daily log template with proper markdown.
```

### Template 2: Detailed Daily Summary (10 mins to prepare)

This is MORE comprehensive and recommended:

```
Documenting Day [X]: [Title]

✅ Completed Micro-Tasks:
- [ ] Task X.1: [description] - Tests: Y, Coverage: Z%
- [ ] Task X.2: [description] - Tests: Y, Coverage: Z%
- [ ] Task X.3: [description] - Tests: Y, Coverage: Z%

🧪 Test Results:
Unit Tests:
$ pytest tests/unit/ -v --cov=src.[module]
[paste output]

Integration Tests:
$ pytest tests/integration/ -v
[paste output]

📊 Metrics Summary:
```
| Metric | Value |
|--------|-------|
| Total Tests | X / X ✅ |
| Code Coverage | X% |
| Lines Added | ~XXX |
| Files Modified | X |
| New Endpoints | X |
| New Classes | X |
| Commits | X |
```

🔗 All Commits:
```
$ git log --oneline -X
[paste all commits from today]
```

💡 Key Learnings:
- Learning 1: [2-3 sentences about what you learned]
- Learning 2: [important insight]
- Learning 3: [architectural decision]

🐛 Challenges Faced:
1. Challenge: [problem]
   Solution: [how you fixed it]
   
2. Challenge: [problem]
   Solution: [how you fixed it]

💭 Code Highlights:
- Best code from today (class/function name): [why it's interesting]
- Design decision: [decision and rationale]

🔜 Ready for Day [X+1]: [brief description of what's next]

Please help me:
1. Format this into the daily log markdown
2. Generate clean tables for metrics
3. Add architecture diagram if applicable
4. Polish commit messages section
```

---

## 🎓 How I'll Help (What to Expect)

When you provide the information above, I will:

### 1. **Format & Structure**
- Convert metrics into clean markdown tables
- Organize commits with descriptions
- Create proper markdown headings and sections

### 2. **Generate Missing Pieces**
- Create architecture text diagrams (if applicable)
- Generate visual representations in ASCII/Mermaid
- Add learning summaries
- Write reflection sections

### 3. **Polish & Enhance**
- Improve language and clarity
- Add context and connections to previous days
- Link to relevant documentation
- Update progress trackers

### 4. **Cross-Link**
- Connect to other days' work
- Reference architecture documents
- Link to GitHub commits
- Add forward references to next day

### Example Output:
Your input → Your beautiful, formatted daily log file ready to commit

---

## 🚀 Fast Prompts (1-minute versions)

### When You're in a Hurry:

**Quick Version (Minimal):**
```
Documenting Day X.

Tests: $(pytest tests/ -v | tail -3)
Coverage: $(pytest --cov=src | tail -1)
Commits: $(git log --oneline -3)

Format into daily log please!
```

**Medium Version:**
```
Day X Complete:
- Component 1: ✅
- Component 2: ✅
- Component 3: ✅

$(pytest tests/ -q)
$(git log --oneline -5)

Help me format?
```

---

## 📋 Specific Asks - How to Request Help

### Ask 1: "Help me generate test summary table"
```
I have these test results:
- test_module_1.py: 10 passed
- test_module_2.py: 5 passed  
- test_api_endpoints.py: 8 passed

Coverage report shows 92% overall.

Can you format this into a markdown table for the daily log?
```

### Ask 2: "Help me write learning summary"
```
Today I learned:
1. FAISS similarity search is fast because it uses vector approximation
2. Chunking strategy affects retrieval quality more than embedding model
3. Two-stage retrieval (recall then precision) is industry standard

Can you convert these into well-written learning sections for the daily log?
```

### Ask 3: "Help me create architecture diagram"
```
Our Day 1 architecture is:

Input → TextChunker → EmbeddingModel → FAISSVectorStore → Search Results

Can you create a nice text/ASCII diagram for the daily log?
```

### Ask 4: "Help me summarize challenges"
```
Challenges today:
1. Faced: FAISS index dimension mismatch
   Fixed: Ensured all embeddings normalized before insert
   
2. Faced: Empty query returning error
   Fixed: Added validation and return empty list instead

Can you format these nicely for the challenges section?
```

### Ask 5: "Help me update progress trackers"
```
$ git log --all --oneline | wc -l
$ find src -name "*.py" | xargs wc -l

I've completed:
- Days 1-3 from Phase 1
- Currently on Day 4

Can you update the overall progress metrics and README progress table?
```

### Ask 6: "Help me create documentation"
```
I've built a RAG system today with these components:
- TextChunker class
- FAISSVectorStore class  
- FastAPI endpoints: /ingest, /search, /health

Can you help me write the architecture documentation for docs/architecture/?
```

---

## 🔗 Advanced Asks - Phase Completion

### When Completing a Phase (Every 7 days):

```
Completing Phase 1: Intelligence Layer

Summary of 7 days:
- Day 1: Vector Store [test results]
- Day 2: Reranking [test results]
- Day 3: Context Assembly [test results]
- Day 4: LangGraph Agent [test results]
- Day 5: Multi-Tool Agent [test results]
- Day 6: Streaming [test results]
- Day 7: Observability [test results]

Total stats:
- Lines of code: ~5000
- Tests written: 80+
- Coverage: 92%
- Commits: 35
- Architecture: 5 components, fully integrated

Can you:
1. Create a Phase 1 completion summary
2. Generate a component diagram showing all 7 days
3. Create a "Phase 1 Learnings" document
4. Update the main README with Phase 1 completion
5. Generate a transition summary for Phase 2 kickoff?
```

---

## 🔄 Iterative Documentation Help

Sometimes you'll need multiple iterations:

### Round 1: Raw Data Submission
```
Here's my Day 1 data:
[paste test output, commits, metrics]

Just organize it into the daily log template.
```

### Round 2: Enhancement Request
```
The daily log is great! But can you also:
- Add a "Why This Matters" section for each learning
- Create a sequence diagram for the data flow
- Link to related documentation
```

### Round 3: Cross-linking
```
Now that Phase 1 is done, can you:
- Create index pages linking all 7 daily logs
- Update architecture docs with final diagrams
- Create transition notes for Phase 2 startup
```

---

## 💾 Copy-Paste Helpers

### Get Your Test Output in Right Format:

```bash
# Unit test results
pytest tests/unit/ -v > /tmp/unit_tests.txt

# Integration test results  
pytest tests/integration/ -v > /tmp/integration_tests.txt

# Coverage report
pytest --cov=src --cov-report=term-missing > /tmp/coverage.txt

# All commits today
git log --oneline $(git log -1 --format=%H)~1..HEAD > /tmp/commits.txt

# Now paste these files when asking for help
```

### Quick Command to Generate Summary:

```bash
#!/bin/bash
# Save as scripts/doc-summary.sh

echo "=== Day X Summary ==="
echo ""
echo "Tests:"
pytest tests/ -q 2>&1 | tail -5
echo ""
echo "Coverage:"
pytest --cov=src --cov-report=term-missing 2>&1 | grep TOTAL
echo ""
echo "Commits:"
git log --oneline -5
echo ""
echo "Files Changed:"
git diff --name-only HEAD~5..HEAD 2>/dev/null | sort -u
```

Then run: `./scripts/doc-summary.sh | pbcopy` (macOS) to copy to clipboard

---

## 🎬 Example: Full Day Documentation Flow

### Step 1: End of Day (5:00 PM)
```bash
# Verify everything works
pytest tests/ -v --cov=src

# Get your summary
./scripts/doc-summary.sh
```

### Step 2: Prepare Documentation Prompt (5:10 PM)
```
Documenting Day 1: Vector Store Foundation + FastAPI Service

✅ Tasks Completed:
- Task 1.1: Project Setup ✅
- Task 1.2: Text Chunking Module ✅  
- Task 1.3: FAISS Vector Store ✅
- Task 1.4: FastAPI Endpoints ✅
- Task 1.5: Docker Setup ✅
- Task 1.6: Documentation ✅

📊 Results:
- Tests Passing: 15/15 ✅
- Code Coverage: 92%
- Lines of Code: ~450
- Commits: 4
- Docker: ✅ Building

Recent Commits:
$(git log --oneline -4)

Key Learnings:
1. Embedding normalization is critical before FAISS insertion
2. Two-stage retrieval (FAISS + reranking) improves precision
3. Chunk overlap maintains context continuity

Challenges:
1. FAISS dimension mismatch - Fixed by normalizing embeddings
2. Empty results threw error - Fixed with proper validation

Please help me format this into docs/daily-logs/day-01-20260215.md
```

### Step 3: I React (5:15 PM)
I use the provided data to:
- Create formatted day-01 file
- Add proper markdown styling
- Generate commit references
- Create any diagrams needed
- Link to architecture docs
- Update progress trackers

### Step 4: You Review & Commit (5:20 PM)
```bash
git add docs/daily-logs/day-01-20260215.md
git add docs/architecture/day01-rag-foundation.md
git commit -m "docs(day-01): daily completion log and architecture"
git tag day-01-complete
git push origin main
```

---

## 🔑 Key Principles for Documentation Help

### Principle 1: Data First
- Always provide actual test output
- Share real git logs
- Include actual metrics
- Don't estimate/guess numbers

### Principle 2: Context Matters
- Explain what you built, not just code
- Share why decisions were made
- Connect to overall goals
- Link to previous days

### Principle 3: Timing
- Ask at EOD, not mid-task
- Only after tests pass
- When you have metrics
- Ready to commit code

### Principle 4: Clarity
- Use the templates provided
- Paste actual output
- Be specific about what you want
- Ask for specific formats

---

## 📞 Examples for Every Day Type

### Regular Day Prompt (Days 1-6, 8-14, 16-20):
```
Documenting Day X: [Title]

Tests: $(pytest tests/ -q)
Coverage: $(pytest --cov=src | tail -1)
Commits: $(git log --oneline -X)

[Include key learnings and challenges]

Format into daily log please!
```

### Component Integration Day (Days 3, 7, 14, 21):
```
Documenting Day X: [Milestone - full system integration]

All components working together:
- Previous components: [list]
- New components: [list]
- Integration points: [list]

Tests: $(pytest tests/ -q)
E2E test results: [paste]

Architecture diagram needed?

Please create comprehensive daily log with system diagrams.
```

### Blocking/Challenge Day:
```
Documenting Day X: [Title]

Faced significant challenge:
[Describe blocker and solution]

Despite challenge:
- Tests passing: $(pytest tests/ -q)
- Component delivered: [description]
- Learned: [key insight]

Help me document the problem-solving approach?
```

---

## ✅ Checklist Before Asking for Help

Before submitting documentation request:

- [ ] All tests pass
- [ ] Coverage >90% 
- [ ] Code committed
- [ ] Git history clean
- [ ] You have test output
- [ ] You have commit log
- [ ] You have actual metrics
- [ ] You've written key learnings
- [ ] You've documented challenges
- [ ] You're ready to commit docs

If all checked, you're ready to ask for documentation help!

---

## 🎯 Documentation Help Response Time Goals

| Type | Complexity | Expected Time |
|------|-----------|----------------|
| Format existing data | Low | 2-5 min |
| Add diagrams | Medium | 5-10 min |
| Polish & enhance | Medium | 5-10 min |
| Cross-link & integrate | High | 10-15 min |
| Phase summary | Complex | 15-20 min |

Keep your requests focused for faster help!

---

## 📞 TL;DR - Quick Reference

**Every day at EOD:**
1. Run: `pytest tests/ -q && pytest --cov=src`
2. Get: `git log --oneline -5`
3. Share: test results + commits + learnings
4. Ask: "Help me format Day X into the daily log?"
5. I: Create your formatted daily-log-XX.md
6. You: Review and commit

That's it! The templates and examples above are your reference for what to include.
