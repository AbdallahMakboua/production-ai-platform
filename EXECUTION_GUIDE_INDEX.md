# 📚 Complete Execution Guide Index

## 🎯 Quick Navigation

This is your **master index** for the entire 21-day plan. All guides are interconnected below.

---

## 📖 Core Execution Documents

### 1. **[DETAILED_EXECUTION_PLAN.md](DETAILED_EXECUTION_PLAN.md)** 
   **What:** Micro-task breakdown for all 21 days
   **When:** Read at START of each day
   **Contains:**
   - Objective for each day
   - Detailed micro-tasks (1.1, 1.2, 1.3, etc.)
   - Unit test requirements for each task
   - Integration test requirements
   - Manual verification commands (curl scripts)
   - Commit messages ready to use
   - What to verify before moving on
   
   **Example Usage:** 
   > Starting Day 1? Open this file, read Tasks 1.1-1.6, work through them in order.

---

### 2. **[TESTING_GUIDE.md](TESTING_GUIDE.md)**
   **What:** How to write, run, and verify tests
   **When:** Reference during coding, before every commit
   **Contains:**
   - Unit test patterns and examples
   - Integration test patterns
   - Manual verification templates
   - Coverage goals and how to check
   - Mocking strategies
   - Debugging failed tests
   - Testing per phase expectations
   
   **Example Usage:**
   > Starting to test your chunking code? Check "Unit Testing Convention" section in TESTING_GUIDE.

---

### 3. **[GIT_COMMIT_STRATEGY.md](GIT_COMMIT_STRATEGY.md)**
   **What:** When and how to commit code
   **When:** Before every commit
   **Contains:**
   - Commit message format
   - Type, Scope, Subject rules
   - Example commits for every day type
   - Daily commit pattern
   - Pre-commit checklist
   - Git workflows
   - How to fix mistakes
   - Tagging strategy
   
   **Example Usage:**
   > Finished Task 1.1? Check COMMIT section in DETAILED_EXECUTION_PLAN, write message following format in GIT_COMMIT_STRATEGY.

---

### 4. **[DOCUMENTATION_HELP_PROMPTS.md](DOCUMENTATION_HELP_PROMPTS.md)**
   **What:** Exactly what to say when asking for documentation help
   **When:** End of each day, after all tests pass
   **Contains:**
   - When to ask (daily, phase completion, etc.)
   - Template prompts (basic and detailed versions)
   - What I'll do to help
   - Fast prompts for when in a hurry
   - Specific ask examples
   - Phase completion template
   - Copy-paste helpers
   - Full example workflow
   
   **Example Usage:**
   > Day 1 complete, all tests pass. Use Template 2 from DOCUMENTATION_HELP_PROMPTS, provide your test output and commits.

---

## 📅 Daily Workflow Reference

### Morning (Start of Day)
1. Read today's section in **DETAILED_EXECUTION_PLAN.md**
2. Understand objective and micro-tasks
3. Review test requirements

### During Day (Develop)
1. Code based on micro-task description
2. Reference **TESTING_GUIDE.md** for test patterns
3. Follow task completion order
4. Test frequently (TDD: test first or test immediately after)

### Before Each Commit
1. Verify tests pass
2. Check coverage >90%
3. Review **GIT_COMMIT_STRATEGY.md** for message format
4. Commit with proper message

### End of Day (Documentation)
1. Run final test suite
2. Gather metrics (coverage, test count, LOC)
3. Get commit log
4. Use template from **DOCUMENTATION_HELP_PROMPTS.md**
5. Ask me for documentation help
6. Commit daily log files

---

## 🗂️ File System Organization

```
project-root/
├── DETAILED_EXECUTION_PLAN.md          ← Read at start of day
├── TESTING_GUIDE.md                    ← Reference while testing
├── GIT_COMMIT_STRATEGY.md             ← Reference before commit
├── DOCUMENTATION_HELP_PROMPTS.md       ← Use at end of day
├── 21_day_ai_devops_plan.md           ← High-level plan (reference)
├── EXECUTION_GUIDE.md                 ← Setup only
│
├── docs/
│   ├── daily-logs/
│   │   ├── ENHANCED_DAY_TEMPLATE.md    ← Template for daily logs
│   │   ├── day-01-20260215.md         ← Your completed daily logs
│   │   ├── day-02-YYYYMMDD.md
│   │   ├── ... (one per day)
│   │
│   ├── architecture/
│   │   ├── day01-rag-foundation.md     ← Technical architecture
│   │   ├── day02-retrieval-quality.md
│   │   └── ... (component docs)
│   │
│   ├── PROGRESS.md                     ← Overall progress tracker
│   ├── LEARNINGS.md                    ← Learnings across all days
│   └── BLOCKERS.md                     ← Issues and solutions
│
├── src/
│   ├── rag_service/
│   ├── agent/
│   ├── mcp_server/
│   └── ...
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
└── scripts/
    ├── doc-summary.sh                 ← Generate daily summary
    └── pre-commit-check.sh            ← Verify before commit
```

---

## 📋 Daily Checklist

### ✅ Morning (5 minutes)
- [ ] Read today's Objective from DETAILED_EXECUTION_PLAN
- [ ] Understand all micro-tasks for today
- [ ] Review test requirements
- [ ] Start coding

### ✅ Mid-Day (Continuous)
- [ ] Complete Task X.1 + test + commit
- [ ] Complete Task X.2 + test + commit
- [ ] Complete Task X.3 + test + commit
- [ ] Check coverage still >90%

### ✅ End of Day (30 minutes)
- [ ] All tests passing: `pytest tests/ -q`
- [ ] Coverage >90%: `pytest --cov=src`
- [ ] All changes committed
- [ ] Rebase if needed
- [ ] Gather metrics
- [ ] Prepare documentation prompt
- [ ] Ask for documentation help
- [ ] Review formatted daily log
- [ ] Commit documentation
- [ ] Tag day complete: `git tag day-XX-complete`
- [ ] Push to GitHub

---

## 🎓 How to Use This Guide

### Scenario 1: "I'm starting Day 1"
1. Open [DETAILED_EXECUTION_PLAN.md](DETAILED_EXECUTION_PLAN.md)
2. Read "DAY 1: Vector Store Foundation + FastAPI Service"
3. Work through TASK 1.1, 1.2, etc. in order
4. For each task:
   - Read description
   - Refer to [TESTING_GUIDE.md](TESTING_GUIDE.md) for test patterns
   - Write code
   - Write tests
   - Before commit, check [GIT_COMMIT_STRATEGY.md](GIT_COMMIT_STRATEGY.md)
   - Commit with proper message

### Scenario 2: "I don't know how to write the tests for my function"
1. In [TESTING_GUIDE.md](TESTING_GUIDE.md), find "Unit Testing Convention"
2. Copy the pattern
3. Modify for your specific function
4. All test examples provided show exactly how to structure tests

### Scenario 3: "I finished Day 1 and need to document it"
1. Run test suite: `pytest tests/ -v --cov=src`
2. Get commits: `git log --oneline -5`
3. Open [DOCUMENTATION_HELP_PROMPTS.md](DOCUMENTATION_HELP_PROMPTS.md)
4. Use Template 2: "Detailed Daily Summary"
5. Provide your test results, metrics, commits
6. Use this prompt: "Documenting Day 1: Vector Store Foundation. [paste data]"

### Scenario 4: "I made a git mistake"
1. Open [GIT_COMMIT_STRATEGY.md](GIT_COMMIT_STRATEGY.md)
2. Find "Fixing Mistakes" section
3. Follow the exact commands for your mistake type
4. Continue work

### Scenario 5: "I need to understand testing strategy"
1. Start with [TESTING_GUIDE.md](TESTING_GUIDE.md)
2. Read "Overview" and "Testing Pyramid"
3. Check what applies to your day
4. Look at "Day-by-Day Testing Expectations"

### Scenario 6: "I'm at day 7 completing Phase 1"
1. In [DETAILED_EXECUTION_PLAN.md](DETAILED_EXECUTION_PLAN.md), find DAY 7
2. Complete remaining phase 1 work
3. In [DOCUMENTATION_HELP_PROMPTS.md](DOCUMENTATION_HELP_PROMPTS.md), find "Advanced Asks - Phase Completion"
4. Use that template to ask for help
5. Get comprehensive phase 1 summary

---

## 🔑 Key Concepts

### Micro-Tasks
Small, testable units of work. Each takes 30min-1.5 hours.
- **Where:** DETAILED_EXECUTION_PLAN.md
- **Pattern:** Task X.Y with specific requirements
- **Verification:** Tests + manual check

### Test Hierarchy
```
Unit (70%) → Integration (25%) → E2E (5%) → Manual
```
- **Where:** TESTING_GUIDE.md
- **When:** Write tests as you code
- **Goal:** >90% coverage always

### Commit Strategy
One logical change = one commit with clear message
- **Where:** GIT_COMMIT_STRATEGY.md
- **When:** After micro-task + tests pass
- **Format:** type(scope): subject

### Documentation
Clear daily logs with metrics, learnings, challenges
- **Where:** docs/daily-logs/day-XX-YYYYMMDD.md
- **Template:** ENHANCED_DAY_TEMPLATE.md
- **Help:** Use prompts in DOCUMENTATION_HELP_PROMPTS.md

---

## 💡 Pro Tips

### Tip 1: Start with Template
Every day, copy ENHANCED_DAY_TEMPLATE.md template and fill it out as you go (not just at end).

### Tip 2: Keep Tests Green
Never commit broken tests. Always run `pytest tests/ -q` before `git commit`.

### Tip 3: Commit Frequently
4-5 commits per day is normal. Each micro-task = potential commit.

### Tip 4: Document as You Go
Don't wait until end of day. As you complete each micro-task:
- Note what you learned
- Describe the challenge (if any)
- Update your daily log notes

### Tip 5: Use the Examples
Every guide has real examples from the 21-day plan:
- DETAILED_EXECUTION_PLAN: exact test code
- TESTING_GUIDE: pytest patterns
- GIT_COMMIT_STRATEGY: real commit messages
- DOCUMENTATION_HELP_PROMPTS: actual prompt templates

---

## 📞 Getting Help

### During the Day (Development Questions)
If you're stuck on code:
1. Check if there's an example in DETAILED_EXECUTION_PLAN
2. Check TESTING_GUIDE for patterns
3. Look at similar code from previous days (if applicable)

### Before Committing (Git Questions)
If unsure about commits:
1. Check GIT_COMMIT_STRATEGY for your situation
2. Use the pre-commit checklist
3. Copy commit message format from examples

### At End of Day (Documentation)
When ready to document:
1. Follow DOCUMENTATION_HELP_PROMPTS template
2. Gather all metrics and test output
3. Provide git log
4. Ask for formatting help

---

## 🚀 Success Criteria Per Day

Each day is "Done" when:

✅ All micro-tasks completed  
✅ All tests passing (100% required count)  
✅ Coverage >90%  
✅ Code committed with proper messages (3-5 commits)  
✅ Daily log created and formatted  
✅ Documentation committed  
✅ Ready to move to next day  

---

## 📊 Overall Progress Tracking

### By Day
Track in docs/PROGRESS.md:
```
✅ Day 1: Complete
✅ Day 2: Complete
🔄 Day 3: In Progress (Task 3.2/3.3)
⏳ Day 4+: Not Started
```

### By Phase
```
Phase 1 (Days 1-7, Intelligence): X/7 complete
Phase 2 (Days 8-14, Execution): X/7 complete
Phase 3 (Days 15-21, Production): X/7 complete
Overall: X/21 complete
```

### Update at:
- End of each day (in daily log)
- State in all daily documentation requests
- Helps track momentum

---

## 🎬 Your First Actions

### Right Now:
1. Review this file (you're reading it! ✓)
2. Skim all 4 core guides
3. Read DETAILED_EXECUTION_PLAN Day 1 section completely

### Tomorrow Morning:
1. Start Day 1 work
2. Have DETAILED_EXECUTION_PLAN open in one window
3. Have TESTING_GUIDE open for reference
4. Code according to plan

### Tomorrow Evening:
1. Complete all tasks
2. All tests pass
3. Review DOCUMENTATION_HELP_PROMPTS
4. Prepare your first documentation request
5. Get Day 1 log formatted and committed

---

## 📞 Quick Reference Links

**In a meeting? Need quick answer?**

- "How do I test this?" → TESTING_GUIDE.md
- "What do I commit?" → GIT_COMMIT_STRATEGY.md
- "What do I build today?" → DETAILED_EXECUTION_PLAN.md
- "How do I ask for help?" → DOCUMENTATION_HELP_PROMPTS.md
- "What's the overall plan?" → 21_day_ai_devops_plan.md
- "What do I do first?" → EXECUTION_GUIDE.md

---

## ✨ Final Words

You have everything you need:
- ✅ Detailed plan with exact tasks
- ✅ Test patterns for every scenario
- ✅ Git strategy ready to use
- ✅ Documentation templates
- ✅ Help prompts formatted
- ✅ Examples for everything

**All that's left: Execute Day 1.**

The templates and guides above are your constant references. Come back to them continuously. They're designed to be your companion for all 21 days.

Good luck! You've got this. 🚀

---

**Last Updated:** February 20, 2026  
**Plan Start:** Day 1 (February 15, 2026)  
**Phase 1 Complete:** Day 7 (February 21, 2026)  
**Full Plan Complete:** Day 21 (March 7, 2026)
