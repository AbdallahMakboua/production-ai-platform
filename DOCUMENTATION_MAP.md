# Documentation Map & Strategy

## 📚 Core Reading Order

For someone new to the project:

1. **[START_HERE.md](START_HERE.md)** (5 min) → Project overview and links
2. **[DAYS_1-3_DETAILED_WORKFLOW.md](DAYS_1-3_DETAILED_WORKFLOW.md)** (Reference) → Your daily task checklist  
3. **[docs/LEARNINGS.md](docs/LEARNINGS.md)** (Reference) → Technical decisions & lessons
4. **[docs/architecture/day01-rag-foundation.md](docs/architecture/day01-rag-foundation.md)** (Reference) → Architecture

## 📋 Document Purposes & Locations

### Primary References (Use These)

| Document | Purpose | When to Use |
|----------|---------|------------|
| [START_HERE.md](START_HERE.md) | Entry point + project overview | First time, quick orientation |
| [DAYS_1-3_DETAILED_WORKFLOW.md](DAYS_1-3_DETAILED_WORKFLOW.md) | Day-by-day micro-tasks with tests & commits | During development - follow each section |
| [docs/LEARNINGS.md](docs/LEARNINGS.md) | Technical discoveries, issues, solutions | When debugging similar problems |
| [docs/architecture/day01-rag-foundation.md](docs/architecture/day01-rag-foundation.md) | System design & rationale | For understanding design choices |
| [docs/daily-logs/](docs/daily-logs/) | Daily execution logs | At end of each day for reflection |

### Secondary References (Reference Only)

| Document | Purpose | Note |
|----------|---------|------|
| [QUICK_START.md](QUICK_START.md) | TL;DR daily routine | Points to main docs, avoid duplication |
| [TESTING_GUIDE.md](TESTING_GUIDE.md) | Test patterns and strategies | Patterns from Day 1 implementation |
| [GIT_COMMIT_STRATEGY.md](GIT_COMMIT_STRATEGY.md) | Commit message format | Used in task execution |

### Historical / Planning Docs (Archive)

These duplicate content from above. Keep for reference but use consolidated docs above:
- `EXECUTION_GUIDE.md` - See DAYS_1-3_DETAILED_WORKFLOW.md instead
- `COMPLETE_SYSTEM_SUMMARY.md` - See START_HERE.md + DAYS_1-3_DETAILED_WORKFLOW.md
- `DETAILED_EXECUTION_PLAN.md` - See DAYS_1-3_DETAILED_WORKFLOW.md instead
- `21_day_ai_devops_plan.md` - High-level overview, rarely needed

## 🎯 How to Avoid Duplication

### Rule 1: One Source of Truth per Topic
- **Daily Tasks** → Only in DAYS_1-3_DETAILED_WORKFLOW.md
- **Architecture** → Only in docs/architecture/<dayX>.md
- **Learnings** → Only in docs/LEARNINGS.md

### Rule 2: Reference, Don't Copy
- When explaining where to find info in other docs, use links
- Example: "See [DAYS_1-3_DETAILED_WORKFLOW.md](DAYS_1-3_DETAILED_WORKFLOW.md#task-11) for setup steps"
- Bad: Copy-pasting the setup steps into multiple files

### Rule 3: Update Once, Links Update Everywhere
- If task 1.1 changes, update only DAYS_1-3_DETAILED_WORKFLOW.md
- All other docs with links automatically reference the new version

## 📁 Recommended File Structure

```
/
├── START_HERE.md                    ← Entry point
├── DAYS_1-3_DETAILED_WORKFLOW.md    ← Daily task reference
├── TESTING_GUIDE.md                 ← Test patterns
├── GIT_COMMIT_STRATEGY.md           ← Commit format
│
├── docs/
│   ├── LEARNINGS.md                 ← Technical insights
│   ├── PROGRESS.md                  ← Overall progress tracking
│   ├── BLOCKERS.md                  ← Issues encountered
│   ├── architecture/
│   │   ├── day01-rag-foundation.md  ← Day 1 architecture
│   │   ├── day02-retrieval.md       ← Day 2 architecture (future)
│   │   └── ...
│   └── daily-logs/
│       ├── DAY_TEMPLATE.md          ← Template for new days
│       ├── day-01-20260220.md       ← Execution log for Day 1
│       └── ...
│
└── [Archive - Optional, for reference only]
    ├── EXECUTION_GUIDE.md           (duplication of DAYS_1-3_...)
    ├── DETAILED_EXECUTION_PLAN.md   (duplication of DAYS_1-3_...)
    └── COMPLETE_SYSTEM_SUMMARY.md   (summary of above)
```

## 🧹 Ongoing Maintenance

### When Adding New Content
1. Check if it fits in an existing doc
2. If new topic, create: `docs/architecture/<dayX>.md` or add to `docs/LEARNINGS.md`
3. Link from `START_HERE.md` or relevant daily section

### When Documentation Gets Long
- If a doc > 1500 lines, split into multiple files
- Create index file linking to sub-documents
- Example: `docs/architecture/index.md` linking to `day01.md`, `day02.md`, etc.

### Monthly Cleanup
- Archive completed week's logs to `docs/daily-logs/archive/`
- Review LEARNINGS.md for patterns to add to README
- Check for dead links in documentation

## 💾 What Each File Currently Contains

### [DAYS_1-3_DETAILED_WORKFLOW.md](DAYS_1-3_DETAILED_WORKFLOW.md) (1125 lines)
- ✅ Day 1: 6 tasks with all steps
- ✅ Day 2: Placeholder
- ✅ Day 3: Placeholder
- **Purpose**: Detailed task breakdown with tests, commits, expected outputs
- **Status**: Day 1 complete and proven, days 2-3 need implementation

### [docs/LEARNINGS.md](docs/LEARNINGS.md) (280 lines)
- ✅ Day 1 learnings: 4 critical issues fixed
- ✅ Architecture decisions validated
- ✅ Performance observations
- ✅ Process improvements made
- **Purpose**: Technical reference for future days
- **Status**: Newly created, comprehensive

### [docs/architecture/day01-rag-foundation.md](docs/architecture/day01-rag-foundation.md) (160 lines)
- ✅ Component diagram
- ✅ Data flow specifications
- ✅ Implementation details
- ✅ Design decisions with rationale
- **Purpose**: Design documentation
- **Status**: Complete for Day 1

### Duplication Issues Identified

#### EXECUTION_GUIDE.md (882 lines)
- **Duplicates**: DAYS_1-3_DETAILED_WORKFLOW.md (Day 1 tasks)
- **Action**: Reference DAYS_1-3_DETAILED_WORKFLOW.md instead

#### COMPLETE_SYSTEM_SUMMARY.md (450+ lines)
- **Duplicates**: Task descriptions from multiple sources
- **Action**: Point to DAYS_1-3_DETAILED_WORKFLOW.md for specifics

#### DETAILED_EXECUTION_PLAN.md (500+ lines)
- **Duplicates**: Day 1 task breakdown
- **Action**: Reference DAYS_1-3_DETAILED_WORKFLOW.md

## ✅ Consolidation Checklist

- [x] Create [docs/LEARNINGS.md](docs/LEARNINGS.md) with today's insights
- [x] Create [docs/architecture/day01-rag-foundation.md](docs/architecture/day01-rag-foundation.md)
- [ ] Add "See [docs/architecture/](docs/architecture/) for design decisions" to EXECUTION_GUIDE.md
- [ ] Add "See [DAYS_1-3_DETAILED_WORKFLOW.md](DAYS_1-3_DETAILED_WORKFLOW.md) for daily tasks" to README
- [ ] Update [START_HERE.md](START_HERE.md) with document map
- [ ] Archive old planning docs to `/archive` folder

## 🔗 Quick Links by Scenario

**I want to...**
- **...understand the project**: Read [START_HERE.md](START_HERE.md)
- **...execute today's tasks**: Open [DAYS_1-3_DETAILED_WORKFLOW.md](DAYS_1-3_DETAILED_WORKFLOW.md)
- **...see Day 1 architecture**: Look at [docs/architecture/day01-rag-foundation.md](docs/architecture/day01-rag-foundation.md)
- **...understand why we did X**: Check [docs/LEARNINGS.md](docs/LEARNINGS.md)
- **...debug a similar issue**: Search [docs/LEARNINGS.md](docs/LEARNINGS.md) for the problem
- **...know overall progress**: See [docs/PROGRESS.md](docs/PROGRESS.md)
