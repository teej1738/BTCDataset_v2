# Knowledge Base Layer: Detailed Design Specification

*Generated: 2026-03-04*
*Companion to: FOUNDATION_DESIGN.md (AD-7)*

---

## 0. Problem Statement

Claude Code starts every session with zero memory. The knowledge base is
the mechanism by which CC reconstructs context, understands history, and
makes informed decisions without repeating past mistakes.

BTCDataset_v2 used three files for this purpose:
- CLAUDE.md (793 lines): Project overview, configs, validation status
- STRATEGY_LOG.md (3,923 lines): 55 decision entries (D01-D55)
- knowledge.md (696 lines): ICT SHAP evidence, research questions

**What worked:**
- CLAUDE.md as the "read first" file. CC consistently reads it and gets context.
- Append-only decision log prevents history loss.
- SHAP evidence hierarchy provides quantitative basis for decisions.

**What failed:**
- STRATEGY_LOG.md grew to 3,923 lines. CC cannot read and process the entire
  file in a single session. Later entries duplicate or contradict earlier ones.
- knowledge.md mixed structured data (SHAP tables) with prose (optimizer
  behavior rules). Parsing requires interpretation, not just reading.
- No STATUS.md equivalent. CC has to infer "what's next" from the end of
  STRATEGY_LOG.md, which is unreliable after context compression.
- No structured format for experiment results. Each D-entry in STRATEGY_LOG
  has a different format depending on when it was written.

---

## 1. File Inventory

### 1.1 CLAUDE.md -- Project Context (READ FIRST, ALWAYS)

**Purpose:** Give CC full project context in one file. Everything CC needs
to start working without reading anything else.

**Target length:** 200-400 lines. If it exceeds 500, split sections into
separate files and reference them.

**Sections:**

```markdown
# SELF-MAINTENANCE INSTRUCTIONS
[When and how to update this file -- same pattern as BTCDataset_v2]

# PROJECT OVERVIEW
[1 paragraph: what this project is and what stage it is at]

# CURRENT STATE
| Metric | Value |
|--------|-------|
| Best experiment | [name] |
| AUC | [value] |
| Daily Sharpe | [value] |
| Gates | [X/10 PASS] |
| Holdout status | [UNTOUCHED / TOUCHED on date] |
| Last experiment | [name, date] |

# FILE STRUCTURE
[Directory tree with annotations]

# INSTRUMENT CONFIGS
[Table: instrument, timeframe, data path, holdout dates]

# PRODUCTION CONFIG
[Current best experiment: all key metrics]

# ENVIRONMENT NOTES
[Python version, OS, encoding, installed packages]

# WHAT NOT TO DO
[Anti-patterns from FOUNDATION_DESIGN.md Section 12]
```

**Update rules:**
- Update after any task that changes project state.
- Only update the specific values that changed.
- Never rewrite the entire file.

---

### 1.2 STATUS.md -- Current State (ephemeral)

**Purpose:** Tell CC exactly what to do right now. This file reflects
the present, not history.

**Target length:** 20-50 lines. This is a dashboard, not a document.

**Format:**

```markdown
# STATUS
Last updated: [date]

## CURRENT TASK
[One sentence: what is being worked on right now]
[Status: IN PROGRESS / BLOCKED / WAITING FOR USER]

## BLOCKERS
- [Blocker 1: what, why, how to unblock]
- (none) if no blockers

## NEXT STEPS (priority order)
1. [Next thing to do after current task completes]
2. [Thing after that]
3. [Thing after that]

## RECENT COMPLETIONS
- [Last 3 completed tasks with dates]

## SESSION CHECKLIST
Before starting work:
- [ ] Read CLAUDE.md
- [ ] Read this STATUS.md
- [ ] Read KNOWLEDGE.md if running experiments
- [ ] Run pytest tests/ -x --tb=short

After completing work:
- [ ] Update CLAUDE.md if state changed
- [ ] Update STATUS.md (current task, next steps)
- [ ] Update KNOWLEDGE.md if experiment completed
- [ ] Run pytest tests/ -x --tb=short
```

**Update rules:**
- Update at the START of every session (read, then update current task).
- Update at the END of every session (mark task complete, set next steps).
- If a session is interrupted, STATUS.md still reflects what was in progress.

---

### 1.3 KNOWLEDGE.md -- Evidence Base

**Purpose:** Structured evidence that informs experiment design decisions.
Not prose. Tables and lists that CC can parse mechanically.

**Target length:** 300-500 lines. Archive old sections when it exceeds 500.

**Sections:**

```markdown
# EVIDENCE BASE
Last updated: [date]

## FEATURE EVIDENCE HIERARCHY
[Table: rank, feature, SHAP, category, status (active/pruned/untested)]

## EXPERIMENT HISTORY
[Table: name, date, AUC, WR, EV, Sharpe_daily, gates, verdict, notes]

## RESEARCH QUESTIONS (open)
[Table: ID, question, priority, status, blocking conditions]

## DEAD ENDS
[Table: ID, hypothesis, result, scope boundary (what NOT to retry)]

## PROVEN FINDINGS
[Table: ID, finding, evidence (which experiment), confidence level]

## PARAMETER EVIDENCE
[Table: parameter, tested values, best value, experiment that proved it]
```

**Structured table format (mandatory):**

```markdown
## EXPERIMENT HISTORY

| Name | Date | AUC | WR | EV_R | SR_daily | Gates | Verdict |
|------|------|-----|----|------|----------|-------|---------|
| btc_long_baseline | 2026-03-15 | 0.7933 | 76.0% | +1.23 | 10.71 | 10/10 | PASS |
| btc_long_pruned | 2026-03-16 | 0.7938 | 74.9% | +1.20 | 8.50 | 10/10 | PASS |
```

No prose experiment descriptions in KNOWLEDGE.md. If an experiment needs
explanation, put it in the experiment's result JSON. KNOWLEDGE.md contains
only the structured summary.

**Update rules:**
- After EVERY experiment: add row to EXPERIMENT HISTORY.
- After EVERY SHAP analysis: update FEATURE EVIDENCE HIERARCHY.
- When a research question is answered: move from RESEARCH QUESTIONS to
  PROVEN FINDINGS or DEAD ENDS.
- Never edit old rows. Append only. If a finding is superseded, add a new
  row with a note referencing the old one.

---

### 1.4 DECISIONS.md -- Architectural Decisions (append-only)

**Purpose:** Record WHY architectural choices were made. If a future session
asks "why did we use Pydantic?" the answer is here.

**Target length:** Unlimited (append-only). Each entry is self-contained.

**Format:**

```markdown
# ARCHITECTURAL DECISIONS

## AD-001: Storage architecture
Date: 2026-03-15
Decision: Parquet on disk + DuckDB for queries
Rationale: [2-3 sentences]
Alternatives considered: TimescaleDB (rejected: server process),
  Arctic (rejected: MongoDB dependency)
Status: ACTIVE

## AD-002: Configuration management
Date: 2026-03-15
Decision: TOML + Pydantic
Rationale: [2-3 sentences]
Status: ACTIVE

## AD-003: [Supersedes AD-001]
Date: 2026-06-15
Decision: Migrate to DuckDB as primary storage
Rationale: Dataset grew beyond pandas comfort zone
Supersedes: AD-001
Status: ACTIVE
```

**Update rules:**
- Append a new entry when making a design choice.
- Never edit old entries.
- If a decision changes, add a new entry that references and supersedes
  the old one. The old entry remains for historical context.

---

## 2. CC Session Protocol

### Session startup (every session, no exceptions)

```
Step 1: Read CLAUDE.md
  -> Understand project state, file structure, current metrics
  -> If CLAUDE.md says "read X before starting", read X

Step 2: Read STATUS.md
  -> Understand current task and blockers
  -> If STATUS.md says "IN PROGRESS: [task]", that task was interrupted
     and may need cleanup before continuing

Step 3: Check user's instruction
  -> If user says "continue", pick up from STATUS.md current task
  -> If user gives a new task, update STATUS.md accordingly

Step 4: Read only the RELEVANT sections of KNOWLEDGE.md for today's task:
  - FEATURE EVIDENCE HIERARCHY: only when working on features
  - EXPERIMENT HISTORY: only when proposing or recording experiments
  - DEAD ENDS: always check before proposing anything new
  - RESEARCH QUESTIONS: when planning next experiment
  Loading the full file every session degrades CC performance as
  the project grows. Context window saturation occurs around
  20,000 tokens (~300 lines). Load sections selectively.
```

### Session shutdown (every session)

```
Step 1: Run pytest tests/ -x --tb=short
  -> If tests fail, fix before ending session
  -> If fix is non-trivial, document in STATUS.md as BLOCKED

Step 2: Update STATUS.md
  -> Mark current task: COMPLETE, IN PROGRESS, or BLOCKED
  -> Set NEXT STEPS for the next session
  -> Note any blockers with resolution hints

Step 3: Update CLAUDE.md (if state changed)
  -> Update metrics if experiment completed
  -> Update file structure if new files created
  -> Update CURRENT STATE table

Step 4: Update KNOWLEDGE.md (if experiment completed)
  -> Add experiment row to EXPERIMENT HISTORY
  -> Update FEATURE EVIDENCE if SHAP ran
  -> Move answered RQs to PROVEN FINDINGS or DEAD ENDS
```

---

## 3. Experiment Documentation Format

### Experiment TOML (input)

Every experiment has a TOML config file in `config/experiments/`. This is
the SOLE input to the training pipeline. If it's not in the TOML, it doesn't
exist.

Template: `config/experiments/_template.toml` (always committed, never deleted).

### Experiment result JSON (output)

Every experiment produces a JSON result in `experiments/results/`. This
contains ALL computed metrics, gate results, fold details, and timing info.

CC reads this JSON to update KNOWLEDGE.md. The JSON is the source of truth;
KNOWLEDGE.md is the summary.

### KNOWLEDGE.md row (summary)

After reading the result JSON, CC adds one row to the EXPERIMENT HISTORY
table. This row contains: name, date, AUC, WR, EV, daily Sharpe, gates
summary, and a one-word verdict (PASS/FAIL/MARGINAL).

---

## 4. Evidence Hierarchy Format

### Feature evidence table

```markdown
| Rank | Feature | SHAP | Category | Source | Status |
|------|---------|------|----------|--------|--------|
| 1 | ict_ob_bull_age | 0.2057 | ICT/Core | D36 | active |
| 2 | ote_dist | 0.1860 | ICT/OTE | D36 | active |
| 3 | ict_ob_bear_age | 0.1474 | ICT/Core | D36 | active |
| ... | ... | ... | ... | ... | ... |
| 64 | [last active] | [value] | [cat] | [source] | active |
| -- | [pruned features below this line] | | | | pruned |
| 65 | ict_macro_ny_open | 0.0001 | ICT/Session | D36 | pruned |
```

**Status values:**
- `active`: Currently used in production config
- `pruned`: Removed in D55 with zero AUC loss
- `untested`: Not yet evaluated (new feature family)
- `dead`: Confirmed no predictive value across multiple experiments

### Research question table

```markdown
| ID | Question | Priority | Status | Blocking | Last tested |
|----|----------|----------|--------|----------|-------------|
| RQ1 | OB quality score vs raw age? | P1 | answered:NO | - | D43 |
| RQ2 | H1 OTE vs 5m OTE? | P1 | open | - | - |
| RQ3 | Triple confluence WR > 70%? | P1 | open | - | - |
```

**Status values:**
- `open`: Not yet tested
- `in_progress`: Experiment running
- `answered:YES`: Hypothesis confirmed (move to PROVEN FINDINGS)
- `answered:NO`: Hypothesis rejected (move to DEAD ENDS)
- `blocked`: Cannot test until prerequisite is met

---

## 5. Archival Protocol

When KNOWLEDGE.md exceeds 300 lines:

1. Identify sections that are purely historical (no active research questions
   reference them).
2. Move those sections to `outputs/knowledge_archive_YYYY.md`.
3. In KNOWLEDGE.md, add a one-line reference:
   `[Archived experiments pre-2026: outputs/knowledge_archive_2026.md]`
4. Never delete archived content. The archive is append-only.

### Sharpe Metric Policy

Daily Sharpe is now the primary Sharpe metric. Per-bar Sharpe is recorded
for historical comparison only. All new experiments report daily Sharpe as
the headline number. Gates use daily Sharpe. The evaluation JSON always
includes both daily and per-bar Sharpe, clearly labeled.

---

## 6. Cross-Reference Policy

**Rule: Every file must be self-contained for its purpose.**

- CLAUDE.md must give CC enough context to start working WITHOUT reading
  any other file.
- STATUS.md must tell CC what to do next WITHOUT reading STRATEGY_LOG.
- KNOWLEDGE.md must provide enough evidence to propose an experiment
  WITHOUT reading individual experiment JSONs.
- DECISIONS.md must explain WHY a choice was made WITHOUT reading source code.

Cross-references are allowed for details (e.g., "see experiment JSON for
fold-level metrics") but the summary in the knowledge file must be
sufficient for decision-making.

---

## 7. Failure Recovery

### CC loses context mid-task

STATUS.md says "IN PROGRESS: [task]". CC reads it, understands where work
was interrupted, and can either:
- Continue the task (if partially complete)
- Restart the task (if state is unclear)

### KNOWLEDGE.md is stale

CC runs an experiment and finds it duplicates a previous one. This means
KNOWLEDGE.md was not updated. CC should:
1. Read `experiments/registry.json` to find the duplicate.
2. Update KNOWLEDGE.md with the missing entry.
3. Continue with the correct next research question.

### Conflicting information

If CLAUDE.md and KNOWLEDGE.md disagree on a metric:
1. The experiment result JSON is the source of truth.
2. Update whichever file is wrong.
3. Add a note to DECISIONS.md about the conflict and resolution.

---

## 8. BTCDataset_v2 Migration

### What transfers to new KNOWLEDGE.md

From BTCDataset_v2's project files, the following evidence transfers:

**Feature evidence:**
- D55 pruning results: 670 -> 64 features, 7/10 ICT families dead weight
- SHAP rankings from D36 (top 30 features)
- Regime feature assessment from D50

**Dead ends:**
- DE01: d1_ict_market_trend as hard filter
- DE02: Symmetric short session rules
- DE03: Regime classification as hard filter
- DE04: Fixed 48-bar cooldown at low frequency
- DE05: All-time cumulative CVD
- DE06: H4 CE limit entries

**Proven findings:**
- FVG is the sole structural edge generator
- h4_sweep is the quality gate
- Direction-session routing is essential
- OTE distance is genuinely predictive (SHAP #2)
- Daily Sharpe 10.71 (D55b holdout), beats BH 9/9 folds

**What does NOT transfer:**
- D-numbered decision IDs (restart numbering in new project)
- E-numbered experiment IDs (different registry)
- Specific file paths (different directory structure)
- The exact 793-line CLAUDE.md (write fresh for new project)

---

*End of KNOWLEDGE_BASE_DESIGN.md*
