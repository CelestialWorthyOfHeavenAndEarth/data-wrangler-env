---
title: data-wrangler-env
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# DataWranglerEnv

**A multi-dimensional, exploit-resistant data quality and cleaning environment for training and evaluating AI agents.**

Submitted to the Meta PyTorch Hackathon x Scaler School of Technology — Round 1.

---

## Overview

Data wrangling accounts for 80% of a data scientist's working time, yet no structured benchmark exists for training AI agents to automate it. DataWranglerEnv fills this gap.

An agent receives a deliberately corrupted dataset and must use natural language commands to diagnose and clean it within a step budget. The environment scores results across eight dimensions, enforces domain-specific business rules, tracks every operation for provenance, and includes anti-exploit mechanisms to prevent agents from gaming the grading system.

The environment is fully compatible with the [OpenEnv specification](https://openenv.meta.com) and deploys as a Docker-based Hugging Face Space.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        inference.py                          │
│                                                              │
│   Phase 1: DIAGNOSE    Phase 2: CLEAN    Phase 3: VERIFY    │
│   profile              fix_dtype         validate            │
│   find_missing         fill_missing      check_rules         │
│   find_duplicates      remove_duplicates submit              │
│   check_rules          clip / replace                        │
│   find_outliers        standardize                           │
└──────────────────────────┬──────────────────────────────────┘
                           │  HTTP  (reset / step / state)
┌──────────────────────────▼──────────────────────────────────┐
│                   OpenEnv HTTP Server (FastAPI)              │
├─────────────────────────────────────────────────────────────┤
│                DataWranglerEnvironment                       │
│                                                              │
│  ┌─────────────────┐   ┌──────────────┐   ┌─────────────┐  │
│  │  CleaningEngine  │   │   Dataset    │   │   Grader    │  │
│  │                  │   │  Generator   │   │             │  │
│  │  20+ commands    │   │  3 tasks     │   │ 8 dimensions│  │
│  │  Quoted args     │   │  Seeded RNG  │   │ Clamped to  │  │
│  │  Error handling  │   │  Ground truth│   │ (0.001,0.999│  │
│  └─────────────────┘   └──────────────┘   └─────────────┘  │
│                                                              │
│  ┌─────────────────┐   ┌──────────────┐   ┌─────────────┐  │
│  │   Undo Stack     │   │   Operation  │   │   Golden    │  │
│  │                  │   │     Log      │   │    Rows     │  │
│  │  Up to 10 states │   │  (Lineage)   │   │ Anti-exploit│  │
│  └─────────────────┘   └──────────────┘   └─────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## Task Design

Three tasks of increasing complexity, each with a fully reproducible dirty/clean dataset pair generated from a seeded random process.

| Task | Domain | Dirty Size | Columns | Step Budget |
|------|--------|-----------|---------|-------------|
| `task_1_easy` | Customer Records | 58 rows | 5 | 30 |
| `task_2_medium` | Sales Transactions | 218 rows | 8 | 50 |
| `task_3_hard` | Healthcare Records | 1,070+ rows | 12 | 80 |

### Issues Injected Per Task

**task_1_easy** — Customer Records
- Missing values (~10%) in `age` and `email`
- 5 exact duplicate rows
- City name typos (e.g., `"New Yrok"`, `"chicgo"`)

**task_2_medium** — Sales Transactions
- Missing values in `price`, `category`, `region`
- ~8% duplicate rows
- Type errors: prices stored as strings with `$` prefix
- Date format inconsistencies: mixed `YYYY-MM-DD` and `DD/MM/YYYY`
- Negative quantities
- Extreme price outliers ($50,000+)

**task_3_hard** — Healthcare Records
- Missing values across 6 columns
- 5% exact duplicates
- 20 fuzzy duplicates (same patient, variant name spelling)
- Gender category inconsistencies: `"Male"`, `"M"`, `"male"`, `"MALE"`
- Blood type format inconsistencies: `"A+"` vs `"a positive"`
- Impossible values: negative height, height > 250cm, negative weight
- Cross-column logic errors: diastolic blood pressure exceeding systolic
- Date format inconsistencies across `visit_date` and `dob`

### Red Herring Data

Each task includes deliberately valid-but-suspicious records to test whether agents over-clean:

- A customer named `"Null Fisher"` — a real person, not a missing value
- A product priced at `$0.00` — a legitimate free promotional item
- A bulk order with `quantity = 500` — a valid enterprise purchase
- A customer aged `0` — a valid infant record

Removing these rows penalises the agent's data preservation score.

---

## Grading System

The grader computes a composite score from eight weighted dimensions. All scores are clamped to the open interval `(0.001, 0.999)` per the OpenEnv validator specification.

```
Composite Score = sum(dimension_score * weight)
```

| Dimension | Weight | Description |
|-----------|--------|-------------|
| Missing Values Fixed | 20% | Proportion of missing values correctly filled |
| Value Accuracy | 20% | Cell-level match against ground truth |
| Duplicates Removed | 15% | Proportion of injected duplicate rows removed |
| Type Correctness | 15% | Column dtypes matching expected types |
| Constraint Compliance | 10% | Business rule satisfaction rate |
| Data Preservation | 10% | Proportion of legitimate rows retained |
| Golden Row Integrity | 5% | Anti-exploit: selected clean rows must survive |
| Step Efficiency | 5% | Reward for completing the task in fewer steps |

### Score Flow

```
reset()
  └── Initial score: 0.001 (uncleaned)

step("fill_missing age mean")
  └── Step reward: improvement delta * 2.0 + 0.03

step("validate")
  └── Full 8-dimension report, small fixed reward

step("submit")
  └── Final composite score becomes episode reward
```

---

## Business Rule Validation

Each task defines domain-specific constraints that go beyond structural cleaning. Agents can inspect violations with `check_rules` and are scored on compliance.

**task_1_easy**
- `age` must be in range [0, 120]
- `email` must be non-null
- `email` must match pattern `.*@.*\..*`

**task_2_medium**
- `price` must be in range [0, 10000]
- `quantity` must be in range [1, 1000]
- `transaction_id` must be non-null
- `region` must be one of: North, South, East, West, Central

**task_3_hard**
- `height_cm` must be in range [30, 250]
- `weight_kg` must be in range [1, 300]
- `bp_systolic` must be greater than `bp_diastolic`
- `gender` must be one of: Male, Female
- `blood_type` must be a valid ABO+Rh value
- `patient_id` must be non-null

---

## Anti-Exploit Mechanisms

### Golden Rows

At reset time, a random subset of clean rows (5–15 per task) is designated as golden. These rows are checked on submit. If an agent deletes rows indiscriminately, fills columns with constants, or otherwise damages golden rows, the `golden_row_integrity` dimension scores near zero. This prevents:

- Submitting an empty or truncated dataset
- Filling all missing values with a single placeholder
- Wholesale column replacement strategies

### Red Herring Penalty

Agents that remove the intentionally valid-but-suspicious rows lose points on the `data_preservation` dimension, since those rows exist in the ground truth clean dataset.

### Score Variance

The three tasks span genuinely different complexity levels (30, 50, 80 step budgets; 5, 8, 12 columns; 3 to 8 issue types). This ensures meaningful score variance across tasks, satisfying the OpenEnv variance check requirement.

---

## Data Lineage

Every command executed in an episode is recorded. On `submit`, the environment appends a full cleaning provenance report to the final observation:

```
=== Cleaning Provenance Report ===
Total steps: 18
  Diagnostic (read-only): 7
  Data-modifying: 10
  Undo operations: 1

Data Transformations Applied:
  Step  4: fix_dtype price float         (rows: 218 -> 218)
  Step  6: fill_missing price median
  Step  8: remove_duplicates             (rows: 218 -> 202)
  Step 10: clip price 0 10000
  Step 12: fill_missing category mode
  Step 14: fill_missing region mode
  Step 15: undo
  Step 16: standardize region titlecase  (rows: 202 -> 202)
  Step 17: remove_rows quantity less_than 1
  Step 18: fix_dtype date datetime
```

Agents can also query the log mid-episode with the `history` command.

---

## Command Reference

### Diagnostic Commands (read-only)

| Command | Description |
|---------|-------------|
| `profile` | Dataset shape, column types, missing percentages, duplicate count |
| `profile_column COL` | Detailed statistics for a single column |
| `find_missing` | Missing value counts and percentages per column |
| `find_duplicates [COL1,COL2]` | Identify duplicate rows, optionally scoped to column subset |
| `find_outliers COL` | IQR-based outlier detection for a numeric column |
| `check_rules` | Report all business rule violations |
| `history` | Display the operation log (data lineage) |
| `view [N]` | Preview the first N rows (default 10, max 50) |

### Cleaning Commands (modify data)

| Command | Description |
|---------|-------------|
| `fill_missing COL STRATEGY [VALUE]` | Fill nulls using mean / median / mode / constant / forward_fill |
| `remove_duplicates [COL1,COL2] [KEEP]` | Drop duplicate rows; keep = first / last / none |
| `fix_dtype COL TYPE` | Cast column to int / float / str / datetime |
| `replace COL OLD NEW` | Replace exact string matches in a column |
| `regex_replace COL PATTERN REPLACEMENT` | Regex-based find-and-replace |
| `standardize COL METHOD` | Normalize text: lowercase / uppercase / titlecase / strip |
| `remove_rows COL CONDITION VALUE` | Remove rows matching a condition: equals / not_equals / less_than / greater_than / contains |
| `clip COL LOWER UPPER` | Clip numeric values to a closed range |
| `rename_column OLD NEW` | Rename a column |
| `drop_column COL` | Remove a column entirely |
| `sort COL [asc\|desc]` | Sort the dataset by a column |
| `undo` | Restore the dataset to the state before the last modifying command |

### Evaluation Commands

| Command | Description |
|---------|-------------|
| `validate` | Compute the current 8-dimension quality score without ending the episode |
| `submit` | Finalise the episode, compute the final score, and emit the provenance report |

---

## Baseline Agent Strategy

The included `inference.py` implements a structured three-phase agent:

```
Phase 1 — DIAGNOSE  (first 15% of steps)
  profile
  find_missing
  find_duplicates
  check_rules
  find_outliers COL

Phase 2 — CLEAN  (middle 70% of steps)
  fix_dtype (strip non-numeric characters, cast types)
  fill_missing (appropriate strategy per column)
  remove_duplicates
  clip (outliers and impossible values)
  standardize (categorical consistency)
  replace (specific value corrections)
  check_rules + targeted fixes

Phase 3 — VERIFY  (final 15% of steps)
  validate
  fix remaining high-impact issues
  submit
```

The agent accumulates a diagnosis summary from Phase 1 and passes it as context throughout Phase 2 and 3, allowing it to reason about the full picture rather than reacting to each step in isolation.

---

## Project Structure

```
data-wrangler-env/
├── inference.py                          # Baseline three-phase LLM agent
├── models.py                             # Pydantic action and observation models
├── client.py                             # OpenEnv-compatible environment client
├── openenv.yaml                          # OpenEnv specification file
├── pyproject.toml                        # Python project dependencies
├── requirements.txt                      # Pip requirements
├── Dockerfile                            # HF Spaces Docker build configuration
├── server/
│   ├── app.py                            # FastAPI application (reset/step/state endpoints)
│   ├── data_wrangler_env_environment.py  # Core environment: undo, lineage, business rules
│   ├── cleaning_engine.py               # 20+ command implementations
│   ├── dataset_generator.py             # Task generators with red herrings and golden rows
│   └── grader.py                         # 8-dimension composite scoring
└── test_local.py                         # Unit tests (no network required)
```

---

## Running Locally

**Install dependencies**

```bash
pip install openenv-core openai requests pandas numpy
```

**Start the environment server**

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

**Verify the server is running**

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

**Run the baseline agent**

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="your-token-here"
export ENV_BASE_URL="http://localhost:8000"

python inference.py
```

**Run unit tests**

```bash
python test_local.py
# RESULTS: 23 passed, 0 failed
```

---

## Example Session

```
reset(task="task_2_medium", seed=42)
> Dataset loaded: 218 rows x 8 columns
> Columns: transaction_id, product, category, price, quantity, date, customer_id, region

step("profile")
> Dataset Shape: 218 rows x 8 columns
> price    object  198  20  9.2%   201
> category object  208  10  4.6%   8
> region   object  212  6   2.8%   5
> Duplicate rows: 16 (7.3%)

step("check_rules")
> Business Rule Check
> - price: 23 values outside [0, 10000]
> - quantity: 5 values outside [1, 1000]
> - region: 8 invalid categories

step("fix_dtype price float")
> Converted 'price' from object -> float. Coercion errors: 10

step("fill_missing price median")
> Filled 20 missing values in 'price' using strategy 'median'.

step("remove_duplicates")
> Removed 16 duplicate rows. Dataset: 218 -> 202 rows.

step("clip price 0 10000")
> Clipped 3 values in 'price' to [0, 10000].

step("remove_rows quantity less_than 1")
> Removed 5 rows where quantity less_than 1. Dataset: 202 -> 197 rows.

step("validate")
> Overall Quality Score: 0.742 / 1.000
> Missing Values Fixed:  0.952  (20%)
> Duplicates Removed:    0.999  (15%)
> Type Correctness:      0.875  (15%)
> Value Accuracy:        0.681  (20%)
> Data Preservation:     0.985  (10%)
> Constraint Compliance: 0.892  (10%)
> Step Efficiency:       0.720  (5%)
> Golden Row Integrity:  1.000  (5%)

step("submit")
> FINAL REPORT (Submitted)
> Final Quality Score: 0.7891 / 1.0000
> [Cleaning Provenance Report follows...]
```

---

## License

BSD 3-Clause License. See LICENSE for details.
