# DataWranglerEnv 🧹

> A rich, multi-dimensional data quality & cleaning environment for training and evaluating AI agents.
> Built for the **Meta PyTorch Hackathon × Scaler School of Technology**.

---

## 🎯 What is DataWranglerEnv?

Data wrangling consumes **80% of every data scientist's time**, yet no benchmark exists for training AI agents to automate it. DataWranglerEnv fills this gap.

Agents receive a messy real-world dataset and must **diagnose**, **clean**, and **validate** it using natural language commands. The environment provides rich feedback through an **8-dimensional grading system**, supports **undo/redo** for experimentation, enforces **business rules**, and includes **anti-exploit mechanisms** to prevent gaming.

### Why This Matters

- **Real-world utility**: Every data team needs automated data cleaning
- **Progressive complexity**: Easy → Medium → Hard tasks with genuine difficulty scaling
- **Rich feedback**: 8 grading dimensions, data lineage tracking, and cleaning provenance reports
- **Exploit-resistant**: Golden row integrity checks, red herring data, and constraint validation

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│                inference.py                  │
│         (3-Phase LLM Agent Strategy)         │
│    DIAGNOSE → CLEAN → VERIFY → SUBMIT       │
└──────────────────┬──────────────────────────┘
                   │ HTTP (reset/step/state)
┌──────────────────▼──────────────────────────┐
│           OpenEnv HTTP Server                │
│              (FastAPI)                       │
├──────────────────────────────────────────────┤
│         DataWranglerEnvironment              │
│  ┌───────────┐  ┌──────────┐  ┌───────────┐ │
│  │ Cleaning   │  │ Dataset  │  │  Grader   │ │
│  │ Engine     │  │Generator │  │ (8-dim)   │ │
│  │ 20+ cmds   │  │ 3 tasks  │  │ composite │ │
│  └───────────┘  └──────────┘  └───────────┘ │
│  ┌───────────┐  ┌──────────┐  ┌───────────┐ │
│  │   Undo    │  │  Data    │  │  Golden   │ │
│  │   Stack   │  │ Lineage  │  │   Rows    │ │
│  └───────────┘  └──────────┘  └───────────┘ │
└──────────────────────────────────────────────┘
```

---

## 📊 Tasks

| Task | Domain | Size | Issues | Difficulty |
|------|--------|------|--------|------------|
| `task_1_easy` | Customer Records | 53 rows × 5 cols | Missing values, duplicates, typos | ⭐ |
| `task_2_medium` | Sales Transactions | 202 rows × 8 cols | Types, dates, negatives, outliers | ⭐⭐ |
| `task_3_hard` | Healthcare Records | 1000+ rows × 12 cols | Fuzzy dupes, logic errors, impossible values, format inconsistencies | ⭐⭐⭐ |

### Data Quality Issues Covered

- **Missing values** — NaN/None across columns
- **Exact duplicates** — Identical rows
- **Fuzzy duplicates** — Same entity, different spelling ("A. Smith" vs "Alice Smith")
- **Type errors** — Strings in numeric columns ("$42.99" instead of 42.99)
- **Date inconsistencies** — Mixed formats (YYYY-MM-DD vs DD/MM/YYYY)
- **Negative values** — Negative quantities/prices where invalid
- **Outliers** — Extreme statistical outliers
- **Category inconsistencies** — "Male" vs "M" vs "male" vs "MALE"
- **Impossible values** — Negative height, weight > 500kg
- **Cross-column logic errors** — Diastolic BP > Systolic BP
- **Red herring data** — Valid data that looks suspicious (person named "Null", $0 promo items)

---

## 🎖️ 8-Dimension Grading System

| Dimension | Weight | What it measures |
|-----------|--------|------------------|
| Missing Values Fixed | 20% | How many missing values were properly filled |
| Duplicates Removed | 15% | Exact duplicate row elimination |
| Type Correctness | 15% | Column data types match expected types |
| Value Accuracy | 20% | Cell-level accuracy vs ground truth |
| Data Preservation | 10% | Didn't delete legitimate data |
| Constraint Compliance | 10% | Business rule satisfaction (domain-specific) |
| Step Efficiency | 5% | Fewer steps = higher score (rewards planning) |
| Golden Row Integrity | 5% | Anti-exploit: selected "golden" rows must survive cleaning |

---

## 🛡️ Anti-Exploit Features

### Golden Rows
A random subset of clean rows is marked as "golden." If an agent damages these rows (e.g., by deleting everything and re-inserting), the golden_row_integrity score drops. This prevents:
- Agents that delete all rows and submit empty datasets
- Agents that fill all missing values with a single constant
- Agents that game the grader by targeting specific metrics

### Red Herring Data
The dataset includes valid-but-suspicious data:
- A customer named **"Null Fisher"** — a real person, not missing data
- A product priced at **$0.00** — a legitimate free promotional item
- A customer aged **0** — a valid baby record

These test whether agents over-clean by removing data that looks dirty but is actually correct.

### Business Rule Validation
Domain-specific constraints that agents must satisfy:
- `age ∈ [0, 120]` (customer records)
- `price ∈ [$0, $10,000]` (sales)
- `systolic_bp > diastolic_bp` (healthcare)
- Valid categories only (gender must be "Male" or "Female")

---

## 🔧 20+ Commands

### Diagnostic (Read-Only)
```
profile                    → Dataset overview
profile_column COL         → Column statistics
find_missing               → Missing value report
find_duplicates [COLS]     → Duplicate detection
find_outliers COL          → Outlier detection (IQR)
check_rules                → Business rule violations
history                    → Operation history / data lineage
view [N]                   → Preview rows
```

### Cleaning (Modifies Data)
```
fill_missing COL STRATEGY  → Fill nulls (mean/median/mode/constant/forward_fill)
remove_duplicates [COLS]   → Drop duplicates
fix_dtype COL TYPE         → Cast types (int/float/str/datetime)
replace COL OLD NEW        → Replace values
regex_replace COL PAT NEW  → Regex-based replacement
standardize COL METHOD     → Format normalization
remove_rows COL COND VAL   → Conditional row removal
clip COL LOW HIGH          → Clip numeric values
rename_column OLD NEW      → Rename columns
drop_column COL            → Remove columns
sort COL [asc|desc]        → Sort data
undo                       → Undo last modification
```

### Evaluation
```
validate                   → Check quality score (8 dimensions)
submit                     → Finalize and grade
```

---

## 📋 Data Lineage & Provenance

Every operation is tracked. On submit, the environment generates a **Cleaning Provenance Report**:

```
=== Cleaning Provenance Report ===
Total steps: 15
  Diagnostic (read-only): 6
  Data-modifying: 9
  Undo operations: 1

Data Transformations Applied:
  Step  3: fix_dtype price float (rows: 218 → 218)
  Step  5: fill_missing price median
  Step  7: remove_duplicates (rows: 218 → 202)
  Step  9: clip price 0 10000
  Step 11: standardize region titlecase
```

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install openenv-core openai requests pandas numpy
```

### Run the Environment
```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Run Inference
```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="your-key"
python inference.py
```

---

## 🧠 Baseline Agent Strategy

The baseline inference agent uses a **3-phase architecture**:

1. **DIAGNOSE** (first 15% of steps): `profile` → `find_missing` → `find_duplicates` → `check_rules` → `find_outliers`
2. **CLEAN** (middle 70%): Fix types → Fill missing → Remove duplicates → Fix outliers → Standardize categories → Fix business rules
3. **VERIFY** (last 15%): `validate` → Fix remaining issues → `submit`

This teaches agents a real-world data cleaning workflow rather than random exploration.

---

## 📁 Project Structure

```
data-wrangler-env/
├── inference.py                    # Baseline LLM agent (3-phase strategy)
├── models.py                       # Pydantic Action/Observation models
├── openenv.yaml                    # OpenEnv specification
├── pyproject.toml                  # Python dependencies
├── requirements.txt                # Pip requirements
├── Dockerfile                      # HF Spaces deployment
├── server/
│   ├── app.py                      # FastAPI application
│   ├── data_wrangler_env_environment.py  # Environment core (undo, lineage, rules)
│   ├── cleaning_engine.py          # 20+ command implementations
│   ├── dataset_generator.py        # 3 task generators (with red herrings)
│   ├── grader.py                   # 8-dimension scoring system
│   └── Dockerfile                  # Server Docker build
└── test_local.py                   # Local unit tests
```

---

## 📜 License

BSD-style license. See [LICENSE](LICENSE) for details.
