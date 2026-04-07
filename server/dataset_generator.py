"""
Dataset Generator for DataWranglerEnv.

Generates synthetic "dirty" datasets with known ground truth for each task.
Uses seeded random generation for full reproducibility.

Tasks:
    - task_1_easy:   Customer Records   (~50 rows,  5 cols)
    - task_2_medium: Sales Transactions  (~200 rows, 8 cols)
    - task_3_hard:   Healthcare Records  (~1000 rows, 12 cols)
"""

import random
import string
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ── Shared data pools ────────────────────────────────────────────────────────

FIRST_NAMES = [
    "Alice", "Bob", "Charlie", "Diana", "Edward", "Fiona", "George", "Hannah",
    "Ivan", "Julia", "Kevin", "Laura", "Michael", "Nina", "Oscar", "Patricia",
    "Quinn", "Rachel", "Samuel", "Tanya", "Ulysses", "Vera", "William", "Xena",
    "Yusuf", "Zara", "Aiden", "Bella", "Caleb", "Dahlia", "Ethan", "Freya",
    "Gavin", "Hazel", "Isaac", "Jade", "Kyle", "Luna", "Mason", "Nora",
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
    "Ramirez", "Lewis", "Robinson",
]

CITIES = [
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
    "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose",
    "Austin", "Jacksonville", "Fort Worth", "Columbus", "Charlotte",
    "Indianapolis", "San Francisco", "Seattle", "Denver", "Washington",
]

CITY_TYPOS = {
    "New York": ["new york", "New Yrok", "Nw York", "NewYork"],
    "Los Angeles": ["los angeles", "Los Angele", "Lo Angeles", "LosAngeles"],
    "Chicago": ["chicgo", "Chcago", "chicago"],
    "Houston": ["houstan", "Hosuton", "houston"],
    "San Francisco": ["san francisco", "San Fransisco", "SanFrancisco"],
    "Seattle": ["seatle", "Seattel", "seattle"],
}

PRODUCTS = [
    "Laptop Pro", "Wireless Mouse", "USB-C Hub", "Monitor 27\"",
    "Mechanical Keyboard", "Webcam HD", "Headset Pro", "SSD 1TB",
    "RAM 16GB", "Graphics Card", "Power Supply", "CPU Cooler",
    "Ethernet Cable", "HDMI Adapter", "Laptop Stand", "Mouse Pad XL",
    "Speaker Set", "External HDD", "Tablet 10\"", "Smartphone Case",
]

CATEGORIES = [
    "Electronics", "Accessories", "Storage", "Components",
    "Peripherals", "Audio", "Networking", "Display",
]

REGIONS = ["North", "South", "East", "West", "Central"]

DIAGNOSES = [
    "Hypertension", "Type 2 Diabetes", "Asthma", "Migraine",
    "Arthritis", "Anxiety Disorder", "Depression", "GERD",
    "Hypothyroidism", "Hyperlipidemia", "Bronchitis", "Allergic Rhinitis",
    "Osteoporosis", "Anemia", "UTI", "Back Pain",
]

MEDICATIONS = [
    "Lisinopril", "Metformin", "Albuterol", "Sumatriptan",
    "Ibuprofen", "Sertraline", "Fluoxetine", "Omeprazole",
    "Levothyroxine", "Atorvastatin", "Amoxicillin", "Cetirizine",
    "Alendronate", "Ferrous Sulfate", "Ciprofloxacin", "Naproxen",
]

BLOOD_TYPES = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]

COMMANDS_HELP = """Available commands:
  help                          - Show this help message
  view [N]                      - Show first N rows (default 10)
  profile                       - Dataset summary: shape, dtypes, missing %, unique counts
  profile_column COL            - Detailed stats for a single column
  find_missing                  - Show missing value counts per column
  find_duplicates [COL1,COL2]   - Find duplicate rows (optional subset of columns)
  find_outliers COL             - Statistical outlier detection for a numeric column
  check_rules                   - Check business rule violations
  history                       - Show operation history (data lineage)
  fill_missing COL STRATEGY [VALUE] - Fill nulls (mean/median/mode/constant VALUE)
  remove_duplicates [COL1,COL2] [KEEP] - Drop duplicates (keep: first/last/none)
  fix_dtype COL TYPE            - Cast column to type (int/float/str/datetime)
  replace COL OLD NEW           - Replace specific values in a column
  regex_replace COL PATTERN REPLACEMENT - Regex-based replacement
  standardize COL METHOD        - Normalize formatting (lowercase/uppercase/titlecase/strip)
  remove_rows COL CONDITION VALUE - Remove rows (CONDITION: equals/not_equals/less_than/greater_than/contains)
  clip COL LOWER UPPER          - Clip numeric values to [LOWER, UPPER]
  rename_column OLD_NAME NEW_NAME - Rename a column
  drop_column COL               - Remove a column
  sort COL [asc|desc]           - Sort data by column
  undo                          - Undo the last data-modifying operation
  validate                      - Check current quality score without submitting
  submit                        - Finalize and grade the cleaned dataset (ends episode)
"""


# ── Task 1: Easy — Customer Records ──────────────────────────────────────────

def generate_task_1_easy(rng: random.Random) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Generate a simple customer records dataset with basic issues."""
    n_rows = 50
    issues = {"missing": [], "duplicates": [], "typos": []}

    # Build clean data
    rows = []
    for i in range(n_rows):
        fname = rng.choice(FIRST_NAMES)
        lname = rng.choice(LAST_NAMES)
        name = f"{fname} {lname}"
        email = f"{fname.lower()}.{lname.lower()}{rng.randint(1,99)}@example.com"
        age = rng.randint(18, 85)
        city = rng.choice(CITIES)
        base_date = datetime(2022, 1, 1)
        signup_date = (base_date + timedelta(days=rng.randint(0, 730))).strftime("%Y-%m-%d")
        rows.append({
            "name": name, "email": email, "age": age,
            "city": city, "signup_date": signup_date,
        })

    # ── Red herring rows: valid-but-suspicious data ──
    # These SHOULD NOT be cleaned — tests agents don't over-clean
    rows.append({"name": "Null Fisher", "email": "null.fisher42@example.com",
                 "age": 45, "city": "New York", "signup_date": "2023-06-15"})
    rows.append({"name": "None Yamada", "email": "none.yamada7@example.com",
                 "age": 28, "city": "Los Angeles", "signup_date": "2022-12-01"})
    rows.append({"name": "Na Lee", "email": "na.lee99@example.com",
                 "age": 0, "city": "Chicago", "signup_date": "2024-01-01"})  # Baby

    clean_df = pd.DataFrame(rows)

    # Create dirty copy
    dirty_df = clean_df.copy()

    # Inject missing values (~10% in age and email)
    for col in ["age", "email"]:
        n_missing = max(2, int(n_rows * 0.10))
        indices = rng.sample(range(n_rows), n_missing)
        for idx in indices:
            issues["missing"].append({"row": idx, "column": col, "original": dirty_df.at[idx, col]})
            dirty_df.at[idx, col] = np.nan if col == "age" else None

    # Inject duplicate rows (~5 rows)
    n_dupes = 5
    dupe_indices = rng.sample(range(n_rows), n_dupes)
    dupe_rows = dirty_df.iloc[dupe_indices].copy()
    dirty_df = pd.concat([dirty_df, dupe_rows], ignore_index=True)
    issues["duplicates"] = [{"source_row": idx} for idx in dupe_indices]

    # Inject city name typos (3 random rows)
    typo_indices = rng.sample(range(n_rows), min(3, n_rows))
    for idx in typo_indices:
        original_city = dirty_df.at[idx, "city"]
        if original_city in CITY_TYPOS:
            typo = rng.choice(CITY_TYPOS[original_city])
            issues["typos"].append({"row": idx, "column": "city", "original": original_city, "typo": typo})
            dirty_df.at[idx, "city"] = typo

    # Convert age to float (because of NaN)
    dirty_df["age"] = pd.to_numeric(dirty_df["age"], errors="coerce")

    # ── Golden rows: select rows that must not be damaged (anti-exploit) ──
    golden_indices = rng.sample(range(len(clean_df)), min(5, len(clean_df)))
    issues["golden_indices"] = golden_indices

    # ── Business rules ──
    issues["business_rules"] = [
        {"type": "range", "column": "age", "min": 0, "max": 120,
         "description": "Age must be between 0 and 120"},
        {"type": "not_null", "column": "email",
         "description": "Email address is required"},
        {"type": "pattern", "column": "email", "pattern": r".*@.*\..*",
         "description": "Email must contain @ and domain"},
    ]

    return dirty_df, clean_df, issues


# ── Task 2: Medium — Sales Transactions ──────────────────────────────────────

def generate_task_2_medium(rng: random.Random) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Generate a sales transactions dataset with multiple issue types."""
    n_rows = 200
    issues = {"missing": [], "duplicates": [], "type_errors": [], "date_inconsistencies": [],
              "negative_values": [], "outliers": []}

    rows = []
    for i in range(n_rows):
        txn_id = f"TXN-{10000 + i}"
        product = rng.choice(PRODUCTS)
        category = rng.choice(CATEGORIES)
        price = round(rng.uniform(9.99, 999.99), 2)
        quantity = rng.randint(1, 20)
        base_date = datetime(2023, 1, 1)
        date = base_date + timedelta(days=rng.randint(0, 365))
        date_str = date.strftime("%Y-%m-%d")
        customer_id = f"CUST-{rng.randint(1000, 1999)}"
        region = rng.choice(REGIONS)
        rows.append({
            "transaction_id": txn_id, "product": product, "category": category,
            "price": price, "quantity": quantity, "date": date_str,
            "customer_id": customer_id, "region": region,
        })

    # ── Red herring rows: legitimate edge cases that should NOT be cleaned ──
    rows.append({"transaction_id": "TXN-FREE01", "product": "Promotional Sticker",
                 "category": "Accessories", "price": 0.00, "quantity": 1,
                 "date": "2023-07-04", "customer_id": "CUST-1500", "region": "North"})
    rows.append({"transaction_id": "TXN-BULK01", "product": "Ethernet Cable",
                 "category": "Networking", "price": 2.99, "quantity": 500,
                 "date": "2023-11-24", "customer_id": "CUST-1001", "region": "Central"})

    clean_df = pd.DataFrame(rows)
    dirty_df = clean_df.copy()

    # Missing values (~15% across price, category, region)
    for col in ["price", "category", "region"]:
        n_missing = max(3, int(n_rows * 0.05))
        indices = rng.sample(range(n_rows), n_missing)
        for idx in indices:
            issues["missing"].append({"row": idx, "column": col, "original": dirty_df.at[idx, col]})
            dirty_df.at[idx, col] = np.nan if col == "price" else None

    # Duplicate rows (~8%)
    n_dupes = int(n_rows * 0.08)
    dupe_indices = rng.sample(range(n_rows), n_dupes)
    dupe_rows = dirty_df.iloc[dupe_indices].copy()
    dirty_df = pd.concat([dirty_df, dupe_rows], ignore_index=True)
    issues["duplicates"] = [{"source_row": idx} for idx in dupe_indices]

    # Convert price to object dtype BEFORE inserting string values (pandas 3.x compat)
    dirty_df["price"] = dirty_df["price"].astype(object)

    # Type errors: prices as strings with "$" prefix (~10 rows)
    type_error_indices = rng.sample(range(n_rows), 10)
    for idx in type_error_indices:
        if pd.notna(dirty_df.at[idx, "price"]):
            original = dirty_df.at[idx, "price"]
            dirty_df.at[idx, "price"] = f"${original}"
            issues["type_errors"].append({"row": idx, "column": "price", "value": f"${original}"})

    # Date format inconsistencies (~15 rows use DD/MM/YYYY)
    date_err_indices = rng.sample(range(n_rows), 15)
    for idx in date_err_indices:
        if pd.notna(dirty_df.at[idx, "date"]):
            try:
                d = datetime.strptime(str(dirty_df.at[idx, "date"]), "%Y-%m-%d")
                new_fmt = d.strftime("%d/%m/%Y")
                dirty_df.at[idx, "date"] = new_fmt
                issues["date_inconsistencies"].append({"row": idx, "original_format": "YYYY-MM-DD", "new_format": "DD/MM/YYYY"})
            except (ValueError, TypeError):
                pass

    # Negative quantities (~5 rows)
    dirty_df["quantity"] = dirty_df["quantity"].astype(object)
    neg_indices = rng.sample(range(n_rows), 5)
    for idx in neg_indices:
        dirty_df.at[idx, "quantity"] = -abs(int(dirty_df.at[idx, "quantity"]))
        issues["negative_values"].append({"row": idx, "column": "quantity"})

    # Outlier prices (~3 rows with extreme values)
    outlier_indices = rng.sample(range(n_rows), 3)
    for idx in outlier_indices:
        if pd.notna(dirty_df.at[idx, "price"]) and not isinstance(dirty_df.at[idx, "price"], str):
            dirty_df.at[idx, "price"] = round(rng.uniform(50000, 99999), 2)
            issues["outliers"].append({"row": idx, "column": "price"})

    # ── Golden rows ──
    golden_indices = rng.sample(range(len(clean_df)), min(8, len(clean_df)))
    issues["golden_indices"] = golden_indices

    # ── Business rules ──
    issues["business_rules"] = [
        {"type": "range", "column": "price", "min": 0, "max": 10000,
         "description": "Price must be between $0 and $10,000"},
        {"type": "range", "column": "quantity", "min": 1, "max": 1000,
         "description": "Quantity must be between 1 and 1000"},
        {"type": "not_null", "column": "transaction_id",
         "description": "Transaction ID is required"},
        {"type": "categorical", "column": "region",
         "allowed_values": ["North", "South", "East", "West", "Central"],
         "description": "Region must be a valid US region"},
    ]

    return dirty_df, clean_df, issues


# ── Task 3: Hard — Healthcare Patient Records ────────────────────────────────

def generate_task_3_hard(rng: random.Random) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Generate a complex healthcare dataset with many intertwined issues."""
    n_rows = 1000
    issues = {
        "missing": [], "duplicates": [], "fuzzy_duplicates": [],
        "type_errors": [], "impossible_values": [], "logic_errors": [],
        "category_inconsistencies": [], "format_inconsistencies": [],
    }

    rows = []
    for i in range(n_rows):
        fname = rng.choice(FIRST_NAMES)
        lname = rng.choice(LAST_NAMES)
        name = f"{fname} {lname}"
        patient_id = f"PAT-{10000 + i}"
        dob_year = rng.randint(1940, 2005)
        dob = datetime(dob_year, rng.randint(1, 12), rng.randint(1, 28))
        dob_str = dob.strftime("%Y-%m-%d")
        gender = rng.choice(["Male", "Female"])
        blood_type = rng.choice(BLOOD_TYPES)
        height = round(rng.gauss(170, 12), 1)
        weight = round(rng.gauss(75, 15), 1)
        bp_systolic = rng.randint(100, 160)
        bp_diastolic = rng.randint(60, min(90, bp_systolic - 10))
        diagnosis = rng.choice(DIAGNOSES)
        medication = rng.choice(MEDICATIONS)
        visit_date = (datetime(2024, 1, 1) + timedelta(days=rng.randint(0, 365))).strftime("%Y-%m-%d")

        rows.append({
            "patient_id": patient_id, "name": name, "dob": dob_str,
            "gender": gender, "blood_type": blood_type,
            "height_cm": height, "weight_kg": weight,
            "bp_systolic": bp_systolic, "bp_diastolic": bp_diastolic,
            "diagnosis": diagnosis, "medication": medication,
            "visit_date": visit_date,
        })

    clean_df = pd.DataFrame(rows)
    dirty_df = clean_df.copy()

    # ── Missing values (~12% spread across multiple columns) ──
    missing_cols = ["name", "blood_type", "height_cm", "weight_kg", "diagnosis", "medication"]
    for col in missing_cols:
        n_miss = max(5, int(n_rows * 0.02))
        indices = rng.sample(range(n_rows), n_miss)
        for idx in indices:
            issues["missing"].append({"row": idx, "column": col})
            if col in ["height_cm", "weight_kg"]:
                dirty_df.at[idx, col] = np.nan
            else:
                dirty_df.at[idx, col] = None

    # ── Exact duplicate rows (~5%) ──
    n_dupes = int(n_rows * 0.05)
    dupe_indices = rng.sample(range(n_rows), n_dupes)
    dupe_rows = dirty_df.iloc[dupe_indices].copy()
    dirty_df = pd.concat([dirty_df, dupe_rows], ignore_index=True)
    issues["duplicates"] = [{"source_row": idx} for idx in dupe_indices]

    # ── Fuzzy duplicates (same patient, different name spelling) (~20 rows) ──
    fuzzy_indices = rng.sample(range(n_rows), 20)
    fuzzy_rows = []
    for idx in fuzzy_indices:
        row = dirty_df.iloc[idx].copy()
        original_name = str(row["name"])
        parts = original_name.split()
        if len(parts) >= 2:
            # Introduce spelling variations
            variant = rng.choice([
                f"{parts[0][0]}. {parts[1]}",          # "A. Smith"
                f"{parts[0]}  {parts[1]}",              # Extra space
                f"{parts[0].lower()} {parts[1].lower()}", # All lowercase
                f"{parts[0]} {parts[1][:-1]}",           # Truncated last name
            ])
            row["name"] = variant
            row["patient_id"] = f"PAT-{rng.randint(90000, 99999)}"  # Different ID
            fuzzy_rows.append(row)
            issues["fuzzy_duplicates"].append({
                "original_row": idx, "original_name": original_name, "variant": variant
            })
    if fuzzy_rows:
        dirty_df = pd.concat([dirty_df, pd.DataFrame(fuzzy_rows)], ignore_index=True)

    # ── Gender category inconsistencies ──
    gender_variants = {"Male": ["M", "male", "MALE", "m"], "Female": ["F", "female", "FEMALE", "f"]}
    gender_indices = rng.sample(range(min(n_rows, len(dirty_df))), min(40, len(dirty_df)))
    for idx in gender_indices:
        original = dirty_df.at[idx, "gender"]
        if original in gender_variants:
            variant = rng.choice(gender_variants[original])
            dirty_df.at[idx, "gender"] = variant
            issues["category_inconsistencies"].append({
                "row": idx, "column": "gender", "original": original, "variant": variant
            })

    # ── Blood type inconsistencies ──
    bt_indices = rng.sample(range(min(n_rows, len(dirty_df))), min(15, len(dirty_df)))
    for idx in bt_indices:
        original = dirty_df.at[idx, "blood_type"]
        if original and pd.notna(original):
            variant = rng.choice([str(original).lower(), str(original).replace("+", " positive").replace("-", " negative")])
            dirty_df.at[idx, "blood_type"] = variant
            issues["category_inconsistencies"].append({
                "row": idx, "column": "blood_type", "original": original, "variant": variant
            })

    # ── Impossible values ──
    # Negative height (~5 rows)
    imp_indices = rng.sample(range(min(n_rows, len(dirty_df))), 5)
    for idx in imp_indices:
        if pd.notna(dirty_df.at[idx, "height_cm"]):
            dirty_df.at[idx, "height_cm"] = -abs(float(dirty_df.at[idx, "height_cm"]))
            issues["impossible_values"].append({"row": idx, "column": "height_cm", "reason": "negative"})

    # Extreme height > 250 (~3 rows)
    ext_indices = rng.sample(range(min(n_rows, len(dirty_df))), 3)
    for idx in ext_indices:
        if pd.notna(dirty_df.at[idx, "height_cm"]):
            dirty_df.at[idx, "height_cm"] = round(rng.uniform(300, 500), 1)
            issues["impossible_values"].append({"row": idx, "column": "height_cm", "reason": "too_high"})

    # Negative weight (~3 rows)
    neg_w = rng.sample(range(min(n_rows, len(dirty_df))), 3)
    for idx in neg_w:
        if pd.notna(dirty_df.at[idx, "weight_kg"]):
            dirty_df.at[idx, "weight_kg"] = round(rng.uniform(-100, -10), 1)
            issues["impossible_values"].append({"row": idx, "column": "weight_kg", "reason": "negative"})

    # ── Cross-column logic errors: diastolic > systolic (~10 rows) ──
    logic_indices = rng.sample(range(min(n_rows, len(dirty_df))), 10)
    for idx in logic_indices:
        systolic = dirty_df.at[idx, "bp_systolic"]
        if pd.notna(systolic):
            dirty_df.at[idx, "bp_diastolic"] = int(systolic) + rng.randint(10, 40)
            issues["logic_errors"].append({
                "row": idx, "type": "bp_diastolic_gt_systolic",
                "systolic": int(systolic),
                "diastolic": int(dirty_df.at[idx, "bp_diastolic"]),
            })

    # ── Date format inconsistencies (~30 rows) ──
    date_indices = rng.sample(range(min(n_rows, len(dirty_df))), 30)
    for idx in date_indices:
        if pd.notna(dirty_df.at[idx, "visit_date"]):
            try:
                d = datetime.strptime(str(dirty_df.at[idx, "visit_date"]), "%Y-%m-%d")
                fmt = rng.choice(["%d/%m/%Y", "%m-%d-%Y", "%d %b %Y"])
                dirty_df.at[idx, "visit_date"] = d.strftime(fmt)
                issues["format_inconsistencies"].append({"row": idx, "column": "visit_date"})
            except (ValueError, TypeError):
                pass

    # ── DOB format inconsistencies (~20 rows) ──
    dob_indices = rng.sample(range(min(n_rows, len(dirty_df))), 20)
    for idx in dob_indices:
        if pd.notna(dirty_df.at[idx, "dob"]):
            try:
                d = datetime.strptime(str(dirty_df.at[idx, "dob"]), "%Y-%m-%d")
                fmt = rng.choice(["%d/%m/%Y", "%m-%d-%Y"])
                dirty_df.at[idx, "dob"] = d.strftime(fmt)
                issues["format_inconsistencies"].append({"row": idx, "column": "dob"})
            except (ValueError, TypeError):
                pass

    # ── Golden rows: important rows for anti-exploit ──
    golden_indices = rng.sample(range(len(clean_df)), min(15, len(clean_df)))
    issues["golden_indices"] = golden_indices

    # ── Business rules (healthcare domain constraints) ──
    issues["business_rules"] = [
        {"type": "range", "column": "height_cm", "min": 30, "max": 250,
         "description": "Height must be between 30cm and 250cm"},
        {"type": "range", "column": "weight_kg", "min": 1, "max": 300,
         "description": "Weight must be between 1kg and 300kg"},
        {"type": "cross_column", "column_a": "bp_systolic", "column_b": "bp_diastolic",
         "relation": "greater_than",
         "description": "Systolic BP must be greater than Diastolic BP"},
        {"type": "categorical", "column": "gender",
         "allowed_values": ["Male", "Female"],
         "description": "Gender must be 'Male' or 'Female'"},
        {"type": "categorical", "column": "blood_type",
         "allowed_values": ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"],
         "description": "Blood type must be a valid ABO+Rh type"},
        {"type": "not_null", "column": "patient_id",
         "description": "Patient ID is required"},
    ]

    return dirty_df, clean_df, issues


# ── Public API ────────────────────────────────────────────────────────────────

TASK_CONFIGS = {
    "task_1_easy": {
        "generator": generate_task_1_easy,
        "max_steps": 30,
        "description": "Customer Records — fix missing values, remove duplicates, correct city name typos",
    },
    "task_2_medium": {
        "generator": generate_task_2_medium,
        "max_steps": 50,
        "description": "Sales Transactions — fix types, dates, missing values, duplicates, and outliers",
    },
    "task_3_hard": {
        "generator": generate_task_3_hard,
        "max_steps": 80,
        "description": "Healthcare Records — fix everything: fuzzy dupes, logic errors, impossible values, format inconsistencies",
    },
}


def generate_dataset(
    task_name: str, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], int, str]:
    """Generate a dirty/clean dataset pair for the given task.

    Args:
        task_name: One of 'task_1_easy', 'task_2_medium', 'task_3_hard'
        seed: Random seed for reproducibility

    Returns:
        Tuple of (dirty_df, clean_df, issue_manifest, max_steps, description)
    """
    if task_name not in TASK_CONFIGS:
        raise ValueError(f"Unknown task: {task_name}. Choose from: {list(TASK_CONFIGS.keys())}")

    config = TASK_CONFIGS[task_name]
    rng = random.Random(seed)
    # Also seed numpy for any numpy operations
    np.random.seed(seed)

    dirty_df, clean_df, issues = config["generator"](rng)

    return dirty_df, clean_df, issues, config["max_steps"], config["description"]
