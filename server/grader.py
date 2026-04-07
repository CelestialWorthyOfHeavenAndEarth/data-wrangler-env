"""
Grader for DataWranglerEnv.

Multi-dimensional scoring system that compares the agent's cleaned dataset
against the ground truth. Produces deterministic scores in (0.001, 0.999).

Dimensions:
    - Missing values fixed    (20%)
    - Duplicates removed      (15%)
    - Type correctness        (15%)
    - Value accuracy           (20%)
    - Data preservation        (10%)
    - Constraint compliance    (10%)  — NEW: business rule satisfaction
    - Step efficiency           (5%)  — NEW: fewer steps = better
    - Golden row integrity      (5%)  — NEW: anti-exploit mechanism
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def compute_score(
    dirty_df: pd.DataFrame,
    current_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    original_dirty_df: pd.DataFrame,
    issue_manifest: Dict[str, Any],
    step_count: int = 0,
    max_steps: int = 30,
    golden_indices: List[int] = None,
) -> Tuple[float, Dict[str, float]]:
    """Compute the composite data quality score.

    Args:
        dirty_df: The original dirty dataset (snapshot at reset)
        current_df: The agent's current working dataset
        clean_df: The ground truth clean dataset
        original_dirty_df: The very first dirty state (for reference)
        issue_manifest: Manifest of all injected issues
        step_count: Current step number
        max_steps: Maximum allowed steps
        golden_indices: Indices of golden rows in clean_df

    Returns:
        Tuple of (composite_score, dimension_scores_dict)
    """
    scores = {}
    if golden_indices is None:
        golden_indices = issue_manifest.get("golden_indices", [])

    # ── Dimension 1: Missing Values Fixed (20%) ──────────────────────────
    original_missing = original_dirty_df.isnull().sum().sum()
    current_missing = current_df.isnull().sum().sum()
    target_missing = clean_df.isnull().sum().sum()

    if original_missing > target_missing:
        fixable = original_missing - target_missing
        fixed = max(0, original_missing - current_missing)
        scores["missing_fixed"] = min(1.0, fixed / fixable) if fixable > 0 else 1.0
    else:
        scores["missing_fixed"] = 1.0

    # ── Dimension 2: Duplicates Removed (15%) ─────────────────────────────
    original_dupes = original_dirty_df.duplicated().sum()
    current_dupes = current_df.duplicated().sum()
    target_dupes = clean_df.duplicated().sum()

    if original_dupes > target_dupes:
        fixable_dupes = original_dupes - target_dupes
        removed = max(0, original_dupes - current_dupes)
        scores["duplicates_removed"] = min(1.0, removed / fixable_dupes) if fixable_dupes > 0 else 1.0
    else:
        scores["duplicates_removed"] = 1.0

    # ── Dimension 3: Type Correctness (15%) ───────────────────────────────
    type_matches = 0
    type_total = len(clean_df.columns)

    for col in clean_df.columns:
        if col not in current_df.columns:
            continue
        clean_dtype = clean_df[col].dtype
        current_dtype = current_df[col].dtype

        if clean_dtype == current_dtype:
            type_matches += 1
        elif pd.api.types.is_numeric_dtype(clean_dtype) and pd.api.types.is_numeric_dtype(current_dtype):
            type_matches += 1
        elif pd.api.types.is_string_dtype(clean_dtype) and pd.api.types.is_string_dtype(current_dtype):
            type_matches += 1
        elif str(clean_dtype) == "object" and str(current_dtype) == "object":
            type_matches += 1

    scores["type_correctness"] = type_matches / type_total if type_total > 0 else 1.0

    # ── Dimension 4: Value Accuracy (20%) ─────────────────────────────────
    try:
        min_rows = min(len(current_df), len(clean_df))
        common_cols = [c for c in clean_df.columns if c in current_df.columns]
        if common_cols and min_rows > 0:
            clean_subset = clean_df[common_cols].head(min_rows).reset_index(drop=True)
            current_subset = current_df[common_cols].head(min_rows).reset_index(drop=True)

            total_cells = 0
            matching_cells = 0

            for col in common_cols:
                for i in range(min_rows):
                    total_cells += 1
                    clean_val = clean_subset.at[i, col]
                    try:
                        current_val = current_subset.at[i, col]
                    except (KeyError, IndexError):
                        continue

                    if pd.isna(clean_val) and pd.isna(current_val):
                        matching_cells += 1
                    elif pd.isna(clean_val) or pd.isna(current_val):
                        continue
                    elif str(clean_val).strip().lower() == str(current_val).strip().lower():
                        matching_cells += 1
                    else:
                        try:
                            cv = float(clean_val)
                            av = float(current_val)
                            if cv != 0 and abs(cv - av) / abs(cv) < 0.01:
                                matching_cells += 0.8
                        except (ValueError, TypeError):
                            pass

            scores["value_accuracy"] = matching_cells / total_cells if total_cells > 0 else 0.0
        else:
            scores["value_accuracy"] = 0.0
    except Exception:
        scores["value_accuracy"] = 0.0

    # ── Dimension 5: Data Preservation (10%) ──────────────────────────────
    expected_rows = len(clean_df)
    current_rows = len(current_df)

    if current_rows >= expected_rows:
        scores["data_preservation"] = 1.0
    elif current_rows == 0:
        scores["data_preservation"] = 0.0
    else:
        scores["data_preservation"] = max(0.0, current_rows / expected_rows)

    # ── Dimension 6: Constraint Compliance (10%) — NEW ────────────────────
    business_rules = issue_manifest.get("business_rules", [])
    if business_rules:
        rules_satisfied = 0
        rules_total = len(business_rules)
        for rule in business_rules:
            rule_type = rule.get("type", "")
            col = rule.get("column", "")

            if rule_type == "range" and col in current_df.columns:
                lo, hi = rule.get("min", float("-inf")), rule.get("max", float("inf"))
                numeric = pd.to_numeric(current_df[col], errors="coerce")
                valid = numeric.dropna()
                if len(valid) > 0:
                    pct_ok = ((valid >= lo) & (valid <= hi)).mean()
                    rules_satisfied += pct_ok

            elif rule_type == "not_null" and col in current_df.columns:
                pct_ok = 1.0 - (current_df[col].isna().sum() / max(1, len(current_df)))
                rules_satisfied += pct_ok

            elif rule_type == "cross_column":
                col_a = rule.get("column_a", "")
                col_b = rule.get("column_b", "")
                if col_a in current_df.columns and col_b in current_df.columns:
                    a = pd.to_numeric(current_df[col_a], errors="coerce")
                    b = pd.to_numeric(current_df[col_b], errors="coerce")
                    valid_mask = a.notna() & b.notna()
                    if valid_mask.sum() > 0:
                        pct_ok = (a[valid_mask] > b[valid_mask]).mean()
                        rules_satisfied += pct_ok

            elif rule_type == "categorical" and col in current_df.columns:
                allowed = set(rule.get("allowed_values", []))
                if allowed:
                    non_null = current_df[col].dropna()
                    if len(non_null) > 0:
                        pct_ok = non_null.isin(allowed).mean()
                        rules_satisfied += pct_ok

            elif rule_type == "pattern" and col in current_df.columns:
                pat = rule.get("pattern", "")
                if pat:
                    non_null = current_df[col].dropna().astype(str)
                    if len(non_null) > 0:
                        pct_ok = non_null.str.match(pat).mean()
                        rules_satisfied += pct_ok

        scores["constraint_compliance"] = rules_satisfied / rules_total if rules_total > 0 else 0.8
    else:
        scores["constraint_compliance"] = 0.8  # Default for tasks without rules

    # ── Dimension 7: Step Efficiency (5%) — NEW ───────────────────────────
    if step_count > 0 and max_steps > 0:
        efficiency = 1.0 - (step_count / max_steps)
        scores["step_efficiency"] = max(0.1, min(1.0, efficiency * 1.5))
    else:
        scores["step_efficiency"] = 0.5

    # ── Dimension 8: Golden Row Integrity (5%) — NEW ──────────────────────
    if golden_indices and len(current_df) > 0:
        golden_ok = 0
        golden_total = len(golden_indices)
        for gi in golden_indices:
            if gi >= len(clean_df):
                continue
            golden_row = clean_df.iloc[gi]
            # Check if this golden row still exists in current_df (approximately)
            found = False
            for _, row in current_df.iterrows():
                match_count = 0
                total_check = 0
                for col in clean_df.columns:
                    if col not in current_df.columns:
                        continue
                    total_check += 1
                    gv = golden_row[col]
                    cv = row[col]
                    if pd.isna(gv) and pd.isna(cv):
                        match_count += 1
                    elif not pd.isna(gv) and not pd.isna(cv):
                        if str(gv).strip().lower() == str(cv).strip().lower():
                            match_count += 1
                if total_check > 0 and match_count / total_check > 0.8:
                    found = True
                    break
            if found:
                golden_ok += 1
        scores["golden_row_integrity"] = golden_ok / golden_total if golden_total > 0 else 1.0
    else:
        scores["golden_row_integrity"] = 1.0

    # ── Composite Score ──────────────────────────────────────────────────
    weights = {
        "missing_fixed": 0.20,
        "duplicates_removed": 0.15,
        "type_correctness": 0.15,
        "value_accuracy": 0.20,
        "data_preservation": 0.10,
        "constraint_compliance": 0.10,
        "step_efficiency": 0.05,
        "golden_row_integrity": 0.05,
    }

    composite = sum(scores[dim] * weights[dim] for dim in weights)
    # Clamp to open interval (0, 1)
    composite = max(0.001, min(0.999, composite))

    return composite, scores


def compute_step_reward(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    original_dirty_df: pd.DataFrame,
    issue_manifest: Dict[str, Any],
    command: str,
    data_modified: bool,
) -> float:
    """Compute the reward for a single step."""
    cmd = command.strip().split()[0].lower() if command.strip() else ""

    # Diagnostic commands get small positive reward
    diagnostic_cmds = {"help", "view", "profile", "profile_column", "find_missing",
                       "find_duplicates", "find_outliers", "check_rules", "history"}
    if cmd in diagnostic_cmds:
        return 0.02

    # Validate and undo get small reward
    if cmd in ("validate", "undo"):
        return 0.01

    # Submit doesn't give step reward (final score is computed separately)
    if cmd == "submit":
        return 0.001

    if not data_modified:
        return -0.01

    # For data-modifying commands, compare improvement
    before_score, _ = compute_score(
        original_dirty_df, before_df, clean_df, original_dirty_df, issue_manifest
    )
    after_score, _ = compute_score(
        original_dirty_df, after_df, clean_df, original_dirty_df, issue_manifest
    )

    improvement = after_score - before_score

    if improvement > 0:
        return min(0.15, improvement * 2.0 + 0.03)
    elif improvement < -0.01:
        return max(-0.20, improvement * 2.0)
    else:
        return 0.001
