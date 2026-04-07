"""
Grader for DataWranglerEnv.

Multi-dimensional scoring system that compares the agent's cleaned dataset
against the ground truth. Produces deterministic scores in [0.0, 1.0].

Dimensions:
    - Missing values fixed  (25%)
    - Duplicates removed     (20%)
    - Type correctness       (20%)
    - Value accuracy          (25%)
    - Data preservation       (10%)
"""

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


def compute_score(
    dirty_df: pd.DataFrame,
    current_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    original_dirty_df: pd.DataFrame,
    issue_manifest: Dict[str, Any],
) -> Tuple[float, Dict[str, float]]:
    """Compute the composite data quality score.

    Args:
        dirty_df: The original dirty dataset (snapshot at reset)
        current_df: The agent's current working dataset
        clean_df: The ground truth clean dataset
        original_dirty_df: The very first dirty state (for reference)
        issue_manifest: Manifest of all injected issues

    Returns:
        Tuple of (composite_score, dimension_scores_dict)
    """
    scores = {}

    # ── Dimension 1: Missing Values Fixed (25%) ──────────────────────────
    original_missing = original_dirty_df.isnull().sum().sum()
    current_missing = current_df.isnull().sum().sum()
    target_missing = clean_df.isnull().sum().sum()

    if original_missing > target_missing:
        fixable = original_missing - target_missing
        fixed = max(0, original_missing - current_missing)
        scores["missing_fixed"] = min(1.0, fixed / fixable) if fixable > 0 else 1.0
    else:
        scores["missing_fixed"] = 1.0

    # ── Dimension 2: Duplicates Removed (20%) ─────────────────────────────
    original_dupes = original_dirty_df.duplicated().sum()
    current_dupes = current_df.duplicated().sum()
    target_dupes = clean_df.duplicated().sum()

    if original_dupes > target_dupes:
        fixable_dupes = original_dupes - target_dupes
        removed = max(0, original_dupes - current_dupes)
        scores["duplicates_removed"] = min(1.0, removed / fixable_dupes) if fixable_dupes > 0 else 1.0
    else:
        scores["duplicates_removed"] = 1.0

    # ── Dimension 3: Type Correctness (20%) ───────────────────────────────
    type_matches = 0
    type_total = len(clean_df.columns)

    for col in clean_df.columns:
        if col not in current_df.columns:
            continue
        clean_dtype = clean_df[col].dtype
        current_dtype = current_df[col].dtype

        # Allow some flexibility in type matching
        if clean_dtype == current_dtype:
            type_matches += 1
        elif pd.api.types.is_numeric_dtype(clean_dtype) and pd.api.types.is_numeric_dtype(current_dtype):
            type_matches += 1
        elif pd.api.types.is_string_dtype(clean_dtype) and pd.api.types.is_string_dtype(current_dtype):
            type_matches += 1
        elif str(clean_dtype) == "object" and str(current_dtype) == "object":
            type_matches += 1

    scores["type_correctness"] = type_matches / type_total if type_total > 0 else 1.0

    # ── Dimension 4: Value Accuracy (25%) ─────────────────────────────────
    # Compare cell-by-cell against the clean dataset
    # We align rows by index where possible
    try:
        # Get the minimum row count for comparison
        min_rows = min(len(current_df), len(clean_df))
        min_cols = min(len(current_df.columns), len(clean_df.columns))
        
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

                    # Handle NaN comparison
                    if pd.isna(clean_val) and pd.isna(current_val):
                        matching_cells += 1
                    elif pd.isna(clean_val) or pd.isna(current_val):
                        continue
                    elif str(clean_val).strip().lower() == str(current_val).strip().lower():
                        matching_cells += 1
                    else:
                        # Partial match for numeric values (within 1% tolerance)
                        try:
                            cv = float(clean_val)
                            av = float(current_val)
                            if cv != 0 and abs(cv - av) / abs(cv) < 0.01:
                                matching_cells += 0.8  # Partial credit
                        except (ValueError, TypeError):
                            pass

            scores["value_accuracy"] = matching_cells / total_cells if total_cells > 0 else 0.0
        else:
            scores["value_accuracy"] = 0.0
    except Exception:
        scores["value_accuracy"] = 0.0

    # ── Dimension 5: Data Preservation (10%) ──────────────────────────────
    # Penalize if valid rows from clean_df were incorrectly removed
    expected_rows = len(clean_df)
    current_rows = len(current_df)

    if current_rows >= expected_rows:
        # No valid data lost (might have extra rows from un-removed dupes)
        scores["data_preservation"] = 1.0
    elif current_rows == 0:
        scores["data_preservation"] = 0.0
    else:
        # Proportional penalty for missing rows
        scores["data_preservation"] = max(0.0, current_rows / expected_rows)

    # ── Composite Score ──────────────────────────────────────────────────
    weights = {
        "missing_fixed": 0.25,
        "duplicates_removed": 0.20,
        "type_correctness": 0.20,
        "value_accuracy": 0.25,
        "data_preservation": 0.10,
    }

    composite = sum(scores[dim] * weights[dim] for dim in weights)
    # Clamp to open interval (0, 1) — the OpenEnv validator rejects
    # scores of exactly 0.0 or 1.0.
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
    """Compute the reward for a single step.

    Args:
        before_df: DataFrame state before this step
        after_df: DataFrame state after this step
        clean_df: Ground truth
        original_dirty_df: Original dirty state
        issue_manifest: Issue manifest
        command: The command that was executed
        data_modified: Whether the command modified the data

    Returns:
        Step reward (can be negative)
    """
    cmd = command.strip().split()[0].lower() if command.strip() else ""

    # Diagnostic commands get small positive reward (encourage exploration)
    diagnostic_cmds = {"help", "view", "profile", "profile_column", "find_missing",
                       "find_duplicates", "find_outliers"}
    if cmd in diagnostic_cmds:
        return 0.02

    # Validate gets small reward
    if cmd == "validate":
        return 0.01

    # Submit doesn't give step reward (final score is computed separately)
    if cmd == "submit":
        return 0.001

    if not data_modified:
        # Command tried to modify but failed (or no change)
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
        # Positive improvement — scale reward
        return min(0.15, improvement * 2.0 + 0.03)
    elif improvement < -0.01:
        # Made things worse — negative reward
        return max(-0.20, improvement * 2.0)
    else:
        # Negligible change
        return 0.001
