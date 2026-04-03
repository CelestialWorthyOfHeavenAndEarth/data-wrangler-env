"""
Cleaning Engine for DataWranglerEnv.

Parses text commands from the agent and executes data cleaning operations
on the working DataFrame. Returns text results for the observation.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .dataset_generator import COMMANDS_HELP


class CleaningEngine:
    """Parses agent text commands and applies cleaning operations to a DataFrame."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def execute(self, command_str: str) -> Tuple[str, bool]:
        """Parse and execute a text command.

        Args:
            command_str: Raw text command from the agent

        Returns:
            Tuple of (response_text, data_was_modified)
        """
        command_str = command_str.strip()
        if not command_str:
            return "Error: Empty command. Type 'help' for available commands.", False

        parts = self._parse_command(command_str)
        cmd = parts[0].lower()
        args = parts[1:]

        dispatch = {
            "help": self._cmd_help,
            "view": self._cmd_view,
            "profile": self._cmd_profile,
            "profile_column": self._cmd_profile_column,
            "find_missing": self._cmd_find_missing,
            "find_duplicates": self._cmd_find_duplicates,
            "find_outliers": self._cmd_find_outliers,
            "fill_missing": self._cmd_fill_missing,
            "remove_duplicates": self._cmd_remove_duplicates,
            "fix_dtype": self._cmd_fix_dtype,
            "replace": self._cmd_replace,
            "standardize": self._cmd_standardize,
            "remove_rows": self._cmd_remove_rows,
            "clip": self._cmd_clip,
            "validate": self._cmd_validate,
            "submit": self._cmd_submit,
        }

        handler = dispatch.get(cmd)
        if handler is None:
            suggestions = [c for c in dispatch.keys() if c.startswith(cmd[:3])] if len(cmd) >= 3 else []
            msg = f"Error: Unknown command '{cmd}'."
            if suggestions:
                msg += f" Did you mean: {', '.join(suggestions)}?"
            msg += " Type 'help' for available commands."
            return msg, False

        try:
            return handler(args)
        except Exception as e:
            return f"Error executing '{cmd}': {str(e)}", False

    def _parse_command(self, command_str: str) -> list:
        """Parse command string, handling quoted arguments."""
        parts = []
        current = ""
        in_quotes = False
        quote_char = None

        for char in command_str:
            if char in ('"', "'") and not in_quotes:
                in_quotes = True
                quote_char = char
            elif char == quote_char and in_quotes:
                in_quotes = False
                quote_char = None
            elif char == " " and not in_quotes:
                if current:
                    parts.append(current)
                    current = ""
            else:
                current += char

        if current:
            parts.append(current)
        return parts

    # ── Diagnostic commands (read-only) ──────────────────────────────────

    def _cmd_help(self, args: list) -> Tuple[str, bool]:
        return COMMANDS_HELP, False

    def _cmd_view(self, args: list) -> Tuple[str, bool]:
        n = 10
        if args:
            try:
                n = int(args[0])
            except ValueError:
                return "Error: 'view' expects an integer argument. Usage: view [N]", False
        n = min(n, 50)  # cap to prevent huge output
        result = self.df.head(n).to_string(max_colwidth=30)
        return f"Showing first {n} rows ({len(self.df)} total):\n\n{result}", False

    def _cmd_profile(self, args: list) -> Tuple[str, bool]:
        lines = []
        lines.append(f"Dataset Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        lines.append(f"\nColumns:")
        lines.append(f"{'Column':<20} {'Type':<12} {'Non-Null':<10} {'Missing':<10} {'Missing%':<10} {'Unique':<8}")
        lines.append("-" * 70)
        for col in self.df.columns:
            dtype = str(self.df[col].dtype)
            non_null = self.df[col].notna().sum()
            missing = self.df[col].isna().sum()
            missing_pct = f"{(missing / len(self.df) * 100):.1f}%"
            unique = self.df[col].nunique()
            lines.append(f"{col:<20} {dtype:<12} {non_null:<10} {missing:<10} {missing_pct:<10} {unique:<8}")

        # Duplicate check
        n_dupes = self.df.duplicated().sum()
        lines.append(f"\nDuplicate rows: {n_dupes} ({n_dupes / len(self.df) * 100:.1f}%)")

        return "\n".join(lines), False

    def _cmd_profile_column(self, args: list) -> Tuple[str, bool]:
        if not args:
            return "Error: Usage: profile_column COLUMN_NAME", False
        col = args[0]
        if col not in self.df.columns:
            return f"Error: Column '{col}' not found. Available: {', '.join(self.df.columns)}", False

        lines = [f"Profile for column '{col}':"]
        series = self.df[col]
        lines.append(f"  Type: {series.dtype}")
        lines.append(f"  Non-null: {series.notna().sum()} / {len(series)}")
        lines.append(f"  Missing: {series.isna().sum()} ({series.isna().mean() * 100:.1f}%)")
        lines.append(f"  Unique values: {series.nunique()}")

        if pd.api.types.is_numeric_dtype(series):
            desc = series.describe()
            lines.append(f"  Min: {desc.get('min', 'N/A')}")
            lines.append(f"  Max: {desc.get('max', 'N/A')}")
            lines.append(f"  Mean: {desc.get('mean', 'N/A'):.2f}" if pd.notna(desc.get('mean')) else "  Mean: N/A")
            lines.append(f"  Std: {desc.get('std', 'N/A'):.2f}" if pd.notna(desc.get('std')) else "  Std: N/A")
            lines.append(f"  Median: {desc.get('50%', 'N/A')}")
        else:
            top_values = series.dropna().value_counts().head(10)
            lines.append(f"  Top values:")
            for val, count in top_values.items():
                lines.append(f"    '{val}': {count}")

        return "\n".join(lines), False

    def _cmd_find_missing(self, args: list) -> Tuple[str, bool]:
        missing = self.df.isnull().sum()
        missing = missing[missing > 0]
        if missing.empty:
            return "No missing values found! The dataset is complete.", False

        lines = ["Missing values by column:"]
        lines.append(f"{'Column':<25} {'Count':<8} {'Percentage':<10}")
        lines.append("-" * 43)
        for col, count in missing.sort_values(ascending=False).items():
            pct = f"{count / len(self.df) * 100:.1f}%"
            lines.append(f"{col:<25} {count:<8} {pct:<10}")
        lines.append(f"\nTotal missing cells: {missing.sum()}")
        return "\n".join(lines), False

    def _cmd_find_duplicates(self, args: list) -> Tuple[str, bool]:
        subset = None
        if args:
            subset = [c.strip() for c in args[0].split(",")]
            invalid = [c for c in subset if c not in self.df.columns]
            if invalid:
                return f"Error: Unknown columns: {invalid}. Available: {list(self.df.columns)}", False

        dupes = self.df[self.df.duplicated(subset=subset, keep=False)]
        n_dupes = self.df.duplicated(subset=subset, keep="first").sum()

        if n_dupes == 0:
            cols_desc = f" (on columns: {subset})" if subset else ""
            return f"No duplicate rows found{cols_desc}.", False

        lines = [f"Found {n_dupes} duplicate rows (keeping first occurrence):"]
        if len(dupes) <= 20:
            lines.append(dupes.to_string(max_colwidth=25))
        else:
            lines.append(f"Showing first 10 of {len(dupes)} duplicate entries:")
            lines.append(dupes.head(10).to_string(max_colwidth=25))
        return "\n".join(lines), False

    def _cmd_find_outliers(self, args: list) -> Tuple[str, bool]:
        if not args:
            return "Error: Usage: find_outliers COLUMN_NAME", False
        col = args[0]
        if col not in self.df.columns:
            return f"Error: Column '{col}' not found. Available: {', '.join(self.df.columns)}", False

        try:
            numeric_col = pd.to_numeric(self.df[col], errors="coerce")
        except Exception:
            return f"Error: Column '{col}' cannot be converted to numeric for outlier detection.", False

        q1 = numeric_col.quantile(0.25)
        q3 = numeric_col.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers_mask = (numeric_col < lower) | (numeric_col > upper)
        n_outliers = outliers_mask.sum()

        if n_outliers == 0:
            return f"No outliers found in '{col}' (IQR method, bounds: [{lower:.2f}, {upper:.2f}]).", False

        lines = [f"Found {n_outliers} outliers in '{col}' (IQR method):"]
        lines.append(f"  Q1: {q1:.2f}, Q3: {q3:.2f}, IQR: {iqr:.2f}")
        lines.append(f"  Lower bound: {lower:.2f}")
        lines.append(f"  Upper bound: {upper:.2f}")
        outlier_values = numeric_col[outliers_mask].dropna()
        lines.append(f"  Outlier values: {list(outlier_values.head(15).values)}")
        return "\n".join(lines), False

    # ── Cleaning commands (modify data) ──────────────────────────────────

    def _cmd_fill_missing(self, args: list) -> Tuple[str, bool]:
        if len(args) < 2:
            return "Error: Usage: fill_missing COLUMN STRATEGY [VALUE]\n  Strategies: mean, median, mode, constant, forward_fill", False
        col = args[0]
        strategy = args[1].lower()
        if col not in self.df.columns:
            return f"Error: Column '{col}' not found. Available: {', '.join(self.df.columns)}", False

        n_before = self.df[col].isna().sum()
        if n_before == 0:
            return f"No missing values in '{col}'. Nothing to fill.", False

        if strategy == "mean":
            try:
                fill_val = pd.to_numeric(self.df[col], errors="coerce").mean()
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce").fillna(fill_val)
            except Exception:
                return f"Error: Cannot compute mean for non-numeric column '{col}'.", False
        elif strategy == "median":
            try:
                fill_val = pd.to_numeric(self.df[col], errors="coerce").median()
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce").fillna(fill_val)
            except Exception:
                return f"Error: Cannot compute median for non-numeric column '{col}'.", False
        elif strategy == "mode":
            mode_val = self.df[col].mode()
            if mode_val.empty:
                return f"Error: No mode found for '{col}'.", False
            self.df[col] = self.df[col].fillna(mode_val.iloc[0])
        elif strategy == "constant":
            if len(args) < 3:
                return "Error: 'constant' strategy requires a VALUE. Usage: fill_missing COL constant VALUE", False
            fill_val = args[2]
            self.df[col] = self.df[col].fillna(fill_val)
        elif strategy == "forward_fill":
            self.df[col] = self.df[col].ffill()
        else:
            return f"Error: Unknown strategy '{strategy}'. Use: mean, median, mode, constant, forward_fill", False

        n_after = self.df[col].isna().sum()
        filled = n_before - n_after
        return f"Filled {filled} missing values in '{col}' using strategy '{strategy}'. Remaining: {n_after}", True

    def _cmd_remove_duplicates(self, args: list) -> Tuple[str, bool]:
        subset = None
        keep = "first"
        if args:
            subset = [c.strip() for c in args[0].split(",")]
            invalid = [c for c in subset if c not in self.df.columns]
            if invalid:
                return f"Error: Unknown columns: {invalid}. Available: {list(self.df.columns)}", False
            if len(args) > 1:
                keep = args[1].lower()
                if keep not in ("first", "last", "none", "false"):
                    return f"Error: keep must be 'first', 'last', or 'none'. Got: '{keep}'", False
                if keep == "none":
                    keep = False

        n_before = len(self.df)
        self.df = self.df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
        n_after = len(self.df)
        removed = n_before - n_after

        if removed == 0:
            return "No duplicate rows found to remove.", False

        return f"Removed {removed} duplicate rows. Dataset: {n_before} → {n_after} rows.", True

    def _cmd_fix_dtype(self, args: list) -> Tuple[str, bool]:
        if len(args) < 2:
            return "Error: Usage: fix_dtype COLUMN TYPE (int/float/str/datetime)", False
        col = args[0]
        target = args[1].lower()
        if col not in self.df.columns:
            return f"Error: Column '{col}' not found.", False

        before_type = str(self.df[col].dtype)
        errors = 0
        if target in ("int", "int64"):
            # First try to clean string values (remove $, commas etc.)
            self.df[col] = self.df[col].astype(str).str.replace(r'[^\d.\-]', '', regex=True)
            numeric = pd.to_numeric(self.df[col], errors="coerce")
            errors = numeric.isna().sum() - self.df[col].isna().sum()
            self.df[col] = numeric.astype("Int64")
        elif target in ("float", "float64"):
            self.df[col] = self.df[col].astype(str).str.replace(r'[^\d.\-]', '', regex=True)
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
            errors = self.df[col].isna().sum()
        elif target in ("str", "string", "object"):
            self.df[col] = self.df[col].astype(str)
        elif target in ("datetime", "date"):
            self.df[col] = pd.to_datetime(self.df[col], errors="coerce", infer_datetime_format=True)
            errors = self.df[col].isna().sum()
        else:
            return f"Error: Unknown type '{target}'. Use: int, float, str, datetime", False

        return f"Converted '{col}' from {before_type} → {target}. Coercion errors: {errors}", True

    def _cmd_replace(self, args: list) -> Tuple[str, bool]:
        if len(args) < 3:
            return "Error: Usage: replace COLUMN OLD_VALUE NEW_VALUE", False
        col = args[0]
        old_val = args[1]
        new_val = args[2]
        if col not in self.df.columns:
            return f"Error: Column '{col}' not found.", False

        # Count matches before replacement
        mask = self.df[col].astype(str) == old_val
        n_matches = mask.sum()

        if n_matches == 0:
            return f"No matches found for '{old_val}' in column '{col}'.", False

        self.df.loc[mask, col] = new_val
        return f"Replaced {n_matches} occurrences of '{old_val}' with '{new_val}' in '{col}'.", True

    def _cmd_standardize(self, args: list) -> Tuple[str, bool]:
        if len(args) < 2:
            return "Error: Usage: standardize COLUMN METHOD (lowercase/uppercase/titlecase/strip)", False
        col = args[0]
        method = args[1].lower()
        if col not in self.df.columns:
            return f"Error: Column '{col}' not found.", False

        before_uniq = self.df[col].nunique()

        if method == "lowercase":
            self.df[col] = self.df[col].astype(str).str.lower()
        elif method == "uppercase":
            self.df[col] = self.df[col].astype(str).str.upper()
        elif method == "titlecase":
            self.df[col] = self.df[col].astype(str).str.title()
        elif method == "strip":
            self.df[col] = self.df[col].astype(str).str.strip()
        else:
            return f"Error: Unknown method '{method}'. Use: lowercase, uppercase, titlecase, strip", False

        after_uniq = self.df[col].nunique()
        consolidated = before_uniq - after_uniq

        return f"Standardized '{col}' using {method}. Unique values: {before_uniq} → {after_uniq} (consolidated {consolidated}).", True

    def _cmd_remove_rows(self, args: list) -> Tuple[str, bool]:
        if len(args) < 3:
            return "Error: Usage: remove_rows COLUMN CONDITION VALUE\n  Conditions: equals, not_equals, less_than, greater_than, contains", False
        col = args[0]
        condition = args[1].lower()
        value = args[2]
        if col not in self.df.columns:
            return f"Error: Column '{col}' not found.", False

        n_before = len(self.df)

        if condition == "equals":
            mask = self.df[col].astype(str) == value
        elif condition == "not_equals":
            mask = self.df[col].astype(str) != value
        elif condition == "less_than":
            try:
                val = float(value)
                mask = pd.to_numeric(self.df[col], errors="coerce") < val
            except ValueError:
                return f"Error: '{value}' is not a valid number for less_than.", False
        elif condition == "greater_than":
            try:
                val = float(value)
                mask = pd.to_numeric(self.df[col], errors="coerce") > val
            except ValueError:
                return f"Error: '{value}' is not a valid number for greater_than.", False
        elif condition == "contains":
            mask = self.df[col].astype(str).str.contains(value, case=False, na=False)
        else:
            return f"Error: Unknown condition '{condition}'. Use: equals, not_equals, less_than, greater_than, contains", False

        n_removed = mask.sum()
        if n_removed == 0:
            return f"No rows match condition '{col} {condition} {value}'.", False

        self.df = self.df[~mask].reset_index(drop=True)
        return f"Removed {n_removed} rows where {col} {condition} {value}. Dataset: {n_before} → {len(self.df)} rows.", True

    def _cmd_clip(self, args: list) -> Tuple[str, bool]:
        if len(args) < 3:
            return "Error: Usage: clip COLUMN LOWER UPPER", False
        col = args[0]
        if col not in self.df.columns:
            return f"Error: Column '{col}' not found.", False
        try:
            lower = float(args[1])
            upper = float(args[2])
        except ValueError:
            return "Error: LOWER and UPPER must be numbers.", False

        numeric_col = pd.to_numeric(self.df[col], errors="coerce")
        n_clipped = ((numeric_col < lower) | (numeric_col > upper)).sum()

        self.df[col] = numeric_col.clip(lower=lower, upper=upper)

        return f"Clipped {n_clipped} values in '{col}' to [{lower}, {upper}].", True

    # ── Special commands ─────────────────────────────────────────────────

    def _cmd_validate(self, args: list) -> Tuple[str, bool]:
        # Placeholder — actual scoring done by the environment
        return "__VALIDATE__", False

    def _cmd_submit(self, args: list) -> Tuple[str, bool]:
        # Placeholder — actual submission done by the environment
        return "__SUBMIT__", False
