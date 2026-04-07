# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
DataWrangler Environment Implementation.

A data quality & cleaning environment where AI agents fix messy real-world
datasets through text commands. The agent must profile, diagnose, and clean
datasets to maximize a multi-dimensional quality score.

Features:
    - 3 progressive tasks: Customer Records → Sales → Healthcare
    - Multi-dimensional grading (missing, duplicates, types, values, preservation, efficiency)
    - Data lineage tracking & cleaning provenance report
    - Undo/redo support for experimentation
    - Business rule validation (domain-specific constraints)
    - Anti-exploit golden rows for grading integrity
    - Red herring data to test over-cleaning robustness
"""

from copy import deepcopy
from typing import Any, Dict, List, Optional
from uuid import uuid4

import pandas as pd
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import DataWranglerAction, DataWranglerObservation
except ImportError:
    try:
        from models import DataWranglerAction, DataWranglerObservation
    except ImportError:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from models import DataWranglerAction, DataWranglerObservation

from .cleaning_engine import CleaningEngine
from .dataset_generator import COMMANDS_HELP, TASK_CONFIGS, generate_dataset
from .grader import compute_score, compute_step_reward


class DataWranglerEnvironment(Environment):
    """Data quality & cleaning environment.

    The agent receives text commands to inspect and clean a messy dataset.
    It is scored on how well the cleaned result matches the ground truth.

    Features:
        - Undo/redo: Agents can undo bad cleaning decisions
        - Data lineage: Every operation is tracked for provenance
        - Business rules: Domain-specific constraints are validated
        - Golden rows: Anti-exploit mechanism to prevent gaming
        - Red herrings: Valid-but-suspicious data tests over-cleaning

    Example:
        >>> env = DataWranglerEnvironment()
        >>> obs = env.reset(task="task_1_easy")
        >>> obs = env.step(DataWranglerAction(message="profile"))
        >>> obs = env.step(DataWranglerAction(message="find_missing"))
        >>> obs = env.step(DataWranglerAction(message="fill_missing age mean"))
        >>> obs = env.step(DataWranglerAction(message="submit"))
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the DataWrangler environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task_name = "task_1_easy"
        self._dirty_df: Optional[pd.DataFrame] = None
        self._clean_df: Optional[pd.DataFrame] = None
        self._original_dirty_df: Optional[pd.DataFrame] = None
        self._issue_manifest: dict = {}
        self._max_steps: int = 30
        self._description: str = ""
        self._engine: Optional[CleaningEngine] = None
        self._done: bool = False
        self._cumulative_reward: float = 0.0
        self._seed: int = 42
        self._validate_count: int = 0
        self._last_score: float = 0.0
        # ── New features ──
        self._operation_log: List[Dict[str, Any]] = []  # Data lineage
        self._undo_stack: List[pd.DataFrame] = []        # Undo support
        self._golden_indices: List[int] = []              # Anti-exploit
        self._business_rules: List[Dict[str, Any]] = []   # Domain constraints

    def reset(self, **kwargs: Any) -> DataWranglerObservation:
        """Reset the environment with a fresh dirty dataset.

        Kwargs:
            task: Task name (task_1_easy, task_2_medium, task_3_hard)
            seed: Random seed for reproducibility

        Returns:
            Initial DataWranglerObservation with dataset overview
        """
        self._task_name = kwargs.get("task", "task_1_easy")
        self._seed = kwargs.get("seed", 42)

        if isinstance(self._seed, str):
            try:
                self._seed = int(self._seed)
            except ValueError:
                self._seed = 42

        if self._task_name not in TASK_CONFIGS:
            self._task_name = "task_1_easy"

        # Generate fresh dataset (now includes golden_indices and business_rules)
        dirty_df, clean_df, issues, max_steps, description = generate_dataset(
            self._task_name, self._seed
        )

        self._dirty_df = dirty_df.copy()
        self._clean_df = clean_df.copy()
        self._original_dirty_df = dirty_df.copy()
        self._issue_manifest = issues
        self._max_steps = max_steps
        self._description = description
        self._engine = CleaningEngine(self._dirty_df)
        self._done = False
        self._cumulative_reward = 0.0
        self._validate_count = 0
        self._last_score = 0.0
        # Reset new features
        self._operation_log = []
        self._undo_stack = []
        self._golden_indices = issues.get("golden_indices", [])
        self._business_rules = issues.get("business_rules", [])

        self._state = State(
            episode_id=str(uuid4()),
            step_count=0,
        )

        # Build initial observation
        shape = f"{self._dirty_df.shape[0]} rows × {self._dirty_df.shape[1]} columns"
        columns_info = ", ".join(self._dirty_df.columns.tolist())
        initial_msg = (
            f"=== DataWrangler Environment ===\n"
            f"Task: {self._task_name}\n"
            f"Description: {self._description}\n\n"
            f"Dataset loaded: {shape}\n"
            f"Columns: {columns_info}\n\n"
            f"Your goal: Clean this dataset to fix all data quality issues.\n"
            f"You have {self._max_steps} steps. Type 'help' for available commands.\n"
            f"Start by exploring the data with 'profile' or 'view'."
        )

        return DataWranglerObservation(
            response=initial_msg,
            dataset_shape=shape,
            current_score=0.001,
            step_number=0,
            max_steps=self._max_steps,
            task_name=self._task_name,
            available_commands=COMMANDS_HELP,
            done=False,
            reward=0.001,
        )

    def step(self, action: DataWranglerAction) -> DataWranglerObservation:  # type: ignore[override]
        """Execute a cleaning command on the dataset."""
        self._state.step_count += 1
        step_num = self._state.step_count

        if self._done:
            return self._make_observation(
                response="Episode already finished. Call reset() to start a new one.",
                reward=0.001,
                done=True,
            )

        if self._dirty_df is None:
            self.reset()

        # Max steps check
        if step_num > self._max_steps:
            self._done = True
            final_score, dim_scores = compute_score(
                self._original_dirty_df, self._dirty_df, self._clean_df,
                self._original_dirty_df, self._issue_manifest,
                step_count=step_num, max_steps=self._max_steps,
                golden_indices=self._golden_indices,
            )
            self._last_score = final_score
            return self._make_observation(
                response=self._format_final_report(final_score, dim_scores, "Max steps reached"),
                reward=final_score * 0.5,
                done=True,
            )

        command = action.message.strip()

        # ── Handle undo command ──
        if command.lower() == "undo":
            return self._handle_undo()

        # ── Handle check_rules command ──
        if command.lower() == "check_rules":
            return self._handle_check_rules()

        # ── Handle history command ──
        if command.lower() == "history":
            return self._handle_history()

        # Save state snapshot for undo (only for data-modifying commands)
        before_df = self._dirty_df.copy()

        # Execute command
        response, data_modified = self._engine.execute(command)

        # Track operation for lineage
        if data_modified:
            self._undo_stack.append(before_df)
            # Limit undo stack to 10 states to save memory
            if len(self._undo_stack) > 10:
                self._undo_stack.pop(0)

        self._operation_log.append({
            "step": step_num,
            "command": command,
            "modified": data_modified,
            "rows_before": len(before_df),
            "rows_after": len(self._dirty_df),
        })

        # Handle special commands
        if response == "__VALIDATE__":
            self._validate_count += 1
            current_score, dim_scores = compute_score(
                self._original_dirty_df, self._dirty_df, self._clean_df,
                self._original_dirty_df, self._issue_manifest,
                step_count=step_num, max_steps=self._max_steps,
                golden_indices=self._golden_indices,
            )
            self._last_score = current_score
            response = self._format_validation_report(current_score, dim_scores)
            reward = 0.01 if self._validate_count <= 5 else 0.001
            self._cumulative_reward += reward
            return self._make_observation(response=response, reward=reward, done=False)

        if response == "__SUBMIT__":
            self._done = True
            final_score, dim_scores = compute_score(
                self._original_dirty_df, self._dirty_df, self._clean_df,
                self._original_dirty_df, self._issue_manifest,
                step_count=step_num, max_steps=self._max_steps,
                golden_indices=self._golden_indices,
            )
            self._last_score = final_score
            reward = final_score
            self._cumulative_reward += reward
            report = self._format_final_report(final_score, dim_scores, "Submitted")
            report += "\n\n" + self._format_lineage_report()
            return self._make_observation(response=report, reward=reward, done=True)

        # Compute step reward
        reward = compute_step_reward(
            before_df, self._dirty_df, self._clean_df,
            self._original_dirty_df, self._issue_manifest,
            command, data_modified,
        )
        self._cumulative_reward += reward
        return self._make_observation(response=response, reward=reward, done=False)

    @property
    def state(self) -> State:
        """Get the current environment state."""
        return self._state

    # ── New command handlers ─────────────────────────────────────────────

    def _handle_undo(self) -> DataWranglerObservation:
        """Undo the last data-modifying operation."""
        if not self._undo_stack:
            return self._make_observation(
                response="Nothing to undo. No data-modifying operations have been performed.",
                reward=0.001,
                done=False,
            )
        self._dirty_df = self._undo_stack.pop()
        self._engine = CleaningEngine(self._dirty_df)
        self._operation_log.append({
            "step": self._state.step_count,
            "command": "undo",
            "modified": True,
            "rows_before": -1,
            "rows_after": len(self._dirty_df),
        })
        return self._make_observation(
            response=f"Undo successful. Dataset restored to {len(self._dirty_df)} rows × {self._dirty_df.shape[1]} columns.",
            reward=0.005,
            done=False,
        )

    def _handle_check_rules(self) -> DataWranglerObservation:
        """Check business rule violations in the current dataset."""
        if not self._business_rules:
            return self._make_observation(
                response="No business rules defined for this task.",
                reward=0.001,
                done=False,
            )

        violations = []
        for rule in self._business_rules:
            rule_type = rule.get("type", "")
            col = rule.get("column", "")

            if rule_type == "range" and col in self._dirty_df.columns:
                lo, hi = rule.get("min", float("-inf")), rule.get("max", float("inf"))
                numeric = pd.to_numeric(self._dirty_df[col], errors="coerce")
                bad = ((numeric < lo) | (numeric > hi)).sum()
                if bad > 0:
                    violations.append(f"  • {col}: {bad} values outside [{lo}, {hi}]")

            elif rule_type == "not_null" and col in self._dirty_df.columns:
                n_null = self._dirty_df[col].isna().sum()
                if n_null > 0:
                    violations.append(f"  • {col}: {n_null} null values (required: not null)")

            elif rule_type == "pattern" and col in self._dirty_df.columns:
                pat = rule.get("pattern", "")
                if pat:
                    non_null = self._dirty_df[col].dropna().astype(str)
                    bad = (~non_null.str.match(pat)).sum()
                    if bad > 0:
                        violations.append(f"  • {col}: {bad} values don't match pattern '{pat}'")

            elif rule_type == "cross_column":
                col_a = rule.get("column_a", "")
                col_b = rule.get("column_b", "")
                relation = rule.get("relation", "")
                if col_a in self._dirty_df.columns and col_b in self._dirty_df.columns:
                    a = pd.to_numeric(self._dirty_df[col_a], errors="coerce")
                    b = pd.to_numeric(self._dirty_df[col_b], errors="coerce")
                    if relation == "greater_than":
                        bad = (a <= b).sum()
                        if bad > 0:
                            violations.append(f"  • {col_a} must be > {col_b}: {bad} violations")

            elif rule_type == "categorical" and col in self._dirty_df.columns:
                allowed = set(rule.get("allowed_values", []))
                if allowed:
                    actual = set(self._dirty_df[col].dropna().unique())
                    bad = actual - allowed
                    if bad:
                        violations.append(f"  • {col}: {len(bad)} invalid categories: {list(bad)[:5]}")

        if not violations:
            response = "=== Business Rule Check ===\nAll business rules satisfied! ✓"
        else:
            response = f"=== Business Rule Check ===\nFound {len(violations)} rule violation(s):\n" + "\n".join(violations)

        return self._make_observation(response=response, reward=0.01, done=False)

    def _handle_history(self) -> DataWranglerObservation:
        """Show the operation history (data lineage)."""
        if not self._operation_log:
            return self._make_observation(
                response="No operations performed yet.",
                reward=0.001,
                done=False,
            )

        lines = ["=== Operation History (Data Lineage) ==="]
        for op in self._operation_log:
            status = "✓ modified" if op["modified"] else "  read-only"
            lines.append(f"  Step {op['step']:2d}: [{status}] {op['command']}")
        lines.append(f"\nTotal operations: {len(self._operation_log)}")
        lines.append(f"Data-modifying: {sum(1 for o in self._operation_log if o['modified'])}")

        return self._make_observation(response="\n".join(lines), reward=0.001, done=False)

    # ── Private helpers ──────────────────────────────────────────────────

    def _clamp_score(self, score: float) -> float:
        """Clamp score to open interval (0, 1) per validator requirement."""
        return max(0.001, min(0.999, score))

    def _make_observation(self, response: str, reward: float, done: bool) -> DataWranglerObservation:
        """Create a DataWranglerObservation with current state info."""
        shape = ""
        if self._dirty_df is not None:
            shape = f"{self._dirty_df.shape[0]} rows × {self._dirty_df.shape[1]} columns"

        return DataWranglerObservation(
            response=response,
            dataset_shape=shape,
            current_score=self._clamp_score(self._last_score),
            step_number=self._state.step_count,
            max_steps=self._max_steps,
            task_name=self._task_name,
            available_commands=COMMANDS_HELP,
            done=done,
            reward=self._clamp_score(reward) if done else max(0.001, reward) if reward >= 0 else reward,
            metadata={
                "cumulative_reward": round(self._cumulative_reward, 4),
                "task": self._task_name,
                "step": self._state.step_count,
                "operations_count": len(self._operation_log),
                "undo_available": len(self._undo_stack) > 0,
            },
        )

    def _format_lineage_report(self) -> str:
        """Format the data lineage/provenance report shown on submit."""
        if not self._operation_log:
            return "Cleaning Report: No operations were performed."

        lines = ["=== Cleaning Provenance Report ==="]
        modify_ops = [o for o in self._operation_log if o["modified"]]
        read_ops = [o for o in self._operation_log if not o["modified"]]

        lines.append(f"Total steps: {len(self._operation_log)}")
        lines.append(f"  Diagnostic (read-only): {len(read_ops)}")
        lines.append(f"  Data-modifying: {len(modify_ops)}")
        lines.append(f"  Undo operations: {sum(1 for o in self._operation_log if o['command'] == 'undo')}")
        lines.append("")

        if modify_ops:
            lines.append("Data Transformations Applied:")
            for op in modify_ops:
                delta = ""
                if op["rows_before"] >= 0 and op["rows_after"] >= 0:
                    diff = op["rows_after"] - op["rows_before"]
                    if diff != 0:
                        delta = f" (rows: {op['rows_before']} → {op['rows_after']})"
                lines.append(f"  Step {op['step']:2d}: {op['command']}{delta}")

        return "\n".join(lines)

    def _format_validation_report(self, score: float, dim_scores: dict) -> str:
        """Format a validation progress report."""
        lines = [
            "=== Validation Report ===",
            f"Overall Quality Score: {score:.3f} / 1.000",
            "",
            "Dimension Breakdown:",
            f"  Missing Values Fixed:  {dim_scores.get('missing_fixed', 0):.3f}  (weight: 20%)",
            f"  Duplicates Removed:    {dim_scores.get('duplicates_removed', 0):.3f}  (weight: 15%)",
            f"  Type Correctness:      {dim_scores.get('type_correctness', 0):.3f}  (weight: 15%)",
            f"  Value Accuracy:        {dim_scores.get('value_accuracy', 0):.3f}  (weight: 20%)",
            f"  Data Preservation:     {dim_scores.get('data_preservation', 0):.3f}  (weight: 10%)",
            f"  Constraint Compliance: {dim_scores.get('constraint_compliance', 0):.3f}  (weight: 10%)",
            f"  Step Efficiency:       {dim_scores.get('step_efficiency', 0):.3f}  (weight: 5%)",
            f"  Golden Row Integrity:  {dim_scores.get('golden_row_integrity', 0):.3f}  (weight: 5%)",
            "",
            f"Steps used: {self._state.step_count} / {self._max_steps}",
            f"Cumulative reward: {self._cumulative_reward:.3f}",
            f"Undo stack: {len(self._undo_stack)} states",
            "",
            "Continue cleaning or type 'submit' to finalize.",
        ]
        return "\n".join(lines)

    def _format_final_report(self, score: float, dim_scores: dict, reason: str) -> str:
        """Format the final grading report."""
        lines = [
            f"=== FINAL REPORT ({reason}) ===",
            f"Final Quality Score: {score:.4f} / 1.0000",
            "",
            "Dimension Breakdown:",
            f"  Missing Values Fixed:  {dim_scores.get('missing_fixed', 0):.4f}  (20%)",
            f"  Duplicates Removed:    {dim_scores.get('duplicates_removed', 0):.4f}  (15%)",
            f"  Type Correctness:      {dim_scores.get('type_correctness', 0):.4f}  (15%)",
            f"  Value Accuracy:        {dim_scores.get('value_accuracy', 0):.4f}  (20%)",
            f"  Data Preservation:     {dim_scores.get('data_preservation', 0):.4f}  (10%)",
            f"  Constraint Compliance: {dim_scores.get('constraint_compliance', 0):.4f}  (10%)",
            f"  Step Efficiency:       {dim_scores.get('step_efficiency', 0):.4f}  (5%)",
            f"  Golden Row Integrity:  {dim_scores.get('golden_row_integrity', 0):.4f}  (5%)",
            "",
            f"Total steps taken: {self._state.step_count}",
            f"Cumulative reward: {self._cumulative_reward:.4f}",
            "",
            "Episode complete.",
        ]
        return "\n".join(lines)
