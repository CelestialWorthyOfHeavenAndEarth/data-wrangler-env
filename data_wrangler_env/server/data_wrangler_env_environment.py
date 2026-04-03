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

Tasks:
    - task_1_easy:   Customer Records (50 rows, 5 cols)
    - task_2_medium: Sales Transactions (200 rows, 8 cols)
    - task_3_hard:   Healthcare Records (1000 rows, 12 cols)
"""

from copy import deepcopy
from typing import Any, Optional
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

        # Validate task name
        if self._task_name not in TASK_CONFIGS:
            self._task_name = "task_1_easy"

        # Generate fresh dataset
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
            current_score=0.0,
            step_number=0,
            max_steps=self._max_steps,
            task_name=self._task_name,
            available_commands=COMMANDS_HELP,
            done=False,
            reward=0.0,
        )

    def step(self, action: DataWranglerAction) -> DataWranglerObservation:  # type: ignore[override]
        """Execute a cleaning command on the dataset.

        Args:
            action: DataWranglerAction with a text command

        Returns:
            DataWranglerObservation with the result
        """
        self._state.step_count += 1
        step_num = self._state.step_count

        # Check if already done
        if self._done:
            return self._make_observation(
                response="Episode already finished. Call reset() to start a new one.",
                reward=0.0,
                done=True,
            )

        # Check max steps
        if step_num > self._max_steps:
            self._done = True
            final_score, dim_scores = compute_score(
                self._original_dirty_df, self._dirty_df, self._clean_df,
                self._original_dirty_df, self._issue_manifest
            )
            self._last_score = final_score
            return self._make_observation(
                response=self._format_final_report(final_score, dim_scores, "Max steps reached"),
                reward=final_score * 0.5,  # Partial reward for timeout
                done=True,
            )

        command = action.message.strip()

        # Take a snapshot before execution for reward calculation
        before_df = self._dirty_df.copy()

        # Execute command
        response, data_modified = self._engine.execute(command)

        # Handle special commands
        if response == "__VALIDATE__":
            self._validate_count += 1
            current_score, dim_scores = compute_score(
                self._original_dirty_df, self._dirty_df, self._clean_df,
                self._original_dirty_df, self._issue_manifest
            )
            self._last_score = current_score

            response = self._format_validation_report(current_score, dim_scores)
            reward = 0.01 if self._validate_count <= 5 else 0.0  # Diminishing reward
            self._cumulative_reward += reward

            return self._make_observation(response=response, reward=reward, done=False)

        if response == "__SUBMIT__":
            self._done = True
            final_score, dim_scores = compute_score(
                self._original_dirty_df, self._dirty_df, self._clean_df,
                self._original_dirty_df, self._issue_manifest
            )
            self._last_score = final_score
            # Submit gives the final score as reward
            reward = final_score
            self._cumulative_reward += reward

            return self._make_observation(
                response=self._format_final_report(final_score, dim_scores, "Submitted"),
                reward=reward,
                done=True,
            )

        # Compute step reward for normal commands
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

    # ── Private helpers ──────────────────────────────────────────────────

    def _make_observation(self, response: str, reward: float, done: bool) -> DataWranglerObservation:
        """Create a DataWranglerObservation with current state info."""
        shape = ""
        if self._dirty_df is not None:
            shape = f"{self._dirty_df.shape[0]} rows × {self._dirty_df.shape[1]} columns"

        return DataWranglerObservation(
            response=response,
            dataset_shape=shape,
            current_score=self._last_score,
            step_number=self._state.step_count,
            max_steps=self._max_steps,
            task_name=self._task_name,
            available_commands=COMMANDS_HELP,
            done=done,
            reward=reward,
            metadata={
                "cumulative_reward": round(self._cumulative_reward, 4),
                "task": self._task_name,
                "step": self._state.step_count,
            },
        )

    def _format_validation_report(self, score: float, dim_scores: dict) -> str:
        """Format a validation progress report."""
        lines = [
            "=== Validation Report ===",
            f"Overall Quality Score: {score:.3f} / 1.000",
            "",
            "Dimension Breakdown:",
            f"  Missing Values Fixed:  {dim_scores.get('missing_fixed', 0):.3f}  (weight: 25%)",
            f"  Duplicates Removed:    {dim_scores.get('duplicates_removed', 0):.3f}  (weight: 20%)",
            f"  Type Correctness:      {dim_scores.get('type_correctness', 0):.3f}  (weight: 20%)",
            f"  Value Accuracy:        {dim_scores.get('value_accuracy', 0):.3f}  (weight: 25%)",
            f"  Data Preservation:     {dim_scores.get('data_preservation', 0):.3f}  (weight: 10%)",
            "",
            f"Steps used: {self._state.step_count} / {self._max_steps}",
            f"Cumulative reward: {self._cumulative_reward:.3f}",
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
            f"  Missing Values Fixed:  {dim_scores.get('missing_fixed', 0):.4f}  (25%)",
            f"  Duplicates Removed:    {dim_scores.get('duplicates_removed', 0):.4f}  (20%)",
            f"  Type Correctness:      {dim_scores.get('type_correctness', 0):.4f}  (20%)",
            f"  Value Accuracy:        {dim_scores.get('value_accuracy', 0):.4f}  (25%)",
            f"  Data Preservation:     {dim_scores.get('data_preservation', 0):.4f}  (10%)",
            "",
            f"Total steps taken: {self._state.step_count}",
            f"Cumulative reward: {self._cumulative_reward:.4f}",
            "",
            "Episode complete.",
        ]
        return "\n".join(lines)
