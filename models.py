# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the DataWranglerEnv Environment.

Defines typed Pydantic models for Action, Observation used by the
data quality & cleaning environment.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class DataWranglerAction(Action):
    """Action for the DataWrangler environment — a text command string.

    The agent sends natural-language-style commands to inspect, diagnose,
    and clean a messy dataset. Examples:
        - "profile"
        - "find_missing"
        - "fill_missing age mean"
        - "remove_duplicates name,email first"
        - "submit"
    """

    message: str = Field(..., description="Text command to execute on the dataset")


class DataWranglerObservation(Observation):
    """Observation returned after each action in the DataWrangler environment.

    Provides the agent with the result of its command plus rich metadata
    about the current state of the dataset and episode progress.
    """

    response: str = Field(
        default="",
        description="Text result of the executed command (data tables, stats, errors)",
    )
    dataset_shape: str = Field(
        default="",
        description="Current dataset dimensions, e.g. '50 rows × 5 columns'",
    )
    current_score: float = Field(
        default=0.0,
        description="Current data quality score (0.0-1.0)",
    )
    step_number: int = Field(
        default=0,
        description="Current step number in the episode",
    )
    max_steps: int = Field(
        default=30,
        description="Maximum steps allowed for this task",
    )
    task_name: str = Field(
        default="",
        description="Active task identifier (task_1_easy, task_2_medium, task_3_hard)",
    )
    available_commands: str = Field(
        default="",
        description="Help text listing valid commands",
    )
