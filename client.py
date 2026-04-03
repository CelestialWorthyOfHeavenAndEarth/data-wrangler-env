# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DataWrangler Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import DataWranglerAction, DataWranglerObservation


class DataWranglerEnv(
    EnvClient[DataWranglerAction, DataWranglerObservation, State]
):
    """
    Client for the DataWrangler Environment.

    Connects to the DataWrangler server over WebSocket for efficient
    multi-step data cleaning interactions.

    Example:
        >>> with DataWranglerEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset(task="task_1_easy")
        ...     print(result.observation.response)
        ...
        ...     result = client.step(DataWranglerAction(message="profile"))
        ...     print(result.observation.response)

    Example with Docker:
        >>> client = DataWranglerEnv.from_docker_image("data_wrangler_env:latest")
        >>> try:
        ...     result = client.reset(task="task_1_easy")
        ...     result = client.step(DataWranglerAction(message="find_missing"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: DataWranglerAction) -> Dict:
        """Convert DataWranglerAction to JSON payload for step message."""
        return {
            "message": action.message,
        }

    def _parse_result(self, payload: Dict) -> StepResult[DataWranglerObservation]:
        """Parse server response into StepResult[DataWranglerObservation]."""
        obs_data = payload.get("observation", {})
        observation = DataWranglerObservation(
            response=obs_data.get("response", ""),
            dataset_shape=obs_data.get("dataset_shape", ""),
            current_score=obs_data.get("current_score", 0.0),
            step_number=obs_data.get("step_number", 0),
            max_steps=obs_data.get("max_steps", 30),
            task_name=obs_data.get("task_name", ""),
            available_commands=obs_data.get("available_commands", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server response into State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
