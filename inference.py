"""
DataWranglerEnv — Baseline Inference Script

Runs an LLM agent against the DataWrangler environment across all 3 tasks.
Uses the OpenAI API client. Emits structured [START]/[STEP]/[END] logs.

Environment variables:
    API_BASE_URL  - The API endpoint for the LLM
    MODEL_NAME    - The model identifier to use
    HF_TOKEN      - API key for authentication

Usage:
    export API_BASE_URL="https://api.openai.com/v1"
    export MODEL_NAME="gpt-4o-mini"
    export HF_TOKEN="your-key"
    python inference.py
"""

import asyncio
import os
import sys
import traceback
from typing import List

from openai import OpenAI

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import DataWranglerAction
from client import DataWranglerEnv

# ── Configuration ────────────────────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.environ.get("MODEL_NAME") or "meta-llama/Llama-3.1-8B-Instruct"
API_KEY = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY") or ""

# ENV_BASE_URL: connect directly to this server instead of launching Docker locally.
# Set to your HF Space URL for testing: https://aswini-kumar-data-wrangler-env.hf.space
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "")
IMAGE_NAME = os.environ.get("IMAGE_NAME", "data-wrangler-env:latest")
BENCHMARK = "DataWranglerEnv"
TEMPERATURE = 0.3
MAX_TOKENS = 512

TASKS = [
    {"name": "task_1_easy", "max_steps": 30, "max_total_reward": 3.0, "success_threshold": 0.6},
    {"name": "task_2_medium", "max_steps": 50, "max_total_reward": 5.0, "success_threshold": 0.5},
    {"name": "task_3_hard", "max_steps": 80, "max_total_reward": 8.0, "success_threshold": 0.3},
]


SYSTEM_PROMPT = """You are an expert data analyst tasked with cleaning a messy dataset.

You interact with the dataset by sending short text commands. Available commands:
  help                          - Show all commands
  view [N]                      - Show first N rows
  profile                       - Dataset summary (shape, dtypes, missing %, duplicates)
  profile_column COL            - Detailed stats for one column
  find_missing                  - Missing value counts per column
  find_duplicates [COL1,COL2]   - Find duplicate rows
  find_outliers COL             - Outlier detection for numeric column
  fill_missing COL STRATEGY     - Fill nulls (mean/median/mode/constant VALUE)
  remove_duplicates [COL1,COL2] - Drop duplicates
  fix_dtype COL TYPE            - Cast column type (int/float/str/datetime)
  replace COL OLD NEW           - Replace specific values
  standardize COL METHOD        - Normalize formatting (lowercase/uppercase/titlecase/strip)
  remove_rows COL CONDITION VAL - Remove rows (equals/less_than/greater_than/contains)
  clip COL LOWER UPPER          - Clip numeric values
  validate                      - Check current quality score
  submit                        - Finalize and get final score

Strategy:
1. Start with 'profile' to understand the dataset
2. Use 'find_missing', 'find_duplicates', 'find_outliers' to diagnose issues
3. Fix issues systematically — missing values first, then duplicates, then types, then values
4. Use 'validate' to check progress periodically
5. When satisfied, use 'submit' to finalize

IMPORTANT: Respond with ONLY the command to execute. No explanations, no markdown, just the command.
For example: fill_missing age mean
"""


# ── Logging — MANDATORY PLAIN-TEXT FORMAT ────────────────────────────────────
# Spec: https://openenv.meta.com (sample inference script)
#   [START] task=<task> env=<env> model=<model>
#   [STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
#   [END]   success=<true|false> steps=<n> rewards=<r1,r2,...>

def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error=None):
    done_str = "true" if done else "false"
    error_str = str(error) if error is not None else "null"
    action_clean = action.replace("\n", " ").replace("\r", "")[:200]
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_str} error={error_str}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={success_str} steps={steps} rewards={rewards_str}",
        flush=True,
    )


# ── Agent Logic ──────────────────────────────────────────────────────────────

def build_user_prompt(step: int, last_response: str, last_reward: float, history: List[str]) -> str:
    """Build the user prompt for the LLM."""
    # Truncate response to prevent token overflow
    truncated = last_response[:1500] if len(last_response) > 1500 else last_response

    # Show recent history (last 5 actions)
    recent = history[-5:] if len(history) > 5 else history
    history_str = "\n".join(recent) if recent else "No actions taken yet."

    return (
        f"Step {step}.\n\n"
        f"Last command result:\n{truncated}\n\n"
        f"Last reward: {last_reward:+.3f}\n\n"
        f"Recent history:\n{history_str}\n\n"
        f"What command should I execute next? Reply with ONLY the command."
    )


def get_model_message(
    client: OpenAI,
    step: int,
    last_response: str,
    last_reward: float,
    history: List[str],
) -> str:
    """Get the next command from the LLM."""
    user_prompt = build_user_prompt(step, last_response, last_reward, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        # Clean up — remove markdown code blocks if present
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        return text if text else "profile"
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "profile"


# ── Main Loop ────────────────────────────────────────────────────────────────

async def run_task(client: OpenAI, task_config: dict) -> float:
    """Run a single task and return the score."""
    task_name = task_config["name"]
    max_steps = task_config["max_steps"]
    max_total_reward = task_config["max_total_reward"]
    success_threshold = task_config["success_threshold"]

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    # Connect to env: use ENV_BASE_URL if set (e.g. HF Space), else launch Docker
    if ENV_BASE_URL:
        env = DataWranglerEnv(base_url=ENV_BASE_URL)
    else:
        env = await DataWranglerEnv.from_docker_image(IMAGE_NAME)

    try:
        result = await env.reset(task=task_name, seed=42)
        last_response = result.observation.response
        last_reward = 0.0

        for step in range(1, max_steps + 1):
            if result.done:
                break

            message = get_model_message(client, step, last_response, last_reward, history)

            result = await env.step(DataWranglerAction(message=message))
            obs = result.observation

            reward = result.reward or 0.0
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken = step
            last_response = obs.response
            last_reward = reward

            log_step(step=step, action=message, reward=reward, done=done, error=error)

            history.append(f"Step {step}: '{message}' -> reward {reward:+.3f}")

            if done:
                break

        score = sum(rewards) / max_total_reward if max_total_reward > 0 else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= success_threshold

    except Exception as e:
        print(f"[DEBUG] Task {task_name} error: {e}", flush=True)
        traceback.print_exc()
    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


async def main() -> None:
    """Run all tasks sequentially."""
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    scores = {}
    for task_config in TASKS:
        score = await run_task(client, task_config)
        scores[task_config["name"]] = score

    print("\n" + "=" * 50, flush=True)
    print("FINAL SCORES:", flush=True)
    for task_name, score in scores.items():
        print(f"  {task_name}: {score:.4f}", flush=True)
    print(f"  Average: {sum(scores.values()) / len(scores):.4f}", flush=True)
    print("=" * 50, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
