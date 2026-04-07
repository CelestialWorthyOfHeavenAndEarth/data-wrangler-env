"""
DataWranglerEnv — Baseline Inference Script

Runs an LLM agent against the DataWrangler environment across all 3 tasks.
Uses the OpenAI API client. Emits structured [START]/[STEP]/[END] logs.

Agent Strategy (3-Phase):
    Phase 1 (DIAGNOSE): profile, find_missing, find_duplicates, check_rules
    Phase 2 (CLEAN):    Systematic cleaning - types → missing → duplicates → values
    Phase 3 (VERIFY):   validate, review, submit

Environment variables:
    API_BASE_URL  - The API endpoint for the LLM
    MODEL_NAME    - The model identifier to use
    HF_TOKEN      - API key for authentication
    ENV_BASE_URL  - (optional) Override the environment server URL

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
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ── Configuration ────────────────────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.environ.get("MODEL_NAME") or "meta-llama/Llama-3.1-8B-Instruct"
API_KEY = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY") or ""

# Default to our deployed HF Space if ENV_BASE_URL not set
ENV_BASE_URL = (
    os.environ.get("ENV_BASE_URL")
    or "https://aswini-kumar-data-wrangler-env.hf.space"
)
IMAGE_NAME = os.environ.get("IMAGE_NAME", "data-wrangler-env:latest")
BENCHMARK = "DataWranglerEnv"
TEMPERATURE = 0.3
MAX_TOKENS = 512

TASKS = [
    {"name": "task_1_easy", "max_steps": 30, "max_total_reward": 3.0, "success_threshold": 0.6},
    {"name": "task_2_medium", "max_steps": 50, "max_total_reward": 5.0, "success_threshold": 0.5},
    {"name": "task_3_hard", "max_steps": 80, "max_total_reward": 8.0, "success_threshold": 0.3},
]


# ── 3-Phase System Prompt ────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert data scientist specializing in data cleaning and quality assurance.
You interact with a messy dataset through text commands to diagnose and fix data quality issues.

Available commands:
  DIAGNOSTIC (read-only):
    profile                       - Dataset overview (shape, types, missing %, duplicates)
    profile_column COL            - Detailed stats for one column
    find_missing                  - Missing value counts per column
    find_duplicates [COL1,COL2]   - Find duplicate rows
    find_outliers COL             - Outlier detection (IQR method)
    check_rules                   - Check business rule violations
    history                       - Show operation history / data lineage
    view [N]                      - Show first N rows

  CLEANING (modifies data):
    fill_missing COL STRATEGY     - Fill nulls (mean/median/mode/constant VALUE/forward_fill)
    remove_duplicates [COL1,COL2] - Drop duplicate rows
    fix_dtype COL TYPE            - Cast column type (int/float/str/datetime)
    replace COL OLD NEW           - Replace exact values
    regex_replace COL PATTERN NEW - Regex-based replacement
    standardize COL METHOD        - Normalize formatting (lowercase/uppercase/titlecase/strip)
    remove_rows COL COND VAL      - Remove rows (equals/less_than/greater_than/contains)
    clip COL LOWER UPPER          - Clip numeric values to range
    rename_column OLD NEW         - Rename a column
    drop_column COL               - Remove a column
    sort COL [asc|desc]           - Sort data
    undo                          - Undo last modification

  EVALUATION:
    validate                      - Check current quality score (8 dimensions)
    submit                        - Finalize and get final score (ends episode)

CLEANING WORKFLOW (follow this order):
  Phase 1 — DIAGNOSE: Run profile, find_missing, find_duplicates, check_rules, find_outliers
  Phase 2 — CLEAN (in this order):
    a. Fix data types first (fix_dtype, regex_replace to strip $ or special chars)
    b. Fill missing values (fill_missing with appropriate strategy per column)
    c. Remove duplicate rows (remove_duplicates)
    d. Fix outliers/impossible values (clip, remove_rows for negative where shouldn't be)
    e. Standardize categorical values (standardize, replace for inconsistent labels)
    f. Fix business rule violations (check_rules, then fix each violation)
  Phase 3 — VERIFY: validate to check score, fix remaining issues, then submit

CRITICAL RULES:
- Respond with ONLY the command. No explanations, no markdown, no commentary.
- Do NOT remove legitimate data — some rows that look odd are intentional red herrings.
- A person named "Null" is a real person, not missing data.
- A price of $0.00 may be a legitimate free promotional item.
- Use 'undo' if a command makes the score worse.
- Always validate before submitting.
"""


# ── Lightweight HTTP-based Environment Client ────────────────────────────────

class DataWranglerHTTPClient:
    """Simple HTTP client for the DataWrangler environment."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def reset(self, task: str = "task_1_easy", seed: int = 42) -> Dict[str, Any]:
        """Reset the environment. Returns {observation, reward, done}."""
        r = self.session.post(
            f"{self.base_url}/reset",
            json={"task": task, "seed": seed},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    def step(self, message: str) -> Dict[str, Any]:
        """Execute an action. Returns {observation, reward, done}."""
        r = self.session.post(
            f"{self.base_url}/step",
            json={"action": {"message": message}},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    def health(self) -> bool:
        """Check if the server is up."""
        try:
            r = self.session.get(f"{self.base_url}/health", timeout=10)
            return r.status_code == 200
        except Exception:
            return False

    def close(self):
        """Close the HTTP session."""
        self.session.close()


# ── Logging — MANDATORY PLAIN-TEXT FORMAT ────────────────────────────────────
# Spec: https://openenv.meta.com (sample inference script)
#   [START] task=<task> env=<env> model=<model>
#   [STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
#   [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...>

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
        f"[END] success={success_str} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ── Agent Logic ──────────────────────────────────────────────────────────────

def build_user_prompt(
    step: int,
    phase: str,
    last_response: str,
    last_reward: float,
    history: List[str],
    task_name: str,
    diagnosis_summary: str,
) -> str:
    """Build a context-rich prompt for the LLM."""
    truncated = last_response[:2000] if len(last_response) > 2000 else last_response
    recent = history[-8:] if len(history) > 8 else history
    history_str = "\n".join(recent) if recent else "No actions taken yet."

    prompt = f"Step {step} | Phase: {phase} | Task: {task_name}\n\n"

    if diagnosis_summary:
        prompt += f"Dataset Diagnosis Summary:\n{diagnosis_summary}\n\n"

    prompt += (
        f"Last command result:\n{truncated}\n\n"
        f"Last reward: {last_reward:+.3f}\n\n"
        f"Action history:\n{history_str}\n\n"
        f"What command should I execute next? Reply with ONLY the command."
    )

    return prompt


def get_model_message(
    client: OpenAI,
    step: int,
    phase: str,
    last_response: str,
    last_reward: float,
    history: List[str],
    task_name: str,
    diagnosis_summary: str,
) -> str:
    """Get the next command from the LLM."""
    user_prompt = build_user_prompt(
        step, phase, last_response, last_reward, history, task_name, diagnosis_summary
    )
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
        # Remove any leading/trailing quotes
        text = text.strip("'\"")
        return text if text else "profile"
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "profile"


def determine_phase(step: int, max_steps: int, history: List[str]) -> str:
    """Determine the current phase based on step number and history."""
    # Phase 1: DIAGNOSE (first ~15% of steps)
    diagnose_budget = max(3, int(max_steps * 0.15))
    # Phase 3: VERIFY (last ~10% of steps)
    verify_start = int(max_steps * 0.85)

    if step <= diagnose_budget:
        return "DIAGNOSE"
    elif step >= verify_start:
        return "VERIFY"
    else:
        return "CLEAN"


# ── Main Loop ────────────────────────────────────────────────────────────────

async def run_task(llm_client: OpenAI, env: DataWranglerHTTPClient, task_config: dict) -> float:
    """Run a single task and return the score."""
    task_name = task_config["name"]
    max_steps = task_config["max_steps"]
    max_total_reward = task_config["max_total_reward"]
    success_threshold = task_config["success_threshold"]

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.001
    success = False
    diagnosis_summary = ""

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Reset environment
        result = env.reset(task=task_name, seed=42)
        obs = result.get("observation", {})
        last_response = obs.get("response", "")
        last_reward = 0.0
        last_score = 0.0

        for step in range(1, max_steps + 1):
            if result.get("done", False):
                break

            phase = determine_phase(step, max_steps, history)

            # Get LLM's decision
            message = get_model_message(
                llm_client, step, phase, last_response, last_reward,
                history, task_name, diagnosis_summary
            )

            # Execute
            result = env.step(message)
            obs = result.get("observation", {})

            reward = result.get("reward", 0.0) or 0.0
            done = result.get("done", False)
            error = None

            rewards.append(reward)
            steps_taken = step
            last_response = obs.get("response", "")
            last_reward = reward
            current_score = obs.get("current_score", 0.0)

            # Update diagnosis summary from profile/find_ commands
            cmd_lower = message.strip().lower()
            if cmd_lower in ("profile", "find_missing", "find_duplicates", "check_rules"):
                diagnosis_summary += f"\n--- {message} ---\n{last_response[:500]}\n"

            # Track score improvement
            if "validate" in cmd_lower or "submit" in cmd_lower:
                last_score = current_score

            log_step(step=step, action=message, reward=reward, done=done, error=error)
            history.append(f"Step {step}: '{message}' → reward {reward:+.3f}")

            if done:
                break

        score = sum(rewards) / max_total_reward if max_total_reward > 0 else 0.001
        # Clamp to open interval (0, 1)
        score = max(0.001, min(0.999, score))
        success = score >= success_threshold

    except Exception as e:
        print(f"[DEBUG] Task {task_name} error: {e}", flush=True)
        traceback.print_exc()
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


async def main() -> None:
    """Run all tasks sequentially."""
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = DataWranglerHTTPClient(base_url=ENV_BASE_URL)

    # Verify environment is reachable
    if not env.health():
        print(f"[DEBUG] Warning: Environment at {ENV_BASE_URL} is not reachable", flush=True)

    try:
        scores = {}
        for task_config in TASKS:
            score = await run_task(llm_client, env, task_config)
            scores[task_config["name"]] = score

        print("\n" + "=" * 50, flush=True)
        print("FINAL SCORES:", flush=True)
        for task_name, score in scores.items():
            print(f"  {task_name}: {score:.4f}", flush=True)
        print(f"  Average: {sum(scores.values()) / len(scores):.4f}", flush=True)
        print("=" * 50, flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    asyncio.run(main())
