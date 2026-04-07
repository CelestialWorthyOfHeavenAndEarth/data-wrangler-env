"""Local unit test - no network needed. Tests the core environment logic."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.data_wrangler_env_environment import DataWranglerEnvironment
from models import DataWranglerAction

results = []

def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    results.append(f"  {status}: {name}" + (f" -> {detail}" if not condition and detail else ""))

results.append("=== LOCAL ENVIRONMENT UNIT TEST ===\n")

# Test 1: Reset task_1_easy
results.append("[1] Reset task_1_easy")
env = DataWranglerEnvironment()
obs = env.reset(task="task_1_easy", seed=42)
check("Reset returns observation", obs is not None)
check("step_number=0", obs.step_number == 0, f"got {obs.step_number}")
check("done=False", obs.done == False)
check("Shape has 58", "58" in obs.dataset_shape, f"got {obs.dataset_shape}")

# Test 2: Step increments
results.append("\n[2] Step increments")
obs2 = env.step(DataWranglerAction(message="profile"))
check("step_number=1", obs2.step_number == 1, f"got {obs2.step_number}")
check("reward>0", obs2.reward > 0, f"got {obs2.reward}")
check("done=False", obs2.done == False)

obs3 = env.step(DataWranglerAction(message="find_missing"))
check("step_number=2", obs3.step_number == 2, f"got {obs3.step_number}")

obs4 = env.step(DataWranglerAction(message="fill_missing age mean"))
check("step_number=3", obs4.step_number == 3, f"got {obs4.step_number}")
check("Filled in response", "Filled" in obs4.response, f"resp: {obs4.response[:60]}")

# Test 3: Validate
results.append("\n[3] Validate")
obs5 = env.step(DataWranglerAction(message="validate"))
check("step_number=4", obs5.step_number == 4, f"got {obs5.step_number}")
check("score>0", obs5.current_score > 0, f"got {obs5.current_score}")

# Test 4: Submit
results.append("\n[4] Submit")
obs6 = env.step(DataWranglerAction(message="submit"))
check("done=True", obs6.done == True)
check("reward>0", obs6.reward > 0, f"got {obs6.reward}")
check("score>0", obs6.current_score > 0, f"got {obs6.current_score}")

# Test 5: Reset task_2_medium
results.append("\n[5] Reset task_2_medium")
try:
    env2 = DataWranglerEnvironment()
    obs_t2 = env2.reset(task="task_2_medium", seed=42)
    check("Task 2 resets", obs_t2 is not None)
    check("Task 2 shape", "rows" in obs_t2.dataset_shape, f"got {obs_t2.dataset_shape}")
    check("Task 2 step=0", obs_t2.step_number == 0, f"got {obs_t2.step_number}")
except Exception as e:
    check("Task 2 reset", False, f"EXCEPTION: {e}")
    import traceback
    results.append(traceback.format_exc())

# Test 6: Reset task_3_hard
results.append("\n[6] Reset task_3_hard")
try:
    env3 = DataWranglerEnvironment()
    obs_t3 = env3.reset(task="task_3_hard", seed=42)
    check("Task 3 resets", obs_t3 is not None)
    check("Task 3 shape", "rows" in obs_t3.dataset_shape, f"got {obs_t3.dataset_shape}")
except Exception as e:
    check("Task 3 reset", False, f"EXCEPTION: {e}")
    import traceback
    results.append(traceback.format_exc())

# Test 7: Task 2 full lifecycle
results.append("\n[7] Task 2 full lifecycle")
try:
    env2b = DataWranglerEnvironment()
    obs = env2b.reset(task="task_2_medium", seed=42)
    obs = env2b.step(DataWranglerAction(message="profile"))
    check("Task 2 profile step=1", obs.step_number == 1, f"got {obs.step_number}")
    obs = env2b.step(DataWranglerAction(message="submit"))
    check("Task 2 submit done", obs.done == True)
    check("Task 2 submit score>0", obs.current_score > 0, f"got {obs.current_score}")
except Exception as e:
    check("Task 2 lifecycle", False, f"EXCEPTION: {e}")
    import traceback
    results.append(traceback.format_exc())

# Summary
passes = [r for r in results if "PASS:" in r]
fails = [r for r in results if "FAIL:" in r]
results.append(f"\n{'='*50}")
results.append(f"RESULTS: {len(passes)} passed, {len(fails)} failed")
if fails:
    results.append("\nFAILURES:")
    for f in fails:
        results.append(f)
else:
    results.append("ALL TESTS PASSED!")

output = "\n".join(results)
with open("local_test_results.txt", "w", encoding="utf-8") as fp:
    fp.write(output)
print(output)
