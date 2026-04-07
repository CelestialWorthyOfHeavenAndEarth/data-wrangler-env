"""Quick test of the rewritten inference.py HTTP client."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference import DataWranglerHTTPClient, ENV_BASE_URL

out = []

out.append(f"ENV_BASE_URL: {ENV_BASE_URL}")

env = DataWranglerHTTPClient(base_url=ENV_BASE_URL)
out.append(f"Health: {env.health()}")

result = env.reset(task="task_1_easy", seed=42)
out.append(f"Reset done: {result.get('done')}")

result = env.step("profile")
out.append(f"Step reward: {result.get('reward')}")

result = env.step("submit")
out.append(f"Submit done: {result.get('done')} reward: {result.get('reward')}")

# Task 2
result = env.reset(task="task_2_medium", seed=42)
out.append(f"Task2 done: {result.get('done')}")

# Task 3
result = env.reset(task="task_3_hard", seed=42)
out.append(f"Task3 done: {result.get('done')}")

env.close()
out.append("ALL OK")

with open("inference_test_result.txt", "w") as f:
    f.write("\n".join(out))
print("\n".join(out))
