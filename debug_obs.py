import requests
import json

BASE = "https://aswini-kumar-data-wrangler-env.hf.space"
s = requests.Session()

# Reset
s.post(f"{BASE}/reset", json={"task": "task_1_easy", "seed": 42})

# Step
r = s.post(f"{BASE}/step", json={"action": {"message": "profile"}})
d = r.json()
obs = d["observation"]

with open("debug_obs.txt", "w") as f:
    for k in sorted(obs.keys()):
        val = repr(obs[k])
        if len(val) > 100:
            val = val[:100] + "..."
        f.write(f"{k}: {type(obs[k]).__name__} = {val}\n")
    f.write(f"\nTop-level done: {d.get('done')}\n")
    f.write(f"Top-level reward: {d.get('reward')}\n")

    # Also check task_2_medium
    f.write("\n--- Task 2 test ---\n")
    r2 = s.post(f"{BASE}/reset", json={"task": "task_2_medium", "seed": 42})
    f.write(f"Task 2 status: {r2.status_code}\n")
    f.write(f"Task 2 text (first 200): {r2.text[:200]}\n")

print("Done - see debug_obs.txt")
