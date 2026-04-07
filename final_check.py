import requests
import json

BASE = "https://aswini-kumar-data-wrangler-env.hf.space"
s = requests.Session()
out = []

# Health
r = s.get(f"{BASE}/health")
out.append(f"HEALTH: {r.status_code} {r.json()}")

# Reset task 1
r = s.post(f"{BASE}/reset", json={"task": "task_1_easy", "seed": 42})
out.append(f"RESET_T1: {r.status_code} done={r.json().get('done')}")

# Step
r = s.post(f"{BASE}/step", json={"action": {"message": "profile"}})
out.append(f"STEP_PROFILE: {r.status_code} reward={r.json().get('reward')}")

# Submit
r = s.post(f"{BASE}/step", json={"action": {"message": "submit"}})
d = r.json()
out.append(f"SUBMIT: {r.status_code} done={d.get('done')} reward={d.get('reward')}")

# Reset task 2
r = s.post(f"{BASE}/reset", json={"task": "task_2_medium", "seed": 42})
out.append(f"RESET_T2: {r.status_code} done={r.json().get('done')}")

# Reset task 3
r = s.post(f"{BASE}/reset", json={"task": "task_3_hard", "seed": 42})
out.append(f"RESET_T3: {r.status_code} done={r.json().get('done')}")

# Schema
r = s.get(f"{BASE}/schema")
keys = list(r.json().keys())
out.append(f"SCHEMA: {r.status_code} keys={keys}")

# State
r = s.get(f"{BASE}/state")
out.append(f"STATE: {r.status_code}")

result = "\n".join(out)
with open("final_check.log", "w") as f:
    f.write(result)
print(result)
