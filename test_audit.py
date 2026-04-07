"""Comprehensive audit test for DataWranglerEnv - writes results to file."""
import requests
import json

BASE = "https://aswini-kumar-data-wrangler-env.hf.space"
results = []

def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    results.append(f"  {status}: {name}" + (f" -> {detail}" if not condition else ""))

s = requests.Session()

results.append("=" * 60)
results.append("DataWranglerEnv Comprehensive Audit")
results.append("=" * 60)

# 1. Health
results.append("\n[1] Health Check")
try:
    r = s.get(f"{BASE}/health")
    check("Health status 200", r.status_code == 200, f"got {r.status_code}")
    data = r.json()
    check("Health body", data.get("status") == "healthy", f"got {data}")
except Exception as e:
    check("Health", False, str(e))

# 2. Schema
results.append("\n[2] Schema Endpoint")
try:
    r = s.get(f"{BASE}/schema")
    check("Schema status 200", r.status_code == 200, f"got {r.status_code}")
    schema = r.json()
    check("Schema has action", "action" in schema)
    check("Schema has observation", "observation" in schema)
except Exception as e:
    check("Schema", False, str(e))

# 3. Reset
results.append("\n[3] Reset - task_1_easy")
try:
    r = s.post(f"{BASE}/reset", json={"task": "task_1_easy", "seed": 42})
    check("Reset status 200", r.status_code == 200, f"got {r.status_code}")
    d = r.json()
    check("Has observation", "observation" in d, f"keys: {list(d.keys())}")
    check("Has done", "done" in d, f"keys: {list(d.keys())}")
    check("Has reward", "reward" in d, f"keys: {list(d.keys())}")
    check("done=False", d.get("done") == False, f"got {d.get('done')}")
    obs = d.get("observation", {})
    check("Obs has response", "response" in obs)
    check("Obs has step_number", "step_number" in obs)
    check("Obs has dataset_shape", "dataset_shape" in obs)
    check("Obs has current_score", "current_score" in obs)
    check("Obs step=0", obs.get("step_number") == 0, f"got {obs.get('step_number')}")
    check("Shape has 55", "55" in str(obs.get("dataset_shape", "")), f"got {obs.get('dataset_shape')}")
except Exception as e:
    check("Reset", False, str(e))

# 4. Step profile
results.append("\n[4] Step - profile")
try:
    r = s.post(f"{BASE}/step", json={"action": {"message": "profile"}})
    check("Step status 200", r.status_code == 200, f"got {r.status_code}")
    d = r.json()
    check("Has observation", "observation" in d, f"keys: {list(d.keys())}")
    obs = d.get("observation", {})
    check("step_number=1", obs.get("step_number") == 1, f"got {obs.get('step_number')}")
    check("reward>0", (d.get("reward") or 0) > 0, f"got {d.get('reward')}")
    check("done=False", d.get("done") == False)
except Exception as e:
    check("Step profile", False, str(e))

# 5. fill_missing
results.append("\n[5] Step - fill_missing age mean")
try:
    r = s.post(f"{BASE}/step", json={"action": {"message": "fill_missing age mean"}})
    d = r.json()
    obs = d.get("observation", {})
    check("step_number=2", obs.get("step_number") == 2, f"got {obs.get('step_number')}")
    check("Filled in response", "Filled" in obs.get("response", ""), f"resp: {obs.get('response', '')[:80]}")
except Exception as e:
    check("fill_missing", False, str(e))

# 6. validate
results.append("\n[6] Step - validate")
try:
    r = s.post(f"{BASE}/step", json={"action": {"message": "validate"}})
    d = r.json()
    obs = d.get("observation", {})
    check("step_number=3", obs.get("step_number") == 3, f"got {obs.get('step_number')}")
    check("score>0", obs.get("current_score", 0) > 0, f"got {obs.get('current_score')}")
except Exception as e:
    check("validate", False, str(e))

# 7. submit
results.append("\n[7] Step - submit")
try:
    r = s.post(f"{BASE}/step", json={"action": {"message": "submit"}})
    d = r.json()
    obs = d.get("observation", {})
    check("done=True", d.get("done") == True, f"got {d.get('done')}")
    check("reward>0", (d.get("reward") or 0) > 0, f"got {d.get('reward')}")
    check("final score>0", obs.get("current_score", 0) > 0, f"got {obs.get('current_score')}")
except Exception as e:
    check("submit", False, str(e))

# 8. State
results.append("\n[8] State Endpoint")
try:
    r = s.get(f"{BASE}/state")
    check("State 200", r.status_code == 200, f"got {r.status_code}")
    state = r.json()
    check("Has episode_id", "episode_id" in state, f"keys: {list(state.keys())}")
except Exception as e:
    check("State", False, str(e))

# 9. Task 2
results.append("\n[9] Reset - task_2_medium")
try:
    r = s.post(f"{BASE}/reset", json={"task": "task_2_medium", "seed": 42})
    d = r.json()
    obs = d.get("observation", {})
    check("Task 2 done=False", d.get("done") == False)
    check("Task 2 has rows", "rows" in str(obs.get("dataset_shape", "")), f"shape={obs.get('dataset_shape')}")
except Exception as e:
    check("Task 2", False, str(e))

# 10. Task 3
results.append("\n[10] Reset - task_3_hard")
try:
    r = s.post(f"{BASE}/reset", json={"task": "task_3_hard", "seed": 42})
    d = r.json()
    obs = d.get("observation", {})
    check("Task 3 done=False", d.get("done") == False)
    check("Task 3 has rows", "rows" in str(obs.get("dataset_shape", "")), f"shape={obs.get('dataset_shape')}")
except Exception as e:
    check("Task 3", False, str(e))

# Summary
passes = [r for r in results if "PASS:" in r]
fails = [r for r in results if "FAIL:" in r]
results.append("\n" + "=" * 60)
results.append(f"RESULTS: {len(passes)} passed, {len(fails)} failed")
if fails:
    results.append("\nFAILURES:")
    for f in fails:
        results.append(f)
else:
    results.append("\nALL TESTS PASSED!")
results.append("=" * 60)

# Write to file
output = "\n".join(results)
with open("audit_results.txt", "w") as f:
    f.write(output)
print("Audit complete. Results written to audit_results.txt")
