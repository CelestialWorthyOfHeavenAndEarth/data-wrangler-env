"""End-to-end test of the live HF Space."""
import requests

BASE = "https://aswini-kumar-data-wrangler-env.hf.space"

print("=" * 60)
print("DataWranglerEnv - Full End-to-End Live Test")
print("=" * 60)

session = requests.Session()

# 1. Health
r = session.get(f"{BASE}/health")
assert r.json()["status"] == "healthy"
print("1.  Health:        PASS")

# 2. Reset
r = session.post(f"{BASE}/reset", json={"task": "task_1_easy", "seed": 42})
d = r.json()
assert d["done"] == False
assert d["observation"]["step_number"] == 0
print(f"2.  Reset:         PASS  (shape={d['observation']['dataset_shape']})")

# 3. Profile
r = session.post(f"{BASE}/step", json={"action": {"message": "profile"}})
d = r.json()
assert d["observation"]["step_number"] == 1
assert d["reward"] > 0
print(f"3.  Profile:       PASS  (reward={d['reward']})")

# 4. Find missing
r = session.post(f"{BASE}/step", json={"action": {"message": "find_missing"}})
d = r.json()
assert d["observation"]["step_number"] == 2
print(f"4.  Find missing:  PASS  (step={d['observation']['step_number']})")

# 5. Fill missing
r = session.post(f"{BASE}/step", json={"action": {"message": "fill_missing age mean"}})
d = r.json()
assert d["observation"]["step_number"] == 3
assert "Filled" in d["observation"]["response"]
print(f"5.  Fill missing:  PASS  ({d['observation']['response'][:50]}...)")

# 6. Validate
r = session.post(f"{BASE}/step", json={"action": {"message": "validate"}})
d = r.json()
assert d["observation"]["current_score"] > 0
print(f"6.  Validate:      PASS  (score={d['observation']['current_score']:.4f})")

# 7. Submit
r = session.post(f"{BASE}/step", json={"action": {"message": "submit"}})
d = r.json()
assert d["done"] == True
assert d["reward"] > 0
print(f"7.  Submit:        PASS  (final={d['observation']['current_score']:.4f}, reward={d['reward']:.4f})")

print()
print("=" * 60)
print("ALL TESTS PASSED - Environment is fully operational!")
print("=" * 60)
