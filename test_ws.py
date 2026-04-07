"""Test WebSocket mode - this is what the evaluator actually uses."""
import asyncio
import json
import websockets

BASE = "wss://aswini-kumar-data-wrangler-env.hf.space/ws"

async def test_ws():
    results = []
    try:
        async with websockets.connect(BASE) as ws:
            # Reset
            await ws.send(json.dumps({"type": "reset", "task": "task_1_easy", "seed": 42}))
            resp = json.loads(await ws.recv())
            obs = resp.get("observation", {})
            step = obs.get("step_number", "N/A")
            results.append(f"RESET: step={step}, shape={obs.get('dataset_shape', 'N/A')}")

            # Step 1
            await ws.send(json.dumps({"type": "step", "action": {"message": "profile"}}))
            resp = json.loads(await ws.recv())
            obs = resp.get("observation", {})
            step = obs.get("step_number", "N/A")
            results.append(f"STEP1: step={step}, reward={resp.get('reward', 'N/A')}")

            # Step 2
            await ws.send(json.dumps({"type": "step", "action": {"message": "find_missing"}}))
            resp = json.loads(await ws.recv())
            obs = resp.get("observation", {})
            step = obs.get("step_number", "N/A")
            results.append(f"STEP2: step={step}")

            # Submit
            await ws.send(json.dumps({"type": "step", "action": {"message": "submit"}}))
            resp = json.loads(await ws.recv())
            obs = resp.get("observation", {})
            results.append(f"SUBMIT: done={resp.get('done')}, score={obs.get('current_score', 'N/A')}")

    except Exception as e:
        results.append(f"ERROR: {e}")

    with open("ws_test_results.txt", "w") as f:
        f.write("\n".join(results))
    print("Done - see ws_test_results.txt")

asyncio.run(test_ws())
