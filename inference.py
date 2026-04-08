import os
import requests

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:7860")

print("[START]")

total_reward = 0

try:
    res = requests.post(f"{API_BASE_URL}/reset", timeout=5)
    data = res.json() if res.status_code == 200 else {}
    state = data.get("state", {})

    for step in range(60):
        action = "EASY"

        try:
            response = requests.post(
                f"{API_BASE_URL}/step",
                json={"action": action},
                timeout=5
            )

            data = response.json() if response.status_code == 200 else {}

            reward = float(data.get("reward", 0))
            done = data.get("done", False)

        except Exception:
            reward = 0
            done = True

        total_reward += reward
        print(f"[STEP] step={step} reward={reward:.4f}")

        if done:
            break

except Exception:
    print("[STEP] step=0 reward=0.0000")

score = max(0, min(1, total_reward / 1000))

print(f"[END] final_score={score:.4f}")
