import os
import requests

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:7860")

print("[START]")

state = requests.post(f"{API_BASE_URL}/reset").json()["state"]

total_reward = 0

for step in range(60):
    action = "EASY"

    res = requests.post(
        f"{API_BASE_URL}/step",
        json={"action": action}
    ).json()

    reward = res["reward"]
    done = res["done"]

    total_reward += reward

    print(f"[STEP] step={step} reward={reward:.4f}")

    if done:
        break

score = max(0, min(1, total_reward / 1000))

print(f"[END] final_score={score:.4f}")
