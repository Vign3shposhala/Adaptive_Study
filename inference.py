import os
import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:7860")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

client = OpenAI()

print("[START]")

state = requests.post(f"{API_BASE_URL}/reset").json()["state"]

total_reward = 0

for step in range(60):
    action = "EASY"

    res = requests.post(
        f"{API_BASE_URL}/step",
        json={"action": action}
    ).json()

    total_reward += res["reward"]

    print(f"[STEP] step={step} reward={res['reward']:.4f}")

    if res["done"]:
        break

score = max(0, min(1, total_reward / 1000))

print(f"[END] final_score={score:.4f}")