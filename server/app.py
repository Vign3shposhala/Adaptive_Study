import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uuid import uuid4
from typing import Optional

from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel

from env import AdaptiveStudyEnv
from agent import smart_policy

# ─── Task definitions ──────────────────────────────────────────────────────────

TASKS = {
    "task1": {
        "task_id": "task1",
        "name": "Basic Mastery",
        "difficulty": "easy",
        "description": "Reach solid mastery (>0.65) using a balanced study policy over 60 steps.",
        "max_steps": 60,
    },
    "task2": {
        "task_id": "task2",
        "name": "Stress Control",
        "difficulty": "medium",
        "description": "Keep final stress below 0.4 while sustaining focus and energy.",
        "max_steps": 60,
    },
    "task3": {
        "task_id": "task3",
        "name": "Peak Performance",
        "difficulty": "hard",
        "description": "Achieve high mastery (>0.75) AND low stress (<0.35) simultaneously.",
        "max_steps": 60,
    },
}

# ─── Episode store ──────────────────────────────────────────────────────────────

_episodes: dict = {}   # episode_id → {"env": AdaptiveStudyEnv, "task_id": str,
                       #                "rewards": list, "done": bool, "final_state": dict}

# ─── Helpers ────────────────────────────────────────────────────────────────────

def _clamp(v: float) -> float:
    return max(0.001, min(0.999, float(v)))


def _score_for_task(task_id: str, final_state: dict, rewards: list) -> float:
    if task_id == "task1":
        return _clamp(final_state["mastery"])
    elif task_id == "task2":
        return _clamp(1.0 - final_state["stress"])
    elif task_id == "task3":
        mastery_ok = max(0.0, final_state["mastery"] - 0.35) / 0.65
        stress_ok  = max(0.0, 0.65 - final_state["stress"]) / 0.65
        return _clamp((mastery_ok + stress_ok) / 2.0)
    avg = sum(rewards) / len(rewards) if rewards else 0.5
    return _clamp(avg)

# ─── Request models ──────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: Optional[str] = None

class ActionRequest(BaseModel):
    action: str
    episode_id: Optional[str] = None

class GraderRequest(BaseModel):
    episode_id: Optional[str] = None
    task_id: Optional[str] = None

# ─── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Adaptive Study Environment", version="0.2.0")

# Shared single-session env (backward-compat with inference.py)
_single_env = AdaptiveStudyEnv()


@app.get("/")
def index():
    return {
        "name": "adaptive-study-env",
        "status": "ok",
        "openenv": True,
        "endpoints": ["/health", "/tasks", "/reset", "/step", "/state", "/grader"],
    }


@app.get("/health")
def health():
    return {"status": "healthy", "service": "adaptive-study-env"}


@app.get("/tasks")
def list_tasks():
    return list(TASKS.values())


@app.post("/reset")
def reset(payload: ResetRequest | None = Body(default=None)):
    task_id = (payload.task_id if payload else None) or "task1"
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail=f"Unknown task_id: {task_id}")

    env = AdaptiveStudyEnv()
    state = env.reset()
    episode_id = str(uuid4())
    _episodes[episode_id] = {
        "env": env,
        "task_id": task_id,
        "rewards": [],
        "done": False,
        "final_state": state,
    }

    # Also reset the shared single-session env
    _single_env.reset()

    return {
        "episode_id": episode_id,
        "task_id": task_id,
        "state": state,
        "task": TASKS[task_id],
    }


@app.post("/step")
def step(req: ActionRequest):
    if req.episode_id and req.episode_id in _episodes:
        ep = _episodes[req.episode_id]
        if ep["done"]:
            raise HTTPException(status_code=400, detail="Episode already done.")
        obs, reward, done, _ = ep["env"].step(req.action)
        ep["rewards"].append(float(reward))
        ep["done"] = done
        ep["final_state"] = obs
        return {
            "episode_id": req.episode_id,
            "state": obs,
            "reward": float(reward),
            "done": done,
        }

    # Fallback: single shared session (keeps old inference.py working)
    obs, reward, done, _ = _single_env.step(req.action)
    return {
        "state": obs,
        "reward": float(reward),
        "done": done,
    }


@app.get("/state")
def get_state():
    try:
        state = _single_env._get_obs()
    except Exception:
        state = {}
    return {"state": state}


@app.post("/grader")
def grader(payload: GraderRequest | None = Body(default=None)):
    """
    Grade a completed episode.
    Accepts episode_id (preferred) or task_id (runs a fresh evaluation).
    Returns score strictly in (0, 1).
    """
    # --- grade by episode_id ---
    if payload and payload.episode_id and payload.episode_id in _episodes:
        ep = _episodes[payload.episode_id]
        score = _score_for_task(ep["task_id"], ep["final_state"], ep["rewards"])
        return {
            "task_id": ep["task_id"],
            "score": score,
            "passed": score >= 0.5,
            "details": {
                "final_state": ep["final_state"],
                "steps": len(ep["rewards"]),
                "avg_reward": sum(ep["rewards"]) / len(ep["rewards"]) if ep["rewards"] else 0,
            },
        }

    # --- grade by running a fresh episode for task_id ---
    task_id = (payload.task_id if payload else None) or "task1"
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail=f"Unknown task_id: {task_id}")

    env = AdaptiveStudyEnv()
    state = env.reset()
    rewards = []
    done = False
    while not done:
        action = smart_policy(state)
        state, reward, done, _ = env.step(action)
        rewards.append(float(reward))

    score = _score_for_task(task_id, state, rewards)
    return {
        "task_id": task_id,
        "score": score,
        "passed": score >= 0.5,
        "details": {
            "final_state": state,
            "steps": len(rewards),
            "avg_reward": sum(rewards) / len(rewards) if rewards else 0,
        },
    }


def main():
    return app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)
