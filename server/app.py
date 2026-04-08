from fastapi import FastAPI
from pydantic import BaseModel
from env import AdaptiveStudyEnv

app = FastAPI()

env = AdaptiveStudyEnv()


# 🔹 Request schema
class ActionRequest(BaseModel):
    action: str


# 🔹 Reset endpoint
@app.post("/reset")
def reset():
    state = env.reset()
    return {
        "state": state
    }


# 🔹 Step endpoint
@app.post("/step")
def step(req: ActionRequest):
    obs, reward, done, _ = env.step(req.action)

    return {
        "state": obs,
        "reward": float(reward),
        "done": done
    }


# 🔹 State endpoint
@app.get("/state")
def get_state():
    return {
        "state": env.state()
    }


# 🔥 REQUIRED FOR VALIDATOR (VERY IMPORTANT)
def main():
    return app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)
