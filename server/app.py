from fastapi import FastAPI
from pydantic import BaseModel
from env import AdaptiveStudyEnv

app = FastAPI()
env = AdaptiveStudyEnv()

class ActionRequest(BaseModel):
    action: str

@app.post("/reset")
def reset():
    return {"state": env.reset()}

@app.post("/step")
def step(req: ActionRequest):
    state, reward, done, info = env.step(req.action)
    return {"state": state, "reward": reward, "done": done, "info": info}

@app.get("/state")
def state():
    return {"state": env.state()}

def main():
    return app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)
