from fastapi import FastAPI
from pydantic import BaseModel
from environment import CodeDebugEnv

app = FastAPI()
env = CodeDebugEnv()

class Action(BaseModel):
    fixed_code: str

@app.post("/reset")
def reset():
    return env.reset()

@app.post("/step")
def step(action: Action):
    return env.step({"fixed_code": action.fixed_code})