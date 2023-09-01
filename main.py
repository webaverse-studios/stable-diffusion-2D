import traceback
import asyncio
import os
from dataclasses import dataclass
from fastapi import FastAPI, Request, BackgroundTasks
import logging
import os
from time import time
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from cog import BasePredictor, BaseModel, File, Input, Path
from run import run_predict

class PredictionData(BaseModel):
    input: Path = None
    prompts: str = "blue house: fire cathedral   "
    strength: float = 0.85
    guidance_scale: float = 7.5
    split : str = None
    req_type: str = "asset"
    negative_prompt: str = "base, ground, terrain, child's drawing, sillhouette, dark, shadowed, green blob, cast shadow on the ground, background pattern"
    num_inference_steps: int = 20
    sd_seed:int = 1000
    width:int = 512
    height:int = 512
    model: str = "furniture"    


EXPERIMENTS_BASE_DIR = "/experiments/"
QUERY_BUFFER = {}

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

loop = asyncio.get_event_loop()

@app.post("/gen")
async def x(request: Request,background_tasks: BackgroundTasks, data: PredictionData):
    return run_predict(data.input, data.prompts, data.strength, data.guidance_scale, data.split, data.req_type, data.negative_prompt, data.num_inference_steps, data.sd_seed, data.width, data.height, data.model)




@app.get("/backlog/")
def return_backlog():
    return {f"return_backlog - Currently {len(QUERY_BUFFER)} jobs in the backlog."}


if __name__ == "__main__":
    import uvicorn
    print('Starting fast-api on 5001')
    uvicorn.run("__main__:app", host="0.0.0.0", port=5000, workers=8)
