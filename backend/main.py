# ----------------------------------------------------------
# Modules
# ----------------------------------------------------------

from Modules.VisionModules import predict_vision
from Modules.LLMModules import predict_llm
from Modules.TypeVariable import *

import torch
import os

from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from uuid import uuid4

import redis
import pickle
import time

from typing import Optional, Dict, Any


# ----------------------------------------------------------
# Variables
# ----------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
VISION_WEIGHTS_PATH = os.path.join(BASE_PATH, "Weights", "vision_weights.pth")

app = FastAPI()

# React 접근 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis DB
vision_memory = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=6379, db=0)
llm_memory = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=6379, db=1)


# ----------------------------------------------------------
# API Endpoints
# ----------------------------------------------------------

@app.post("/upload")
async def upload(id: Optional[str] = Form(None),
                file: Optional[UploadFile] = File(None), 
                text: Optional[str] = Form(None)
                ) -> Dict[str, Any]:
    
    if id is None:
        id = str(uuid4())

    vision_data = vision_memory.get(id)
    llm_data = llm_memory.get(id)
    
    if vision_data:
        vision_data = pickle.loads(vision_data)
    else:
        vision_data = {}

    if llm_data:
        llm_data = pickle.loads(llm_data)
    else:
        llm_data = {}


    if "inputs" not in vision_data:
        vision_data["inputs"] = []
    
    if "outputs" not in vision_data:
        vision_data["outputs"] = []

    if "inputs" not in llm_data:
        llm_data["inputs"] = []

    if "outputs" not in llm_data:
        llm_data["outputs"] = []

    if "symptom" not in llm_data:
        llm_data["symptom"] = []
    

    img = await file.read()

    vision_data["inputs"].append(img)

    vision_memory.set(id, pickle.dumps(vision_data))
    llm_memory.set(id, pickle.dumps(llm_data))

    await predict_vision(id, vision_memory, llm_memory)
    await predict_llm(id, llm_memory)

    return {"id": id, "file": file.filename, "prompt": text, "message": "업로드 성공!"}

@app.get("/visionOutputs/{id}")
def get_vision_output(id: str) -> VisionOutputType:
    time.sleep(3)
    data = pickle.loads(vision_memory.get(id))

    if not data:
        return {"outputs": []}

    outputs = data.get("outputs", [])

    latest_output = outputs[-1] if outputs else ""
    
    return {"outputs": [latest_output or ""]}

@app.get("/llmOutputs/{id}")
def get_llm_output(id: str):

    data = pickle.loads(llm_memory.get(id))

    if not data:
        return {"outputs": []}
    
    outputs = data.get("outputs", [])

    latest_output = outputs[-1] if outputs else ""

    return {"outputs": [latest_output or ""]}

