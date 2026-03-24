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
    
    # ID 및 Data Load 동작 처리
    if id is None:
        id = str(uuid4())

    vision_data = vision_memory.get(id)
    llm_data = llm_memory.get(id)

    try:
        vision_data = pickle.loads(vision_data) if vision_data else {}
    except Exception:
        vision_data = {}

    try:
        llm_data = pickle.loads(llm_data) if llm_data else {}
    except Exception:
        llm_data = {}


    # Redis Dictionary 동작 처리
    if not isinstance(vision_data.get("inputs"), list):
        vision_data["inputs"] = []

    if not isinstance(vision_data.get("outputs"), list):
        vision_data["outputs"] = []

    if not isinstance(llm_data.get("inputs"), list):
        llm_data["inputs"] = []

    if not isinstance(llm_data.get("outputs"), list):
        llm_data["outputs"] = []

    if not isinstance(llm_data.get("symptom"), list):
        llm_data["symptom"] = []


    # Image File 동작 처리
    if file is None:
        return {"success": False, "id": id, "message": "파일 존재하지 않음"}

    try:
        img = await file.read()
    except Exception as e:
        print(f"파일 읽기 실패: {e}")

        return {"success": False, "id": id, "message": "파일 읽기 실패"}

    if not img:
        return {"success": False, "id": id, "message": "빈 파일"}
    
    
    vision_data["inputs"].append(img)
    llm_data["inputs"].append(text)

    vision_memory.set(id, pickle.dumps(vision_data))
    llm_memory.set(id, pickle.dumps(llm_data))

    await predict_vision(id, vision_memory, llm_memory)
    await predict_llm(id, llm_memory)

    return {"success": True, "id": id, "file": file.filename, "prompt": text, "message": "업로드 성공!"}

@app.get("/visionOutputs/{id}")
def get_vision_output(id: str) -> OutputType:
    data = pickle.loads(vision_memory.get(id))

    if not data:
        return {"outputs": []}

    outputs = data.get("outputs", [])

    latest_output = outputs[-1] if outputs else ""
    
    return {"outputs": [latest_output or ""]}

@app.get("/llmOutputs/{id}")
def get_llm_output(id: str) -> OutputType:

    data = pickle.loads(llm_memory.get(id))

    if not data:
        return {"outputs": []}
    
    outputs = data.get("outputs", [])

    latest_output = outputs[-1] if outputs else ""

    return {"outputs": [latest_output or ""]}

