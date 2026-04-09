# ----------------------------------------------------------
# Modules
# ----------------------------------------------------------

from Modules.VisionModules import predict_vision
from Modules.LLMModules import predict_llm
from Modules.TypeVariable import *

import torch
import os

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from uuid import uuid4

import redis
import pickle
import logging

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
                ) -> ResponseType:
    
    # ID 및 Data Load 동작 처리
    if id is None:
        id = str(uuid4())

    try:
        vision_data = vision_memory.get(id)
        llm_data = llm_memory.get(id)
    except Exception as e:
        logging.error(f"Redis 연결 실패: {e}")
        raise HTTPException(status_code=500, detail="서버 오류")

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


    # Image File 없는 경우
    # LLM만 추론
    if file is None:
        llm_data["inputs"].append(text)

        try:
            llm_memory.set(id, pickle.dumps(llm_data))
        except Exception as e:
            logging.error(f"Redis 연결 실패: {e}")
            raise HTTPException(status_code=500, detail="서버 오류")

        await predict_llm(id, llm_memory)

        return {
            "id": id, 
            "file": None, 
            "prompt": text, 
            
            "message": "업로드 성공!"
        }
    
    # Image File 있는 경우
    # Vision Model로 진단명 추출 후 LLM으로 추론
    else:
        try:
            img = await file.read()
        except Exception as e:
            logging.error(f"파일 읽기 실패: {e}")
            raise HTTPException(status_code=422, detail="파일이 유효하지 않음")

        vision_data["inputs"].append(img)
        llm_data["inputs"].append(text)

        try:
            vision_memory.set(id, pickle.dumps(vision_data))
            llm_memory.set(id, pickle.dumps(llm_data))
        except Exception as e:
            logging.error(f"Redis 연결 실패: {e}")
            raise HTTPException(status_code=500, detail="서버 오류")

        await predict_vision(id, vision_memory, llm_memory)
        await predict_llm(id, llm_memory)

        return {
            "id": id, 
            "file": file.filename, 
            "prompt": text, 

            "message": "업로드 성공!"
        }

@app.get("/Outputs/{id}")
def get_outputs(id: str) -> ResponseType:
    try:
        vision_data = vision_memory.get(id)
        llm_data = llm_memory.get(id)
    except Exception as e:
        logging.error(f"Redis 연결 실패: {e}")
        raise HTTPException(status_code=500, detail="서버 오류")

    try:
        vision_data = pickle.loads(vision_data) if vision_data else {}
    except Exception:
        vision_data = {}

    try:
        llm_data = pickle.loads(llm_data) if llm_data else {}
    except Exception:
        llm_data = {}

    vision_outputs = vision_data.get("outputs", [])
    llm_outputs = llm_data.get("outputs", [])

    latest_vision = vision_outputs[-1] if vision_outputs else ""
    latest_llm = llm_outputs[-1] if llm_outputs else ""

    return {
        "vision_output": [latest_vision],
        "llm_output": [latest_llm],
        
        "status": {
            "vision_done": len(vision_outputs) > 0,
            "llm_done": len(llm_outputs) > 0
        }
    }


