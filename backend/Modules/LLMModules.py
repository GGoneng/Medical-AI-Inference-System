# ----------------------------------------------------------
# Modules
# ----------------------------------------------------------

from Modules.TypeVariable import *

from langchain_core.prompts.prompt import PromptTemplate
from langchain_community.llms.vllm import VLLM
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda

from transformers import AutoTokenizer

import redis
import pickle
import os 
import asyncio

# ----------------------------------------------------------
# Internal Variables (do not call externally)
# ----------------------------------------------------------
_BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ----------------------------------------------------------
# External Variables (can be called from outside)
# ----------------------------------------------------------

model_name = "/models/hari-q3-8b-awq"
tokenizer_name = "Models/hari-q3-8b-awq"

tokenizer_path = os.path.join(_BASE_PATH, tokenizer_name)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

openai_api_key = "EMPTY"
openai_api_base = "http://vllm:8000/v1"

max_tokens = 512 

llm = ChatOpenAI(
    base_url=openai_api_base,
    api_key=openai_api_key,
    model=model_name,
    max_tokens=max_tokens
)


# ----------------------------------------------------------
# Internal Classes (do not call externally)
# ----------------------------------------------------------


# ----------------------------------------------------------
# External Classes (can be called from outside)
# ----------------------------------------------------------

class PromptBuilder:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def build(self, messages):
        prompt_template = self.tokenizer.apply_chat_template(
            messages,
            tokenize = False,
            add_generation_prompt = True,
            enable_thinking = False
        )

        return prompt_template

# ----------------------------------------------------------
# Internal Functions (do not call externally)
# ----------------------------------------------------------

def _build_messages(question: str):
    return [
        {
            "role": "system",
            "content": '''
                당신은 임상 지식을 갖춘 유능하고 신뢰할 수 있는 한국어 기반 의료 어시스턴트입니다.
                사용자의 질문에 대해 정확하고 신중한 임상 추론을 바탕으로 진단 가능성을 제시해 주세요.
                반드시 환자의 연령, 증상, 검사 결과, 통증 부위 등 모든 단서를 종합적으로 고려하여 추론 과정과 진단명을 제시해야 합니다.
                의학적으로 정확한 용어를 사용하되, 필요하다면 일반인이 이해하기 쉬운 용어도 병행해 최대 200 토큰으로 설명해 주세요.
            '''.strip()
        },
        {
            "role": "user",
            "content": question.strip()
        }
    ]



# ----------------------------------------------------------
# External Functions (can be called from outside)
# ----------------------------------------------------------

async def predict_llm(id: str, llm_memory: redis.Redis) -> ResponseType:
    llm_data = pickle.loads(llm_memory.get(id))
    question = llm_data["inputs"][-1] if llm_data["inputs"] else None
    symptom = llm_data["symptom"][-1] if llm_data["symptom"] else None

    if symptom:
        messages = _build_messages(symptom)
        llm.temperature = 0.1
        llm.top_p = 1.0
    
    elif question:
        messages = _build_messages(question)
        llm.temperature = 0.5
        llm.top_p = 0.8

    else:
        return {"id": id, "llm_result": "Text Data가 없습니다."}
    
    prompt_builder = PromptBuilder(tokenizer)

    result = await llm.ainvoke(prompt_builder.build(messages))

    llm_data["outputs"].append(result.content)
    llm_memory.set(id, pickle.dumps(llm_data))

    print(result)

    return {"id": id, "llm_result": "LLM 모델 추론 성공!"}