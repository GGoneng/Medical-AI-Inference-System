from vllm import LLM, SamplingParams
import os
from transformers import AutoTokenizer

_BASE_PATH = os.path.dirname(os.path.abspath(__file__))
_MODEL_NAME = os.path.join(_BASE_PATH, "hari-q3-8b-awq")
_QUESTION_PATH = os.path.join(_BASE_PATH, "question.txt")
_ANSWER_PATH = os.path.join(_BASE_PATH, "answer.txt")

def main():
    model = LLM(
        _MODEL_NAME,
        gpu_memory_utilization=0.86,
        max_model_len=1024,
        max_num_seqs=1,
        max_num_batched_tokens=1024,
        enforce_eager=True
    )

    tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)

    with open(_QUESTION_PATH, "r", encoding="utf-8") as f:
        question_list = f.readlines()

    temperature_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    top_p_list = [0.8, 0.9, 1.0]

    with open(_ANSWER_PATH, "w", encoding="utf-8") as f:
        for temperature in temperature_list:
            for top_p in top_p_list:
                f.write(f"===== Temperature: {temperature}, Top-p: {top_p} =====\n\n")

                for question in question_list:
                    messages = [
                        {"role": "system", "content": '''
                        당신은 임상 지식을 갖춘 유능하고 신뢰할 수 있는 한국어 기반 의료 어시스턴트입니다.
                        사용자의 질문에 대해 정확하고 신중한 임상 추론을 바탕으로 진단 가능성을 제시해 주세요.
                        반드시 환자의 연령, 증상, 검사 결과, 통증 부위 등 모든 단서를 종합적으로 고려하여 추론 과정과 진단명을 제시해야 합니다.
                        의학적으로 정확한 용어를 사용하되, 필요하다면 일반인이 이해하기 쉬운 용어도 병행해 최대 200 토큰으로 설명해 주세요.
                        '''.strip()},
                        {"role": "user", "content": question.strip()}
                    ]

                    prompt_text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False
                    )


                    sampling_params = SamplingParams(
                                    max_tokens=512,
                                    # min_tokens=64,
                                    temperature=temperature,
                                    top_p=top_p,
                                )

                    outputs = model.generate([prompt_text], sampling_params)
                    for output in outputs:
                        f.write(output.outputs[0].text.strip() + "\n\n\n\n")

if __name__ == "__main__":
    main()
