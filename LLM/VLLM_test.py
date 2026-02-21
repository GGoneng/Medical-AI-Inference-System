from vllm import LLM, SamplingParams
import os

_BASE_PATH = os.path.dirname(os.path.abspath(__file__))
_MODEL_NAME = os.path.join(_BASE_PATH, "hari-q3-8b-awq")

def main():
    model = LLM(
    _MODEL_NAME,
    gpu_memory_utilization=0.86,
    max_model_len=1024,
    max_num_seqs=1,
    max_num_batched_tokens=1024,
    enforce_eager=True
)

    prompt = '''
    ### Instruction:
    당신은 임상 지식을 갖춘 유능하고 신뢰할 수 있는 한국어 기반 의료 어시스턴트입니다.
    사용자의 질문에 대해 정확하고 신중한 임상 추론을 바탕으로 진단 가능성을 제시해 주세요.
    반드시 환자의 연령, 증상, 검사 결과, 통증 부위 등 모든 단서를 종합적으로 고려하여 추론 과정과 진단명을 제시해야 합니다.
    의학적으로 정확한 용어를 사용하되, 필요하다면 일반인이 이해하기 쉬운 용어도 병행해 설명해 주세요.
    답변은 300토큰 이내로 완결된 문장으로 작성하세요.

    ### Question:
    60세 남성이 복통과 발열을 호소하며 내원하였습니다.
    혈액 검사 결과 백혈구 수치가 상승했고, 우측 하복부 압통이 확인되었습니다.
    가장 가능성이 높은 진단명은 무엇인가요?
    '''.strip()

    sampling_params = SamplingParams(max_tokens=512)

    outputs = model.generate(prompt, sampling_params)
    for output in outputs:
        print(output.outputs[0].text)

if __name__ == "__main__":
    main()
