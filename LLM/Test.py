import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def benchmark(model, tokenizer, messages):

    text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True
)
    # tokenize
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    torch.cuda.reset_peak_memory_stats()

    # start
    start = time.time()

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=4096
    )

    end = time.time()

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # output token length
    output_tokens = generated_ids[0].shape[0]

    # latency
    latency = end - start

    # tokens/sec
    tok_per_sec = output_tokens / latency

    # vram
    vram = torch.cuda.max_memory_allocated() / 1024**3

    return latency, tok_per_sec, vram, response


prompt = '''
### Instruction:
당신은 임상 지식을 갖춘 유능하고 신뢰할 수 있는 한국어 기반 의료 어시스턴트입니다.
사용자의 질문에 대해 정확하고 신중한 임상 추론을 바탕으로 진단 가능성을 제시해 주세요.
반드시 환자의 연령, 증상, 검사 결과, 통증 부위 등 모든 단서를 종합적으로 고려하여 추론 과정과 진단명을 제시해야 합니다.
의학적으로 정확한 용어를 사용하되, 필요하다면 일반인이 이해하기 쉬운 용어도 병행해 설명해 주세요.

### Question:
60세 남성이 복통과 발열을 호소하며 내원하였습니다.
혈액 검사 결과 백혈구 수치가 상승했고, 우측 하복부 압통이 확인되었습니다.
가장 가능성이 높은 진단명은 무엇인가요?
'''.strip()

# -------- model 1 --------
model1_name = "snuh/hari-q3-8b"
model1 = AutoModelForCausalLM.from_pretrained(
    model1_name,
    load_in_4bit=True,
    device_map="auto"
)
tok1 = AutoTokenizer.from_pretrained(model1_name)

lat, toksec, vram, response = benchmark(model1, tok1, prompt)

print("\n=== bitsandbytes 4bit ===")
print(f"Latency: {lat:.3f}s")
print(f"Tokens/sec: {toksec:.2f}")
print(f"Peak VRAM: {vram:.2f} GB")
print(f"Response: \n {response}")


# -------- model 2 --------
model2_name = "./hari-q3-8b-awq"
model2 = AutoModelForCausalLM.from_pretrained(
    model2_name,
    device_map="auto"
)
tok2 = AutoTokenizer.from_pretrained(model2_name)

lat, toksec, vram, response = benchmark(model2, tok2, prompt)

print("\n=== AWQ 4bit ===")
print(f"Latency: {lat:.3f}s")
print(f"Tokens/sec: {toksec:.2f}")
print(f"Peak VRAM: {vram:.2f} GB")
print(f"Response: \n {response}")