# -----------------------------------------------------------------------------------
# 파일명       : ONNX_Transform.py
# 설명         : Image Segmentation(U-Net) 모델 경량화를 위한 ONNX 변환  
# 작성자       : 이민하
# 작성일       : 2026-04-10
# -----------------------------------------------------------------------------------
# >> 주요 기능
# - Image Segmentation 모델의 ONNX 변환
# - 기존 모델과의 성능 비교
#
# >> 성능 
# 0 → mean: 0.000402, max: 0.002815, pixel mismatch: 0.000000
# 1 → mean: 0.000412, max: 0.003423, pixel mismatch: 0.000000
# 2 → mean: 0.000432, max: 0.007195, pixel mismatch: 0.000000
# 3 → mean: 0.000415, max: 0.003870, pixel mismatch: 0.000000
# 4 → mean: 0.000409, max: 0.002540, pixel mismatch: 0.000000
# 5 → mean: 0.000412, max: 0.003342, pixel mismatch: 0.000000
# 6 → mean: 0.000409, max: 0.003719, pixel mismatch: 0.000000
# 7 → mean: 0.000416, max: 0.002794, pixel mismatch: 0.000000
# 8 → mean: 0.000414, max: 0.003315, pixel mismatch: 0.000000
# 9 → mean: 0.000419, max: 0.003700, pixel mismatch: 0.000000
# -----------------------------------------------------------------------------------


from XRaySegModules import *

import os

import torch

import onnx
import onnxruntime as ort
import numpy as np

import albumentations as A

from PIL import Image

# Config 파일 불러오기
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_PATH, "tuning_config.yaml")

try:
    config = load_config(CONFIG_PATH)

except FileNotFoundError as e:
    raise FileNotFoundError(
        f"\nCheck an inputted config path."
    ) from e

except TypeError as e:
    raise TypeError(
        f"\nPath must be a string type."
    ) from e


# 실험 조건 고정
SEED = config["parameters"]["seed"]
set_seed(SEED)

# HyperParameter 설정
NUM_CLASSES = config["parameters"]["num_classes"]
DEVICE = config["parameters"]["device"]

# 모델 불러오기
MODEL_NAME = config["model"]
model = load_model(MODEL_NAME, NUM_CLASSES, DEVICE)

# 가중치 불러오기
WEIGHTS = config["path"]["weights"]
original_model = load_weights(model, WEIGHTS, DEVICE)

# ONNX export
IMG_SIZE = config["parameters"]["size"]
dummy_input = torch.randn(1, 1, IMG_SIZE, IMG_SIZE).to(DEVICE)

SAVE_PATH = config["path"]["save"]

torch.onnx.export(
    original_model,
    dummy_input,
    SAVE_PATH + "/pediatric_x_ray_segmentation.onnx",
    input_names=["input"],
    verbose=True,
)

onnx_model = onnx.load(SAVE_PATH + "/pediatric_x_ray_segmentation.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX model is valid")

# 실제 데이터 불러오기
DATA_PATH = BASE_PATH + "/dataset"
data_list = []

for file in os.listdir(DATA_PATH):
    data_list.append(os.path.join(DATA_PATH, file))

transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.pytorch.ToTensorV2()
])


# 원본과의 성능 비교
ort_session = ort.InferenceSession(SAVE_PATH + "/pediatric_x_ray_segmentation.onnx")

input_name = ort_session.get_inputs()[0].name

for i, img_path in enumerate(data_list[:10]): 
    img = Image.open(img_path).convert('L')
    img = np.array(img, dtype=np.float32) / 255.0

    aug = transform(image=img)
    img_tensor = aug["image"].unsqueeze(0).to(DEVICE)

    # PyTorch 추론
    with torch.no_grad():
        torch_output = original_model(img_tensor).cpu().numpy()

    # ONNX 추론
    onnx_output = ort_session.run(
        None,
        {input_name: img_tensor.cpu().numpy()}
    )[0]

    diff = np.abs(torch_output - onnx_output)

    torch_pred = torch_output.argmax(axis=1)
    onnx_pred = onnx_output.argmax(axis=1)

    pixel_diff = np.mean(torch_pred != onnx_pred)

    print(f"{i} → mean: {diff.mean():.6f}, max: {diff.max():.6f}, pixel mismatch: {pixel_diff:.6f}")