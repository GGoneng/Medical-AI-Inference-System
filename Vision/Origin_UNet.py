# -----------------------------------------------------------------------------------
# 파일명       : Origin_UNet.py
# 설명         : Image Segmentation(U-Net)을 통한 소아 복부 질환 탐지       
# 작성자       : 이민하
# 작성일       : 2025-08-26
# -----------------------------------------------------------------------------------
# >> 주요 기능
# - 정상 1, 비정상 4가지의 상황을 Image Segmentation을 통해 구별
# - Computer Vision 분야의 U-Net 구조를 직접 커스텀하여 구현 
#
# >> 성능
# Train Multi-Class Dice Score : 0.898 -> 0.908
# Test Multi-Class Dice Score : 0.883 -> 0.912
# -----------------------------------------------------------------------------------


import os
import json

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from torch.utils.data import DataLoader

import albumentations as A

from functools import reduce

from XRaySegModules import *

# Config 파일 불러오기
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_PATH, "config.yaml")

config = load_config(CONFIG_PATH)

# 데이터 경로 설정
TRAIN_DATA_PATH = config["path"]["train"]["source"]
TRAIN_LABEL_PATH = config["path"]["train"]["label"]

VAL_DATA_PATH = config["path"]["val"]["source"]
VAL_LABEL_PATH = config["path"]["val"]["label"]

TEST_DATA_PATH = config["path"]["test"]["source"]
TEST_LABEL_PATH = config["path"]["test"]["label"]

# Training 데이터 준비
folder_list = []
label_file_list = []
label_list = []

for folder in os.listdir(TRAIN_LABEL_PATH):
    folder_list.append(os.path.join(TRAIN_LABEL_PATH, folder))

for path in folder_list:
    for file_name in os.listdir(path):
        label_file_list.append(os.path.join(path, file_name))

for file in label_file_list:
    with open(file, "r", encoding="utf-8") as f:
        label_list.append(json.load(f))


# Validation 데이터 준비
val_folder_list = []
val_label_file_list = []
val_label_list = []

for folder in os.listdir(VAL_LABEL_PATH):
    val_folder_list.append(os.path.join(VAL_LABEL_PATH, folder))

for path in val_folder_list:
    for file_name in os.listdir(path):
        val_label_file_list.append(os.path.join(path, file_name))

for file in val_label_file_list:
    with open(file, "r", encoding="utf-8") as f:
        val_label_list.append(json.load(f))


# Test 데이터 준비
test_folder_list = []
test_label_file_list = []
test_label_list = []

for folder in os.listdir(TEST_LABEL_PATH):
    test_folder_list.append(os.path.join(TEST_LABEL_PATH, folder))

for path in test_folder_list:
    for file_name in os.listdir(path):
        test_label_file_list.append(os.path.join(path, file_name))

for file in test_label_file_list:
    with open(file, "r", encoding="utf-8") as f:
        test_label_list.append(json.load(f))


replace_dict = {"Labeling_Data": "Source_Data", ".json": ".png"}

train_file_list = [reduce(lambda x, y: x.replace(*y), replace_dict.items(), file) for file in label_file_list]
val_file_list = [reduce(lambda x, y: x.replace(*y), replace_dict.items(), file) for file in val_label_file_list]
test_file_list = [reduce(lambda x, y: x.replace(*y), replace_dict.items(), file) for file in test_label_file_list]

transform = A.Compose([
    A.Resize(512, 512),
    A.pytorch.ToTensorV2()
])

BATCH_SIZE = config["parameters"]["batch_size"]

trainDS = XRayDataset(train_file_list, label_list, transform)
trainDL = DataLoader(trainDS, batch_size=BATCH_SIZE, shuffle=True)

valDS = XRayDataset(val_file_list, val_label_list, transform)
valDL = DataLoader(valDS, batch_size=BATCH_SIZE)

testDS = XRayDataset(test_file_list, test_label_list, transform)
testDL = DataLoader(testDS, batch_size=BATCH_SIZE)

SEED = config["parameters"]["seed"]
EPOCH = config["parameters"]["epochs"]
LR = config["parameters"]["learning_rate"]

NUM_CLASSES = config["parameters"]["num_classes"]
DEVICE = config["parameters"]["device"]

PATIENCE = config["parameters"]["patience"]
THRESHOLD = config["parameters"]["threshold"]

optimizer = config["parameters"]["optimizer"].lower()

model = load_model("SegmentationUNe", 5, "cuda")

