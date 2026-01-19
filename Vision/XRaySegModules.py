import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset

from PIL import Image, ImageDraw

import os
import numpy as np



class XRayDataset(Dataset):
    def __init__(self, img_path: , json_list, transform=None):
        self.img_path = img_path
        self.label = json_list
        self.transform = transform

    def __len__(self):
        return len(self.img_path)
    
    def __getitem__(self, idx):
        img_path = self.img_path[idx]
        item = self.label[idx]

        img = Image.open(img_path).convert('L')
        mask = Image.new('L', img.size, 0)
        draw = ImageDraw.Draw(mask)

        for shape in item["shapes"]:
            points = [tuple(point) for point in shape["points"]]
            class_id = shape["class"]
            draw.polygon(points, fill=class_id)
        
        img = np.array(img, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.uint8)


        if self.transform:
            augment = self.transform(image=img, mask=mask)
            img = augment["image"]
            mask = augment["mask"].long()

        return img, mask



class Conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
    
    def forward(self, input):
        return self.conv(input)
    
class Expand(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.conv = Conv(in_ch, out_ch)

    def forward(self, input, skip):
        x = self.up(input)
        x = torch.cat((x, skip), dim=1)
        x = self.conv(x)

        return x

class OriginUNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.encoder1 = Conv(1, 64)
        self.encoder2 = Conv(64, 128)
        self.encoder3 = Conv(128, 256)
        self.encoder4 = Conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)

        self.bottleneck = Conv(512, 1024)

        self.decoder1 = Expand(1024, 512)
        self.decoder2 = Expand(512, 256)
        self.decoder3 = Expand(256, 128)
        self.decoder4 = Expand(128, 64)

        self.output = nn.Conv2d(64, num_classes, 1)

        self.dropout = nn.Dropout2d(0.2)

    def forward(self, input):
        in1 = self.encoder1(input)
        in2 = self.encoder2(self.maxpool(in1))
        in3 = self.encoder3(self.maxpool(in2))
        in4 = self.encoder4(self.maxpool(in3))

        bn = self.bottleneck(self.dropout(self.maxpool(in4)))

        out1 = self.decoder1(bn, in4)
        out2 = self.decoder2(out1, in3)
        out3 = self.decoder3(out2, in2)
        out4 = self.decoder4(out3, in1)

        final_output = self.output(out4)

        return final_output


def dice_coefficient(pred, target, smooth=1):
    num_classes = pred.shape[1]
    pred = F.softmax(pred, dim=1) 
    
    target_onehot = F.one_hot(target, num_classes=num_classes)     # [B, H, W, C]
    target_onehot = target_onehot.permute(0, 3, 1, 2).float()
    
    dice_scores = []
    
    for i in range(num_classes):
        intersection = (pred[:, i] * target_onehot[:, i]).sum()
        union = pred[:, i].sum() + target_onehot[:, i].sum()
        dice = (2 * intersection + smooth) / (union + smooth)
        dice_scores.append(dice)

    return torch.mean(torch.stack(dice_scores)).item()


def multiclass_dice_loss(pred, target):
    num_classes = pred.shape[1]

    dice_coef = dice_coefficient(pred, target)

    return 1 - dice_coef / num_classes 



class CustomWeightedLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.class_weights = torch.tensor([0.1, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32, device=self.device)
        self.CELoss = nn.CrossEntropyLoss(weight=self.class_weights)

    def forward(self, pred, target):
        ce_loss = self.CELoss(pred, target)

        target_onehot = F.one_hot(target, num_classes=pred.shape[1])  # [B, H, W, C]
        target_onehot = target_onehot.permute(0, 3, 1, 2).float()  # [B, C, H, W]

        dice_loss = multiclass_dice_loss(pred, target_onehot)

        return ce_loss + 2 * dice_loss
    

# 모델 Test 함수
def testing(model, valDL, data_size, loss_fn, device):
    # Dropout, BatchNorm 등 가중치 규제 비활성화
    model.eval()

    loss_total, score_total = 0, 0

    with torch.no_grad():
        # Val DataLoader에 저장된 Feature, Target 텐서로 학습 진행
        for featureTS, targetTS in valDL:
            featureTS, targetTS = featureTS.to(device), targetTS.to(device)

            batch_size = len(targetTS)
    
            pre_val = model(featureTS)
            
            loss = loss_fn(pre_val, targetTS).to(device)

            # DICE Score 확인 (클래스 객체와 예측값의 면적 비교)
            score = dice_coefficient(pre_val, targetTS)

            loss_total += loss.item() * batch_size
            score_total += score * batch_size

    avg_loss = loss_total / data_size
    avg_score = score_total / data_size

    return avg_loss, avg_score


def training(model, trainDL, valDL, optimizer, epoch, 
             data_size, val_data_size, loss_fn,
             scheduler, device):
    # 가중치 파일 저장 위치 정의
    SAVE_PATH = './saved_models'
    os.makedirs(SAVE_PATH, exist_ok=True)

    # Early Stopping을 위한 변수
    BREAK_CNT_LOSS = 0
    LIMIT_VALUE = 50

    # Loss가 더 낮은 가중치 파일을 저장하기 위하여 Loss 로그를 담을 리스트
    LOSS_HISTORY, SCORE_HISTORY = [[], []], [[], []]

    for count in range(1, epoch  + 1):
        # GPU 환경에서 training과 testing을 반복하므로 eval 모드 -> train 모드로 전환
        # testing에서는 train 모드 -> eval 모드
        model.train()

        SAVE_WEIGHT = os.path.join(SAVE_PATH, f"best_model_weights.pth")

        loss_total, score_total = 0, 0

        # Train DataLoader에 저장된 Feature, Target 텐서로 학습 진행
        for featureTS, targetTS in trainDL:
            featureTS, targetTS = featureTS.to(device), targetTS.to(device)

            batch_size = len(targetTS)

            # 결과 추론
            pre_val = model(featureTS)

            # 추론값으로 Loss값 계산
            loss = loss_fn(pre_val, targetTS).to(device)

            # DICE Score 확인 (클래스 객체와 예측값의 면적 비교)
            score = dice_coefficient(pre_val, targetTS)

            loss_total += loss.item() * batch_size
            score_total += score * batch_size

            # 이전 gradient 초기화
            optimizer.zero_grad()

            # 역전파로 gradient 계산
            loss.backward()

            # 계산된 gradient로 가중치 업데이트
            optimizer.step()
        
        # Val Loss, Score 계산
        val_loss, val_score = testing(model, valDL, val_data_size, loss_fn, device)

        LOSS_HISTORY[0].append(loss_total / data_size)
        SCORE_HISTORY[0].append(score_total / data_size)

        LOSS_HISTORY[1].append(val_loss)
        SCORE_HISTORY[1].append(val_score)

        print(f"[{count} / {epoch}]\n - TRAIN LOSS : {LOSS_HISTORY[0][-1]}")
        print(f"- TRAIN DICE SCORE : {SCORE_HISTORY[0][-1]}")

        print(f"\n - TEST LOSS : {LOSS_HISTORY[1][-1]}")
        print(f"- TEST DICE SCORE : {SCORE_HISTORY[1][-1]}")

        # Val Score 결과로 스케줄러 업데이트
        scheduler.step(val_loss)

        # Early Stopping 구현
        if len(LOSS_HISTORY[1]) >= 2:
            if LOSS_HISTORY[1][-1] >= LOSS_HISTORY[1][-2]: BREAK_CNT_LOSS += 1
        
        if len(LOSS_HISTORY[1]) == 1:
            torch.save(model.state_dict(), SAVE_WEIGHT)

        else:
            if LOSS_HISTORY[1][-1] < min(LOSS_HISTORY[1][:-1]):
                torch.save(model.state_dict(), SAVE_WEIGHT)

        if BREAK_CNT_LOSS > LIMIT_VALUE:
            print(f"성능 및 손실 개선이 없어서 {count} EPOCH에 학습 중단")
            break
    
    return LOSS_HISTORY, SCORE_HISTORY