
import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchvision import transforms
import torch.nn.functional as F
import sys

# 匯入 ResidualUNet 模型
sys.path.append(r"C:\project\Localization_and_Segmentation\app\VertebraSegmentation\net")
from model.resunet import ResidualUNet

# === 使用者設定 ===
image_path = r"C:\project\Localization_and_Segmentation\test.jpg"
yolo_weights = r"C:\project\Localization_and_Segmentation\yolov11.pt"
unet_weights = r"C:\project\Localization_and_Segmentation\best.pt"
output_path = r"C:\project\Localization_and_Segmentation\seg_output.png"

# === 模型載入 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

yolo = YOLO(yolo_weights)
unet = ResidualUNet(n_channels=1, n_classes=1)
unet.load_state_dict(torch.load(unet_weights, map_location=device))
unet.to(device)
unet.eval()

# === 預處理 ===
transform = transforms.Compose([transforms.ToTensor()])

# === 載入圖像 ===
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError(f"無法讀取圖像：{image_path}")
h, w = image.shape
full_mask = np.zeros((h, w), dtype=np.uint8)

# === YOLOv11 偵測 ===
results = yolo(image)
boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

# === 對每個框做 segmentation 並合併結果 ===
for i, (x1, y1, x2, y2) in enumerate(boxes):
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        continue

    patch = transform(crop).unsqueeze(0).to(device)  # [1, 1, H, W]
    with torch.no_grad():
        out = unet(patch)
        out = torch.sigmoid(out)
        pred_mask = (out > 0.5).float().squeeze().cpu().numpy() * 255
        pred_mask = pred_mask.astype(np.uint8)

    # 放回全圖對應區域
    mask_resized = cv2.resize(pred_mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    full_mask[y1:y2, x1:x2] = np.maximum(full_mask[y1:y2, x1:x2], mask_resized)

# === 儲存結果 ===
cv2.imwrite(output_path, full_mask)
print(f"✅ segmentation mask 已儲存：{output_path}")
