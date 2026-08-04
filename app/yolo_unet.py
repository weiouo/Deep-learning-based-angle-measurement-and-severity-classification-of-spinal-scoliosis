import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchvision import transforms
import sys

# 匯入 ResidualUNet 模型
sys.path.append(r"C:\project\Localization_and_Segmentation\app\VertebraSegmentation\net")
from model.resunet import ResidualUNet

# === 使用者設定 ===
input_folder = r"C:\project\Localization_and_Segmentation\app\dataset\scol"
output_folder = r"C:\project\Localization_and_Segmentation\app\seg_result\scol"
yolo_weights = r"C:\project\Localization_and_Segmentation\app\PyTorch_YOLOv3\checkpoints\v11_clahe_best.pt"
unet_weights = r"C:\project\Localization_and_Segmentation\app\best.pt"

# 確保輸出資料夾存在
os.makedirs(output_folder, exist_ok=True)

# === 模型載入 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolo = YOLO(yolo_weights)
unet = ResidualUNet(n_channels=1, n_classes=1)
unet.load_state_dict(torch.load(unet_weights, map_location=device))
unet.to(device)
unet.eval()

transform = transforms.Compose([transforms.ToTensor()])

# === 批次處理圖像 ===
for filename in os.listdir(input_folder):
    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    image_path = os.path.join(input_folder, filename)
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        print(f"⚠️ 無法讀取圖像：{filename}")
        continue

    gray_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    rgb_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)

    h, w = gray_img.shape
    full_mask = np.zeros((h, w), dtype=np.uint8)

    # YOLO 偵測
    results = yolo(rgb_img)
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

    for x1, y1, x2, y2 in boxes:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = gray_img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        orig_h, orig_w = crop.shape[:2]
        crop_resized = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_LINEAR)
        patch = transform(crop_resized).unsqueeze(0).to(device)

        with torch.no_grad():
            out = unet(patch)
            out = torch.sigmoid(out)
            pred_mask = (out > 0.5).float().squeeze().cpu().numpy() * 255
            pred_mask = pred_mask.astype(np.uint8)

        pred_mask_resized = cv2.resize(pred_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        full_mask[y1:y2, x1:x2] = np.maximum(full_mask[y1:y2, x1:x2], pred_mask_resized)

    # 儲存結果
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, full_mask)
    print(f"✅ 已儲存：{output_path}")
