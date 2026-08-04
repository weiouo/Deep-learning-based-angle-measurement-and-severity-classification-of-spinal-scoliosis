import os
import cv2
import torch
import numpy as np
from torchvision import transforms
import sys

# 加入 model 路徑
sys.path.append(r"C:\project\Localization_and_Segmentation\app\VertebraSegmentation\net")
from model.resunet import ResidualUNet

# === 設定 ===
image_dir = r"C:\project\Localization_and_Segmentation\app\patch_img\images"
model_path = r"C:\project\Localization_and_Segmentation\app\best.pt"
output_dir = r"C:\project\Localization_and_Segmentation\app\patch_img\pred"
os.makedirs(output_dir, exist_ok=True)

# === 模型 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResidualUNet(n_channels=1, n_classes=1)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# === 轉換器 ===
transform = transforms.Compose([
    transforms.ToTensor(),
])

def predict(img):
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
        out = torch.sigmoid(out)
        out = (out > 0.5).float()
        mask = out.squeeze().cpu().numpy() * 255
    return mask.astype(np.uint8)

# === 遍歷圖片 ===
for fname in os.listdir(image_dir):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    path = os.path.join(image_dir, fname)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("⚠️ 無法讀取：", fname)
        continue

    mask = predict(img)
    cv2.imwrite(os.path.join(output_dir, fname), mask)

print("✅ 推論完成，結果已儲存於：", output_dir)
