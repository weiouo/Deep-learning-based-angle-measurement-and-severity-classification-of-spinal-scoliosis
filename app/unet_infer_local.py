
import os
import cv2
import torch
import numpy as np
from model.resunet import ResidualUNet
from torchvision import transforms
import torch.nn.functional as F

# === 使用者設定 ===
image_dir = r"C:\project\Localization_and_Segmentation\app\patch_img\images"
model_path = r"C:\project\Localization_and_Segmentation\app\best.pt"
output_dir = r"C:\project\Localization_and_Segmentation\app\patch_img\pred"
os.makedirs(output_dir, exist_ok=True)

# === 模型設定 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResidualUNet(n_channels=1, n_classes=1)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# === 圖片轉換 ===
transform = transforms.Compose([
    transforms.ToTensor(),
])

def predict(img):
    tensor = transform(img).unsqueeze(0).to(device)  # [1, 1, H, W]
    with torch.no_grad():
        out = model(tensor)
        out = torch.sigmoid(out)
        out = (out > 0.5).float()  # 二值化
        mask = out.squeeze().cpu().numpy() * 255
    return mask.astype(np.uint8)

# === 處理所有圖 ===
for fname in os.listdir(image_dir):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    fpath = os.path.join(image_dir, fname)
    img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("⚠️ 無法讀取圖片：", fpath)
        continue

    pred_mask = predict(img)
    out_path = os.path.join(output_dir, fname)
    cv2.imwrite(out_path, pred_mask)

print("✅ 完成推論結果，儲存於：", output_dir)
