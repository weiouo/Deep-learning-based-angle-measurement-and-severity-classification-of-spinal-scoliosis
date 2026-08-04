import os
import json
import numpy as np
import cv2
from PIL import Image
from labelme.utils import shape_to_mask

# 用戶輸入參數（修改這個路徑）
json_path = "your_path_to_json/0001.json"  # 修改為你的 JSON 標註檔案路徑
images_dir = os.path.join(os.path.dirname(json_path), "images")
labels_dir = os.path.join(os.path.dirname(json_path), "labels")
output_dir_img = os.path.join(os.path.dirname(json_path), "patch_img", "images")
output_dir_mask = os.path.join(os.path.dirname(json_path), "patch_img", "masks")

os.makedirs(output_dir_img, exist_ok=True)
os.makedirs(output_dir_mask, exist_ok=True)

# 讀取 JSON
with open(json_path, 'r') as f:
    data = json.load(f)

image_filename = data['imagePath']
image_path = os.path.join(images_dir, image_filename)
image = np.asarray(Image.open(image_path).convert("RGB"))
h, w = image.shape[:2]

# 產生大張 segmentation mask
mask = np.zeros((h, w), dtype=np.uint8)
for shape in data['shapes']:
    poly_mask = shape_to_mask((h, w), shape['points'])
    mask = np.logical_or(mask, poly_mask)

mask = (mask * 255).astype(np.uint8)

# 讀取對應 YOLO 標籤
label_path = os.path.join(labels_dir, os.path.splitext(image_filename)[0] + ".txt")
with open(label_path, "r") as f:
    yolo_lines = f.readlines()

def yolo_to_box(line, img_w, img_h, padding=10):
    cls, x, y, w, h = map(float, line.strip().split())
    x1 = int((x - w / 2) * img_w) - padding
    y1 = int((y - h / 2) * img_h) - padding
    x2 = int((x + w / 2) * img_w) + padding
    y2 = int((y + h / 2) * img_h) + padding
    return max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)

# 開始裁 patch
for idx, line in enumerate(yolo_lines):
    x1, y1, x2, y2 = yolo_to_box(line, w, h, padding=10)
    patch_img = image[y1:y2, x1:x2]
    patch_mask = mask[y1:y2, x1:x2]

    patch_name = os.path.splitext(image_filename)[0] + f"_{idx}.png"
    cv2.imwrite(os.path.join(output_dir_img, patch_name), patch_img)
    cv2.imwrite(os.path.join(output_dir_mask, patch_name), patch_mask)

print("✅ 已完成 patch 裁切，圖片儲存於：")
print(output_dir_img)
print(output_dir_mask)
