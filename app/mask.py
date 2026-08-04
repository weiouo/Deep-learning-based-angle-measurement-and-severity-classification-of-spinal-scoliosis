import os
import json
import numpy as np
import cv2
from labelme.utils import shape_to_mask
from PIL import Image

# 資料夾設定
json_dir = r"C:\project\Localization_and_Segmentation\app\patch_img\images"
output_mask_dir = r"C:\project\Localization_and_Segmentation\app\patch_img\masks"
os.makedirs(output_mask_dir, exist_ok=True)

# 處理所有 .json
for file in os.listdir(json_dir):
    if not file.endswith(".json"):
        continue

    json_path = os.path.join(json_dir, file)

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    image_height = data['imageHeight']
    image_width = data['imageWidth']
    mask = np.zeros((image_height, image_width), dtype=np.uint8)

    for shape in data['shapes']:
        poly_mask = shape_to_mask((image_height, image_width), shape['points'])
        mask = np.logical_or(mask, poly_mask)

    # 轉為 uint8 binary mask
    binary_mask = (mask * 255).astype(np.uint8)

    # 儲存
    mask_filename = os.path.splitext(file)[0] + ".png"
    mask_path = os.path.join(output_mask_dir, mask_filename)
    cv2.imwrite(mask_path, binary_mask)

print("✅ 所有 JSON 已成功轉成 binary mask 並輸出到：")
print(output_mask_dir)
