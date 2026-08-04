import os
import cv2
from PIL import Image

# 來源完整圖片（你挑的 30 張）
input_dir = r"C:\project\Localization_and_Segmentation\app\patch"
# YOLO bbox 標註位置
label_dir = r"C:\project\Localization_and_Segmentation\app\PyTorch_YOLOv3\data\custom\good\labels"
# 輸出 vertebra patch 的位置
output_dir = r"C:\project\Localization_and_Segmentation\app\patch_img\images"
os.makedirs(output_dir, exist_ok=True)

# 支援圖像格式
image_exts = [".jpg", ".jpeg", ".png", ".bmp"]

def yolo_to_box(yolo_line, img_w, img_h, padding=0):
    cls, x, y, w, h = map(float, yolo_line.strip().split())
    x1 = int((x - w / 2) * img_w) - padding
    y1 = int((y - h / 2) * img_h) - padding
    x2 = int((x + w / 2) * img_w) + padding
    y2 = int((y + h / 2) * img_h) + padding
    return max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)

# 處理每張圖
for filename in os.listdir(input_dir):
    if not any(filename.lower().endswith(ext) for ext in image_exts):
        continue

    name = os.path.splitext(filename)[0]
    img_path = os.path.join(input_dir, filename)
    label_path = os.path.join(label_dir, f"{name}.txt")

    if not os.path.exists(label_path):
        print(f"⚠️ 無對應 label：{label_path}")
        continue

    image = cv2.imread(img_path)
    if image is None:
        print(f"❌ 圖片無法讀取：{img_path}")
        continue

    h, w = image.shape[:2]

    with open(label_path, "r") as f:
        lines = f.readlines()

    for idx, line in enumerate(lines):
        x1, y1, x2, y2 = yolo_to_box(line, w, h, padding=20)
        crop = image[y1:y2, x1:x2]
        out_name = f"{name}_{idx}.png"
        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, crop)

print("✅ 已完成 vertebra patch 裁切並儲存到 patch_img/images/")
