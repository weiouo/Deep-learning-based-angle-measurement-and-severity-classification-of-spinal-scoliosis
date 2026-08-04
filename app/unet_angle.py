import cv2
import numpy as np
from sklearn.linear_model import LinearRegression
import math
import os

# === 圖片路徑 ===
mask_path = r"C:\project\Localization_and_Segmentation\app\seg_result\scol\N34,S,12,F_1_0.jpg"
if not os.path.exists(mask_path):
    raise FileNotFoundError(f"❌ 找不到圖像：{mask_path}")

mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
if mask is None:
    raise ValueError("❌ 圖像讀取失敗")

_, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

# === 擷取每個 vertebra 的中心點位置 ===
vertebrae = []
for i in range(1, num_labels):
    x, y, w, h, area = stats[i]
    if area < 100:
        continue
    cx = x + w // 2
    cy = y + h // 2
    vertebrae.append({"cx": cx, "cy": cy, "idx": i})

vertebrae.sort(key=lambda v: v["cy"])  # 上到下排序

# === 擬合斜率（角度） ===
def fit_angle(center_list):
    X = np.array([v["cx"] for v in center_list])
    Y = np.array([v["cy"] for v in center_list])
    dx = X[-1] - X[0]
    dy = Y[-1] - Y[0]
    return math.degrees(math.atan2(dy, dx))

# === 找出 y 差夠遠 且角度差最大的 vertebra 對 ===
max_angle_diff = -1
best_pair = None
angle1 = angle2 = 0

for i in range(len(vertebrae)):
    for j in range(i + 1, len(vertebrae)):
        vi = vertebrae[i]
        vj = vertebrae[j]

        if abs(vj["cy"] - vi["cy"]) < 100:
            continue

        # 取鄰近節點做線性擬合
        top_neighbors = vertebrae[max(0, i-2):min(len(vertebrae), i+3)]
        bot_neighbors = vertebrae[max(0, j-2):min(len(vertebrae), j+3)]


        theta1 = fit_angle(top_neighbors)
        theta2 = fit_angle(bot_neighbors)

        diff = abs(theta1 - theta2)
        if diff > 180:
            diff = 360 - diff

        if diff > max_angle_diff:
            max_angle_diff = diff
            best_pair = (vi["cy"], vj["cy"])
            angle1, angle2 = theta1, theta2

top_vertebra = next(v for v in vertebrae if v["cy"] == best_pair[0])
bot_vertebra = next(v for v in vertebrae if v["cy"] == best_pair[1])

# === 輸出結果 ===
print(f"✅ 選中 vertebra：第 {top_vertebra['idx']} 塊（y={best_pair[0]}） 和 第 {bot_vertebra['idx']} 塊（y={best_pair[1]}）")
print(f"⬆️ 選定 vertebra 上邊緣角度: {angle1:.2f}°")
print(f"⬇️ 選定 vertebra 下邊緣角度: {angle2:.2f}°")
print(f"📐 Cobb angle (最大差): {max_angle_diff:.2f}°")
