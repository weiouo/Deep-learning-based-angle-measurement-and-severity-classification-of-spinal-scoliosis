import cv2
import numpy as np
from sklearn.linear_model import LinearRegression
import math
import os
from sklearn.metrics import confusion_matrix, classification_report

def compute_cobb_angle(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

    vertebrae = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < 100:
            continue
        cx = x + w // 2
        cy = y + h // 2
        vertebrae.append({"cx": cx, "cy": cy, "idx": i})

    vertebrae.sort(key=lambda v: v["cy"])
    if len(vertebrae) < 4:
        return None  # 無法判斷

    def fit_angle(center_list):
        X = np.array([v["cx"] for v in center_list])
        Y = np.array([v["cy"] for v in center_list])
        dx = X[-1] - X[0]
        dy = Y[-1] - Y[0]
        return math.degrees(math.atan2(dy, dx))

    max_angle_diff = -1
    for i in range(len(vertebrae)):
        for j in range(i + 1, len(vertebrae)):
            vi = vertebrae[i]
            vj = vertebrae[j]
            if abs(vj["cy"] - vi["cy"]) < 100:
                continue
            top_neighbors = vertebrae[max(0, i-2):min(len(vertebrae), i+3)]
            bot_neighbors = vertebrae[max(0, j-2):min(len(vertebrae), j+3)]
            theta1 = fit_angle(top_neighbors)
            theta2 = fit_angle(bot_neighbors)
            diff = abs(theta1 - theta2)
            if diff > 180:
                diff = 360 - diff
            if diff > max_angle_diff:
                max_angle_diff = diff
    return max_angle_diff

# === 資料夾設定 ===
normal_dir = r"C:\project\Localization_and_Segmentation\app\seg_result\normal"
scol_dir = r"C:\project\Localization_and_Segmentation\app\seg_result\scol"

y_true = []
y_pred = []

for fname in os.listdir(normal_dir):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
        continue
    fpath = os.path.join(normal_dir, fname)
    angle = compute_cobb_angle(fpath)
    if angle is None:
        continue
    y_true.append(0)  # 正常
    y_pred.append(1 if angle > 10 else 0)

for fname in os.listdir(scol_dir):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
        continue
    fpath = os.path.join(scol_dir, fname)
    angle = compute_cobb_angle(fpath)
    if angle is None:
        continue
    y_true.append(1)  # 側彎
    y_pred.append(1 if angle > 10 else 0)

# === 顯示結果 ===
cm = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=["Normal", "Scoliosis"], digits=2)

print("📊 Confusion Matrix:")
print(cm)
print("\n📋 Classification Report:")
print(report)
