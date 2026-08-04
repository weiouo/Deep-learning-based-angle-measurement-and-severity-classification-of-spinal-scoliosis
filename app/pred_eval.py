import os
import cv2
import numpy as np

# === 資料夾設定 ===
pred_dir = r"C:\project\Localization_and_Segmentation\app\patch_img\pred"
gt_dir = r"C:\project\Localization_and_Segmentation\app\patch_img\masks"

# === 評估函數 ===
def dice_coef(pred, mask):
    pred = pred.astype(bool)
    mask = mask.astype(bool)
    intersection = np.logical_and(pred, mask).sum()
    return 2. * intersection / (pred.sum() + mask.sum() + 1e-8)

def iou_score(pred, mask):
    pred = pred.astype(bool)
    mask = mask.astype(bool)
    intersection = np.logical_and(pred, mask).sum()
    union = np.logical_or(pred, mask).sum()
    return intersection / (union + 1e-8)

def pixel_accuracy(pred, mask):
    return (pred == mask).sum() / pred.size

# === 統計收集 ===
dice_list, iou_list, acc_list = [], [], []

for fname in os.listdir(pred_dir):
    pred_path = os.path.join(pred_dir, fname)
    mask_path = os.path.join(gt_dir, fname)
    if not os.path.exists(mask_path):
        print(f"⚠️ 找不到 GT：{fname}")
        continue

    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if pred is None or mask is None or pred.shape != mask.shape:
        print(f"⚠️ 錯誤圖：{fname}")
        continue

    pred_bin = (pred > 127).astype(np.uint8)
    mask_bin = (mask > 127).astype(np.uint8)

    dice_list.append(dice_coef(pred_bin, mask_bin))
    iou_list.append(iou_score(pred_bin, mask_bin))
    acc_list.append(pixel_accuracy(pred_bin, mask_bin))

# === 結果輸出 ===
print("📊 Segmentation 評估結果：")
print(f"  Dice coefficient：平均 = {np.mean(dice_list):.4f}，標準差 = {np.std(dice_list):.4f}")
print(f"  IoU (Jaccard)   ：平均 = {np.mean(iou_list):.4f}，標準差 = {np.std(iou_list):.4f}")
print(f"  Pixel Accuracy  ：平均 = {np.mean(acc_list):.4f}，標準差 = {np.std(acc_list):.4f}")
