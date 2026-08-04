import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, UnivariateSpline
from sklearn.metrics import mean_squared_error, r2_score

import torch
from torchvision import transforms
from ultralytics import YOLO

def clahe_hist(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)

def preprocess(img):
    return transforms.ToTensor()(img / 255.0).float()

def run_analysis(opt):
    model = YOLO(opt.weights)
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'

    point_counts = []
    mse_list = []
    r2_list = []

    for filename in os.listdir(opt.image_dir):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            continue

        image_path = os.path.join(opt.image_dir, filename)
        label_path = os.path.join(opt.label_dir, os.path.splitext(filename)[0] + ".txt")

        if not os.path.exists(label_path):
            continue

        image = cv2.imread(image_path)
        if image is None:
            continue

        if opt.use_clahe:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe_applied = clahe_hist(gray)
            image = cv2.cvtColor(clahe_applied, cv2.COLOR_GRAY2BGR)

        orig_h, orig_w = image.shape[:2]

        results = model.predict(source=image, imgsz=opt.img_size, conf=opt.conf_thres,
                                iou=opt.nms_thres, device=device_str, verbose=False)
        
        pred_centers = []
        for result in results:
            for box in result.boxes.data.tolist():
                x1, y1, x2, y2, score, cls = box
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                pred_centers.append((cx, cy))

        if len(pred_centers) < 4:
            continue

        pred_centers.sort(key=lambda pt: pt[1])
        x_pred = [pt[0] for pt in pred_centers]
        y_pred = [pt[1] for pt in pred_centers]

        gt_centers = []
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id, cx, cy, w, h = parts
                if cls_id != '0':
                    continue
                cx = float(cx) * orig_w
                cy = float(cy) * orig_h
                gt_centers.append((cx, cy))

        if len(gt_centers) < 4:
            continue

        gt_centers.sort(key=lambda pt: pt[1])
        x_gt = [pt[0] for pt in gt_centers]
        y_gt = [pt[1] for pt in gt_centers]

        y_eval = np.linspace(min(y_pred), max(y_pred), 300)

        spline_pred = UnivariateSpline(y_pred, x_pred, s=1)
        x_spline_pred = spline_pred(y_eval)

        spline_gt = CubicSpline(y_gt, x_gt)
        x_spline_gt = spline_gt(y_eval)

        # 評估
        mse = mean_squared_error(x_spline_gt, x_spline_pred)
        r2 = r2_score(x_spline_gt, x_spline_pred)

        point_counts.append(len(pred_centers))
        mse_list.append(mse)
        r2_list.append(r2)

    # 畫散佈圖
    plt.figure(figsize=(6, 5))
    plt.scatter(point_counts, mse_list, color='red')
    plt.xlabel("Number of YOLO-Detected Points")
    plt.ylabel("MSE")
    plt.title("MSE vs. Number of Detected Points")
    plt.grid(True)
    plt.savefig("mse_vs_points.png")
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.scatter(point_counts, r2_list, color='blue')
    plt.xlabel("Number of YOLO-Detected Points")
    plt.ylabel("R²")
    plt.title("R² vs. Number of Detected Points")
    plt.grid(True)
    plt.savefig("r2_vs_points.png")
    plt.close()

    print("✅ 分析完成，圖已儲存為 mse_vs_points.png 與 r2_vs_points.png")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True, help='Path to image folder')
    parser.add_argument('--label_dir', type=str, required=True, help='Path to label folder')
    parser.add_argument('--weights', type=str, required=True, help='Path to YOLOv11 weights (.pt)')
    parser.add_argument('--img_size', type=int, default=416, help='Image size')
    parser.add_argument('--conf_thres', type=float, default=0.8, help='Confidence threshold')
    parser.add_argument('--nms_thres', type=float, default=0.3, help='IoU threshold for NMS')
    parser.add_argument('--use_clahe', action='store_true', help='Apply CLAHE to input images')
    opt = parser.parse_args()
    run_analysis(opt)
