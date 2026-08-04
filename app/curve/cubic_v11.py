import os
import cv2
import numpy as np
from PIL import Image
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

import torch
from torchvision import transforms
from ultralytics import YOLO

def clahe_hist(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)

def preprocess(img):
    return transforms.ToTensor()(img / 255.0).float()

def run_batch_evaluation(opt):
    model = YOLO(opt.weights)
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'

    image_dir = opt.image_dir
    label_dir = opt.label_dir
    os.makedirs("result", exist_ok=True)

    mse_list = []
    r2_list = []

    for filename in os.listdir(image_dir):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            continue

        image_path = os.path.join(image_dir, filename)
        label_path = os.path.join(label_dir, os.path.splitext(filename)[0] + ".txt")

        if not os.path.exists(label_path):
            continue

        image = cv2.imread(image_path)
        if image is None:
            continue

        if opt.use_clahe:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe_applied = clahe_hist(gray)
            image = cv2.cvtColor(clahe_applied, cv2.COLOR_GRAY2BGR)

        orig_img = image.copy()
        orig_h, orig_w = image.shape[:2]

        results = model.predict(source=image, imgsz=opt.img_size, conf=opt.conf_thres, iou=opt.nms_thres, device=device_str, verbose=False)
        pred_centers = []
        for result in results:
            for box in result.boxes.data.tolist():
                x1, y1, x2, y2, score, cls = box
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                pred_centers.append((cx, cy))

        if len(pred_centers) < 3:
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

        if len(gt_centers) < 3:
            continue

        gt_centers.sort(key=lambda pt: pt[1])
        x_gt = [pt[0] for pt in gt_centers]
        y_gt = [pt[1] for pt in gt_centers]

        y_eval = np.linspace(min(y_pred), max(y_pred), 300)

        spline_pred = CubicSpline(y_pred, x_pred)
        x_spline_pred = spline_pred(y_eval)

        spline_gt = CubicSpline(y_gt, x_gt)
        x_spline_gt = spline_gt(y_eval)

        x_spline_pred_centered = x_spline_pred - np.mean(x_spline_pred)
        x_spline_gt_centered = x_spline_gt - np.mean(x_spline_gt)

        mse = mean_squared_error(x_spline_gt_centered, x_spline_pred_centered)
        r2 = r2_score(x_spline_gt_centered, x_spline_pred_centered)
        mse_list.append(mse)
        r2_list.append(r2)

        plt.figure(figsize=(6, 6))
        plt.imshow(orig_img, cmap='gray')
        plt.plot(x_spline_gt, y_eval, '--', color='green', linewidth=2.5, label='GT (Cubic Spline)')
        plt.plot(x_spline_pred, y_eval, '-', color='red', linewidth=2.5, label='Cubic Spline Fit')
        plt.scatter(x_pred, y_pred, c='black', s=15, marker='x', label='Predicted Centers')
        plt.legend()
        plt.axis('off')
        plt.savefig(os.path.join("result", f"{os.path.splitext(filename)[0]}_curve.png"), bbox_inches='tight', pad_inches=0)
        plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True, help='Path to image folder')
    parser.add_argument('--label_dir', type=str, required=True, help='Path to label folder')
    parser.add_argument('--weights', type=str, required=True, help='Path to YOLOv11 weights (.pt)')
    parser.add_argument('--img_size', type=int, default=416, help='Image size')
    parser.add_argument('--conf_thres', type=float, default=0.8, help='Object confidence threshold')
    parser.add_argument('--nms_thres', type=float, default=0.3, help='IoU threshold for NMS')
    parser.add_argument('--use_clahe', action='store_true', help='Apply CLAHE to input images')
    opt = parser.parse_args()
    run_batch_evaluation(opt)
