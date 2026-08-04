import os
import cv2
import numpy as np
from PIL import Image
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

import torch
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F

from PyTorch_YOLOv3.models import Darknet
from PyTorch_YOLOv3.utils.utils import load_classes, rescale_boxes, non_max_suppression

def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    img = F.pad(img, pad, "constant", value=pad_value)
    return img, pad

def clahe_hist(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)

def preprocess(img):
    return transforms.ToTensor()(img / 255.0).float()

def resize(image, size):
    return torch.nn.functional.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)

def run_batch_evaluation(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(opt.cfg, img_size=opt.img_size).to(device)
    model.load_state_dict(torch.load(opt.weights, map_location=device))
    model.eval()
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

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

        image = cv2.imread(image_path, 0)
        if image is None:
            continue

        orig_h, orig_w = image.shape[:2]
        orig_img = image.copy()
        image = clahe_hist(image)
        image = preprocess(image)
        image, pad = pad_to_square(image, 0)
        image = resize(image, opt.img_size)
        input_tensor = Variable(image.type(Tensor)).unsqueeze(0)

        with torch.no_grad():
            detections = model(input_tensor)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

        pred_centers = []
        if detections[0] is not None:
            detections = rescale_boxes(detections[0], opt.img_size, (orig_h, orig_w))
            for x1, y1, x2, y2, *_ in detections:
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                pred_centers.append((cx.item(), cy.item()))

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
        plt.plot(x_spline_gt, y_eval, 'r-', label='Ground Truth (Cubic)')
        plt.plot(x_spline_pred, y_eval, 'g-', label='Cubic Fit')
        plt.text(10, 20, f"MSE: {mse:.2f}\nR²: {r2:.2f}", color='yellow', fontsize=9,
                 bbox=dict(facecolor='black', alpha=0.5))
        plt.legend()
        plt.axis('off')
        plt.savefig(os.path.join("result", f"{os.path.splitext(filename)[0]}_curve.png"), bbox_inches='tight', pad_inches=0)
        plt.close()

    print(f"共評估 {len(mse_list)} 張圖")
    print(f"MSE 平均: {np.mean(mse_list):.4f}，標準差: {np.std(mse_list):.4f}")
    print(f"R² 平均: {np.mean(r2_list):.4f}，標準差: {np.std(r2_list):.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True, help='Path to image folder')
    parser.add_argument('--label_dir', type=str, required=True, help='Path to label folder')
    parser.add_argument('--weights', type=str, default='PyTorch_YOLOv3/checkpoints/new_best.pt', help='Path to weights')
    parser.add_argument('--cfg', type=str, default='PyTorch_YOLOv3/config/yolov3-custom.cfg', help='Path to config')
    parser.add_argument('--names', type=str, default='PyTorch_YOLOv3/data/custom/classes.names', help='Path to class names')
    parser.add_argument('--img_size', type=int, default=416, help='Image size')
    parser.add_argument('--conf_thres', type=float, default=0.8, help='Object confidence threshold')
    parser.add_argument('--nms_thres', type=float, default=0.3, help='IoU threshold for NMS')
    opt = parser.parse_args()
    run_batch_evaluation(opt)
