import os
import cv2
import math
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

import torch
from torch.autograd import Variable
from torchvision import transforms

from PyTorch_YOLOv3.models import Darknet
from PyTorch_YOLOv3.utils.utils import load_classes, rescale_boxes, non_max_suppression
import torch.nn.functional as F


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


def calculate_angle(poly, y_top, y_bottom):
    dpoly = np.polyder(poly)
    slope_top = dpoly(y_top)
    slope_bottom = dpoly(y_bottom)
    theta_rad = np.arctan(abs((slope_bottom - slope_top) / (1 + slope_bottom * slope_top)))
    theta_deg = np.degrees(theta_rad)
    return theta_deg


def severity_level(angle):
    if angle < 10:
        return "Normal (<10°)"
    else:
        return "scoliosis (>10°)"


def run_inference(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Darknet(opt.cfg, img_size=opt.img_size).to(device)
    model.load_state_dict(torch.load(opt.weights, map_location=device))
    model.eval()

    classes = load_classes(opt.names)
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    # Read and preprocess image
    image = cv2.imread(opt.image, 0)
    assert image is not None, f"Image not found: {opt.image}"

    image = clahe_hist(image)
    image = preprocess(image)
    image, _ = pad_to_square(image, 0)
    image = resize(image, opt.img_size)

    input_tensor = Variable(image.type(Tensor)).unsqueeze(0)

    print("Running inference...")
    with torch.no_grad():
        detections = model(input_tensor)
        detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

    orig_image = np.array(Image.open(opt.image).convert('L'))
    fig, ax = plt.subplots(1)
    ax.imshow(orig_image, cmap='gray')

    centers = []

    if detections[0] is not None:
        detections = rescale_boxes(detections[0], opt.img_size, orig_image.shape[:2])
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            box_w, box_h = x2 - x1, y2 - y1
            bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(bbox)

            # 中心點
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            centers.append((cx, cy))
            ax.plot(cx, cy, 'bo', markersize=4)

    if len(centers) >= 3:
        # 按 y 排序（從上到下）
        centers.sort(key=lambda pt: pt[1])
        x = [pt[0] for pt in centers]
        y = [pt[1] for pt in centers]

        # 三次多項式擬合
        coeffs = np.polyfit(y, x, 3)
        poly = np.poly1d(coeffs)

        # 畫擬合曲線
        y_fit = np.linspace(min(y), max(y), 300)
        x_fit = poly(y_fit)
        ax.plot(x_fit, y_fit, 'b-', linewidth=1)

        # 計算角度
        angle = calculate_angle(poly, min(y), max(y))
        severity = severity_level(angle)
        print(f"Cobb angle (approx): {angle:.2f} degrees")
        print(f"Severity: {severity}")

        # 在圖上標註角度 & 嚴重度
        ax.text(10, 20, f"Cobb angle: {angle:.2f}°", color='yellow', fontsize=10, bbox=dict(facecolor='black', alpha=0.5))
        ax.text(10, 70, f"Severity: {severity}", color='yellow', fontsize=10, bbox=dict(facecolor='black', alpha=0.5))
    else:
        print("Not enough points to fit curve and calculate angle.")

    plt.axis('off')
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    plt.savefig("inference_result.png", bbox_inches="tight", pad_inches=0.0)
    plt.close()
    print("Saved result to inference_result.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Path to image')
    parser.add_argument('--weights', type=str, default='PyTorch_YOLOv3/checkpoints/new_best.pt', help='Path to weights')
    parser.add_argument('--cfg', type=str, default='PyTorch_YOLOv3/config/yolov3-custom.cfg', help='Path to config')
    parser.add_argument('--names', type=str, default='PyTorch_YOLOv3/data/custom/classes.names', help='Path to class names')
    parser.add_argument('--img_size', type=int, default=416, help='Image size')
    parser.add_argument('--conf_thres', type=float, default=0.8, help='Object confidence threshold')
    parser.add_argument('--nms_thres', type=float, default=0.3, help='IoU threshold for NMS')

    opt = parser.parse_args()
    run_inference(opt)
