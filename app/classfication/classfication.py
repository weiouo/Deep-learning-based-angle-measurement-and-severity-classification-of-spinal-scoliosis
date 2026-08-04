import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from PIL import Image
import torch
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F
from scipy.interpolate import CubicSpline

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


def calculate_angle(spline, y_top, y_bottom):
    slope_top = spline.derivative()(y_top)
    slope_bottom = spline.derivative()(y_bottom)
    theta_rad = np.arctan(abs((slope_bottom - slope_top) / (1 + slope_bottom * slope_top)))
    theta_deg = np.degrees(theta_rad)
    return theta_deg


def classify_severity(angle):
    if angle < 10:
        return "normal"
    else:
        return "scol"


def detect_angle(image_path, model, opt, Tensor):
    image = cv2.imread(image_path, 0)
    if image is None:
        print(f"Failed to read {image_path}")
        return None

    image = clahe_hist(image)
    image = preprocess(image)
    image, _ = pad_to_square(image, 0)
    image = resize(image, opt.img_size)
    input_tensor = Variable(image.type(Tensor)).unsqueeze(0)

    with torch.no_grad():
        detections = model(input_tensor)
        detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

    centers = []
    orig_image = np.array(Image.open(image_path).convert('L'))

    if detections[0] is not None:
        detections = rescale_boxes(detections[0], opt.img_size, orig_image.shape[:2])
        for x1, y1, x2, y2, *_ in detections:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            centers.append((cx, cy))

    if len(centers) >= 3:
        centers.sort(key=lambda pt: pt[1])
        x = [pt[0] for pt in centers]
        y = [pt[1] for pt in centers]
        try:
            spline = CubicSpline(y, x)
            angle = calculate_angle(spline, min(y), max(y))
            return angle
        except Exception as e:
            print(f"Error in spline fitting: {e}")
            return 0
    else:
        print(f"Not enough points in {image_path}")
        return 0


def main():
    # === YOLO 設定 ===
    class Opt:
        weights = 'PyTorch_YOLOv3/checkpoints/new_best.pt'
        cfg = 'PyTorch_YOLOv3/config/yolov3-custom.cfg'
        names = 'PyTorch_YOLOv3/data/custom/classes.names'
        img_size = 416
        conf_thres = 0.8
        nms_thres = 0.3

    opt = Opt()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(opt.cfg, img_size=opt.img_size).to(device)
    model.load_state_dict(torch.load(opt.weights, map_location=device))
    model.eval()
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    # === 資料夾 ===
    dataset_dir = "dataset"
    class_folders = ["normal", "scol"]

    y_true = []
    y_pred = []

    for class_name in class_folders:
        class_dir = os.path.join(dataset_dir, class_name)
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img_path = os.path.join(class_dir, filename)
                print(f"Processing {img_path}...")

                angle = detect_angle(img_path, model, opt, Tensor)
                if angle is None:
                    continue

                print(f"Detected angle: {angle:.2f}")
                pred_class = classify_severity(angle)
                print(f"Predicted class: {pred_class}")

                y_true.append(class_name)
                y_pred.append(pred_class)

    # === Confusion Matrix ===
    labels = ["normal", "scol"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("\nConfusion Matrix:")
    print(cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix of Angle Classification")
    plt.show()

    # === 四個指標 ===
    print("\nEvaluation Metrics:")
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label="scol")
    rec = recall_score(y_true, y_pred, pos_label="scol")
    f1 = f1_score(y_true, y_pred, pos_label="scol")

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")


if __name__ == "__main__":
    main()
