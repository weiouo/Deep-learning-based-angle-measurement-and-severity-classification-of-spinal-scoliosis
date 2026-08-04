import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from scipy.interpolate import UnivariateSpline  # ✅ 修正這裡
from ultralytics import YOLO


def clahe_hist(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


def calculate_angle(spline, y_top, y_bottom):
    slope_top = spline.derivative()(y_top)
    slope_bottom = spline.derivative()(y_bottom)
    theta_rad = np.arctan(abs((slope_bottom - slope_top) / (1 + slope_bottom * slope_top)))
    theta_deg = np.degrees(theta_rad)
    return theta_deg


def classify_severity(angle):
    return "normal" if angle < 10 else "scol"


def detect_angle(image_path, model, use_clahe=False):
    image = cv2.imread(image_path, 0)
    if image is None:
        print(f"Failed to read {image_path}")
        return None

    if use_clahe:
        image = clahe_hist(image)

    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    results = model.predict(image_color, conf=0.8, imgsz=416, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    centers = []
    for x1, y1, x2, y2 in boxes:
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        centers.append((cx, cy))

    if len(centers) >= 3:
        centers.sort(key=lambda pt: pt[1])  # sort by y
        x = [pt[0] for pt in centers]
        y = [pt[1] for pt in centers]
        try:
            spline = UnivariateSpline(y, x, k=3, s=0)
            angle = calculate_angle(spline, min(y), max(y))
            return angle
        except Exception as e:
            print(f"Error fitting spline in {image_path}: {e}")
            return 0
    else:
        print(f"Not enough points in {image_path}")
        return 0


def main():
    model_path = "C:/project/Localization_and_Segmentation/app/PyTorch_YOLOv3/checkpoints/v11_clahe_best.pt"
    use_clahe = True  # Toggle CLAHE enhancement

    model = YOLO(model_path)

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

                angle = detect_angle(img_path, model, use_clahe=use_clahe)
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
    plt.title(f"Confusion Matrix (CLAHE={use_clahe})")
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
