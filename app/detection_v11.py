import os
import cv2
import torch
import argparse
import matplotlib.pyplot as plt
from ultralytics import YOLO

def clahe_hist(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)

def draw_bboxes(image, boxes):
    for box in boxes:
        x1, y1, x2, y2, score, cls = box
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{int(cls)} {score:.2f}"
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return image

def detect_and_draw(opt):
    model = YOLO(opt.weights)
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'

    image = cv2.imread(opt.image_path)
    if image is None:
        raise FileNotFoundError(f"無法讀取圖片：{opt.image_path}")

    if opt.use_clahe:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe_applied = clahe_hist(gray)
        image = cv2.cvtColor(clahe_applied, cv2.COLOR_GRAY2BGR)

    orig_img = image.copy()
    results = model.predict(source=image, imgsz=opt.img_size,
                            conf=opt.conf_thres, iou=opt.nms_thres,
                            device=device_str, verbose=False)

    for result in results:
        boxes = result.boxes.data.tolist()
        image_with_boxes = draw_bboxes(orig_img.copy(), boxes)

        # 儲存與顯示
        os.makedirs("result", exist_ok=True)
        output_path = os.path.join("result", os.path.basename(opt.image_path).replace(".", "_bbox."))
        cv2.imwrite(output_path, image_with_boxes)
        print(f"已儲存結果至：{output_path}")

        plt.imshow(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title("YOLO Bounding Box Result")
        plt.show()

# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image')
    parser.add_argument('--weights', type=str, required=True, help='Path to YOLOv11 weights (.pt)')
    parser.add_argument('--img_size', type=int, default=416, help='YOLO input image size')
    parser.add_argument('--conf_thres', type=float, default=0.8, help='Confidence threshold')
    parser.add_argument('--nms_thres', type=float, default=0.3, help='IoU threshold for NMS')
    parser.add_argument('--use_clahe', action='store_true', help='Apply CLAHE preprocessing')
    opt = parser.parse_args()

    detect_and_draw(opt)
