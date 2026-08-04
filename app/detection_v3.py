import cv2
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt

from PyTorch_YOLOv3.models import Darknet
from PyTorch_YOLOv3.utils.utils import rescale_boxes, non_max_suppression

# --- 輔助函數 ---

def clahe_hist(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)

def preprocess(img):
    return transforms.ToTensor()(img / 255.0).float()

def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    img = F.pad(img, pad, "constant", value=pad_value)
    return img, pad

def resize(image, size):
    return torch.nn.functional.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)

# --- 主處理函數 ---

def detect_and_draw(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 載入模型
    model = Darknet(opt.cfg, img_size=opt.img_size).to(device)
    model.load_state_dict(torch.load(opt.weights, map_location=device))
    model.eval()

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    # 讀圖 + CLAHE
    image = cv2.imread(opt.image_path, 0)
    if image is None:
        raise FileNotFoundError(f"無法讀取圖片：{opt.image_path}")
    orig_h, orig_w = image.shape[:2]
    orig_img_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    image = clahe_hist(image)
    image = preprocess(image)
    image, pad = pad_to_square(image, 0)
    image = resize(image, opt.img_size)
    input_tensor = Variable(image.type(Tensor)).unsqueeze(0)

    # 推論
    with torch.no_grad():
        detections = model(input_tensor)
        detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

    # 繪製結果
    if detections[0] is not None:
        detections = rescale_boxes(detections[0], opt.img_size, (orig_h, orig_w))
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(orig_img_color, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(orig_img_color, f"{int(cls_pred)} {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 儲存與顯示結果
    os.makedirs("result", exist_ok=True)
    output_path = os.path.join("result", os.path.basename(opt.image_path).replace(".", "_bbox."))
    cv2.imwrite(output_path, orig_img_color)
    print(f"已儲存結果至：{output_path}")

    # 顯示
    plt.imshow(cv2.cvtColor(orig_img_color, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Detection Result")
    plt.show()

# --- CLI 入口 ---

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image')
    parser.add_argument('--weights', type=str, default='PyTorch_YOLOv3/checkpoints/new_best.pt')
    parser.add_argument('--cfg', type=str, default='PyTorch_YOLOv3/config/yolov3-custom.cfg')
    parser.add_argument('--img_size', type=int, default=416)
    parser.add_argument('--conf_thres', type=float, default=0.8)
    parser.add_argument('--nms_thres', type=float, default=0.3)
    opt = parser.parse_args()

    detect_and_draw(opt)
