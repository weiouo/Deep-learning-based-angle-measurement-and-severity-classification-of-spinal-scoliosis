import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_clahe(image_path, clip_limit=2.0, tile_grid_size=(8, 8)):
    # 讀取灰階影像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"無法讀取影像：{image_path}")
    
    # 建立 CLAHE 物件
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    # 套用 CLAHE
    clahe_image = clahe.apply(image)
    
    # 顯示原圖與處理後影像
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("CLAHE Result")
    plt.imshow(clahe_image, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 儲存處理後影像（可選）
    output_path = image_path.replace('.jpg', '_clahe.jpg')
    cv2.imwrite(output_path, clahe_image)
    print(f"已儲存 CLAHE 影像至：{output_path}")

# 使用範例
apply_clahe(r"C:\project\Localization_and_Segmentation\app\dataset\scol\N34,S,12,F_1_0.jpg")
