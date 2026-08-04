## MOTIVATION
Conventionally, the Cobb angle is calculated by drawing extension lines along the superior endplate of the most tilted upper vertebra and the inferior endplate of the most tilted lower vertebra. However, this manual approach suffers from substantial measurement variation, inefficiency, and an inability to process large volumes of clinical cases rapidly—motivating the inception of this study.

## WORKFLOW
<img width="1455" height="641" alt="final1 drawio" src="https://github.com/user-attachments/assets/c2a5d6f8-d97a-4a5d-8b62-a6654736e1eb" />

1. **Data Preparation & Preprocessing**
   - **Data Annotation**: Collected spinal X-ray images and labeled them with both **bounding boxes** (for detection) and **segmentation masks**.
   - **Image Enhancement**: Applied **CLAHE** (Contrast Limited Adaptive Histogram Equalization) to boost local contrast and highlight faint bone contours and fine details.

2. **Vertebral Detection (Object Detection)**
   - Compared **YOLOv3** and **YOLOv11** to evaluate their performance in detecting small, closely packed vertebral objects in high-resolution X-rays.

3. **Spinal Angle Calculation Approaches**
   - **Method 1 (Curve Fitting)**: 
     - Fits a smooth curve (polynomial/spline) using the **center coordinates ($x, y$)** of all YOLO-detected vertebrae to model spinal curvature.
   - **Method 2 (Clinical Simulation)**: 
     - First generates precise vertebral masks using **U-Net**.
     - Identifies the **most tilted upper and lower end vertebrae** from the segmentation masks.
     - Calculates the **Cobb angle** by computing the intersection angle between their boundary extension lines (mimicking clinical diagnostic criteria).

4. **Evaluation & Classification**
   - Classified the dataset into scoliosis severity levels using both angle estimation methods.
   - Compared diagnostic accuracy and correlation with clinical ground truth to identify the optimal pipeline.
  
## RESULT

<p align="center">
  <img src="https://github.com/user-attachments/assets/f95e9f4e-55d4-456d-8ee7-775d97d7f0d8" width="60%"><br>
  <sub>Polynomial fitting fails to cover all vertebral center points, whereas LOWESS exhibits unnatural sharp bends.</sub>
</p>

---

<p align="center">
  <img src="https://github.com/user-attachments/assets/b11713af-7be2-4109-84e1-d12f8ff22c47" width="60%"><br>
  <sub>Spline achieves the best fitting performance but is prone to failure when the number of center points is insufficient; Cubic Spline, on the other hand, demonstrates higher stability.</sub>
</p>

---

<p align="center">
  <img src="https://github.com/user-attachments/assets/720a7d49-ec89-417c-8dbc-f833595670b6" width="60%"><br>
  <sub>Combining CLAHE with YOLOv11 enhances classification performance for curve fitting, while U-Net further boosts overall accuracy.</sub>
</p>

---

<p align="center">
  <img src="https://github.com/user-attachments/assets/636bc841-3b80-4813-8582-9931ad424cfa" width="60%"><br>
  <sub>Although the overall MSE and $R^2$ remain stable, outliers inflate the standard deviation, leading to evaluation bias.</sub>
</p>

