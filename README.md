<!--
*** Thanks for checking out this README Template. If you have a suggestion that would
*** make this better, please fork the repo and create a pull request or simply open
*** an issue with the tag "enhancement".
*** Thanks again! Now go create something AMAZING! :D
***
***
***
*** To avoid retyping too much info. Do a search and replace for the following:
*** github_username, repo_name, twitter_handle, email
-->
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
