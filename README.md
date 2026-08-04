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
Data Preparation: Collected spine X-rays and annotated bounding boxes and segmentation masks. Applied CLAHE for contrast enhancement and detail sharpening.

Vertebral Detection: Leveraged YOLOv3 vs. YOLOv11 to compare small object detection performance.

Angle Calculation Approaches:

*Method 1 (Curve Fitting): Fits a polynomial/spline curve using the center coordinates of YOLO-detected vertebrae.

*Method 2 (Clinical Simulation): Segments vertebral masks using U-Net, then computes the Cobb angle from the most tilted top and bottom endplates.

Evaluation: Classified the dataset under both methods to identify the most accurate angle estimation pipeline.
