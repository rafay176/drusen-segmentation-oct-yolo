# Drusen Segmentation in Retinal OCT Images using YOLO

## Overview
This project develops a **YOLO-based segmentation pipeline** for detecting **drusen lesions** in retinal OCT images, which are key biomarkers for **Age-related Macular Degeneration (AMD)**. Accurate drusen segmentation supports early diagnosis and treatment planning.

The pipeline includes:

- Data preprocessing and medically valid augmentations
- Model training and evaluation
- Visualization of predictions

---

## Dataset
- Source: SD-OCT retinal images from AMD and control subjects
- Total B-scans: 35,400 (269 AMD patients, 115 normal subjects)
- Annotations: Pixel-level segmentation masks for drusen

---

## Methodology

- **Model**: YOLOv11 segmentation variant (`YOLOv11m-seg`)
- **Preprocessing**:
  - Resize images to 832×832
  - Normalize pixel values
  - Apply data augmentations:
    - Horizontal flip
    - Mosaic
    - Copy-paste (medical validity ensured)
- **Training**:
  - Baseline and augmented strategies
  - Pretrained weights used for transfer learning
  - Hyperparameter optimization
- **Evaluation Metrics**:
  - mAP@50–95
  - Precision, Recall, F1-score
  - Visual comparison of predicted masks vs ground truth

---

## Results

- **Baseline mAP@50–95**: 0.375  
- **After augmentation and optimization**: 0.415  
- Performance validated using **precision-recall curves**  
- Annotation quality verified via mask visualization
