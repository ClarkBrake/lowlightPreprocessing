# Low-Light Preprocessing for YOLOv8 Pedestrian Detection

This repository investigates the impact of low-light image preprocessing techniques on YOLOv8 pedestrian detection performance. The project focuses on improving detection versatility under dark and degraded lighting conditions using simple yet effective image transformations.

---

## Overview

Low-light environments significantly reduce object detection accuracy due to poor visibility, reduced contrast, and noise. This project evaluates multiple preprocessing strategies to enhance image quality before inference and compares their effect on detection performance.

The goal is to demonstrate that preprocessing can meaningfully improve detection results in challenging lighting conditions.

---

## Repository Structure

.
├── README.md
├── eval_dark_variants.py
├── make_detection_panel.py
├── make_preprocessing_panel.py
├── make_results_figures.py
├── preprocess_alpha_beta.py
├── preprocess_dark_detection.py
├── preprocess_gray_clahe.py
└── preprocess_grayscale.py

---

## Dataset Setup (IMPORTANT)

To run this project, your dataset must follow the YOLOv8 format.

### Required Structure

data/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/

* `images/` contains `.png` or `.jpg` files
* `labels/` contains corresponding `.txt` annotation files
* Each image must have a matching label file with the same name

---

## YAML Configuration

You must create a dataset configuration file (e.g., `dataset.yaml`):

```yaml
train: data/images/train
val: data/images/val

nc: 1
names: ['person']
```

### Notes:

* `train` and `val` must point to your dataset folders
* `nc` = number of classes
* `names` = class labels

---

## Methodology

The pipeline evaluates several preprocessing approaches:

### 1. Baseline

* Original images with no preprocessing

### 2. Brightness / Contrast Adjustment

* Controlled intensity scaling using alpha-beta transformation

### 3. Dark Image Enhancement

* Targeted adjustments for low-light conditions

### 4. CLAHE (Contrast Limited Adaptive Histogram Equalization)

* Improves local contrast in dark regions

### 5. Grayscale Conversion

* Tests robustness of detection without color information

---

## Workflow

1. Prepare dataset and YAML configuration
2. Apply preprocessing methods
3. Run YOLOv8 detection
4. Evaluate performance across all variants
5. Generate figures and visual comparisons

---

## Usage

### 1. Run Preprocessing

```bash
python preprocess_alpha_beta.py
python preprocess_dark_detection.py
python preprocess_gray_clahe.py
python preprocess_grayscale.py
```

---

### 2. Evaluate Detection Performance

```bash
python eval_dark_variants.py
```

---

### 3. Generate Visualizations

Detection outputs:

```bash
python make_detection_panel.py
```

Preprocessing comparison:

```bash
python make_preprocessing_panel.py
```

Final figures:

```bash
python make_results_figures.py
```

---

## Results

The experiments show that preprocessing can improve detection performance in low-light conditions.

Key observations:

* Brightness and contrast adjustments improve detection confidence
* CLAHE enhances visibility in dark regions
* Extremely dark images significantly degrade baseline performance
* Preprocessing provides consistent improvements over unprocessed input

---

## Notes

* Dataset must follow YOLOv8 format with proper train/val split
* YAML configuration is required for evaluation
* Preprocessing parameters can be tuned within each script
* Results depend on lighting severity and dataset quality

---

## Applications

* Autonomous driving in low-light environments
* Night-time pedestrian detection
* Surveillance and security systems
* Robotics and perception systems

---

## Author
Clark Brake 
Developed as part of a computer vision project exploring preprocessing techniques for improving detection performance under challenging environmental conditions.
