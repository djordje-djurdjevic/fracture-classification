# Fracture Classification using YOLOv8

## Project Overview

This project implements a binary image classification system for detecting bone fractures in X-ray images using YOLOv8 (classification model).

The system is designed to evaluate how different image preprocessing techniques affect model performance and error distribution.

---

## Problem Statement

Manual analysis of X-ray images is:
- time-consuming
- dependent on expert interpretation
- prone to delays and inconsistencies

The goal is to develop a machine learning model that can assist in fracture detection and improve diagnostic efficiency.

---

## Dataset

Dataset used:
https://www.kaggle.com/datasets/bmadushanirodrigo/fracture-multi-region-x-ray-data


Structure:
dataset/

├── original/

│ ├── train/

│ ├── val/

│ └── test/

├── processed/

Classes:
- fractured
- not fractured

The dataset is not included in the repository due to size limitations.


Classes:
- fractured
- not fractured

The dataset is not included in the repository due to size limitations.

---

## Methodology

### Model
- YOLOv8 classification model (yolov8n-cls)

### Training setup
- Image size: 224
- Batch size: 16
- Epochs: 30
- Optimizer: default (Ultralytics)

---

## Preprocessing

The following preprocessing techniques were evaluated:

- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Gaussian Blur
- Histogram Equalization
- Sharpening filter

Different combinations were tested to analyze their effect on model performance.

---

## Project Structure
├── dataset/

│ ├── original/

│ └── processed/

├── scripts/    

├── results/

├── docs/

├── README.md


Install dependencies:

pip install ultralytics opencv-python numpy


Train model:

python scripts/train.py


Run preprocessing:

python scripts/preprocessing.py

---
## Notes

- Dataset must be downloaded separately from Kaggle
- Each preprocessing experiment should be run independently
