# NYCU-Computer-Vision-2025-Spring-HW1
StudentID: 110550122  
Name: 柯凱軒

## Introduction
This project focuses on image classification using deep learning, specifically leveraging ResNet50. The task involves classifying images into 100 different categories. The primary goal is to train a robust model that can generalize well to unseen images. The core idea is to fine-tune a pre-trained ResNet50 model using a custom dataset while incorporating data augmentation and learning rate scheduling to improve performance.

## How to install
1. Install Dependencies  
```python
pip install torch torchvision torchaudio matplotlib tqdm pandas
```
2. Ensure you have the dataset structured as follows:
```python
./data/
    ├── train/
    ├── val/
    ├── test/
```
