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
3. Run the code
```python
python train.py
```
## Performance snapshot
![performance]([https://example.com/image.png](https://github.com/Khsuanko/NYCU-Computer-Vision-2025-Spring-HW1/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2%202025-03-26%20235718.png))
