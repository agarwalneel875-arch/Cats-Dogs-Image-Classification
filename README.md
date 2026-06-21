# Cats vs Dogs Image Classification using ResNet18 (PyTorch)

## Project Overview

This project implements a Deep Learning-based Image Classification model to distinguish between cats and dogs using Transfer Learning with ResNet18. The model was trained using PyTorch and achieved approximately 98% test accuracy.

The project demonstrates the use of pretrained convolutional neural networks, data augmentation, model evaluation, and inference on new images.

---

## Features

- Transfer Learning using pretrained ResNet18
- Image Data Augmentation
- PyTorch Implementation
- Training & Validation Accuracy Tracking
- Loss Curve Visualization
- Classification Metrics
- Confusion Matrix
- Saved Model for Inference
- Custom Image Prediction Script

---

## Dataset

Dataset: Cats vs Dogs Dataset

Directory Structure:

dataset/
├── training_set/
│ ├── cats/
│ └── dogs/
│
└── test_set/
├── cats/
└── dogs/

Training Images: ~8000

Testing Images: ~2000

Classes:
- Cats
- Dogs

---

## Model Architecture

Base Model:
- ResNet18 (Pretrained on ImageNet)

Transfer Learning Strategy:
- Frozen feature extraction layers
- Replaced final fully connected layer with 2 output classes

Output Classes:
- Cat
- Dog

---

## Data Augmentation

The following augmentation techniques were applied:

- Random Horizontal Flip
- Random Rotation
- Resize to 224 × 224
- Normalization using ImageNet statistics

---

## Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Scikit-Learn
- PIL

---

## Training Results

| Metric | Value |
|----------|----------|
| Best Test Accuracy | 98.07% |
| Training Accuracy | 96.46% |
| Final Test Accuracy | 97.92% |

Epoch Results:

| Epoch | Train Accuracy | Test Accuracy |
|---------|---------|---------|
| 1 | 96.29% | 97.58% |
| 2 | 95.89% | 97.97% |
| 3 | 96.40% | 98.07% |
| 4 | 96.46% | 98.02% |
| 5 | 96.34% | 97.92% |

---

## Evaluation Metrics

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

## Project Structure

Cats-Dogs-Image-Classification/

├── Cats_Dogs_Image_Classification_ResNet18.ipynb

├── inference.py

├── cats_dogs_resnet18.pth

├── requirements.txt

└── README.md

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Cats-Dogs-Image-Classification.git

cd Cats-Dogs-Image-Classification
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running Inference

Place a cat or dog image in the project folder.

Run:

```bash
python inference.py
```

Example Output:

```text
Loading model...
Model loaded successfully
Loading image...
Prediction: cats
```

---

## Future Improvements

- Fine-tune the entire ResNet18 network
- Deploy using Streamlit or Flask
- Add support for multiple pet breeds
- Convert model to ONNX/TensorFlow Lite for deployment

---

## Conclusion

A deep learning image classification model was successfully developed using Transfer Learning with ResNet18. The model achieved approximately 98% test accuracy while demonstrating the effectiveness of pretrained convolutional neural networks for image classification tasks.

This project showcases practical applications of Deep Learning, Computer Vision, Transfer Learning, and Model Deployment techniques.

---

## Author

Neel Agarwal

AI/ML Undergraduate | Deep Learning & Computer Vision Enthusiast
