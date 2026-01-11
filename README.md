# ☕ Coffee Bean Classification

Image-based Multi-Class Classification with Transfer Learning & Hyperparameter Tuning

## 📌 Project Overview

This project focuses on classifying coffee beans into multiple classes using image data.
The goal is to build a robust and well-diagnosed image classification pipeline, starting from deep EDA, followed by careful preprocessing, baseline modeling, model diagnosis, and systematic hyperparameter tuning.

The project is designed with a Kaggle-style, ML Engineer–oriented workflow, where each phase is implemented in a separate notebook to ensure clarity, reproducibility, and traceability.

## 🎯 Objectives
- Perform deep Exploratory Data Analysis (EDA) to understand data quality, imbalance, and risks
- Design preprocessing pipeline aligned with EDA findings
- Establish strong baseline models using transfer learning
- Apply model diagnosis to identify performance bottlenecks
- Improve performance using Optuna-based hyperparameter tuning
- Compare multiple CNN architectures in a fair and systematic manner

## 🧠 Problem Formulation
- Task: Image Classification
- Type: Multi-class classification
- Input: RGB images of coffee beans
- Output: Coffee bean class label
- Evaluation Metrics: Accuracy

## 🧪 Models Used
Transfer Learning with pretrained CNN backbones:
- ResNet50
- EfficientNet (B0–B3)
- MobileNetV2

Each model is evaluated under:
- Baseline configuration
- Hyperparameter tuning using Optuna

Precision, Recall, F1-score (macro & weighted)

Confusion Matrix
