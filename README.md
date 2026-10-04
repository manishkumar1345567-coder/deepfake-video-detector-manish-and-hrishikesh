# Forensic Video Manipulation Detection via Frozen Embeddings

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)

A decoupled, two-stage digital forensic framework that detects manipulated videos (DeepFakes, FaceSwap, Face2Face, NeuralTextures). The pipeline isolates spatial representation learning from sequence classification by using a strictly **frozen pre-trained ResNet-50 backbone** for feature extraction and an **Extreme Gradient Boosting (XGBoost)** classifier for forensic attribution.

---

## 📌 Project Overview

End-to-end deep learning forensic models frequently overfit to specific generator architectures and struggle under lossy video compression. This repository presents a computationally efficient alternative:

1. **Temporal Subsampling & Face Localization:** Samples video frames at fixed intervals and crops the facial region with an expanded boundary margin ($\alpha = 0.20$) using **MTCNN** to preserve blending boundary artifacts.
2. **Frozen Feature Extraction:** Extracts high-dimensional forensic embeddings ($\mathbb{R}^{2048}$) using an ImageNet-pretrained **ResNet-50** with zero backpropagation or parameter updates.
3. **Supervised Classification:** Trains **XGBoost** on static feature vectors to assign frame-level manipulation probabilities.
4. **Temporal Score Aggregation:** Aggregates frame probabilities into video-level decisions using mean pooling and top-$k$ trimmed peak anomaly scoring.

---

## 🏗️ System Architecture

```text
Raw Video (.mp4 / .avi)
         │
         ▼
[Temporal Subsampling (Stride S)]
         │
         ▼
[MTCNN Face Detection & Crop Margin (+20%)]
         │
         ▼ Aligned Face Patches (224x224)
[Frozen ResNet-50 Feature Extractor]  <── ImageNet-1K Pretrained (Weights Fixed)
         │
         ▼ Embeddings (2048-d vectors)
[XGBoost Classification Head]
         │
         ▼
[Frame Probability Predictions (pt)] ──► [Temporal Aggregation] ──► Final Verdict
