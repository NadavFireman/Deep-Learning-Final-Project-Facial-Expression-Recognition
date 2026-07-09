# Deep Learning Final Project - Facial Expression Recognition

**Final Project (M.Sc. Data Science, HIT). An end-to-end facial expression recognition pipeline on 48×48 grayscale faces — five models in increasing order of complexity, from a from-scratch linear SVM to a fine-tuned Swin Transformer reaching **69.7% test accuracy**, above the estimated human-level baseline (~65%) on this dataset. Developed as a two-person team project.**

## Overview
The task is to classify each face image into one of seven emotion categories (angry, disgust, fear, happy, neutral, sad, surprise). The dataset is genuinely hard: low-resolution grayscale images, noisy labels, and severe class imbalance — the rarest class (disgust) makes up only ~1.5% of the training data. The project builds and compares five models under a single consistent protocol (stratified 60-20-20 split, best-validation snapshots, per-class confusion matrices), isolating how much each ingredient — depth, regularization, augmentation, class weighting, and ImageNet pretraining — actually contributes.

## Key Features
- **Five-Model Comparison:** Models built and evaluated in increasing order of complexity:
  - **Linear SVM (from scratch):** 37.3% test accuracy — the linear baseline.
  - **Simple CNN (2 conv blocks, from scratch):** 50.8% test accuracy.
  - **Deep CNN (5 conv blocks, from scratch):** **62.8% test accuracy** — the standout from-scratch result, within ~7 points of the pretrained transformer.
  - **ResNet18 (transfer learning):** 66.2% test accuracy.
  - **Swin-T (transfer learning):** **69.7% test accuracy** — the best model in the project, approaching published SOTA (~73–76%).
- **From-Scratch Linear SVM:** Vectorized multiclass hinge loss trained with mini-batch SGD, supporting L2, L1 and Elastic Net regularization.
- **Custom Deep CNN:** Five Conv → BatchNorm → GELU → MaxPool blocks with Global Average Pooling and an FC-Dropout head — trained with zero pretrained weights.
- **GPU-Native Data Augmentation:** Per-image random flip, rotation (±10°) and translation implemented as a single batched affine-grid transform — runs entirely on the GPU, with no CPU/DataLoader round-trip.
- **Class-Imbalance Handling:** Class-weighted CrossEntropyLoss computed from the training distribution — lifted the rare *disgust* class from 4.5% (SVM) to **72.1%** (Swin-T) test accuracy.
- **Transfer Learning:** ResNet18 with a small-input stem (3×3 stride-1 conv, no maxpool) that preserves 48×48 resolution, and Swin-T fine-tuned at 224×224 with LR warmup + cosine decay, label smoothing and gradient clipping. The `pretrained.py` wrapper also supports ResNet50, ViT, Swin-S/V2 and MaxViT backbones.
- **Staged Hyperparameter Search:** Chained grid search — architecture stages first, then optimization — with short runs selected by validation accuracy.
- **Training Engineering:** Mixed precision (AMP), preprocessed-tensor caching, best-validation snapshots and early stopping.
- **Error Analysis:** Normalized confusion matrices for every model and a per-emotion comparison across all five — happy/surprise emerge as the easiest classes, fear/sad as the hardest.

## Tech Stack
- **Language:** Python
- **Framework:** PyTorch + Torchvision (pretrained backbones)
- **Evaluation & Visualization:** Scikit-learn, Matplotlib, Seaborn
- **Environment:** Google Colab (GPU)

## Repository Structure
- `final_project_deep_learning.ipynb`: Driver notebook (Introduction → Dataset → Method → Experiments → Conclusion) with all executed results.
- `final_project_deep_learning.pdf`: PDF export of the executed notebook.
- `data/`: The full dataset — `train/` and `test/` folders, one subfolder per emotion class.
- `data.py`: Dataset counting, stratified 60-20-20 split, image→tensor loading, mean-image normalization and caching.
- `linear_classifier.py`: From-scratch multiclass SVM (L2/L1/Elastic Net) with SGD trainer and grid search.
- `simple_cnn.py`: Baseline two-block CNN with trainer and grid search.
- `pretrained.py`: Transfer-learning wrapper and fine-tuning trainer (warmup + cosine schedule, label smoothing, gradient clipping, AMP).
- `deep_cnn.py`: Five-block DeepCNN, GPU batch augmentation, class-weighted trainer and chained grid search.
- `helpers.py`: Seeding, parameter counting, training-curve plots and confusion-matrix utilities.

## Dataset
A facial expression (FER) dataset of ~35.9K grayscale 48×48 face images in seven emotion categories (≈28.7K training / ≈7.2K test), included in this repository under `data/train/` and `data/test/` — one folder per class. Point the notebook's data path to the `data/` directory.
