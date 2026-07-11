# Deep Learning Final Project - Facial Expression Recognition

**Final Project (M.Sc. Data Science, HIT). Facial expression recognition on 48×48 grayscale faces — five models in increasing complexity, from a from-scratch linear SVM to a fine-tuned Swin Transformer at 69.7% test accuracy, above the estimated human-level baseline (~65%). Developed as a two-person team project.**

## Overview
Seven emotion classes on a genuinely hard dataset — low-resolution, noisy labels, severe imbalance (disgust ≈ 1.5% of training) — with all five models evaluated under one consistent protocol: stratified 60-20-20 split, best-validation snapshots, per-class confusion matrices.

## The Five Models
- **Linear SVM (from scratch):** vectorized multiclass hinge loss with SGD and L2/L1/Elastic Net — **37.3%** test accuracy.
- **Simple CNN (from scratch):** two conv blocks — **50.8%**.
- **Deep CNN (from scratch):** five Conv → BatchNorm → GELU → MaxPool blocks, GAP + Dropout head, GPU augmentation and class weighting — **62.8%**, within ~7 points of the pretrained transformer.
- **ResNet18 (transfer learning):** small-input stem (3×3 stride-1, no maxpool) preserving 48×48 resolution — **66.2%**.
- **Swin-T (transfer learning):** fine-tuned at 224×224 with LR warmup + cosine decay, label smoothing and gradient clipping — **69.7%**, approaching published SOTA (~73–76%).

## Key Features
- **Staged Hyperparameter Search:** chained grids — architecture first, then optimization — selected by validation accuracy.
- **Training Engineering:** mixed precision (AMP), preprocessed-tensor caching, early stopping; `pretrained.py` also supports ResNet50, ViT, Swin-S/V2 and MaxViT backbones.
- **Error Analysis:** normalized confusion matrices for every model — happy/surprise easiest, fear/sad hardest.

## Repository Structure
- `final_project_deep_learning.ipynb`: Driver notebook (Introduction → Dataset → Method → Experiments → Conclusion), all results executed.
- `final_project_deep_learning.pdf`: PDF export of the executed notebook.
- `data/`: The full dataset — `train/` and `test/`, one subfolder per emotion class.
- `data.py`: Dataset counting, stratified split, image→tensor loading, normalization and caching.
- `linear_classifier.py`: From-scratch multiclass SVM with SGD trainer and grid search.
- `simple_cnn.py`: Baseline two-block CNN with trainer and grid search.
- `deep_cnn.py`: Five-block DeepCNN, GPU batch augmentation, class-weighted trainer, chained grid search.
- `pretrained.py`: Transfer-learning wrapper and fine-tuning trainer (warmup + cosine, label smoothing, clipping, AMP).
- `helpers.py`: Seeding, parameter counting, training curves and confusion-matrix utilities.

## Dataset
~35.9K grayscale 48×48 face images in seven emotion categories (≈28.7K train / ≈7.2K test), included under `data/` — point the notebook's data path there.
