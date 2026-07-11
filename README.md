# Deep Learning Final Project - Facial Expression Recognition

**Final Project (M.Sc. Data Science, HIT). Facial expression recognition on 48×48 grayscale faces — seven emotion classes, noisy labels, severe imbalance (disgust ≈ 1.5%). Five models compared under one protocol (stratified 60-20-20 split, best-validation snapshots, per-class confusion matrices), from a from-scratch linear SVM to a fine-tuned Swin Transformer at **69.7% test accuracy**.**

## The Five Models
- **Linear SVM (from scratch):** **37.3%** test accuracy.
- **Simple CNN (from scratch, 2 conv blocks):** **50.8%**.
- **Deep CNN (from scratch, 5 blocks + class weighting):** **62.8%** — within ~7 points of the pretrained transformer.
- **ResNet18 (transfer learning, 48×48-preserving stem):** **66.2%**.
- **Swin-T (transfer learning, fine-tuned at 224×224):** **69.7%** — approaching published SOTA (~73–76%).

## Key Features
- **Training Engineering:** staged hyperparameter search, mixed precision (AMP), tensor caching, early stopping; `pretrained.py` also supports ResNet50, ViT, Swin-S/V2 and MaxViT.
- **Error Analysis:** normalized confusion matrices for every model — happy/surprise easiest, fear/sad hardest.

## Repository Structure
- `final_project_deep_learning.ipynb`: Driver notebook, all results executed.
- `final_project_deep_learning.pdf`: PDF export of the executed notebook.
- `data/` — Full dataset:
  - `train/`: seven class folders - `angry/`, `disgust/`, `fear/`, `happy/`, `neutral/`, `sad/`, `surprise/`
  - `test/`: seven class folders - `angry/`, `disgust/`, `fear/`, `happy/`, `neutral/`, `sad/`, `surprise/`
- `data.py`: Stratified split, loading, normalization and caching.
- `linear_classifier.py`: From-scratch multiclass SVM + SGD trainer.
- `simple_cnn.py`: Baseline two-block CNN.
- `deep_cnn.py`: Five-block DeepCNN, GPU augmentation, class-weighted trainer.
- `pretrained.py`: Transfer-learning wrapper (warmup + cosine, label smoothing, AMP).
- `helpers.py`: Seeding, training curves, confusion-matrix utilities.

## Dataset
~35.9K grayscale 48×48 face images (≈28.7K train / ≈7.2K test), included under `data/`.
