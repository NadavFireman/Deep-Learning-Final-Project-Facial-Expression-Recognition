"""
Shared helpers for training and evaluation: seeding, plotting and metrics.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix as sklearn_cm


def reset_seed(seed=42):
    """
    Reset the random seeds of random, numpy and torch for reproducibility.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    """
    Return the number of trainable parameters in a model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_stats(stats):
    """
    Plot loss and accuracy history. `stats` is a dict with keys:
    'loss_history', 'train_acc_history', 'val_acc_history' (any may be absent).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    if 'loss_history' in stats:
        ax1.plot(stats['loss_history'], 'o', markersize=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss history')

    if 'train_acc_history' in stats:
        ax2.plot(stats['train_acc_history'], '-o', label='train')
    if 'val_acc_history' in stats:
        ax2.plot(stats['val_acc_history'], '-o', label='val')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy history')
    ax2.legend()

    plt.tight_layout()
    plt.show()
    

def plot_confusion_matrix(y_true, y_pred, classes, normalize=True):
    """
    Plot a confusion matrix as a seaborn heatmap. If normalize=True, rows sum to 1.
    """
    import seaborn as sns
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    cm = sklearn_cm(y_true, y_pred, labels=range(len(classes)))
    
    fmt = 'd'
    if normalize:
        cm = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True).clip(min=1)
        fmt = '.2f'

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', cbar=True,
                xticklabels=classes, yticklabels=classes,
                square=True, linewidths=0.5)
    plt.xlabel('predicted')
    plt.ylabel('true')
    plt.title('Confusion matrix')
    plt.tight_layout()
    plt.show()

def print_confusion_matrix(y_true, y_pred, classes, normalize=True):
    """
    Computes the confusion matrix and prints it as a clean text-based table.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    cm = sklearn_cm(y_true, y_pred, labels=range(len(classes)))
    
    if normalize:
        cm = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True).clip(min=1)
        cm = np.round(cm, 2)

    print("\n--- Confusion Matrix (Text Table) ---")

    header = f"{'True \\ Pred':<15}" + "".join([f"{c:>10}" for c in classes])
    print(header)
    print("-" * len(header))
    
    for i, cls_true in enumerate(classes):
        row_str = f"{cls_true:<15}"
        for j in range(len(classes)):
            val = cm[i, j]
            if normalize:
                row_str += f"{val:>10.2f}"
            else:
                row_str += f"{int(val):>10}"
        print(row_str)
    
    print("-" * len(header) + "\n")
    return cm


def evaluate(model, X, y, batch_size=256):
    """
    Run the model on X in batches and return accuracy and predictions.
    Returns (accuracy, y_pred) where y_pred is a 1D numpy array.
    """
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            scores = model(X[i:i + batch_size])
            preds.append(scores.argmax(dim=1).cpu())
    y_pred = torch.cat(preds).numpy()
    y_true = y.cpu().numpy()
    acc = (y_pred == y_true).mean()
    return acc, y_pred