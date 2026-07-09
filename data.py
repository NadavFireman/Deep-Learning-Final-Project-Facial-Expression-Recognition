"""
Dataset utilities: count, split, load and preprocess the images.
"""
import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image


VALID_EXTS = {'.jpg', '.jpeg', '.png'}


def hello_data():
    print("Hello from data.py!")


def count_images(root, classes, exts=VALID_EXTS):
    """
    Count images in the train and test folders. Returns (n_train, n_test).
    """
    n_train = n_test = 0
    for cls in classes:
        n_train += sum(1 for f in (Path(root) / 'train' / cls).iterdir() if f.suffix.lower() in exts)
        n_test  += sum(1 for f in (Path(root) / 'test'  / cls).iterdir() if f.suffix.lower() in exts)
    return n_train, n_test


def split_data(root, classes, seed=42, exts=VALID_EXTS):
    """
    Stratified 60-20-20 split. The validation set is carved from the train
    folder to match the test-set size, so val and test end up the same size.
    Returns six lists: train_paths, train_labels, val_paths, val_labels,
    test_paths, test_labels.
    """
    n_train, n_test = count_images(root, classes, exts)
    val_frac = n_test / n_train

    rng = np.random.default_rng(seed)
    train_paths, train_labels = [], []
    val_paths,   val_labels   = [], []
    test_paths,  test_labels  = [], []

    for cls_idx, cls in enumerate(classes):
        train_files = sorted([f for f in (Path(root) / 'train' / cls).iterdir() if f.suffix.lower() in exts])
        test_files  = sorted([f for f in (Path(root) / 'test'  / cls).iterdir() if f.suffix.lower() in exts])
        perm = rng.permutation(len(train_files))
        n_val_cls = int(round(len(train_files) * val_frac))
        for i in perm[n_val_cls:]:
            train_paths.append(train_files[i]); train_labels.append(cls_idx)
        for i in perm[:n_val_cls]:
            val_paths.append(train_files[i]); val_labels.append(cls_idx)
        for f in test_files:
            test_paths.append(f); test_labels.append(cls_idx)

    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels


def class_distribution(labels, classes):
    """
    Return a per-class count and percentage table (DataFrame) for one split.
    """
    counts = {'count': {cls: labels.count(i) for i, cls in enumerate(classes)}}
    df = pd.DataFrame(counts)
    df.loc['TOTAL'] = df.sum()
    df['%'] = (df['count'] / df.loc['TOTAL', 'count'] * 100).round(2)
    return df


def _load_images_to_tensor(paths, labels, dtype=torch.float32, desc=None):
    """
    Load grayscale image files into a (N, 1, 48, 48) tensor in [0, 1].
    RGB files are collapsed to a single channel by averaging.
    """
    from tqdm.auto import tqdm
    n = len(paths)
    X = torch.empty(n, 1, 48, 48, dtype=dtype)
    y = torch.tensor(labels, dtype=torch.int64)
    for i, p in enumerate(tqdm(paths, desc=desc, leave=False)):
        img = Image.open(p)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        if arr.ndim == 3:
            arr = arr[:, :, :3].mean(axis=2)
        X[i, 0] = torch.from_numpy(arr)
    return X, y


def preprocess_data(
    train_paths, train_labels,
    val_paths, val_labels,
    test_paths, test_labels,
    cuda=True,
    flatten=False,
    dtype=torch.float32,
):
    """
    Load the images, optionally move to GPU, subtract the train mean image,
    and optionally flatten to (N, 2304). Returns a dict of six tensors plus
    the mean image.
    """
    X_train, y_train = _load_images_to_tensor(train_paths, train_labels, dtype, desc='train')
    X_val,   y_val   = _load_images_to_tensor(val_paths,   val_labels,   dtype, desc='val')
    X_test,  y_test  = _load_images_to_tensor(test_paths,  test_labels,  dtype, desc='test')

    if cuda and torch.cuda.is_available():
        X_train = X_train.cuda(); y_train = y_train.cuda()
        X_val   = X_val.cuda();   y_val   = y_val.cuda()
        X_test  = X_test.cuda();  y_test  = y_test.cuda()

    mean_image = X_train.mean(dim=0, keepdim=True)
    X_train = X_train - mean_image
    X_val   = X_val   - mean_image
    X_test  = X_test  - mean_image

    if flatten:
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_val   = X_val.reshape(X_val.shape[0], -1)
        X_test  = X_test.reshape(X_test.shape[0], -1)

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "y_test":  y_test,
        "mean_image": mean_image,
    }


def stage_to_local(drive_root, local_root, classes, exts=VALID_EXTS):
    """
    Copy the train/test image folders from Drive to a fast local disk once.
    Reading ~36k small files directly from Drive is extremely slow; copying
    the whole tree first and then reading from local disk is much faster.
    Returns local_root. Skips copying if local_root already exists.
    """
    import shutil
    if os.path.exists(local_root):
        return local_root
    for split in ('train', 'test'):
        for cls in classes:
            src = Path(drive_root) / split / cls
            dst = Path(local_root) / split / cls
            dst.mkdir(parents=True, exist_ok=True)
            for f in src.iterdir():
                if f.suffix.lower() in exts:
                    shutil.copy2(f, dst / f.name)
    return local_root


def load_or_build(cache_path, train_paths, train_labels, val_paths, val_labels,
                  test_paths, test_labels, cuda=True, flatten=False, dtype=torch.float32,
                  drive_cache_path=None):
    """
    Load preprocessed tensors from a .pt cache, otherwise build and save them.
    cache_path is a fast local path (e.g. /content/...). If drive_cache_path is
    given, the cache is kept on Drive across sessions: it is copied from Drive
    to the local path at the start, and copied to Drive once after building.
    cuda and flatten are applied after loading.
    """
    import shutil

    # Bring the cache from Drive to the fast local disk if needed.
    if drive_cache_path and os.path.exists(drive_cache_path) and not os.path.exists(cache_path):
        print(f'Copying cache from Drive ...')
        shutil.copy(drive_cache_path, cache_path)

    if os.path.exists(cache_path):
        print(f'Loading cache from {cache_path} ...')
        data_dict = torch.load(cache_path, map_location='cpu')
        print('Cache loaded.')
    else:
        print('No cache found. Loading images (this happens once)...')
        data_dict = preprocess_data(train_paths, train_labels, val_paths, val_labels,
                                    test_paths, test_labels, cuda=False, flatten=False, dtype=dtype)
        print(f'Saving cache to {cache_path} ...')
        torch.save(data_dict, cache_path)
        print('Cache saved.')
        # Copy the single cache file to Drive so it survives future sessions.
        if drive_cache_path and not os.path.exists(drive_cache_path):
            print('Copying cache to Drive ...')
            shutil.copy(cache_path, drive_cache_path)
            print('Cache copied to Drive.')

    if cuda and torch.cuda.is_available():
        for k in data_dict:
            data_dict[k] = data_dict[k].cuda()

    if flatten:
        for k in ("X_train", "X_val", "X_test"):
            data_dict[k] = data_dict[k].reshape(data_dict[k].shape[0], -1)

    return data_dict


def flatten_data(data_dict):
    """
    Return a copy of data_dict with the image tensors flattened to (N, 2304).
    """
    flat = dict(data_dict)
    for k in ("X_train", "X_val", "X_test"):
        flat[k] = data_dict[k].reshape(data_dict[k].shape[0], -1)
    return flat


def show_examples(paths, labels, classes, per_class=5, seed=42):
    """
    Plot a grid of example images: one row per class, `per_class` columns.
    """
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(seed)
    by_class = {i: [] for i in range(len(classes))}
    for p, l in zip(paths, labels):
        by_class[l].append(p)

    n_rows, n_cols = len(classes), per_class
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.3, n_rows * 1.3))
    for row, cls in enumerate(classes):
        chosen = rng.choice(len(by_class[row]), size=per_class, replace=False)
        for col, idx in enumerate(chosen):
            ax = axes[row, col]
            img = Image.open(by_class[row][idx]).convert('L')
            ax.imshow(np.asarray(img), cmap='gray')
            ax.axis('off')
            if col == 0:
                ax.set_ylabel(cls, rotation=0, ha='right', va='center', fontsize=11)
                ax.axis('on')
                ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.show()