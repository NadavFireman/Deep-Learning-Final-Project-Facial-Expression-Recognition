"""
Deep CNN for facial expression recognition (task 3).
Five conv blocks (Conv -> BatchNorm -> GELU -> MaxPool) then FC -> Dropout -> FC,
trained with AdamW, GPU data augmentation, lr decay, best-val snapshot and early
stopping. This is the regularized, deeper model that breaks the simple
baseline's overfitting ceiling.
"""
import copy
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True


def hello_deep_cnn():
    print("Hello from deep_cnn.py!")


class DeepCNN(nn.Module):
    """
    Five conv blocks (Conv-BN-GELU-MaxPool) then FC-Dropout-FC.
    Input: (N, 1, 48, 48). After 5 pools: 48->24->12->6->3->1. Output: (N, num_classes).
    """

    def __init__(self, conv1=48, conv2=96, conv3=192, conv4=256, conv5=256,
                 hidden_dim=512, dropout=0.5, num_classes=7, kernel_size=3, stride=1):
        super().__init__()
        self.stride = stride
        pad = kernel_size // 2
        self.conv1 = nn.Conv2d(1, conv1, kernel_size=kernel_size, stride=stride, padding=pad)
        self.bn1 = nn.BatchNorm2d(conv1)
        self.conv2 = nn.Conv2d(conv1, conv2, kernel_size=kernel_size, stride=stride, padding=pad)
        self.bn2 = nn.BatchNorm2d(conv2)
        self.conv3 = nn.Conv2d(conv2, conv3, kernel_size=kernel_size, stride=stride, padding=pad)
        self.bn3 = nn.BatchNorm2d(conv3)
        self.conv4 = nn.Conv2d(conv3, conv4, kernel_size=kernel_size, stride=stride, padding=pad)
        self.bn4 = nn.BatchNorm2d(conv4)
        self.conv5 = nn.Conv2d(conv4, conv5, kernel_size=kernel_size, stride=stride, padding=pad)
        self.bn5 = nn.BatchNorm2d(conv5)
        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(conv5, hidden_dim)
        self.bn_fc = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def _block(self, x, conv, bn):
        x = F.gelu(bn(conv(x)))
        if self.stride == 1:
            x = self.pool(x)
        return x

    def forward(self, x):
        x = self._block(x, self.conv1, self.bn1)
        x = self._block(x, self.conv2, self.bn2)
        x = self._block(x, self.conv3, self.bn3)
        x = self._block(x, self.conv4, self.bn4)
        x = self._block(x, self.conv5, self.bn5)
        x = self.gap(x).flatten(start_dim=1)
        x = F.gelu(self.bn_fc(self.fc1(x)))
        x = self.dropout(x)
        return self.fc2(x)


def augment_batch(x, max_shift=4, p_flip=0.5, max_angle=10.0):
    """
    Fast GPU augmentation on a whole batch (N, 1, 48, 48), each image
    independently: random horizontal flip, random small rotation and random
    translation. Runs entirely on x's device (no CPU / DataLoader round-trip).
    """
    N, C, H, W = x.shape
    device = x.device

    # per-image random horizontal flip
    flip = torch.rand(N, device=device) < p_flip
    x[flip] = torch.flip(x[flip], dims=[3])

    # per-image random rotation + translation via an affine grid
    angles = (torch.rand(N, device=device) * 2 - 1) * (max_angle * 3.14159265 / 180.0)
    cos, sin = torch.cos(angles), torch.sin(angles)
    tx = (torch.rand(N, device=device) * 2 - 1) * (2.0 * max_shift / W)
    ty = (torch.rand(N, device=device) * 2 - 1) * (2.0 * max_shift / H)
    theta = torch.zeros(N, 2, 3, device=device, dtype=x.dtype)
    theta[:, 0, 0] = cos;  theta[:, 0, 1] = -sin; theta[:, 0, 2] = tx
    theta[:, 1, 0] = sin;  theta[:, 1, 1] = cos;  theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, x.size(), align_corners=False)
    x = F.grid_sample(x, grid, align_corners=False, padding_mode='reflection')
    return x


def _accuracy(model, X, y, batch_size=4096):
    """Compute accuracy (%) over X in large batches (GPU-friendly)."""
    model.eval()
    correct = torch.zeros((), device=X.device)
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            pred = model(X[i:i + batch_size]).argmax(dim=1)
            correct += (pred == y[i:i + batch_size]).sum()
    return 100.0 * correct.item() / X.shape[0]


def train_deep_cnn(model, X_train, y_train, X_val, y_val,
                   lr=1e-3, weight_decay=1e-4, batch_size=256,
                   num_epochs=60, lr_decay=0.99, augment=True,
                   patience=0, amp=True, seed=42, verbose=True,
                   class_weights=None):
    """
    Train a DeepCNN with AdamW. If augment is True, each training batch gets a
    fast GPU flip+rotate+shift augmentation. Decays lr each epoch, keeps the
    best-val snapshot (restored at the end) and optionally early-stops after
    `patience` epochs without improvement (0 = off). If amp and CUDA, uses
    mixed precision. If class_weights (a 1-D tensor of length num_classes) is
    given, the CrossEntropyLoss is weighted per class - used to counteract the
    class imbalance (e.g. the rare 'disgust' class).

    Returns a dict with 'loss_history', 'train_acc_history', 'val_acc_history',
    'best_val_acc' and 'best_epoch'.
    """
    torch.manual_seed(seed)
    device = X_train.device
    model = model.to(device)
    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    use_amp = amp and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    N = X_train.shape[0]
    loss_history, train_acc_history, val_acc_history = [], [], []
    best_val, best_state, best_epoch = -1.0, None, 0
    patience_counter = 0
    cur_lr = lr

    for epoch in range(1, num_epochs + 1):
        model.train()
        perm = torch.randperm(N, device=device)
        running_loss, num_batches = 0.0, 0
        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            xb = X_train[idx]
            if augment:
                xb = augment_batch(xb.clone())
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = criterion(model(xb), y_train[idx])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            num_batches += 1
            loss_history.append(loss.item())

        train_acc = _accuracy(model, X_train, y_train)
        val_acc = _accuracy(model, X_val, y_val)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        marker = ""
        if val_acc > best_val:
            best_val, best_state, best_epoch = val_acc, copy.deepcopy(model.state_dict()), epoch
            patience_counter = 0
            marker = "  <- NEW BEST"
        else:
            patience_counter += 1

        if verbose:
            print(f'Epoch {epoch:3d}/{num_epochs} | loss: {running_loss/num_batches:.4f} | '
                  f'train: {train_acc:.2f}% | val: {val_acc:.2f}%{marker}')

        cur_lr *= lr_decay
        for g in optimizer.param_groups:
            g['lr'] = cur_lr

        if patience and patience_counter >= patience:
            if verbose:
                print(f'*** early stopping at epoch {epoch} ***')
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        'loss_history': loss_history,
        'train_acc_history': train_acc_history,
        'val_acc_history': val_acc_history,
        'best_val_acc': best_val,
        'best_epoch': best_epoch,
    }


def grid_search_deep_cnn(X_train, y_train, X_val, y_val, param_grid,
                         num_epochs=5, augment=True, amp=True,
                         seed=42, verbose=True, class_weights=None):
    """
    Train a DeepCNN for every combination in param_grid (short runs) and keep
    the best by val accuracy. Model keys: conv1-conv5, hidden_dim, dropout,
    kernel_size, stride. Train keys: lr, weight_decay, batch_size, lr_decay.
    Use a few epochs for fast hyperparameter search.

    Returns (best_model, best_params, results), results sorted by val acc.
    """
    model_keys = {'conv1', 'conv2', 'conv3', 'conv4', 'conv5',
                  'hidden_dim', 'dropout', 'kernel_size', 'stride'}
    keys = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))

    best_model, best_params, best_val = None, None, -1.0
    results = []
    for combo in combos:
        params = dict(zip(keys, combo))
        m_args = {k: v for k, v in params.items() if k in model_keys}
        t_args = {k: v for k, v in params.items() if k not in model_keys}
        model = DeepCNN(**m_args)
        stats = train_deep_cnn(model, X_train, y_train, X_val, y_val,
                               num_epochs=num_epochs, augment=augment,
                               amp=amp, seed=seed, verbose=False,
                               class_weights=class_weights, **t_args)
        va = stats['best_val_acc']
        results.append({**params, 'val_acc': va})
        if verbose:
            ps = ', '.join(f'{k}={v}' for k, v in params.items())
            print(f'{ps:75s} | val {va:5.2f}')
        if va > best_val:
            best_val, best_model, best_params = va, model, params

    results.sort(key=lambda r: r['val_acc'], reverse=True)
    if verbose:
        print(f'\nBest: {best_params} | val acc {best_val:.2f}')
    return best_model, best_params, results