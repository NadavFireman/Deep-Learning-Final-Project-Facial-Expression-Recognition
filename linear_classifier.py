"""
Linear SVM classifier (multiclass) trained from scratch with SGD.
Supports L2, L1 and Elastic Net regularization.
"""
import itertools
import torch


def hello_linear_classifier():
    print("Hello from linear_classifier.py!")


def svm_loss_vectorized(W, X, y, reg, reg_type='l2', alpha=0.5):
    """
    Multiclass SVM loss (vectorized) with L2, L1 or Elastic Net regularization.
    For elastic net: reg * (alpha * |W| + (1 - alpha) * W^2).
    W: (D, C), X: (N, D), y: (N,). Returns (loss, dW).
    """
    N = X.shape[0]
    scores = X.mm(W)
    correct = scores[torch.arange(N), y].view(-1, 1)
    margins = (scores - correct + 1.0).clamp(min=0)
    margins[torch.arange(N), y] = 0.0
    loss = margins.sum() / N

    mask = (margins > 0).to(W.dtype)
    mask[torch.arange(N), y] = -mask.sum(dim=1)
    dW = X.t().mm(mask) / N

    if reg_type == 'l2':
        loss += reg * (W * W).sum()
        dW += 2 * reg * W
    elif reg_type == 'l1':
        loss += reg * W.abs().sum()
        dW += reg * torch.sign(W)
    elif reg_type == 'elastic':
        loss += reg * (alpha * W.abs().sum() + (1 - alpha) * (W * W).sum())
        dW += reg * (alpha * torch.sign(W) + (1 - alpha) * 2 * W)
    return loss, dW


class LinearSVM:
    """
    Multiclass linear SVM trained with mini-batch SGD.
    """

    def __init__(self):
        self.W = None

    def train(self, X, y, lr=1e-2, reg=1e-4, reg_type='l2', alpha=0.5,
              num_iters=2000, batch_size=512, seed=42, lr_decay=1.0,
              X_val=None, y_val=None, track_every=0, verbose=True):
        """
        Train with SGD. X: (N, D), y: (N,). lr is multiplied by lr_decay each
        iteration. If X_val/y_val and track_every>0 are given, train and val
        accuracy are recorded every track_every iterations.

        Returns a dict with 'loss_history', and (if tracked) 'train_acc_history'
        and 'val_acc_history'.
        """
        g = torch.Generator(device=X.device).manual_seed(seed)
        N, D = X.shape
        C = int(y.max().item()) + 1
        if self.W is None:
            self.W = 1e-6 * torch.randn(D, C, generator=g, device=X.device, dtype=X.dtype)

        loss_history = []
        train_acc_history, val_acc_history = [], []
        cur_lr = lr
        for it in range(num_iters):
            idx = torch.randint(0, N, (batch_size,), generator=g, device=X.device)
            loss, dW = svm_loss_vectorized(self.W, X[idx], y[idx], reg, reg_type, alpha)
            self.W -= cur_lr * dW
            cur_lr *= lr_decay
            loss_history.append(loss.item())

            if track_every and (it % track_every == 0):
                train_acc_history.append((self.predict(X) == y).double().mean().item() * 100)
                if X_val is not None:
                    val_acc_history.append((self.predict(X_val) == y_val).double().mean().item() * 100)

            if verbose and (it % 200 == 0):
                print(f'iteration {it:5d} / {num_iters}: loss {loss.item():.4f}')

        stats = {'loss_history': loss_history}
        if track_every:
            stats['train_acc_history'] = train_acc_history
            if X_val is not None:
                stats['val_acc_history'] = val_acc_history
        return stats

    def predict(self, X):
        """
        Return predicted class indices for X. X: (N, D) -> (N,).
        """
        return X.mm(self.W).argmax(dim=1)


def accuracy(model, X, y):
    """
    Return classification accuracy in percent (0-100).
    """
    return 100.0 * (model.predict(X) == y).double().mean().item()


def grid_search_svm(X_train, y_train, X_val, y_val, param_grid,
                    num_iters=2000, batch_size=512, seed=42, verbose=True):
    """
    Train a LinearSVM for every combination in param_grid and keep the one
    with the best validation accuracy. Any LinearSVM.train argument
    (lr, reg, reg_type, alpha, num_iters, batch_size) can be put in the grid.

    Returns (best_svm, best_params, results), results sorted by val acc.
    """
    keys = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))

    best_svm, best_params, best_val = None, None, -1.0
    results = []
    for combo in combos:
        params = dict(zip(keys, combo))
        train_kwargs = dict(num_iters=num_iters, batch_size=batch_size, seed=seed, verbose=False)
        train_kwargs.update(params)
        svm = LinearSVM()
        svm.train(X_train, y_train, **train_kwargs)
        tr = accuracy(svm, X_train, y_train)
        va = accuracy(svm, X_val, y_val)
        results.append({**params, 'train_acc': tr, 'val_acc': va})
        if verbose:
            ps = ', '.join(f'{k}={v}' for k, v in params.items())
            print(f'{ps:60s} | train {tr:5.2f} | val {va:5.2f}')
        if va > best_val:
            best_val, best_svm, best_params = va, svm, params

    results.sort(key=lambda r: r['val_acc'], reverse=True)
    if verbose:
        print(f'\nBest: {best_params} | val acc {best_val:.2f}')
    return best_svm, best_params, results