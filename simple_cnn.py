"""
Simple baseline CNN for facial expression recognition.
Two conv blocks (Conv -> ReLU -> MaxPool) followed by a single fully
connected layer. No batch norm, dropout or other tricks - this is the
plain baseline that the deeper network in task 3 improves upon.
"""
import copy
import itertools
import torch
import torch.nn as nn

# Let cuDNN pick the fastest convolution algorithms for the fixed 48x48 input.
torch.backends.cudnn.benchmark = True


def hello_simple_cnn():
    print("Hello from simple_cnn.py!")


class SimpleCNN(nn.Module):
    """
    Two conv blocks (Conv-ReLU-MaxPool) then a single FC layer.
    Input: (N, 1, 48, 48). Output: (N, num_classes).
    """

    def __init__(self, conv1=32, conv2=64, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, conv1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                       # 48 -> 24
            nn.Conv2d(conv1, conv2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                       # 24 -> 12
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv2 * 12 * 12, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def _accuracy(model, X, y, batch_size=8192):
    """Compute accuracy (%) over X in large batches (GPU-friendly)."""
    model.eval()
    correct = torch.zeros((), device=X.device)
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            pred = model(X[i:i + batch_size]).argmax(dim=1)
            correct += (pred == y[i:i + batch_size]).sum()
    return 100.0 * correct.item() / X.shape[0]


def train_cnn(model, X_train, y_train, X_val, y_val,
              lr=1e-3, weight_decay=0.0, batch_size=128, num_epochs=30, lr_decay=1.0,
              train_eval_size=0, amp=True, seed=42, verbose=True):
    
    torch.manual_seed(seed)
    device = X_train.device
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    use_amp = amp and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    N = X_train.shape[0]
    if train_eval_size and train_eval_size < N:
        Xtr_eval, ytr_eval = X_train[:train_eval_size], y_train[:train_eval_size]
    else:
        Xtr_eval, ytr_eval = X_train, y_train

    loss_history, train_acc_history, val_acc_history = [], [], []
    best_val, best_state, best_epoch = -1.0, None, 0
    cur_lr = lr

    for epoch in range(1, num_epochs + 1):
        model.train()
        perm = torch.randperm(N, device=device)
        epoch_losses = []
        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = criterion(model(X_train[idx]), y_train[idx])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_losses.append(loss.detach())

        epoch_losses = torch.stack(epoch_losses)
        loss_history.extend(epoch_losses.cpu().tolist())
        mean_loss = epoch_losses.mean().item()

        train_acc = _accuracy(model, Xtr_eval, ytr_eval)
        val_acc = _accuracy(model, X_val, y_val)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        marker = ""
        if val_acc > best_val:
            best_val, best_state, best_epoch = val_acc, copy.deepcopy(model.state_dict()), epoch
            marker = "  <- NEW BEST"
        if verbose:
            print(f'Epoch {epoch:3d}/{num_epochs} | loss: {mean_loss:.4f} | '
                  f'train: {train_acc:.2f}% | val: {val_acc:.2f}%{marker}')

        cur_lr *= lr_decay
        for g in optimizer.param_groups:
            g['lr'] = cur_lr

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        'loss_history': loss_history,
        'train_acc_history': train_acc_history,
        'val_acc_history': val_acc_history,
        'best_val_acc': best_val,
        'best_epoch': best_epoch,
    }


def grid_search_cnn(X_train, y_train, X_val, y_val, param_grid,
                    num_epochs=30, train_eval_size=4000, amp=True,
                    seed=42, verbose=True):
    """
    Train a SimpleCNN for every combination in param_grid and keep the best by
    val accuracy. Grid keys can be model args (conv1, conv2) and train args
    (lr, batch_size, num_epochs, lr_decay). num_epochs in the grid overrides
    the default. train_eval_size limits the train-accuracy estimate to the
    first K samples for speed; amp enables mixed precision. val accuracy, which
    selects the winner, is always measured on the full set.

    Returns (best_model, best_params, results), results sorted by val acc.
    """
    model_keys = {'conv1', 'conv2'}
    keys = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))

    best_model, best_params, best_val = None, None, -1.0
    results = []
    for combo in combos:
        params = dict(zip(keys, combo))
        m_args = {k: v for k, v in params.items() if k in model_keys}
        t_args = {k: v for k, v in params.items() if k not in model_keys}
        t_args.setdefault('num_epochs', num_epochs)
        t_args.setdefault('train_eval_size', train_eval_size)
        t_args.setdefault('amp', amp)
        model = SimpleCNN(**m_args)
        stats = train_cnn(model, X_train, y_train, X_val, y_val,
                          seed=seed, verbose=False, **t_args)
        va = stats['best_val_acc']
        results.append({**params, 'val_acc': va})
        if verbose:
            ps = ', '.join(f'{k}={v}' for k, v in params.items())
            print(f'{ps:60s} | val {va:5.2f}')
        if va > best_val:
            best_val, best_model, best_params = va, model, params

    results.sort(key=lambda r: r['val_acc'], reverse=True)
    if verbose:
        print(f'\nBest: {best_params} | val acc {best_val:.2f}')
    return best_model, best_params, results