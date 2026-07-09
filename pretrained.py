"""
Transfer-learning wrapper for the Facial Expression dataset. Loads an
ImageNet-pretrained torchvision backbone - resnet18, resnet50, vit_b_16,
vit_b_32, swin_t, swin_s, swin_v2_t, swin_v2_s or maxvit_t - repeats the 1-channel input
to 3, resizes if needed, and replaces the head with Dropout + a 7-class linear
layer. For ResNet on small inputs it swaps in a small-image stem (3x3 stride-1,
no maxpool) to keep resolution. Ships its own trainer (train_pretrained);
depends only on torch and torchvision.
"""
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def hello_pretrained():
    print("Hello from pretrained.py!")


class PretrainedFER(nn.Module):
    """
    backbone: pretrained network (resnet18/resnet50, vit_b_16/vit_b_32,
              swin_t/swin_s/swin_v2_t/swin_v2_s, maxvit_t).
    pretrained: ImageNet weights (True) or random init (False).
    freeze_backbone: True = train only the head; False = fine-tune all.
    img_size: resize before the backbone (transformers force 224).
    dropout: dropout before the final linear head.
    small_input: small-image ResNet stem for img_size<=64 when fine-tuning.
    """

    def __init__(self, backbone='resnet18', num_classes=7, img_size=48,
                 pretrained=True, freeze_backbone=False, dropout=0.2,
                 small_input=True):
        super().__init__()
        self.img_size = img_size
        weights = 'DEFAULT' if pretrained else None
        is_resnet = backbone in ('resnet18', 'resnet50')

        if backbone == 'resnet18':
            self.net = torchvision.models.resnet18(weights=weights)
            in_features = self.net.fc.in_features
            self.net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
            self._head = self.net.fc
        elif backbone == 'resnet50':
            self.net = torchvision.models.resnet50(weights=weights)
            in_features = self.net.fc.in_features
            self.net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
            self._head = self.net.fc
        elif backbone in ('vit_b_16', 'vit_b_32'):
            self.net = getattr(torchvision.models, backbone)(weights=weights)
            in_features = self.net.heads.head.in_features
            self.net.heads.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
            self._head = self.net.heads.head
            self.img_size = 224
            small_input = False
        elif backbone in ('swin_t', 'swin_s', 'swin_v2_t', 'swin_v2_s'):
            self.net = getattr(torchvision.models, backbone)(weights=weights)
            in_features = self.net.head.in_features
            self.net.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
            self._head = self.net.head
            self.img_size = 224
            small_input = False
        elif backbone == 'maxvit_t':
            # תוספת עבור MaxViT
            self.net = torchvision.models.maxvit_t(weights=weights)
            in_features = self.net.classifier[-1].in_features
            self.net.classifier[-1] = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
            self._head = self.net.classifier[-1]
            self.img_size = 224
            small_input = False
        else:
            raise ValueError(f'unknown backbone: {backbone}')

        if is_resnet and small_input and (not freeze_backbone) and self.img_size <= 64:
            self.net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.net.maxpool = nn.Identity()

        if freeze_backbone:
            for p in self.net.parameters():
                p.requires_grad = False
            for p in self._head.parameters():
                p.requires_grad = True

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if x.shape[-1] != self.img_size:
            x = F.interpolate(x, size=self.img_size, mode='bilinear', align_corners=False)
        return self.net(x)


def augment_batch(x, max_shift=4, p_flip=0.5, max_angle=10.0):
    """
    Fast GPU augmentation on a whole batch (N, 1, 48, 48), each image
    independently: random horizontal flip, small rotation and translation.
    Runs entirely on x's device (no CPU / DataLoader round-trip).
    """
    N, C, H, W = x.shape
    device = x.device

    flip = torch.rand(N, device=device) < p_flip
    x[flip] = torch.flip(x[flip], dims=[3])

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


def _accuracy(model, X, y, batch_size=256):
    """Compute accuracy (%) over X in large batches (GPU-friendly)."""
    model.eval()
    correct = torch.zeros((), device=X.device)
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            pred = model(X[i:i + batch_size]).argmax(dim=1)
            correct += (pred == y[i:i + batch_size]).sum()
    return 100.0 * correct.item() / X.shape[0]


def train_pretrained(model, X_train, y_train, X_val, y_val,
                     lr=1e-4, weight_decay=1e-4, batch_size=128,
                     num_epochs=60, lr_decay=0.95, warmup_epochs=2, augment=True,
                     patience=0, amp=True, seed=42, verbose=True, class_weights=None,
                     label_smoothing=0.1, max_grad_norm=1.0, use_cosine=True,
                     min_lr_ratio=0.01):
    """
    Fine-tune a PretrainedFER (or any model(x)->logits) with AdamW. Linear lr
    warmup over `warmup_epochs`, then cosine decay to lr*min_lr_ratio
    (use_cosine=True, default) or multiply by `lr_decay` each epoch
    (use_cosine=False). Gradients clipped to `max_grad_norm` (0 to disable),
    loss uses `label_smoothing` and optional per-class `class_weights`. GPU
    augmentation, per-batch loss logging, best-val snapshot (restored at the
    end) and optional early stopping. Self-contained - does not import any
    other project module.

    Returns a dict with 'loss_history', 'train_acc_history', 'val_acc_history',
    'best_val_acc' and 'best_epoch'.
    """
    torch.manual_seed(seed)
    device = X_train.device
    model = model.to(device)
    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    use_amp = amp and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    N = X_train.shape[0]
    loss_history, train_acc_history, val_acc_history = [], [], []
    best_val, best_state, best_epoch = -1.0, None, 0
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        if epoch <= warmup_epochs:
            cur_lr = lr * epoch / max(1, warmup_epochs)
        elif use_cosine:
            progress = (epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
            cur_lr = lr * (min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))
        else:
            cur_lr = lr * (lr_decay ** (epoch - warmup_epochs))
        for g in optimizer.param_groups:
            g['lr'] = cur_lr

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
            if max_grad_norm and max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            num_batches += 1
            loss_history.append(loss.item())

        train_acc = _accuracy(model, X_train, y_train, batch_size)
        val_acc = _accuracy(model, X_val, y_val, batch_size)
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