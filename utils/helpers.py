"""
Utility functions shared across training scripts.
"""

import os
import random
import torch
import numpy as np
import config


# ── Reproducibility ────────────────────────────────────────────────────────────

def set_seed(seed=None):
    seed = seed or config.SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── Device selection ───────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"[device] Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
        print("[device] Using Apple MPS (M1/M2 GPU)")
    else:
        dev = torch.device("cpu")
        print("[device] No GPU found — using CPU (training will be slow)")
    return dev


# ── Running average meter ──────────────────────────────────────────────────────

class AverageMeter:
    """Tracks a running mean — useful for loss / accuracy over an epoch."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val, n=1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def save_checkpoint(state: dict, filename: str):
    """state should contain at least {'epoch', 'model_state', 'best_acc'}"""
    path = os.path.join(config.CHECKPOINT_DIR, filename)
    torch.save(state, path)
    print(f"[ckpt] Saved → {path}")


def load_checkpoint(filename: str, model, optimizer=None, device="cpu"):
    """
    Loads checkpoint into model (and optionally optimizer).
    Returns (start_epoch, best_acc).
    """
    path = os.path.join(config.CHECKPOINT_DIR, filename)
    if not os.path.isfile(path):
        print(f"[ckpt] No checkpoint at {path} — starting fresh")
        return 0, 0.0

    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])

    epoch    = ckpt.get("epoch",    0)
    best_acc = ckpt.get("best_acc", 0.0)
    print(f"[ckpt] Resumed from {path}  (epoch {epoch}, best_acc {best_acc:.2f}%)")
    return epoch, best_acc


# ── Model size ─────────────────────────────────────────────────────────────────

def count_parameters(model):
    """Returns (total_params, trainable_params, size_mb)"""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb   = total * 4 / 1e6   # float32 = 4 bytes
    return total, trainable, size_mb


# ── Top-k accuracy ─────────────────────────────────────────────────────────────

def topk_accuracy(output, target, topk=(1, 5)):
    """Returns list of top-k accuracy scalars."""
    with torch.no_grad():
        maxk  = max(topk)
        batch = target.size(0)
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        results = []
        for k in topk:
            c = correct[:k].reshape(-1).float().sum(0)
            results.append((c / batch * 100).item())
        return results