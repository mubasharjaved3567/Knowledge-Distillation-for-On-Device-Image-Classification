"""
Experiment 8 — Attention Transfer (All 3 Layers + Higher Beta)
===============================================================
Improves on Exp7 by:
1. Matching ALL 3 layers (layer1 + layer2 + layer3) simultaneously
2. Higher beta (1000.0) so attention loss actually contributes

In Exp7, at_loss=0.0003 with beta=0.1 → contribution was ~0.00003
Here,   at_loss=0.0003 with beta=1000 → contribution is ~0.3 (meaningful)

Total loss = alpha * CE(student, hard_labels)
           + (1-alpha) * T^2 * KL(student_soft || teacher_soft)
           + beta * [AT(layer1) + AT(layer2) + AT(layer3)]

Settings:
    T=4, alpha=0.95, beta=1000.0, epochs=300

Run:
    python train_student_attention_distill.py
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import config
from models  import resnet20, resnet110
from utils   import (get_cifar100_loaders, set_seed, get_device,
                     AverageMeter, save_checkpoint, load_checkpoint,
                     count_parameters)
from utils.helpers import topk_accuracy


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 8 hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
TEMPERATURE  = 4.0     # same as all previous best runs
ALPHA        = 0.95    # matches Exp5/6/7
BETA         = 1000.0  # much higher — makes attention loss meaningful
EPOCHS       = 300     # matches Exp5/6/7
MATCH_LAYERS = ["layer1", "layer2", "layer3"]   # ALL 3 layers
CKPT_NAME    = "student_attn_all3_T4_a0.95_b1000_Exp8_e300.pth"


# ─────────────────────────────────────────────────────────────────────────────
# Feature hook
# ─────────────────────────────────────────────────────────────────────────────
class FeatureExtractor:
    def __init__(self, model: nn.Module, layer_name: str):
        self.features = None
        layer = dict(model.named_modules())[layer_name]
        self._handle = layer.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        self.features = output

    def remove(self):
        self._handle.remove()


# ─────────────────────────────────────────────────────────────────────────────
# Attention map + loss
# ─────────────────────────────────────────────────────────────────────────────
def attention_map(feat: torch.Tensor) -> torch.Tensor:
    """
    feat: (B, C, H, W)
    returns: (B, H*W) normalized spatial attention map
    """
    return F.normalize(feat.pow(2).mean(dim=1).view(feat.size(0), -1), dim=1)


def attention_loss_all(s_hooks, t_hooks) -> torch.Tensor:
    """Sum attention loss across all matched layers."""
    total = 0.0
    for s_hook, t_hook in zip(s_hooks, t_hooks):
        total += F.mse_loss(
            attention_map(s_hook.features),
            attention_map(t_hook.features.detach()),
        )
    return total


# ─────────────────────────────────────────────────────────────────────────────
# Distillation loss — identical to all previous experiments
# ─────────────────────────────────────────────────────────────────────────────
def distillation_loss(student_logits, teacher_logits, labels, temperature, alpha):
    T = temperature
    soft_student = F.log_softmax(student_logits / T, dim=1)
    soft_teacher = F.softmax(teacher_logits   / T, dim=1)
    kl_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (T * T)
    ce_loss = F.cross_entropy(student_logits, labels)
    return alpha * ce_loss + (1.0 - alpha) * kl_loss


# ─────────────────────────────────────────────────────────────────────────────
# Train one epoch
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(student, teacher, s_hooks, t_hooks,
                    optimizer, loader, device):
    student.train()
    teacher.eval()

    loss_meter    = AverageMeter()
    kd_loss_meter = AverageMeter()
    at_loss_meter = AverageMeter()
    acc_meter     = AverageMeter()

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        # Teacher — no gradient, frozen
        with torch.no_grad():
            teacher_logits = teacher(images)
            # cache teacher features (detached inside attention_loss_all)

        # Student forward
        student_logits = student(images)

        # Logit KD loss
        kd_loss = distillation_loss(student_logits, teacher_logits,
                                    labels, TEMPERATURE, ALPHA)

        # Attention loss across all 3 layers
        at_loss = attention_loss_all(s_hooks, t_hooks)

        # Combined loss — beta=1000 makes at_loss meaningful
        loss = kd_loss + BETA * at_loss
        loss.backward()
        optimizer.step()

        top1 = topk_accuracy(student_logits, labels, topk=(1,))[0]
        bs   = images.size(0)
        loss_meter.update(loss.item(),       bs)
        kd_loss_meter.update(kd_loss.item(), bs)
        at_loss_meter.update(at_loss.item(), bs)
        acc_meter.update(top1,               bs)

    return loss_meter.avg, kd_loss_meter.avg, at_loss_meter.avg, acc_meter.avg


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_meter = AverageMeter()
    acc_meter  = AverageMeter()
    criterion  = nn.CrossEntropyLoss()

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss   = criterion(logits, labels)
        top1   = topk_accuracy(logits, labels, topk=(1,))[0]
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(top1,         images.size(0))

    return loss_meter.avg, acc_meter.avg


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    set_seed()
    device = get_device()
    train_loader, test_loader = get_cifar100_loaders()

    # ── Teacher (frozen) ──────────────────────────────────────────────────────
    teacher = resnet110(num_classes=config.NUM_CLASSES).to(device)
    load_checkpoint("teacher_resnet110.pth", teacher, optimizer=None, device=device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print("[teacher] ResNet-110 loaded and frozen.")

    # ── Student ───────────────────────────────────────────────────────────────
    student = resnet20(num_classes=config.NUM_CLASSES).to(device)
    total, trainable, size_mb = count_parameters(student)
    print(f"[student] ResNet-20  |  {total/1e6:.2f} M params  |  {size_mb:.1f} MB")

    # ── Hooks for all 3 layers ────────────────────────────────────────────────
    s_hooks = [FeatureExtractor(student, l) for l in MATCH_LAYERS]
    t_hooks = [FeatureExtractor(teacher, l) for l in MATCH_LAYERS]

    # Verify attention map sizes
    with torch.no_grad():
        dummy = torch.zeros(2, 3, 32, 32).to(device)
        teacher(dummy)
        student(dummy)
    print(f"[attention] Matching layers: {MATCH_LAYERS}")
    for l, sh, th in zip(MATCH_LAYERS, s_hooks, t_hooks):
        s_map = attention_map(sh.features)
        t_map = attention_map(th.features)
        print(f"  {l}: student {sh.features.shape} → "
              f"attn map {s_map.shape}  ✓")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = optim.SGD(
        student.parameters(),
        lr=config.LR,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY,
        nesterov=True,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )

    start_epoch, best_acc = load_checkpoint(CKPT_NAME, student, optimizer, device)

    history = {
        "train_loss": [], "train_acc": [],
        "test_loss":  [], "test_acc":  [],
        "kd_loss":    [], "at_loss":   [],
    }

    print(f"\nExperiment 8 — Attention Transfer KD (All 3 Layers)")
    print(f"T={TEMPERATURE}  alpha={ALPHA}  beta={BETA}  epochs={EPOCHS}")
    print(f"Matching layers: {MATCH_LAYERS}\n")

    for epoch in range(start_epoch, EPOCHS):
        tr_loss, kd_loss, at_loss, tr_acc = train_one_epoch(
            student, teacher, s_hooks, t_hooks,
            optimizer, train_loader, device,
        )
        te_loss, te_acc = evaluate(student, test_loader, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)
        history["kd_loss"].append(kd_loss)
        history["at_loss"].append(at_loss)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch [{epoch+1:3d}/{EPOCHS}]  "
              f"lr={lr_now:.5f}  "
              f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.2f}%  "
              f"test_loss={te_loss:.4f}  test_acc={te_acc:.2f}%  "
              f"[kd={kd_loss:.4f}  at={at_loss:.4f}]")

        if te_acc > best_acc:
            best_acc = te_acc
            save_checkpoint({
                "epoch":           epoch + 1,
                "model_state":     student.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_acc":        best_acc,
                "temperature":     TEMPERATURE,
                "alpha":           ALPHA,
                "beta":            BETA,
                "history":         history,
            }, CKPT_NAME)

    print(f"\n[done] Best test accuracy: {best_acc:.2f}%")
    torch.save(history,
               os.path.join(config.RESULTS_DIR,
                            "distill_attn_all3_T4.0_a0.95_b1000_Exp8_e300_history.pt"))

    for h in s_hooks + t_hooks:
        h.remove()


if __name__ == "__main__":
    main()