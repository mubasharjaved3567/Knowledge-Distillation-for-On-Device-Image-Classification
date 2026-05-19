"""
Experiment 6 — Feature-Level Knowledge Distillation (FitNets style)
=====================================================================
Adds intermediate feature matching loss on top of standard logit-level KD.

Total loss = alpha * CE(student, hard_labels)
           + (1-alpha) * T^2 * KL(student_soft || teacher_soft)
           + beta      * MSE(projector(student_layer2), teacher_layer2)

Settings (matching Exp5 as base):
    T=4, alpha=0.95, beta=0.1, epochs=300

Run:
    python train_student_feature_distill.py
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
# Experiment 6 hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
TEMPERATURE = 4.0    # same as all previous best runs
ALPHA       = 0.95   # matches Exp5 (your current best)
BETA        = 0.1    # feature loss weight
EPOCHS      = 300    # matches Exp5
MATCH_LAYER = "layer2"
CKPT_NAME = "student_feat_layer2_T4_a0.95_b0.1_Exp6_e300.pth"


# ─────────────────────────────────────────────────────────────────────────────
# Feature hook — captures intermediate layer output during forward pass
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
# Channel projector — aligns student channels to teacher channels (1x1 conv)
# ─────────────────────────────────────────────────────────────────────────────
class ChannelProjector(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ─────────────────────────────────────────────────────────────────────────────
# Distillation loss — identical to your existing train_student_distill.py
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
def train_one_epoch(student, teacher, s_hook, t_hook, projector,
                    optimizer, loader, device):
    student.train()
    projector.train()
    teacher.eval()

    loss_meter      = AverageMeter()
    kd_loss_meter   = AverageMeter()
    feat_loss_meter = AverageMeter()
    acc_meter       = AverageMeter()

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        # Teacher — no gradient, frozen
        with torch.no_grad():
            teacher_logits = teacher(images)
            t_feat = t_hook.features.detach()

        # Student forward
        student_logits = student(images)
        s_feat = s_hook.features

        # Logit-level KD loss (same as previous experiments)
        kd_loss = distillation_loss(student_logits, teacher_logits,
                                    labels, TEMPERATURE, ALPHA)

        # Feature-level loss (NEW in Exp6)
        s_feat_proj = projector(s_feat)
        feat_loss   = F.mse_loss(s_feat_proj, t_feat)

        # Combined loss
        loss = kd_loss + BETA * feat_loss
        loss.backward()
        optimizer.step()

        top1 = topk_accuracy(student_logits, labels, topk=(1,))[0]
        bs   = images.size(0)
        loss_meter.update(loss.item(),           bs)
        kd_loss_meter.update(kd_loss.item(),     bs)
        feat_loss_meter.update(feat_loss.item(), bs)
        acc_meter.update(top1,                   bs)

    return loss_meter.avg, kd_loss_meter.avg, feat_loss_meter.avg, acc_meter.avg


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate — same pattern as train_teacher.py
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

    # ── Hooks ─────────────────────────────────────────────────────────────────
    s_hook = FeatureExtractor(student, MATCH_LAYER)
    t_hook = FeatureExtractor(teacher, MATCH_LAYER)

    # Auto-detect channel sizes
    with torch.no_grad():
        dummy = torch.zeros(2, 3, 32, 32).to(device)
        teacher(dummy); t_ch = t_hook.features.shape[1]
        student(dummy); s_ch = s_hook.features.shape[1]
    print(f"[projector] {MATCH_LAYER}: student {s_ch}ch → teacher {t_ch}ch")

    # ── Projector ─────────────────────────────────────────────────────────────
    projector = ChannelProjector(s_ch, t_ch).to(device)

    # ── Optimizer (student + projector) ───────────────────────────────────────
    optimizer = optim.SGD(
        list(student.parameters()) + list(projector.parameters()),
        lr=config.LR,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY,
        nesterov=True,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS      # CosineAnnealing — same as Exp5
    )

    start_epoch, best_acc = load_checkpoint(CKPT_NAME, student, optimizer, device)

    history = {
        "train_loss": [], "train_acc": [],
        "test_loss":  [], "test_acc":  [],
        "kd_loss":    [], "feat_loss": [],
    }

    print(f"\nExperiment 6 — Feature-Level KD")
    print(f"T={TEMPERATURE}  alpha={ALPHA}  beta={BETA}  epochs={EPOCHS}")
    print(f"Matching layer: {MATCH_LAYER}\n")

    for epoch in range(start_epoch, EPOCHS):
        tr_loss, kd_loss, feat_loss, tr_acc = train_one_epoch(
            student, teacher, s_hook, t_hook, projector,
            optimizer, train_loader, device,
        )
        te_loss, te_acc = evaluate(student, test_loader, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)
        history["kd_loss"].append(kd_loss)
        history["feat_loss"].append(feat_loss)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch [{epoch+1:3d}/{EPOCHS}]  "
              f"lr={lr_now:.5f}  "
              f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.2f}%  "
              f"test_loss={te_loss:.4f}  test_acc={te_acc:.2f}%  "
              f"[kd={kd_loss:.4f}  feat={feat_loss:.4f}]")

        if te_acc > best_acc:
            best_acc = te_acc
            save_checkpoint({
                "epoch":           epoch + 1,
                "model_state":     student.state_dict(),
                "projector_state": projector.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_acc":        best_acc,
                "temperature":     TEMPERATURE,
                "alpha":           ALPHA,
                "beta":            BETA,
                "history":         history,
            }, CKPT_NAME)

    print(f"\n[done] Best test accuracy: {best_acc:.2f}%")
   # change to
    torch.save(history,
           os.path.join(config.RESULTS_DIR, "distill_feat_layer2_T4.0_a0.95_Exp6_e300_history.pt"))   

    s_hook.remove()
    t_hook.remove()


if __name__ == "__main__":
    main()