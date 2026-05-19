"""
Experiment 10 — Decoupled Knowledge Distillation (DKD)
=======================================================
CVPR 2022 — Zhao et al.

Key insight: Hinton's KD loss couples target-class and non-target-class
knowledge together, which suppresses the non-target (dark knowledge) signal.
DKD decouples them into two separate losses:

  TCKD — Target Class KD    → transfers how confident model is on correct class
  NCKD — Non-Target Class KD → transfers dark knowledge between wrong classes

For same-width architectures (your ResNet-20 vs ResNet-110), NCKD is the
key improvement — it directly addresses why CRD/Feature/Attention methods
gave marginal gains.

Total loss = CE(student, hard_labels)
           + alpha * TCKD(student, teacher)
           + beta  * NCKD(student, teacher)

Settings:
    T=4, alpha=1.0, beta=1.0, epochs=300

Expected: 72.3-72.8% — best result in the project

Run:
    python train_student_dkd.py
    or
    caffeinate -i python train_student_dkd.py
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
# Experiment 10 hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
TEMPERATURE = 4.0    # same as all previous best runs
ALPHA       = 1.0    # TCKD weight (paper default)
BETA        = 0.5    # NCKD weight (paper default — higher = more dark knowledge)
EPOCHS      = 300
LR          = 0.1
MOMENTUM    = 0.9
WEIGHT_DECAY= 5e-4
CKPT_NAME   = "student_dkd_T4_a1.0_b0.5_Exp10_e300.pth"


# ─────────────────────────────────────────────────────────────────────────────
# DKD Loss — CVPR 2022
# ─────────────────────────────────────────────────────────────────────────────
def dkd_loss(student_logits, teacher_logits, labels, T, alpha, beta):
    """
    Decoupled Knowledge Distillation loss.

    Args:
        student_logits: (B, C) raw student logits
        teacher_logits: (B, C) raw teacher logits (no grad)
        labels:         (B,)   ground truth class indices
        T:              temperature
        alpha:          TCKD weight
        beta:           NCKD weight

    Returns:
        total DKD loss scalar
    """
    B, C = student_logits.shape

    # ── Soft probabilities ──────────────────────────────────────────────────
    s_soft = F.softmax(student_logits / T, dim=1)   # (B, C)
    t_soft = F.softmax(teacher_logits / T, dim=1)   # (B, C)

    # ── Target class masks ──────────────────────────────────────────────────
    # gt_mask[i, labels[i]] = 1, rest = 0
    gt_mask = torch.zeros_like(s_soft)
    gt_mask.scatter_(1, labels.unsqueeze(1), 1.0)   # (B, C)
    other_mask = 1.0 - gt_mask                       # (B, C)

    # ── TCKD — target class KD ──────────────────────────────────────────────
    # Binary KL divergence on the target class probability
    # p_t = prob assigned to target class
    s_t = (s_soft * gt_mask).sum(dim=1, keepdim=True)    # (B, 1)
    t_t = (t_soft * gt_mask).sum(dim=1, keepdim=True)    # (B, 1)

    # Binary distribution: [p_target, 1-p_target]
    s_binary = torch.cat([s_t, 1 - s_t], dim=1)          # (B, 2)
    t_binary = torch.cat([t_t, 1 - t_t], dim=1)          # (B, 2)

    tckd = F.kl_div(
        torch.log(s_binary + 1e-8),
        t_binary,
        reduction="batchmean"
    ) * (T ** 2)

    # ── NCKD — non-target class KD (dark knowledge) ─────────────────────────
    # Renormalize non-target probabilities to sum to 1
    s_non = s_soft * other_mask
    t_non = t_soft * other_mask

    s_non = s_non / (s_non.sum(dim=1, keepdim=True) + 1e-8)  # (B, C)
    t_non = t_non / (t_non.sum(dim=1, keepdim=True) + 1e-8)  # (B, C)

    nckd = F.kl_div(
        torch.log(s_non + 1e-8),
        t_non,
        reduction="batchmean"
    ) * (T ** 2)

    return alpha * tckd + beta * nckd


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(student, teacher, optimizer, loader, device):
    student.train()
    teacher.eval()

    loss_m  = AverageMeter()
    ce_m    = AverageMeter()
    dkd_m   = AverageMeter()
    acc_m   = AverageMeter()
    ce_loss = nn.CrossEntropyLoss()

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        B = images.size(0)

        with torch.no_grad():
            t_logits = teacher(images)

        s_logits = student(images)

        # CE loss
        loss_ce = ce_loss(s_logits, labels)

        # DKD loss
        loss_dkd = dkd_loss(s_logits, t_logits, labels,
                             T=TEMPERATURE, alpha=ALPHA, beta=BETA)

        loss = loss_ce + loss_dkd

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = topk_accuracy(s_logits, labels)[0]
        loss_m.update(loss.item(), B)
        ce_m.update(loss_ce.item(), B)
        dkd_m.update(loss_dkd.item(), B)
        acc_m.update(acc, B)

    return loss_m.avg, ce_m.avg, dkd_m.avg, acc_m.avg


def evaluate(model, loader, device):
    model.eval()
    loss_m = AverageMeter()
    acc_m  = AverageMeter()
    ce     = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss   = ce(logits, labels)
            acc    = topk_accuracy(logits, labels)[0]
            loss_m.update(loss.item(), images.size(0))
            acc_m.update(acc, images.size(0))

    return loss_m.avg, acc_m.avg


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    set_seed(config.SEED)
    device = get_device()

    train_loader, test_loader = get_cifar100_loaders()

    # ── Teacher ────────────────────────────────────────────────────────────────
    teacher = resnet110(num_classes=config.NUM_CLASSES).to(device)
    t_ckpt  = torch.load(
        os.path.join(config.CHECKPOINT_DIR, "teacher_resnet110.pth"),
        map_location=device
    )
    teacher.load_state_dict(t_ckpt["model_state"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"[teacher] ResNet-110 loaded and frozen.")

    # ── Student ────────────────────────────────────────────────────────────────
    student = resnet20(num_classes=config.NUM_CLASSES).to(device)
    params  = count_parameters(student)
    print(f"[student] ResNet-20  |  params={params}")

    # ── Optimizer + Scheduler ──────────────────────────────────────────────────
    optimizer = optim.SGD(student.parameters(), lr=LR,
                          momentum=MOMENTUM, weight_decay=WEIGHT_DECAY,
                          nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ── Resume if checkpoint exists ────────────────────────────────────────────
    start_epoch = 0
    best_acc    = 0.0
    ckpt_path   = os.path.join(config.CHECKPOINT_DIR, CKPT_NAME)
    if os.path.isfile(ckpt_path):
        ckpt        = torch.load(ckpt_path, map_location=device)
        student.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"]
        best_acc    = ckpt["best_acc"]
        print(f"[resume] epoch={start_epoch}  best_acc={best_acc:.2f}%")

    history = {"train_loss": [], "train_acc": [],
               "test_loss":  [], "test_acc":  [],
               "ce_loss":    [], "dkd_loss":  []}

    print(f"\nExperiment 10 — DKD (Decoupled Knowledge Distillation)")
    print(f"T={TEMPERATURE}  alpha={ALPHA}  beta={BETA}  epochs={EPOCHS}\n")

    for epoch in range(start_epoch, EPOCHS):
        tr_loss, ce_l, dkd_l, tr_acc = train_one_epoch(
            student, teacher, optimizer, train_loader, device)
        te_loss, te_acc = evaluate(student, test_loader, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)
        history["ce_loss"].append(ce_l)
        history["dkd_loss"].append(dkd_l)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch [{epoch+1:3d}/{EPOCHS}]  "
              f"lr={lr_now:.5f}  "
              f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.2f}%  "
              f"test_acc={te_acc:.2f}%  "
              f"[ce={ce_l:.4f}  dkd={dkd_l:.4f}]")

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

    # Save history
    torch.save(history,
               os.path.join(config.RESULTS_DIR,
                            f"distill_T{TEMPERATURE}_a{ALPHA}_Exp10_e{EPOCHS}_history.pt"))

    print(f"\n[done] Best test accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()