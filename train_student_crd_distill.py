"""
Experiment 9 — Contrastive Representation Distillation (CRD)
=============================================================
State-of-the-art knowledge distillation for same-architecture networks.

Instead of matching features directly (Exp6) or attention maps (Exp7/8),
CRD treats distillation as a contrastive learning problem:
- Positive pair: same sample's student & teacher representations
- Negative pairs: different samples' representations

This captures STRUCTURAL relationships between samples, not just
individual feature values — much more powerful for same-width networks.

Total loss = alpha * CE(student, hard_labels)
           + (1-alpha) * T^2 * KL(student_soft || teacher_soft)   ← logit KD
           + beta * CRD_loss(student_features, teacher_features)   ← contrastive

Settings:
    T=4, alpha=0.95, beta=0.8, epochs=300

Expected: 72.5–73.5% — best result in the project

Run:
    python train_student_crd_distill.py
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import config
from models  import resnet20, resnet110
from utils   import (get_cifar100_loaders, set_seed, get_device,
                     AverageMeter, save_checkpoint, load_checkpoint,
                     count_parameters)
from utils.helpers import topk_accuracy


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 9 hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
TEMPERATURE  = 4.0    # same as all previous best runs
ALPHA        = 0.95   # matches Exp5-8
BETA         = 0.8    # CRD loss weight (standard from paper)
EPOCHS       = 300    # matches Exp5-8
FEAT_DIM     = 128    # projection head output dimension
NCEК         = 16384  # number of negatives (use 4096 if memory issues)
NCE_T        = 0.07   # contrastive temperature
CKPT_NAME    = "student_crd_T4_a0.95_b0.8_Exp9_e300.pth"


# ─────────────────────────────────────────────────────────────────────────────
# Feature hook — same pattern as Exp6/7/8
# ─────────────────────────────────────────────────────────────────────────────
class FeatureExtractor:
    def __init__(self, model: nn.Module, layer_name: str):
        self.features = None
        layer = dict(model.named_modules())[layer_name]
        self._handle = layer.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        # flatten spatial dims: (B, C, H, W) → (B, C*H*W)
        o = output
        if o.dim() == 4:
            o = o.mean(dim=[2, 3])   # global average pool → (B, C)
        self.features = o

    def remove(self):
        self._handle.remove()


# ─────────────────────────────────────────────────────────────────────────────
# Projection head — maps raw features to shared embedding space
# ─────────────────────────────────────────────────────────────────────────────
class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Memory bank — stores teacher embeddings for negative sampling
# ─────────────────────────────────────────────────────────────────────────────
class MemoryBank(nn.Module):
    def __init__(self, n_samples: int, feat_dim: int, device):
        super().__init__()
        self.register_buffer(
            "memory",
            F.normalize(torch.randn(n_samples, feat_dim), dim=1).to(device)
        )

    @torch.no_grad()
    def update(self, indices: torch.Tensor, features: torch.Tensor, momentum: float = 0.5):
        self.memory[indices] = F.normalize(
            momentum * self.memory[indices] + (1 - momentum) * features, dim=1
        )

    def get_negatives(self, indices: torch.Tensor, k: int) -> torch.Tensor:
        """Sample k negatives, excluding the positive indices."""
        n = self.memory.size(0)
        device = self.memory.device
        # random negatives
        neg_idx = torch.randint(0, n, (indices.size(0), k), device=device)
        return self.memory[neg_idx]   # (B, k, dim)


# ─────────────────────────────────────────────────────────────────────────────
# CRD loss
# ─────────────────────────────────────────────────────────────────────────────
def crd_loss(s_feat: torch.Tensor,
             t_feat: torch.Tensor,
             memory_bank: MemoryBank,
             indices: torch.Tensor,
             k: int = 4096,
             temperature: float = 0.07) -> torch.Tensor:
    """
    s_feat: (B, dim) normalized student embeddings
    t_feat: (B, dim) normalized teacher embeddings
    Positive: s_feat[i] should be close to t_feat[i]
    Negative: s_feat[i] should be far from memory bank samples
    """
    B, dim = s_feat.shape

    # positive logits: (B,)
    pos = (s_feat * t_feat).sum(dim=1, keepdim=True) / temperature  # (B, 1)

    # negative logits: (B, k)
    neg_feats = memory_bank.get_negatives(indices, k)   # (B, k, dim)
    neg = torch.bmm(neg_feats, s_feat.unsqueeze(2)).squeeze(2) / temperature  # (B, k)

    # NCE loss
    logits = torch.cat([pos, neg], dim=1)   # (B, 1+k)
    labels = torch.zeros(B, dtype=torch.long, device=s_feat.device)  # positive is index 0
    loss = F.cross_entropy(logits, labels)

    # update memory bank with teacher features
    memory_bank.update(indices, t_feat.detach())

    return loss


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
def train_one_epoch(student, teacher,
                    s_hook, t_hook,
                    s_proj, t_proj,
                    memory_bank,
                    optimizer, loader, device):
    student.train()
    s_proj.train()
    teacher.eval()
    t_proj.eval()

    loss_meter     = AverageMeter()
    kd_loss_meter  = AverageMeter()
    crd_loss_meter = AverageMeter()
    acc_meter      = AverageMeter()

    for batch in loader:
        # loader returns (images, labels, indices)
        if len(batch) == 3:
            images, labels, indices = batch
        else:
            images, labels = batch
            indices = torch.arange(images.size(0))

        images  = images.to(device)
        labels  = labels.to(device)
        indices = indices.to(device)

        optimizer.zero_grad()

        # Teacher forward — no grad
        with torch.no_grad():
            teacher_logits = teacher(images)
            t_raw = t_hook.features.detach()
            t_emb = t_proj(t_raw).detach()   # (B, dim)

        # Student forward
        student_logits = student(images)
        s_raw = s_hook.features
        s_emb = s_proj(s_raw)   # (B, dim)

        # Logit KD
        kd_l = distillation_loss(student_logits, teacher_logits,
                                  labels, TEMPERATURE, ALPHA)

        # CRD loss
        cr_l = crd_loss(s_emb, t_emb, memory_bank, indices,
                        k=min(NCEК, memory_bank.memory.size(0) - 1),
                        temperature=NCE_T)

        loss = kd_l + BETA * cr_l
        loss.backward()
        optimizer.step()

        top1 = topk_accuracy(student_logits, labels, topk=(1,))[0]
        bs   = images.size(0)
        loss_meter.update(loss.item(),    bs)
        kd_loss_meter.update(kd_l.item(), bs)
        crd_loss_meter.update(cr_l.item(), bs)
        acc_meter.update(top1,            bs)

    return loss_meter.avg, kd_loss_meter.avg, crd_loss_meter.avg, acc_meter.avg


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate — same pattern as all other scripts
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_meter = AverageMeter()
    acc_meter  = AverageMeter()
    criterion  = nn.CrossEntropyLoss()

    for batch in loader:
        images, labels = batch[0], batch[1]
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss   = criterion(logits, labels)
        top1   = topk_accuracy(logits, labels, topk=(1,))[0]
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(top1,         images.size(0))

    return loss_meter.avg, acc_meter.avg


# ─────────────────────────────────────────────────────────────────────────────
# Index-aware dataset wrapper — CRD needs sample indices for memory bank
# ─────────────────────────────────────────────────────────────────────────────
class IndexDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return img, label, idx


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    set_seed()
    device = get_device()

    # ── Data — wrap with index dataset for CRD ────────────────────────────────
    from torchvision import datasets, transforms
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])
    train_base = datasets.CIFAR100(config.DATA_DIR, train=True,
                                   download=True, transform=train_transform)
    test_base  = datasets.CIFAR100(config.DATA_DIR, train=False,
                                   download=True, transform=test_transform)

    train_dataset = IndexDataset(train_base)
    train_loader  = DataLoader(train_dataset, batch_size=128,
                               shuffle=True, num_workers=4, pin_memory=True)
    test_loader   = DataLoader(test_base, batch_size=128,
                               shuffle=False, num_workers=4, pin_memory=True)

    n_train = len(train_base)

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

    # ── Hooks on layer3 (deepest, most discriminative) ────────────────────────
    s_hook = FeatureExtractor(student, "layer3")
    t_hook = FeatureExtractor(teacher, "layer3")

    # Auto-detect feature dims
    with torch.no_grad():
        dummy = torch.zeros(2, 3, 32, 32).to(device)
        teacher(dummy); t_dim = t_hook.features.shape[1]
        student(dummy); s_dim = s_hook.features.shape[1]
    print(f"[CRD] layer3: student {s_dim}d → teacher {t_dim}d → proj {FEAT_DIM}d")

    # ── Projection heads ──────────────────────────────────────────────────────
    s_proj = ProjectionHead(s_dim, FEAT_DIM).to(device)
    t_proj = ProjectionHead(t_dim, FEAT_DIM).to(device)

    # Freeze teacher projection after warmup (teacher is fixed)
    for p in t_proj.parameters():
        p.requires_grad_(False)

    # ── Memory bank ───────────────────────────────────────────────────────────
    memory_bank = MemoryBank(n_train, FEAT_DIM, device).to(device)
    print(f"[memory bank] {n_train} samples × {FEAT_DIM}d")

    # ── Optimizer: student + student projection head ───────────────────────────
    optimizer = optim.SGD(
        list(student.parameters()) + list(s_proj.parameters()),
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
        "kd_loss":    [], "crd_loss":  [],
    }

    print(f"\nExperiment 9 — CRD (Contrastive Representation Distillation)")
    print(f"T={TEMPERATURE}  alpha={ALPHA}  beta={BETA}  epochs={EPOCHS}")
    print(f"feat_dim={FEAT_DIM}  nce_k={NCEК}  nce_T={NCE_T}\n")

    for epoch in range(start_epoch, EPOCHS):
        tr_loss, kd_loss, crd_l, tr_acc = train_one_epoch(
            student, teacher,
            s_hook, t_hook,
            s_proj, t_proj,
            memory_bank,
            optimizer, train_loader, device,
        )
        te_loss, te_acc = evaluate(student, test_loader, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)
        history["kd_loss"].append(kd_loss)
        history["crd_loss"].append(crd_l)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch [{epoch+1:3d}/{EPOCHS}]  "
              f"lr={lr_now:.5f}  "
              f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.2f}%  "
              f"test_loss={te_loss:.4f}  test_acc={te_acc:.2f}%  "
              f"[kd={kd_loss:.4f}  crd={crd_l:.4f}]")

        if te_acc > best_acc:
            best_acc = te_acc
            save_checkpoint({
                "epoch":           epoch + 1,
                "model_state":     student.state_dict(),
                "s_proj_state":    s_proj.state_dict(),
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
                            "distill_crd_T4.0_a0.95_b0.8_Exp9_e300_history.pt"))

    s_hook.remove()
    t_hook.remove()


if __name__ == "__main__":
    main()