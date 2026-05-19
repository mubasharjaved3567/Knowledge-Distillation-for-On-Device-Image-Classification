"""
Train student (ResNet-20) with Knowledge Distillation.

Single run (uses config.py defaults):
    python train_student_distill.py

Hyperparameter sweep (all T × alpha combinations in config.py):
    python train_student_distill.py --sweep

Results are saved to results/sweep_results.csv for analysis.
"""

import os
import csv
import argparse
import torch
import torch.optim as optim
from torch.profiler import schedule

import config
from models   import resnet20, load_teacher
from utils    import (get_cifar100_loaders, set_seed, get_device,
                       AverageMeter, save_checkpoint, load_checkpoint,
                       count_parameters, DistillationLoss)
from utils.helpers import topk_accuracy


# ── Training / evaluation loops ───────────────────────────────────────────────

def train_one_epoch(student, teacher, loader, optimizer, criterion, device):
    student.train()
    loss_m = AverageMeter()
    acc_m  = AverageMeter()

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        # Teacher forward (frozen — no gradients needed)
        with torch.no_grad():
            teacher_logits = teacher(images)

        optimizer.zero_grad()
        student_logits = student(images)

        loss, kl, ce = criterion(student_logits, teacher_logits, labels)
        loss.backward()
        optimizer.step()

        top1 = topk_accuracy(student_logits, labels)[0]
        loss_m.update(loss.item(), images.size(0))
        acc_m.update(top1,          images.size(0))

    return loss_m.avg, acc_m.avg


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    acc_m = AverageMeter()

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        top1   = topk_accuracy(logits, labels)[0]
        acc_m.update(top1, images.size(0))

    return acc_m.avg


# ── Single training run ────────────────────────────────────────────────────────

def run_distillation(temperature, alpha, device, train_loader, test_loader,
                     teacher, epochs=None, ckpt_name=None, verbose=True):
    """
    Returns best test accuracy achieved.
    """
    epochs    = epochs    or config.STUDENT_EPOCHS
    ckpt_name = ckpt_name or f"student_T{temperature}_a{alpha}_Exp5_e{epochs}.pth"

    student = resnet20(num_classes=config.NUM_CLASSES).to(device)
    criterion = DistillationLoss(temperature=temperature, alpha=alpha)
    optimizer = torch.optim.SGD(student.parameters(),
                                lr=config.LR,
                                momentum=config.MOMENTUM,
                                weight_decay=config.WEIGHT_DECAY,
                                nesterov=True)
    
    # # Experiment 1 - standard step LR schedule

    # scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     optimizer,
    #     milestones=config.LR_MILESTONES,
    #     gamma=config.LR_GAMMA
    # )
    #Experiment 2 - cosine annealing seems to help a lot when distilling with hard labels
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=epochs
)

    best_acc = 0.0
    history  = {"train_loss": [], "train_acc": [], "test_acc": []}

    for epoch in range(epochs):
        tr_loss, tr_acc = train_one_epoch(
            student, teacher, train_loader, optimizer, criterion, device
        )
        te_acc = evaluate(student, test_loader, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_acc"].append(te_acc)

        if verbose:
            print(f"  [{epoch+1:3d}/{epochs}]  "
                  f"train_acc={tr_acc:.2f}%  "
                  f"test_acc={te_acc:.2f}%  "
                  f"(best={best_acc:.2f}%)")

        if te_acc > best_acc:
            best_acc = te_acc
            save_checkpoint({
                "epoch":           epoch + 1,
                "model_state":     student.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_acc":        best_acc,
                "temperature":     temperature,
                "alpha":           alpha,
                "history":         history,
            }, ckpt_name)

    # Save history
    torch.save(history,
               os.path.join(config.RESULTS_DIR,
                        f"distill_T{temperature}_a{alpha}_Exp5_e{epochs}_history.pt"))
    return best_acc


# ── Sweep ──────────────────────────────────────────────────────────────────────

def run_sweep(device, train_loader, test_loader, teacher):
    csv_path = os.path.join(config.RESULTS_DIR, "sweep_results.csv")
    results  = []

    for T in config.TEMPERATURES:
        for alpha in config.ALPHAS:
            print(f"\n{'='*60}")
            print(f"  Distillation sweep  T={T}  alpha={alpha}")
            print(f"{'='*60}")

            best_acc = run_distillation(
                temperature=T,
                alpha=alpha,
                device=device,
                train_loader=train_loader,
                test_loader=test_loader,
                teacher=teacher,
                verbose=True,
            )
            results.append({"T": T, "alpha": alpha, "best_acc": best_acc})
            print(f"  → Best acc: {best_acc:.2f}%")

    # Save to CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["T", "alpha", "best_acc"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[sweep] Results saved → {csv_path}")
    print("\nSummary:")
    print(f"{'T':>6} {'alpha':>6} {'best_acc':>10}")
    print("-" * 26)
    for r in sorted(results, key=lambda x: -x["best_acc"]):
        print(f"{r['T']:>6} {r['alpha']:>6} {r['best_acc']:>9.2f}%")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep",       action="store_true",
                        help="Run full T × alpha grid")
    parser.add_argument("--temperature", type=float, default=config.TEMPERATURE)
    parser.add_argument("--alpha",       type=float, default=config.ALPHA)
    args = parser.parse_args()

    set_seed()
    device = get_device()

    train_loader, test_loader = get_cifar100_loaders()
    teacher = load_teacher(device=device)

    total, _, size_mb = count_parameters(teacher)
    print(f"[teacher] {total/1e6:.2f} M params  |  {size_mb:.1f} MB")

    if args.sweep:
        run_sweep(device, train_loader, test_loader, teacher)
    else:
        print(f"\nSingle run: T={args.temperature}  alpha={args.alpha}")
        best = run_distillation(
            temperature=args.temperature,
            alpha=args.alpha,
            device=device,
            train_loader=train_loader,
            test_loader=test_loader,
            teacher=teacher,
        )
        print(f"\n[done] Best distilled student accuracy: {best:.2f}%")


if __name__ == "__main__":
    main()