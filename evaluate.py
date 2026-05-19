"""
Evaluate a saved checkpoint and report top-1 accuracy, model size, and
parameter count.

Usage examples:
    python evaluate.py --model teacher
    python evaluate.py --model student_hard
    python evaluate.py --model student_distill --ckpt student_T4_a0.9.pth
"""

import argparse
import os
import torch

import config
from models  import resnet20, resnet110, load_teacher
from utils   import get_cifar100_loaders, get_device, AverageMeter, count_parameters
from utils.helpers import topk_accuracy


@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        top1, top5 = topk_accuracy(logits, labels, topk=(1, 5))
        top1_m.update(top1, images.size(0))
        top5_m.update(top5, images.size(0))

    return top1_m.avg, top5_m.avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["teacher", "student_hard",
                                             "student_distill"],
                        required=True)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Checkpoint filename (inside checkpoints/). "
                             "Auto-detected if omitted.")
    args = parser.parse_args()

    device = get_device()
    _, test_loader = get_cifar100_loaders()

    # ── Load model ─────────────────────────────────────────────────────────────
    if args.model == "teacher":
        model = load_teacher(device=device)
        label = "ResNet-110 (teacher)"

    elif args.model == "student_hard":
        ckpt_file = args.ckpt or "student_hard.pth"
        model = resnet20(num_classes=config.NUM_CLASSES).to(device)
        ckpt  = torch.load(os.path.join(config.CHECKPOINT_DIR, ckpt_file),
                           map_location=device)
        model.load_state_dict(ckpt["model_state"])
        label = f"ResNet-20 hard-label  [{ckpt_file}]"

    else:  # student_distill
        ckpt_file = args.ckpt or "student_T4_a0.9.pth"
        model = resnet20(num_classes=config.NUM_CLASSES).to(device)
        ckpt  = torch.load(os.path.join(config.CHECKPOINT_DIR, ckpt_file),
                           map_location=device)
        model.load_state_dict(ckpt["model_state"])
        T     = ckpt.get("temperature", "?")
        alpha = ckpt.get("alpha",       "?")
        label = f"ResNet-20 distilled  T={T}  alpha={alpha}  [{ckpt_file}]"

    # ── Stats ──────────────────────────────────────────────────────────────────
    total, _, size_mb = count_parameters(model)
    top1, top5 = evaluate_model(model, test_loader, device)

    print(f"\n{'─'*60}")
    print(f"  Model  : {label}")
    print(f"  Params : {total/1e6:.3f} M  ({size_mb:.1f} MB)")
    print(f"  Top-1  : {top1:.2f}%")
    print(f"  Top-5  : {top5:.2f}%")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()