"""
Train student (ResNet-20) with hard labels only — NO distillation.

This is the baseline.  The accuracy here is what we compare against
the distilled student to measure how much dark knowledge helps.

Run:
    python train_student_hard.py

Expected accuracy: ~68-70 % top-1 after 200 epochs.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim

import config
from models  import resnet20
from utils   import (get_cifar100_loaders, set_seed, get_device,
                      AverageMeter, save_checkpoint, load_checkpoint,
                      count_parameters)
from utils.helpers import topk_accuracy


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    loss_meter = AverageMeter()
    acc_meter  = AverageMeter()

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        top1 = topk_accuracy(logits, labels)[0]
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(top1,         images.size(0))

    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    acc_meter  = AverageMeter()

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss   = criterion(logits, labels)
        top1   = topk_accuracy(logits, labels)[0]
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(top1,         images.size(0))

    return loss_meter.avg, acc_meter.avg


def main():
    set_seed()
    device = get_device()

    train_loader, test_loader = get_cifar100_loaders()

    model = resnet20(num_classes=config.NUM_CLASSES).to(device)
    total, _, size_mb = count_parameters(model)
    print(f"[model] ResNet-20 (hard-label baseline)  "
          f"|  {total/1e6:.3f} M params  |  {size_mb:.2f} MB")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          lr=config.LR,
                          momentum=config.MOMENTUM,
                          weight_decay=config.WEIGHT_DECAY,
                          nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config.LR_MILESTONES,
        gamma=config.LR_GAMMA
    )

    start_epoch, best_acc = load_checkpoint("student_hard.pth",
                                             model, optimizer, device)

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in range(start_epoch, config.STUDENT_EPOCHS):
        tr_loss, tr_acc = train_one_epoch(model, train_loader,
                                          optimizer, criterion, device)
        te_loss, te_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)

        print(f"Epoch [{epoch+1:3d}/{config.STUDENT_EPOCHS}]  "
              f"train_acc={tr_acc:.2f}%  test_acc={te_acc:.2f}%  "
              f"(best={best_acc:.2f}%)")

        if te_acc > best_acc:
            best_acc = te_acc
            save_checkpoint({
                "epoch":           epoch + 1,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_acc":        best_acc,
                "history":         history,
            }, "student_hard.pth")

    print(f"\n[done] Best hard-label student accuracy: {best_acc:.2f}%")
    torch.save(history,
               os.path.join(config.RESULTS_DIR, "student_hard_history.pt"))


if __name__ == "__main__":
    main()