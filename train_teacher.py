"""
Train ResNet-110 teacher on CIFAR-100.

Run:
    python train_teacher.py

Tip: skip this if you use the pretrained weights from hub
     (teacher_loader.py downloads them automatically).
Expected accuracy: ~74-76 % top-1 after 200 epochs.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import config
from models  import resnet110
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

        top1 = topk_accuracy(logits, labels, topk=(1,))[0]
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

        top1 = topk_accuracy(logits, labels, topk=(1,))[0]
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(top1,         images.size(0))

    return loss_meter.avg, acc_meter.avg


def main():
    set_seed()
    device = get_device()

    train_loader, test_loader = get_cifar100_loaders()

    model = resnet110(num_classes=config.NUM_CLASSES).to(device)
    total, trainable, size_mb = count_parameters(model)
    print(f"[model] ResNet-110  |  {total/1e6:.2f} M params  |  {size_mb:.1f} MB")

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

    # Resume if checkpoint exists
    start_epoch, best_acc = load_checkpoint("teacher_resnet110.pth",
                                             model, optimizer, device)

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in range(start_epoch, config.EPOCHS):
        tr_loss, tr_acc = train_one_epoch(model, train_loader,
                                          optimizer, criterion, device)
        te_loss, te_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch [{epoch+1:3d}/{config.EPOCHS}]  "
              f"lr={lr_now:.5f}  "
              f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.2f}%  "
              f"test_loss={te_loss:.4f}  test_acc={te_acc:.2f}%")

        if te_acc > best_acc:
            best_acc = te_acc
            save_checkpoint({
                "epoch":          epoch + 1,
                "model_state":    model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_acc":       best_acc,
                "history":        history,
            }, "teacher_resnet110.pth")

    print(f"\n[done] Best teacher accuracy: {best_acc:.2f}%")

    # Save history for plotting
    import torch, os
    torch.save(history, os.path.join(config.RESULTS_DIR, "teacher_history.pt"))


if __name__ == "__main__":
    main()