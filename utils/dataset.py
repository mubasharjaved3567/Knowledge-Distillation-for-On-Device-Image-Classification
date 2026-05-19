"""
CIFAR-100 data loaders with standard augmentations.
"""

import torch
from torchvision import datasets, transforms
import config


def get_cifar100_loaders(data_dir=None, batch_size=None, num_workers=None):
    """
    Returns (train_loader, test_loader).

    Train:  RandomCrop + RandomHorizontalFlip + Normalize
    Test:   Normalize only  (no augmentation — clean evaluation)
    """
    data_dir    = data_dir    or config.DATA_DIR
    batch_size  = batch_size  or config.BATCH_SIZE
    num_workers = num_workers or config.NUM_WORKERS

    # Experiment 1 - standard augmentations
    # train_transform = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize(config.CIFAR100_MEAN, config.CIFAR100_STD),
    # ])

    #Experiment 2 - AutoAugment seems to help a lot when distilling with hard labels
    train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.Normalize(config.CIFAR100_MEAN, config.CIFAR100_STD),
])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config.CIFAR100_MEAN, config.CIFAR100_STD),
    ])

    train_dataset = datasets.CIFAR100(
        root=data_dir, train=True,  download=True, transform=train_transform
    )
    test_dataset  = datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"[data] Train: {len(train_dataset):,} samples  "
          f"| Test: {len(test_dataset):,} samples  "
          f"| Batch: {batch_size}")
    return train_loader, test_loader