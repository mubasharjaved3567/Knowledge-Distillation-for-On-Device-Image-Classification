"""
CIFAR-10/100 ResNet implementation.
Follows the original He et al. (2016) design for 32×32 images.
Depth formula:  depth = 6*n + 2
  n=3  → ResNet-20   (~0.27 M params)
  n=9  → ResNet-56   (~0.85 M params)
  n=18 → ResNet-110  (~1.70 M params)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Building block ─────────────────────────────────────────────────────────────
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride,
                               padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1,
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# ── Main model ─────────────────────────────────────────────────────────────────
class ResNet_CIFAR(nn.Module):
    """
    Generic CIFAR ResNet.  Pass n to control depth.
    Returns raw logits; no softmax applied here.
    """
    def __init__(self, block, num_blocks, num_classes=100):
        super().__init__()
        self.in_planes = 16
        self.conv1  = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32,  num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64,  num_blocks[2], stride=2)
        self.linear = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers  = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return self.linear(out)


# ── Public constructors ────────────────────────────────────────────────────────
def resnet20(num_classes=100):
    return ResNet_CIFAR(BasicBlock, [3, 3, 3], num_classes=num_classes)

def resnet32(num_classes=100):
    return ResNet_CIFAR(BasicBlock, [5, 5, 5], num_classes=num_classes)

def resnet44(num_classes=100):
    return ResNet_CIFAR(BasicBlock, [7, 7, 7], num_classes=num_classes)

def resnet56(num_classes=100):
    return ResNet_CIFAR(BasicBlock, [9, 9, 9], num_classes=num_classes)

def resnet110(num_classes=100):
    return ResNet_CIFAR(BasicBlock, [18, 18, 18], num_classes=num_classes)