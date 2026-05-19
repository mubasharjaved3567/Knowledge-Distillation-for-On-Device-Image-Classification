import ssl
ssl._create_default_https_context = ssl._create_unverified_context


"""
Teacher loader.

Priority order:
  1. Local checkpoint  (checkpoints/teacher_resnet110.pth)
  2. chenyaofo/pytorch-cifar-models  (downloads automatically, needs internet)
  3. Train from scratch fallback  (slow — only if both above fail)
"""

import os
import torch
from .resnet import resnet110

CKPT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "checkpoints", "teacher_resnet110.pth"
)


def load_teacher(device="cpu", num_classes=100):
    model = resnet110(num_classes=num_classes).to(device)

    if os.path.isfile(CKPT_PATH):
        print(f"[teacher] Loading local checkpoint: {CKPT_PATH}")
        state = torch.load(CKPT_PATH, map_location=device)
        if "state_dict" in state:
            state = state["state_dict"]
        elif "model_state" in state:
            state = state["model_state"]
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
    else:
        print("[teacher] No checkpoint found")

    for p in model.parameters():
        p.requires_grad = False

    model.eval()
    return model