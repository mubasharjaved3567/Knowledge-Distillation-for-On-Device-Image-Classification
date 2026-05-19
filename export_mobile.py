import torch
import sys
sys.path.insert(0, '.')
import torchvision
import torchvision.transforms as T

# Temporarily patch resnet forward to not use return_features
import models.resnet as resnet_module

# Load checkpoint
from models import resnet20
ckpt  = torch.load('checkpoints/student_attn_layer2_T4_a0.95_b0.1_Exp7_e300.pth', map_location='cpu')
model = resnet20(num_classes=100)
model.load_state_dict(ckpt['model_state'])
model.eval()

# Use trace with fixed input — avoids script issues
dummy    = torch.rand(1, 3, 32, 32)
traced   = torch.jit.trace(model, dummy)
traced._save_for_lite_interpreter('student_kd.ptl')
print('Exported student_kd.ptl')

# Verify PTL
dataset   = torchvision.datasets.CIFAR100(root='data', train=False, download=False)
transform = T.Compose([T.ToTensor(), T.Normalize([0.5071,0.4865,0.4409],[0.2675,0.2640,0.2633])])
classes   = dataset.classes
ptl = torch.jit.load('student_kd.ptl', map_location='cpu')
ptl.eval()
print('Verifying:')
for i in [0, 10, 50]:
    img, label = dataset[i]
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        out = ptl(tensor)
        # handle both tuple and tensor output
        if isinstance(out, tuple):
            out = out[0]
        probs = torch.softmax(out, dim=1)[0]
        top1  = probs.argmax().item()
    print(f'True: {classes[label]:15s} Pred: {classes[top1]:15s} {probs[top1]*100:.1f}%')
