# Knowledge Distillation for Efficient On-Device Image Classification

> **ResNet-110 Teacher → ResNet-20 Student | CIFAR-100 | 10 Experiments**

A systematic study of knowledge distillation techniques to compress a large deep neural network into a compact model suitable for on-device deployment, with end-to-end demonstration on Android (Samsung S21 / Redmi A5).

---

## Results Summary

| Exp | Method | T | Alpha | Beta | Epochs | Accuracy | Gap |
|-----|--------|---|-------|------|--------|----------|-----|
| — | Teacher (ResNet-110) | — | — | — | 200 | **74.17%** | — |
| — | Hard Labels (baseline) | — | — | — | 155 | 69.37% | −4.80% |
| 1 | Logit-KD | 2 | 0.9 | — | 200 | 70.78% | −3.39% |
| 1 | Logit-KD | 4 | 0.9 | — | 200 | 71.21% | −2.96% |
| 1 | Logit-KD | 8 | 0.9 | — | 200 | 70.78% | −3.39% |
| 2 | Logit-KD | 4 | 0.9 | — | 200 | 71.10% | −3.07% |
| 3 | Logit-KD | 4 | 0.95 | — | 200 | 70.72% | −3.45% |
| 4 | Logit-KD | 4 | 0.95 | — | 300 | 71.25% | −2.92% |
| 5 | Logit-KD | 4 | 0.95 | — | 300 | 71.70% | −2.47% |
| 6 | Feat-KD | 4 | 0.95 | 0.1 | 300 | 71.74% | −2.43% |
| 7 | Attn-KD | 4 | 0.95 | 0.1 | 300 | 71.82% | −2.35% |
| 8 | Attn-KD (all layers) | 4 | 0.95 | 1000 | 300 | 71.41% | −2.76% |
| 9 | CRD | 4 | 0.95 | 0.8 | 300 | 71.26% | −2.91% |
| **10** | **DKD** | **4** | **1.0** | **0.5** | **300** | **72.62%** | **−1.55%** |

**Best result: Exp10 DKD — 72.62% accuracy, closing the teacher gap to just 1.55%**

---

## Architecture

```
Teacher: ResNet-110  (1.74M params, 74.17% accuracy)
Student: ResNet-20   (0.27M params, 72.62% best accuracy)
Dataset: CIFAR-100   (50,000 train / 10,000 test, 100 classes, 32×32)
Compression: 6.4× fewer parameters
```

---

## Project Structure

```
knowledge_distillation/
├── models/
│   ├── resnet.py               # ResNet-20/110 for CIFAR
│   ├── teacher_loader.py       # Load pretrained teacher
│   └── __init__.py
├── utils/
│   ├── dataset.py              # CIFAR-100 data loaders
│   ├── distillation_loss.py    # KD loss (KL + CE)
│   ├── helpers.py              # Seed, device, checkpointing
│   └── __init__.py
├── android/                    # Android deployment app
│   └── app/src/main/
│       ├── java/com/kd/classifier/
│       │   ├── MainActivity.kt
│       │   └── Classifier.kt
│       ├── res/layout/
│       │   └── activity_main.xml
│       └── assets/
│           └── student_kd.ptl  # Exported PyTorch Mobile model
├── checkpoints/                # Saved model weights
├── results/                    # Generated plots (PNG + HTML)
├── config.py                   # All hyperparameters
├── train_teacher.py            # Train ResNet-110 teacher
├── train_student_hard.py       # Baseline student (no KD)
├── train_student_distill.py    # Logit-KD (Hinton et al.)
├── train_student_feature_distill.py  # Feature-KD (FitNets)
├── train_student_attention_distill.py # Attention-KD
├── train_student_crd_distill.py      # CRD
├── train_student_dkd.py              # DKD (CVPR 2022)
├── export_mobile.py            # Export to PyTorch Mobile (.ptl)
├── save_test_images.py         # Save CIFAR-100 test images
├── visualize_final.py          # All 9 report plots (Plotly)
├── evaluate.py                 # Evaluate any checkpoint
└── requirements.txt
```

---

## Setup

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/knowledge-distillation.git
cd knowledge-distillation

# Create virtual environment
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Training

### Step 1 — Train teacher
```bash
python train_teacher.py
# Expected: ~74% accuracy | ~3hrs on M1 Max
```

### Step 2 — Train hard label baseline
```bash
python train_student_hard.py
# Expected: ~69% accuracy
```

### Step 3 — Run distillation experiments

```bash
# Logit-KD (Hinton et al.)
python train_student_distill.py --temperature 4 --alpha 0.95 --epochs 300

# Feature-KD
python train_student_feature_distill.py

# Attention-KD
python train_student_attention_distill.py

# CRD
python train_student_crd_distill.py

# DKD (best result)
python train_student_dkd.py
```

### Step 4 — Evaluate
```bash
python evaluate.py
```

### Step 5 — Generate plots
```bash
python visualize_final.py --skip-slow   # skip t-SNE
python visualize_final.py               # all 9 plots including t-SNE
```

---

## Mobile Deployment

### Export model
```bash
python export_mobile.py
# Outputs: student_kd.ptl (~1.1MB)
```

### Build Android APK
```bash
cp student_kd.ptl android/app/src/main/assets/
cd android
./gradlew assembleDebug
# APK: android/app/build/outputs/apk/debug/app-debug.apk
```

The Android app supports:
- 📷 **Take Photo** — capture and classify in real time
- 🖼️ **Upload** — pick from gallery and classify
- Top-3 predictions with confidence bars
- Inference time display

---

## Key Findings

1. **Temperature T=4 is optimal** — T=2 and T=8 both give 70.78% while T=4 gives 71.21%
2. **More epochs help** — α=0.95 at 200ep gives 70.72% but at 300ep gives 71.70%
3. **DKD outperforms all methods** — 72.62% by decoupling target-class and non-target-class knowledge
4. **AutoAugment hurts small models** — ResNet-20 lacks capacity to regularise against it
5. **CRD underperformed expectations** — same-width architectures limit contrastive gains
6. **6.4× compression** — ResNet-20 (0.27M) vs ResNet-110 (1.74M) with only 1.55% accuracy drop

---

## References

| Paper | Used in |
|-------|---------|
| Hinton et al. (2015) — Distilling the Knowledge in a Neural Network | Exp1-5 |
| Romero et al. (2015) — FitNets | Exp6 |
| Zagoruyko & Komodakis (2017) — Attention Transfer | Exp7-8 |
| Tian et al. (2020) — Contrastive Representation Distillation | Exp9 |
| Zhao et al. (2022) — Decoupled Knowledge Distillation (CVPR) | Exp10 |
| He et al. (2016) — Deep Residual Learning | Architecture |
| Sandler et al. (2018) — MobileNetV2 | Mobile deployment |

---

## Requirements

```
torch>=2.0
torchvision
matplotlib
seaborn
plotly
kaleido
scikit-learn
numpy
pillow
pandas
```

---

## License

MIT License — free to use for academic and research purposes.

---

*Semester project — Knowledge Distillation for Efficient On-Device Image Classification*