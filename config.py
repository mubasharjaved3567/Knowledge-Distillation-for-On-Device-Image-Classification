import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
CHECKPOINT_DIR  = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR     = os.path.join(BASE_DIR, "results")

os.makedirs(DATA_DIR,       exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR,    exist_ok=True)

# ── Dataset ────────────────────────────────────────────────────────────────────
NUM_CLASSES  = 100
IMAGE_SIZE   = 32
NUM_WORKERS  = 2

# CIFAR-100 normalisation stats
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)

# ── Training ───────────────────────────────────────────────────────────────────
BATCH_SIZE   = 128
EPOCHS       = 200          # teacher  (reduce to 100 if pressed for time)
STUDENT_EPOCHS = 200        # student

LR           = 0.1
MOMENTUM     = 0.9
WEIGHT_DECAY = 5e-4

# LR schedule: divide by 10 at these epoch milestones
LR_MILESTONES = [100, 150]
LR_GAMMA      = 0.1

# ── Distillation ───────────────────────────────────────────────────────────────
# Temperature controls how "soft" the teacher probabilities are.
# Higher T → softer distribution → more dark knowledge revealed.
TEMPERATURE  = 4.0

# Weight on the soft-label KL term.
# (1 - ALPHA) is the weight on the hard-label CE term.
# ALPHA        = 0.9 #Experiment 1 - standard setting
ALPHA = 0.95  #Experiment 2 - optimized setting (more weight on teacher since we're using AutoAugment which is a stronger augmentation)

# ── Experiment grid (used by train_student_distill.py sweep mode) ──────────────
TEMPERATURES = [1, 2, 4, 8, 16]
ALPHAS       = [0.1, 0.5, 0.9]

# ── Misc ───────────────────────────────────────────────────────────────────────
SEED         = 42
DEVICE       = "cuda"       # falls back to cpu automatically in helpers.py
