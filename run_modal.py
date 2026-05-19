"""
run_modal.py — Run Experiment 9 (CRD) on Modal cloud GPU
=========================================================
Usage:
    modal run run_modal.py

Downloads results after training:
    modal volume get kd-checkpoints student_crd_T4_a0.95_b0.8_Exp9_e300.pth ./checkpoints/
    modal volume get kd-results distill_crd_T4.0_a0.95_b0.8_Exp9_e300_history.pt ./results/
"""

import modal
import os

# ── App ───────────────────────────────────────────────────────────────────────
app = modal.App("knowledge-distillation-exp9")

# ── Persistent volumes ────────────────────────────────────────────────────────
checkpoints_vol = modal.Volume.from_name("kd-checkpoints", create_if_missing=True)
results_vol     = modal.Volume.from_name("kd-results",     create_if_missing=True)
data_vol        = modal.Volume.from_name("kd-data",        create_if_missing=True)

# ── Image ─────────────────────────────────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.2.0",
        "torchvision==0.17.0",
        "tqdm",
        "numpy",
        "Pillow",
    )
    .add_local_dir(".", remote_path="/root/project",
                   ignore=["venv", "__pycache__", ".git",
                            "results", "checkpoints", "*.png"])
)

# ── Training function ─────────────────────────────────────────────────────────
@app.function(
    image=image,
    gpu="T4",
    timeout=60 * 60 * 8,
    volumes={
        "/root/project/checkpoints": checkpoints_vol,
        "/root/project/results"    : results_vol,
        "/root/project/data"       : data_vol,
    },
)
def train_exp9():
    import subprocess, sys

    result = subprocess.run(
        [sys.executable, "train_student_crd_distill.py"],
        cwd="/root/project",
        check=True,
    )

    checkpoints_vol.commit()
    results_vol.commit()

    return "Exp9 CRD training complete"


# ── Local entrypoint ──────────────────────────────────────────────────────────
@app.local_entrypoint()
def main():
    print("Starting Experiment 9 — CRD on Modal T4 GPU...")
    print("You can close your laptop.\n")

    msg = train_exp9.remote()
    print(f"\n{msg}")
    print("\nDownload results with:")
    print("  modal volume get kd-checkpoints student_crd_T4_a0.95_b0.8_Exp9_e300.pth ./checkpoints/")
    print("  modal volume get kd-results distill_crd_T4.0_a0.95_b0.8_Exp9_e300_history.pt ./results/")