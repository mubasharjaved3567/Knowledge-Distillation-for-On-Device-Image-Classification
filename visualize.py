"""
visualize_pro.py — Publication-quality KD report plots.
Dark theme, professional typography, no overlapping labels.

Usage:
    python visualize_pro.py
    python visualize_pro.py --skip-slow   # skip t-SNE
"""

import os, re, sys, glob, argparse, torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from collections import defaultdict

# ── Config ─────────────────────────────────────────────────────────────────────
CKPT_DIR    = "checkpoints"
RESULTS_DIR = "results"
TEACHER_ACC = 74.17
HARD_ACC    = 69.37
HARD_EPOCH  = 155
NUM_CLASSES = 100

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Color System ───────────────────────────────────────────────────────────────
BG      = "#0F1117"
BG2     = "#161B22"
BG3     = "#21262D"
BORDER  = "#30363D"
TEXT    = "#E6EDF3"
TEXT2   = "#7D8590"
BLUE    = "#58A6FF"
RED     = "#F85149"
GREEN   = "#3FB950"
GOLD    = "#D29922"
PURPLE  = "#A371F7"
TEAL    = "#39D353"
ORANGE  = "#F0883E"
PINK    = "#DB61A2"

METHOD_COLORS = {
    "Logit-KD"    : "#3FB950",
    "Feat-KD"     : "#58A6FF",
    "Attn-KD"     : "#A371F7",
    "Attn-KD(all)": "#F0883E",
    "CRD"         : "#DB61A2",
    "DKD"         : "#D29922",
}

def setup():
    plt.rcParams.update({
        "figure.facecolor"  : BG,
        "axes.facecolor"    : BG2,
        "axes.edgecolor"    : BORDER,
        "axes.labelcolor"   : TEXT2,
        "axes.titlecolor"   : TEXT,
        "axes.titlesize"    : 13,
        "axes.titleweight"  : "bold",
        "axes.titlepad"     : 16,
        "axes.labelsize"    : 10,
        "axes.grid"         : True,
        "axes.axisbelow"    : True,
        "grid.color"        : BORDER,
        "grid.linewidth"    : 0.4,
        "grid.alpha"        : 0.5,
        "xtick.color"       : TEXT2,
        "ytick.color"       : TEXT2,
        "xtick.labelsize"   : 8.5,
        "ytick.labelsize"   : 8.5,
        "legend.facecolor"  : BG3,
        "legend.edgecolor"  : BORDER,
        "legend.labelcolor" : TEXT2,
        "legend.fontsize"   : 8,
        "legend.framealpha" : 0.9,
        "text.color"        : TEXT,
        "font.family"       : "DejaVu Sans",
        "figure.constrained_layout.use": True,
        "savefig.facecolor" : BG,
        "savefig.edgecolor" : BG,
    })

setup()


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
def load_checkpoints():
    results = []
    seen    = set()
    all_files = sorted(
        glob.glob(os.path.join(CKPT_DIR, "student_T*.pth"))    +
        glob.glob(os.path.join(CKPT_DIR, "student_feat*.pth")) +
        glob.glob(os.path.join(CKPT_DIR, "student_attn*.pth")) +
        glob.glob(os.path.join(CKPT_DIR, "student_crd*.pth"))  +
        glob.glob(os.path.join(CKPT_DIR, "student_dkd*.pth"))
    )
    pat = re.compile(r"_T([\d.]+)_a([\d.]+).*_Exp(\d+)_e(\d+)\.pth")
    for path in all_files:
        fname = os.path.basename(path)
        if fname in seen or fname.endswith(".bak"): continue
        seen.add(fname)
        m = pat.search(fname)
        if not m: continue
        try:
            ckpt = torch.load(path, map_location="cpu")
        except Exception as e:
            print(f"  [warn] {fname}: {e}"); continue
        T, alpha   = float(m.group(1)), float(m.group(2))
        exp_num    = int(m.group(3))
        epochs     = int(m.group(4))
        acc        = float(ckpt.get("best_acc", 0.0))
        fl         = fname.lower()
        if "dkd"        in fl: method = "DKD"
        elif "crd"      in fl: method = "CRD"
        elif "attn_all" in fl: method = "Attn-KD(all)"
        elif "attn"     in fl: method = "Attn-KD"
        elif "feat"     in fl: method = "Feat-KD"
        else:                  method = "Logit-KD"
        # skip incomplete runs (acc < 65%)
        if acc < 65.0:
            print(f"  [skip] Exp{exp_num} acc={acc:.2f}% (incomplete)"); continue
        results.append({
            "fname": fname, "T": T, "alpha": alpha,
            "exp": exp_num, "epochs": epochs,
            "acc": acc, "gap": round(TEACHER_ACC - acc, 2),
            "method": method,
            "color": METHOD_COLORS.get(method, GREEN),
            "ls": "ls" in fl, "aa": "aa" in fl, "cos": "cos" in fl,
        })
        print(f"  Exp{exp_num} [{method}]: T={T} α={alpha} ep={epochs}  acc={acc:.2f}%")
    results.sort(key=lambda r: (r["exp"], r["T"]))
    return results


def load_history(fname):
    path = os.path.join(RESULTS_DIR, fname)
    if os.path.isfile(path):
        try: return torch.load(path, map_location="cpu")
        except: pass
    return None


def save(fig, name):
    out = os.path.join(RESULTS_DIR, name)
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  ✓ {out}")


def ref_lines(ax):
    ax.axhline(TEACHER_ACC, color=BLUE, linestyle="--",
               linewidth=1.2, alpha=0.8, label=f"Teacher  {TEACHER_ACC:.2f}%", zorder=1)
    ax.axhline(HARD_ACC, color=RED, linestyle=":",
               linewidth=1.2, alpha=0.8, label=f"Hard labels  {HARD_ACC:.2f}%", zorder=1)


def annotate_bar(ax, bar, val, color=TEXT, size=8, offset=0.12):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            f"{val:.2f}%",
            ha="center", va="bottom",
            fontsize=size, fontweight="bold", color=color)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Summary: method comparison bar chart (grouped by method)
# ══════════════════════════════════════════════════════════════════════════════
def plot_1_summary(results):
    best_acc = max(r["acc"] for r in results)

    # Build clean labels — short
    labels = ["Teacher\nResNet-110", "Hard\nLabels"]
    accs   = [TEACHER_ACC, HARD_ACC]
    colors = [BLUE, RED]
    edges  = [BLUE, RED]

    for r in results:
        labels.append(f"Exp{r['exp']}\n{r['method']}")
        accs.append(r["acc"])
        c = r["color"]
        colors.append(c)
        edges.append("#FFD60A" if r["acc"] == best_acc else c)

    n   = len(labels)
    fig, ax = plt.subplots(figsize=(max(12, n * 1.1), 6))
    fig.patch.set_facecolor(BG)

    x    = np.arange(n)
    bars = ax.bar(x, accs, color=colors, width=0.65,
                  edgecolor=edges, linewidth=1.2, zorder=3)

    # Highlight best with gold edge + star
    for i, (bar, acc, color) in enumerate(zip(bars, accs, colors)):
        offset = 0.1
        txt_color = TEXT
        if i >= 2 and acc == best_acc:
            txt_color = "#FFD60A"
            offset = 0.15
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                f"{acc:.2f}%",
                ha="center", va="bottom",
                fontsize=7.5, fontweight="bold", color=txt_color)

    ref_lines(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5, linespacing=1.4, color=TEXT2)
    ax.set_ylim(63, 77)
    ax.set_ylabel("Top-1 Accuracy (%)", color=TEXT2)
    ax.set_title("Knowledge Distillation — Accuracy Across All Experiments", color=TEXT)

    # Custom legend
    patches = [
        mpatches.Patch(color=BLUE,   label="Teacher (ResNet-110)"),
        mpatches.Patch(color=RED,    label="Hard labels baseline"),
    ] + [mpatches.Patch(color=c, label=m)
         for m, c in METHOD_COLORS.items()
         if any(r["method"] == m for r in results)]
    patches.append(mpatches.Patch(color=BG3, edgecolor="#FFD60A",
                                  linewidth=2, label="★ Best result"))
    ax.legend(handles=patches, loc="lower right", fontsize=7.5, ncol=2)

    save(fig, "1_summary_bar.png")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Temperature sweep
# ══════════════════════════════════════════════════════════════════════════════
def plot_2_temperature_sweep(results):
    sweep = sorted(
        [r for r in results if r["alpha"] == 0.9 and r["method"] == "Logit-KD"
         and r["epochs"] == 200],
        key=lambda r: r["T"]
    )
    if len(sweep) < 2:
        print("  [skip] Not enough T sweep points"); return

    temps = [r["T"] for r in sweep]
    accs  = [r["acc"] for r in sweep]

    fig, ax = plt.subplots(figsize=(7, 5))

    # Shaded area between hard and teacher
    ax.axhspan(HARD_ACC, TEACHER_ACC, alpha=0.03, color=GREEN, zorder=0)

    ax.plot(temps, accs, "o-", color=GREEN, linewidth=2.5,
            markersize=11, zorder=4, label="Distilled student",
            markerfacecolor=BG2, markeredgecolor=GREEN, markeredgewidth=2.5)

    # Annotations with background box
    for T, acc in zip(temps, accs):
        ax.annotate(f"{acc:.2f}%",
                    xy=(T, acc), xytext=(0, 18),
                    textcoords="offset points",
                    ha="center", fontsize=9, fontweight="bold", color=GREEN,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=BG3,
                              edgecolor=GREEN, alpha=0.8))

    ref_lines(ax)
    ax.set_xlabel("Distillation Temperature T", color=TEXT2)
    ax.set_ylabel("Top-1 Accuracy (%)", color=TEXT2)
    ax.set_title("Effect of Temperature on Student Accuracy\n(α=0.9, 200 epochs, Logit-KD)", color=TEXT)
    ax.set_xticks(temps)
    ax.set_ylim(68.5, 73)
    ax.legend(loc="lower right")

    save(fig, "2_temperature_sweep.png")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Training curves
# ══════════════════════════════════════════════════════════════════════════════
def plot_3_training_curves():
    entries = [
        ("teacher_history.pt",                       "Teacher (ResNet-110)",      BLUE,   "--", 2.0),
        ("student_hard_history.pt",                  "Hard labels",               RED,    "--", 1.8),
        ("distill_T2.0_a0.9_Exp1_e200_history.pt",   "Exp1: T=2 α=0.9",          "#74c476", "-", 1.0),
        ("distill_T4.0_a0.9_Exp1_e200_history.pt",   "Exp1: T=4 α=0.9",          GREEN,  "-",  1.5),
        ("distill_T8.0_a0.9_Exp1_e200_history.pt",   "Exp1: T=8 α=0.9",          "#a1d99b", "-", 1.0),
        ("distill_T4.0_a0.95_Exp5_e300_history.pt",  "Exp5: Logit-KD best",      TEAL,   "-",  2.0),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    plotted = False
    for fname, label, color, ls, lw in entries:
        h = load_history(fname)
        if h is None: continue
        if "train_loss" in h and len(h["train_loss"]) > 0:
            ax1.plot(h["train_loss"], label=label, color=color,
                     linestyle=ls, linewidth=lw, alpha=0.9)
        if "test_acc" in h and len(h["test_acc"]) > 0:
            ax2.plot(h["test_acc"], label=label, color=color,
                     linestyle=ls, linewidth=lw, alpha=0.9)
        plotted = True

    if not plotted:
        print("  [skip] No history files"); plt.close(fig); return

    ax1.set_title("Training Loss", color=TEXT)
    ax1.set_xlabel("Epoch", color=TEXT2)
    ax1.set_ylabel("Loss", color=TEXT2)
    ax1.legend(fontsize=7.5)
    ax1.set_xlim(0)

    ax2.axhline(TEACHER_ACC, color=BLUE, linestyle="--", linewidth=1, alpha=0.5)
    ax2.set_title("Test Accuracy (%)", color=TEXT)
    ax2.set_xlabel("Epoch", color=TEXT2)
    ax2.set_ylabel("Top-1 Accuracy (%)", color=TEXT2)
    ax2.legend(fontsize=7.5, loc="lower right")
    ax2.set_xlim(0)

    fig.suptitle("Training Dynamics — Teacher vs Student Configurations",
                 fontsize=13, fontweight="bold", color=TEXT, y=1.01)
    save(fig, "3_training_curves.png")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 4 — Method comparison (grouped, clean)
# ══════════════════════════════════════════════════════════════════════════════
def plot_4_method_comparison(results):
    if not results: return

    # Group by method — take best per method
    method_best = defaultdict(lambda: {"acc": 0, "exp": 0})
    for r in results:
        if r["acc"] > method_best[r["method"]]["acc"]:
            method_best[r["method"]] = r

    # Order: Logit-KD, Feat-KD, Attn-KD, Attn-KD(all), CRD, DKD
    order = ["Logit-KD", "Feat-KD", "Attn-KD", "Attn-KD(all)", "CRD", "DKD"]
    methods = [m for m in order if m in method_best]
    accs    = [method_best[m]["acc"] for m in methods]
    colors  = [METHOD_COLORS[m] for m in methods]
    exps    = [f"Exp{method_best[m]['exp']}" for m in methods]
    best_acc = max(accs)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    x    = np.arange(len(methods))
    bars = ax.bar(x, accs, color=colors, width=0.55,
                  edgecolor=[("#FFD60A" if a == best_acc else c) for a, c in zip(accs, colors)],
                  linewidth=1.5, zorder=3)

    for bar, acc, exp, color in zip(bars, accs, exps, colors):
        is_best = acc == best_acc
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{acc:.2f}%",
                ha="center", va="bottom",
                fontsize=9, fontweight="bold",
                color="#FFD60A" if is_best else TEXT)
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.45,
                exp,
                ha="center", va="bottom",
                fontsize=7, color=TEXT2)

    ref_lines(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9.5, color=TEXT)
    ax.set_ylim(68.5, 75.5)
    ax.set_ylabel("Top-1 Accuracy (%)", color=TEXT2)
    ax.set_title("Best Result per Distillation Method\n(ResNet-110 → ResNet-20, CIFAR-100)", color=TEXT)
    ax.legend(loc="lower right")

    save(fig, "4_method_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 5 — Accuracy gap (inverted — higher bar = worse)
# ══════════════════════════════════════════════════════════════════════════════
def plot_5_accuracy_gap(results):
    if not results: return

    labels  = ["Hard\nLabels"] + [f"Exp{r['exp']}\n{r['method']}" for r in results]
    gaps    = [round(TEACHER_ACC - HARD_ACC, 2)] + [r["gap"] for r in results]
    colors  = [RED] + [r["color"] for r in results]
    min_gap = min(gaps)

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.1), 5.5))

    x    = np.arange(len(labels))
    bars = ax.bar(x, gaps, color=colors, width=0.6,
                  edgecolor=["#FFD60A" if g == min_gap else c
                              for g, c in zip(gaps, colors)],
                  linewidth=1.2, zorder=3)

    for bar, gap, color in zip(bars, gaps, colors):
        is_best = gap == min_gap
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.06,
                f"−{gap:.2f}%",
                ha="center", va="bottom",
                fontsize=8, fontweight="bold",
                color="#FFD60A" if is_best else TEXT)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5, linespacing=1.4, color=TEXT2)
    ax.set_ylabel("Accuracy Gap vs Teacher (%)", color=TEXT2)
    ax.set_ylim(0, 6.0)
    ax.set_title("Student–Teacher Accuracy Gap  (lower = better  |  ★ = best)",
                 color=TEXT)
    ax.grid(axis="y", alpha=0.3)

    save(fig, "5_accuracy_gap.png")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 6 — Results table (dark theme)
# ══════════════════════════════════════════════════════════════════════════════
def plot_6_results_table(results):
    col_labels   = ["Model", "Method", "T", "Alpha", "Epochs", "Accuracy", "Gap"]
    best_acc_val = max(r["acc"] for r in results) if results else 0

    rows, row_facecolors = [], []

    rows.append(["Teacher (ResNet-110)", "—", "—", "—", "—",
                 f"{TEACHER_ACC:.2f}%", "—"])
    row_facecolors.append("#0D2137")

    rows.append(["Student — Hard Labels", "—", "—", "—", str(HARD_EPOCH),
                 f"{HARD_ACC:.2f}%", f"−{TEACHER_ACC - HARD_ACC:.2f}%"])
    row_facecolors.append("#2D0D0D")

    for r in results:
        rows.append([
            f"Exp{r['exp']}  ResNet-20",
            r["method"],
            f"T={r['T']}",
            f"α={r['alpha']}",
            str(r["epochs"]),
            f"{r['acc']:.2f}%",
            f"−{r['gap']:.2f}%",
        ])
        row_facecolors.append("#0D2A14" if r["acc"] == best_acc_val else BG2)

    n_rows  = len(rows)
    fig_h   = 1.6 + n_rows * 0.48
    fig, ax = plt.subplots(figsize=(15, fig_h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis("off")

    tbl = ax.table(cellText=rows, colLabels=col_labels,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 2.1)

    # Header
    for j in range(len(col_labels)):
        cell = tbl[0, j]
        cell.set_facecolor("#1F6FEB")
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor(BORDER)

    # Rows
    for i, (fc) in enumerate(row_facecolors):
        is_best = (i >= 2 and rows[i][5].replace("%", "") != "" and
                   float(rows[i][5].replace("%", "")) == best_acc_val)
        for j in range(len(col_labels)):
            cell = tbl[i+1, j]
            cell.set_facecolor(fc)
            cell.set_edgecolor(BORDER)
            if is_best:
                cell.set_text_props(fontweight="bold", color="#FFD60A")
            elif i == 0:
                cell.set_text_props(color="#58A6FF")
            elif i == 1:
                cell.set_text_props(color="#F85149")
            else:
                cell.set_text_props(color=TEXT2)

    ax.set_title("Knowledge Distillation — Complete Results Table\n"
                 "ResNet-110 Teacher  →  ResNet-20 Student  |  CIFAR-100",
                 fontsize=12, fontweight="bold", color=TEXT, pad=20)
    save(fig, "6_results_table.png")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 7 — Alpha comparison (only Logit-KD experiments)
# ══════════════════════════════════════════════════════════════════════════════
def plot_7_alpha_comparison(results):
    # Only pure Logit-KD at T=4 for clean alpha comparison
    logit_t4 = [r for r in results if r["T"] == 4.0 and r["method"] == "Logit-KD"]
    if len(logit_t4) < 2:
        print("  [skip] Not enough Logit-KD T=4 runs"); return

    alpha_best = defaultdict(float)
    for r in logit_t4:
        if r["acc"] > alpha_best[r["alpha"]]:
            alpha_best[r["alpha"]] = r["acc"]

    alphas = sorted(alpha_best.keys())
    accs   = [alpha_best[a] for a in alphas]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.fill_between(range(len(alphas)), accs,
                    [HARD_ACC] * len(alphas),
                    alpha=0.1, color=GREEN, zorder=1)
    ax.plot(range(len(alphas)), accs, "s-", color=GREEN,
            linewidth=2.5, markersize=12, zorder=4,
            markerfacecolor=BG2, markeredgecolor=GREEN, markeredgewidth=2.5,
            label="Best acc per α (Logit-KD, T=4)")

    for i, (a, acc) in enumerate(zip(alphas, accs)):
        ax.annotate(f"α={a}\n{acc:.2f}%",
                    xy=(i, acc), xytext=(0, 18),
                    textcoords="offset points",
                    ha="center", fontsize=9, fontweight="bold", color=GREEN,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=BG3,
                              edgecolor=GREEN, alpha=0.8))

    ref_lines(ax)
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f"α = {a}" for a in alphas], fontsize=10, color=TEXT)
    ax.set_ylim(68.5, 73.5)
    ax.set_ylabel("Top-1 Accuracy (%)", color=TEXT2)
    ax.set_title("Effect of Alpha (KD Loss Weight)\n(T=4, Logit-KD, best result per α)", color=TEXT)
    ax.legend(loc="lower right")

    save(fig, "7_alpha_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 8 — Epochs effect + method progression (combined)
# ══════════════════════════════════════════════════════════════════════════════
def plot_8_progression(results):
    if not results: return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: epoch effect at T=4 α=0.95
    runs = sorted(
        [r for r in results if r["T"] == 4.0 and r["alpha"] == 0.95
         and r["method"] in ("Logit-KD", "Feat-KD", "Attn-KD")],
        key=lambda r: (r["method"], r["epochs"])
    )

    # Group by method
    method_runs = defaultdict(list)
    for r in runs:
        method_runs[r["method"]].append(r)

    for method, mrs in method_runs.items():
        mrs.sort(key=lambda r: r["epochs"])
        eps  = [r["epochs"] for r in mrs]
        accs = [r["acc"]    for r in mrs]
        color = METHOD_COLORS.get(method, GREEN)
        ax1.plot(eps, accs, "o-", color=color, linewidth=2,
                 markersize=9, label=method,
                 markerfacecolor=BG2, markeredgecolor=color, markeredgewidth=2)
        for ep, acc in zip(eps, accs):
            ax1.annotate(f"{acc:.2f}%", xy=(ep, acc),
                         xytext=(0, 12), textcoords="offset points",
                         ha="center", fontsize=8, color=color)

    ref_lines(ax1)
    ax1.set_xlabel("Training Epochs", color=TEXT2)
    ax1.set_ylabel("Top-1 Accuracy (%)", color=TEXT2)
    ax1.set_title("Effect of Training Duration\n(T=4, α=0.95)", color=TEXT)
    ax1.set_ylim(68.5, 73.5)
    ax1.legend(loc="lower right")

    # Right: all methods ordered by accuracy
    sorted_r = sorted(results, key=lambda r: r["acc"])
    accs     = [r["acc"]    for r in sorted_r]
    labels   = [f"Exp{r['exp']}\n{r['method']}" for r in sorted_r]
    colors   = [r["color"]  for r in sorted_r]
    best_acc = max(accs)

    y = np.arange(len(sorted_r))
    bars = ax2.barh(y, accs, color=colors, height=0.6,
                    edgecolor=["#FFD60A" if a == best_acc else c
                                for a, c in zip(accs, colors)],
                    linewidth=1.2, zorder=3)

    for bar, acc, color in zip(bars, accs, colors):
        ax2.text(bar.get_width() + 0.05,
                 bar.get_y() + bar.get_height() / 2,
                 f"{acc:.2f}%",
                 va="center", fontsize=7.5, fontweight="bold",
                 color="#FFD60A" if acc == best_acc else TEXT)

    ax2.axvline(TEACHER_ACC, color=BLUE, linestyle="--", linewidth=1.2,
                alpha=0.8, label=f"Teacher {TEACHER_ACC:.2f}%")
    ax2.axvline(HARD_ACC, color=RED, linestyle=":", linewidth=1.2,
                alpha=0.8, label=f"Hard labels {HARD_ACC:.2f}%")
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels, fontsize=7.5, color=TEXT2)
    ax2.set_xlim(68, 75.5)
    ax2.set_xlabel("Top-1 Accuracy (%)", color=TEXT2)
    ax2.set_title("All Experiments Ranked by Accuracy", color=TEXT)
    ax2.legend(loc="lower right", fontsize=7.5)

    fig.suptitle("Training Duration & Experiment Ranking",
                 fontsize=13, fontweight="bold", color=TEXT, y=1.01)
    save(fig, "8_progression_ranked.png")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 9 — t-SNE
# ══════════════════════════════════════════════════════════════════════════════
def plot_9_tsne():
    try:
        from sklearn.manifold import TSNE
        sys.path.insert(0, ".")
        from models import resnet20
        from utils  import get_cifar100_loaders, get_device
    except ImportError as e:
        print(f"  [skip] t-SNE: {e}"); return

    hard_ckpt    = os.path.join(CKPT_DIR, "student_hard.pth")
    distill_ckpt = os.path.join(CKPT_DIR, "student_T4.0_a0.9_Exp1_e200.pth")
    if not os.path.isfile(hard_ckpt) or not os.path.isfile(distill_ckpt):
        print("  [skip] t-SNE: checkpoints not found"); return

    print("  Running t-SNE (~1-2 min) ...")
    device = get_device()
    _, test_loader = get_cifar100_loaders()
    N_SAMPLES, N_CLS = 1500, 20

    def get_features(ckpt_path):
        m = resnet20(num_classes=NUM_CLASSES).to(device)
        m.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state"])
        m.eval()
        feats, lbls = [], []
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = imgs.to(device)
                f = m(imgs)
                feats.append(f.cpu())
                lbls.append(labels)
                if sum(len(x) for x in feats) >= N_SAMPLES:
                    break
        feats = torch.cat(feats)[:N_SAMPLES]
        lbls  = torch.cat(lbls)[:N_SAMPLES]
        mask  = lbls < N_CLS
        return feats[mask].numpy(), lbls[mask].numpy()

    hard_f,    hard_l    = get_features(hard_ckpt)
    distill_f, distill_l = get_features(distill_ckpt)

    tsne       = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    hard_2d    = tsne.fit_transform(hard_f)
    distill_2d = tsne.fit_transform(distill_f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    cmap = plt.get_cmap("tab20")

    for cls in range(N_CLS):
        col = [cmap(cls / N_CLS)]
        m1  = hard_l == cls
        m2  = distill_l == cls
        ax1.scatter(hard_2d[m1, 0],    hard_2d[m1, 1],    c=col, s=12, alpha=0.7)
        ax2.scatter(distill_2d[m2, 0], distill_2d[m2, 1], c=col, s=12, alpha=0.7)

    for ax, title in [
        (ax1, "Hard-Label Student\n(scattered, overlapping clusters)"),
        (ax2, "Distilled Student T=4 α=0.9\n(tighter, more separated clusters)")
    ]:
        ax.set_title(title, color=TEXT, fontsize=11, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)

    fig.suptitle("t-SNE Feature Embeddings — First 20 CIFAR-100 Classes\n"
                 "Knowledge distillation produces tighter, better-separated representations",
                 fontsize=12, fontweight="bold", color=TEXT)
    save(fig, "9_tsne.png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-slow", action="store_true", help="Skip t-SNE")
    args = parser.parse_args()

    print("\n" + "═" * 62)
    print("  KD Visualizer Pro — Professional Report Plots")
    print("═" * 62)

    results = load_checkpoints()
    if not results:
        print("\n  No complete checkpoints found.\n"); return

    best = max(results, key=lambda r: r["acc"])
    print(f"\n  {len(results)} checkpoints loaded.")
    print(f"  Best  : Exp{best['exp']} [{best['method']}] — {best['acc']:.2f}%")
    print(f"  Gap   : −{best['gap']:.2f}% vs teacher")
    print(f"  Gain  : +{best['acc'] - HARD_ACC:.2f}% vs hard labels\n")

    print("  Generating plots...")
    plot_1_summary(results)
    plot_2_temperature_sweep(results)
    plot_3_training_curves()
    plot_4_method_comparison(results)
    plot_5_accuracy_gap(results)
    plot_6_results_table(results)
    plot_7_alpha_comparison(results)
    plot_8_progression(results)
    if not args.skip_slow:
        plot_9_tsne()
    else:
        print("  [skip] t-SNE (--skip-slow)")

    print(f"\n  All plots saved to: {RESULTS_DIR}/")
    print("═" * 62 + "\n")


if __name__ == "__main__":
    main()