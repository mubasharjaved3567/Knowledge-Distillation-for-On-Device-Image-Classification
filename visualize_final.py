"""
visualize_final.py — Clean, correct, professional KD plots.
All values hardcoded from verified checkpoints. No dynamic loading issues.

Usage:
    python visualize_final.py
    python visualize_final.py --skip-slow
"""

import os, sys, glob, argparse, torch
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from collections import defaultdict

pio.templates.default = "plotly_white"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# VERIFIED DATA — hardcoded from checkpoints, 100% correct
# ══════════════════════════════════════════════════════════════════════════════

TEACHER = {"name": "Teacher (ResNet-110)", "acc": 74.17, "epochs": 200, "best_ep": 156}
HARD    = {"name": "Student Hard Labels",  "acc": 69.37, "epochs": 200, "best_ep": 155}

EXPERIMENTS = [
    # Exp1 — Temperature sweep (α=0.9, 200ep, Logit-KD)
    {"exp": 1,  "T": 2.0, "alpha": 0.9,  "beta": "-",  "epochs": 200, "best_ep": 174, "acc": 70.78, "method": "Logit-KD", "note": "T=2"},
    {"exp": 1,  "T": 4.0, "alpha": 0.9,  "beta": "-",  "epochs": 200, "best_ep": 182, "acc": 71.21, "method": "Logit-KD", "note": "T=4"},
    {"exp": 1,  "T": 8.0, "alpha": 0.9,  "beta": "-",  "epochs": 200, "best_ep": 200, "acc": 70.78, "method": "Logit-KD", "note": "T=8"},
    # Exp2 — Regularisation (T=4, α=0.9, 200ep)
    {"exp": 2,  "T": 4.0, "alpha": 0.9,  "beta": "-",  "epochs": 200, "best_ep": 193, "acc": 71.10, "method": "Logit-KD", "note": ""},
    # Exp3 — Alpha 0.95 (T=4, 200ep)
    {"exp": 3,  "T": 4.0, "alpha": 0.95, "beta": "-",  "epochs": 200, "best_ep": 197, "acc": 70.72, "method": "Logit-KD", "note": ""},
    # Exp4 — More epochs (T=4, α=0.95, 300ep)
    {"exp": 4,  "T": 4.0, "alpha": 0.95, "beta": "-",  "epochs": 300, "best_ep": 288, "acc": 71.25, "method": "Logit-KD", "note": ""},
    # Exp5 — Best Logit-KD (T=4, α=0.95, 300ep)
    {"exp": 5,  "T": 4.0, "alpha": 0.95, "beta": "-",  "epochs": 300, "best_ep": 293, "acc": 71.70, "method": "Logit-KD", "note": ""},
    # Exp6 — Feature KD
    {"exp": 6,  "T": 4.0, "alpha": 0.95, "beta": 0.1,  "epochs": 300, "best_ep": 299, "acc": 71.74, "method": "Feat-KD",  "note": ""},
    # Exp7 — Attention KD layer2
    {"exp": 7,  "T": 4.0, "alpha": 0.95, "beta": 0.1,  "epochs": 300, "best_ep": 298, "acc": 71.82, "method": "Attn-KD",  "note": ""},
    # Exp8 — Attention KD all layers
    {"exp": 8,  "T": 4.0, "alpha": 0.95, "beta": 1000, "epochs": 300, "best_ep": 289, "acc": 71.41, "method": "Attn-KD(all)", "note": ""},
    # Exp9 — CRD
    {"exp": 9,  "T": 4.0, "alpha": 0.95, "beta": 0.8,  "epochs": 300, "best_ep": 293, "acc": 71.26, "method": "CRD",      "note": ""},
    # Exp10 — DKD (BEST)
    {"exp": 10, "T": 4.0, "alpha": 1.0,  "beta": 0.5,  "epochs": 300, "best_ep": 297, "acc": 72.62, "method": "DKD",      "note": "β=0.5"},
]

BEST_ACC = max(e["acc"] for e in EXPERIMENTS)

METHOD_COLORS = {
    "Logit-KD"    : "#2ecc71",
    "Feat-KD"     : "#3498db",
    "Attn-KD"     : "#9b59b6",
    "Attn-KD(all)": "#e67e22",
    "CRD"         : "#e91e8c",
    "DKD"         : "#f39c12",
}

FONT   = dict(family="Inter, Arial, sans-serif", size=12, color="#2c3e50")
LAYOUT = dict(
    font=FONT,
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=70, r=150, t=110, b=80),
)


def save_fig(fig, name, w=1400, h=620):
    html = os.path.join(RESULTS_DIR, name.replace(".png", ".html"))
    png  = os.path.join(RESULTS_DIR, name)
    fig.write_html(html)
    try:
        fig.write_image(png, width=w, height=h, scale=2)
    except Exception as e:
        print(f"  [warn] PNG: {e}")
    print(f"  ✓ {name}")


def ref_lines(fig):
    fig.add_hline(y=74.17, line_dash="dash", line_color="#2c3e50",
                  line_width=1.5, opacity=0.7,
                  annotation_text="  Teacher 74.17%",
                  annotation_position="right",
                  annotation_font=dict(size=10, color="#2c3e50"))
    fig.add_hline(y=69.37, line_dash="dot", line_color="#e74c3c",
                  line_width=1.5, opacity=0.7,
                  annotation_text="  Hard labels 69.37%",
                  annotation_position="right",
                  annotation_font=dict(size=10, color="#e74c3c"))


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Full summary bar chart
# ══════════════════════════════════════════════════════════════════════════════
def plot_1_summary():
    labels  = ["Teacher<br>ResNet-110", "Hard<br>Labels"]
    accs    = [74.17, 69.37]
    colors  = ["#2c3e50", "#e74c3c"]
    texts   = ["74.17%", "69.37%"]
    borders = ["#2c3e50", "#e74c3c"]
    bwidths = [1, 1]

    for e in EXPERIMENTS:
        # Use note for T label if available (fixes Exp1 T=2/4/8)
        t_label = f"T={int(e['T']) if e['T'] == int(e['T']) else e['T']}"
        if e.get("note") and e["note"].startswith("T="):
            t_label = e["note"]
        labels.append(f"Exp{e['exp']}<br>{e['method']}<br>{t_label} α={e['alpha']}<br>{e['epochs']}ep")
        accs.append(e["acc"])
        c = METHOD_COLORS[e["method"]]
        colors.append(c)
        texts.append(f"<b>{e['acc']:.2f}%</b>" if e["acc"] == BEST_ACC else f"{e['acc']:.2f}%")
        borders.append("#FFD700" if e["acc"] == BEST_ACC else c)
        bwidths.append(3 if e["acc"] == BEST_ACC else 1)

    fig = go.Figure(go.Bar(
        x=labels, y=accs,
        marker_color=colors,
        marker_line_color=borders,
        marker_line_width=bwidths,
        text=texts, textposition="outside", textfont=dict(size=9),
        hovertemplate="<b>%{x}</b><br>Accuracy: %{y:.2f}%<extra></extra>",
    ))
    ref_lines(fig)
    fig.update_layout(**LAYOUT,
        title=dict(
            text="<b>Accuracy Comparison — All Models</b><br>"
                 "<sup>ResNet-110 Teacher → ResNet-20 Student | CIFAR-100 | Gold border = best result</sup>",
            font=dict(size=15), x=0.5, xanchor="center"),
        yaxis=dict(range=[63, 78], title="Top-1 Accuracy (%)",
                   showgrid=True, gridcolor="#f0f0f0"),
        xaxis=dict(tickfont=dict(size=8)),
        showlegend=False, height=700,
    )
    save_fig(fig, "1_summary_bar.png", w=1700, h=700)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Temperature sweep (Exp1: T=2,4,8)
# ══════════════════════════════════════════════════════════════════════════════
def plot_2_temperature_sweep():
    # Verified: T=2→70.78%, T=4→71.21%, T=8→70.78% (all Exp1, α=0.9, 200ep)
    temps = [2.0, 4.0, 8.0]
    accs  = [70.78, 71.21, 70.78]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=temps, y=accs,
        mode="lines+markers",
        name="Exp1 (α=0.9, 200ep)",
        line=dict(color="#2ecc71", width=3),
        marker=dict(size=16, color="white",
                    line=dict(color="#2ecc71", width=3)),
        hovertemplate="T=%{x}<br>Accuracy: %{y:.2f}%<extra></extra>",
    ))

    # Non-overlapping annotations: T=2 above, T=4 below, T=8 above
    offsets = [(0, -55), (0, 55), (0, -55)]
    ann_colors = ["#2ecc71", "#e67e22", "#2ecc71"]
    for i, (T, acc) in enumerate(zip(temps, accs)):
        ax, ay = offsets[i]
        label = f"★ T={int(T)} BEST: {acc:.2f}%" if i == 1 else f"T={int(T)}: {acc:.2f}%"
        fig.add_annotation(
            x=T, y=acc, text=f"<b>{label}</b>",
            showarrow=True, arrowhead=2, arrowsize=1,
            arrowwidth=1.5, arrowcolor=ann_colors[i],
            ax=ax, ay=ay,
            font=dict(size=12, color=ann_colors[i]),
        )

    ref_lines(fig)
    fig.update_layout(**LAYOUT,
        title=dict(
            text="<b>Effect of Temperature on Distillation Quality</b><br>"
                 "<sup>Exp1 | α=0.9 | 200 epochs | Logit-KD | T=4 is optimal</sup>",
            font=dict(size=15), x=0.5, xanchor="center"),
        xaxis=dict(title="Distillation Temperature T",
                   tickvals=[2, 4, 8], ticktext=["T=2", "T=4", "T=8"],
                   showgrid=True, gridcolor="#f0f0f0", range=[1, 9]),
        yaxis=dict(title="Top-1 Accuracy (%)", range=[68, 75.5],
                   showgrid=True, gridcolor="#f0f0f0"),
        height=560, width=900,
    )
    save_fig(fig, "2_temperature_sweep.png", w=900, h=560)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Training curves
# ══════════════════════════════════════════════════════════════════════════════
def plot_3_training_curves():
    entries = [
        ("teacher_history.pt",                      "Teacher (74.17%)",          "#2c3e50", "dash",  2.5),
        ("student_hard_history.pt",                 "Hard labels (69.37%)",      "#e74c3c", "dot",   2.0),
        ("distill_T2.0_a0.9_Exp1_e200_history.pt",  "Exp1 T=2 (70.78%)",        "#a8d8a8", "solid", 1.2),
        ("distill_T4.0_a0.9_Exp1_e200_history.pt",  "Exp1 T=4 (71.21%)",        "#2ecc71", "solid", 2.0),
        ("distill_T8.0_a0.9_Exp1_e200_history.pt",  "Exp1 T=8 (70.78%)",        "#82c882", "solid", 1.2),
        ("distill_T4.0_a0.95_Exp5_e300_history.pt", "Exp5 Logit-KD (71.70%)",   "#1a7a1a", "solid", 2.0),
        ("distill_T4.0_a0.95_Exp7_e300_history.pt", "Exp7 Attn-KD (71.82%)",    "#9b59b6", "solid", 2.0),
    ]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["<b>Training Loss</b>",
                                        "<b>Test Accuracy (%)</b>"],
                        horizontal_spacing=0.1)
    plotted = False
    for fname, label, color, dash, width in entries:
        path = os.path.join("results", fname)
        if not os.path.isfile(path): continue
        try:
            h = torch.load(path, map_location="cpu")
        except: continue
        show = True
        if "train_loss" in h and len(h["train_loss"]) > 0:
            fig.add_trace(go.Scatter(
                x=list(range(len(h["train_loss"]))), y=h["train_loss"],
                name=label, line=dict(color=color, dash=dash, width=width),
                showlegend=show, legendgroup=label,
                hovertemplate=f"<b>{label}</b><br>Epoch: %{{x}}<br>Loss: %{{y:.3f}}<extra></extra>",
            ), row=1, col=1)
            show = False
        if "test_acc" in h and len(h["test_acc"]) > 0:
            fig.add_trace(go.Scatter(
                x=list(range(len(h["test_acc"]))), y=h["test_acc"],
                name=label, line=dict(color=color, dash=dash, width=width),
                showlegend=show, legendgroup=label,
                hovertemplate=f"<b>{label}</b><br>Epoch: %{{x}}<br>Acc: %{{y:.2f}}%<extra></extra>",
            ), row=1, col=2)
        plotted = True

    if not plotted:
        print("  [skip] No history .pt files in results/"); return

    fig.add_hline(y=74.17, line_dash="dash", line_color="#2c3e50",
                  line_width=1.2, opacity=0.5,
                  annotation_text="  Teacher 74.17%",
                  annotation_font=dict(size=9), row=1, col=2)

    fig.update_layout(
        font=FONT, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=60, r=200, t=100, b=60),
        title=dict(
            text="<b>Training Dynamics — Teacher vs All Student Configurations</b>",
            font=dict(size=15), x=0.5, xanchor="center"),
        height=540, width=1450,
        legend=dict(x=1.01, y=0.5, bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#e0e0e0", borderwidth=1),
    )
    fig.update_xaxes(title_text="Epoch", showgrid=True, gridcolor="#f0f0f0")
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
    save_fig(fig, "3_training_curves.png", w=1450, h=540)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 4 — Method comparison (best per method)
# ══════════════════════════════════════════════════════════════════════════════
def plot_4_method_comparison():
    order = ["Logit-KD", "Feat-KD", "Attn-KD", "Attn-KD(all)", "CRD", "DKD"]
    method_best = {}
    for e in EXPERIMENTS:
        m = e["method"]
        if m not in method_best or e["acc"] > method_best[m]["acc"]:
            method_best[m] = e

    methods  = [m for m in order if m in method_best]
    accs     = [method_best[m]["acc"] for m in methods]
    colors   = [METHOD_COLORS[m] for m in methods]
    best_acc = max(accs)
    exps     = [f"Exp{method_best[m]['exp']}" for m in methods]

    fig = go.Figure(go.Bar(
        x=methods, y=accs,
        marker_color=colors,
        marker_line_color=["#FFD700" if a == best_acc else c
                           for a, c in zip(accs, colors)],
        marker_line_width=[3 if a == best_acc else 1 for a in accs],
        text=[f"<b>{a:.2f}%</b><br><sup>{e}</sup>"
              for a, e in zip(accs, exps)],
        textposition="outside", textfont=dict(size=11),
        width=0.5,
        hovertemplate="<b>%{x}</b><br>Best accuracy: %{y:.2f}%<extra></extra>",
    ))
    ref_lines(fig)
    fig.update_layout(**LAYOUT,
        title=dict(
            text="<b>Best Result per Distillation Method</b><br>"
                 "<sup>ResNet-110 → ResNet-20 | CIFAR-100 | Gold border = best overall</sup>",
            font=dict(size=15), x=0.5, xanchor="center"),
        yaxis=dict(range=[70.5, 74.0], title="Top-1 Accuracy (%)",
                   showgrid=True, gridcolor="#f0f0f0"),
        xaxis=dict(title="Distillation Method", tickfont=dict(size=12)),
        showlegend=False, height=580, width=1000,
    )
    save_fig(fig, "4_method_comparison.png", w=1000, h=580)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 5 — Accuracy gap (sorted worst to best)
# ══════════════════════════════════════════════════════════════════════════════
def plot_5_accuracy_gap():
    # Best result per experiment number — no duplicates
    exp_best = {}
    for e in EXPERIMENTS:
        key = e["exp"]
        if key not in exp_best or e["acc"] > exp_best[key]["acc"]:
            exp_best[key] = e
    sorted_exp = sorted(exp_best.values(), key=lambda x: -(74.17 - x["acc"]))

    def gap_label(e):
        beta_str = f" β={e['beta']}" if e["beta"] != "-" else ""
        return f"Exp{e['exp']} {e['method']}<br>T={e['T']} α={e['alpha']}{beta_str} {e['epochs']}ep"

    labels = ["Hard Labels<br>No distillation"] + [gap_label(e) for e in sorted_exp]
    gaps   = [round(74.17 - 69.37, 2)] + [round(74.17 - e["acc"], 2) for e in sorted_exp]
    colors = ["#e74c3c"] + [METHOD_COLORS[e["method"]] for e in sorted_exp]
    min_gap = min(gaps)

    fig = go.Figure(go.Bar(
        x=labels, y=gaps,
        marker_color=colors,
        marker_line_color=["#FFD700" if g == min_gap else c
                           for g, c in zip(gaps, colors)],
        marker_line_width=[3 if g == min_gap else 1 for g in gaps],
        text=[f"−{g:.2f}%" for g in gaps],
        textposition="outside", textfont=dict(size=10), width=0.6,
        hovertemplate="<b>%{x}</b><br>Gap: −%{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(**LAYOUT,
        title=dict(
            text="<b>Student–Teacher Accuracy Gap</b><br>"
                 "<sup>Lower is better | Sorted worst → best | Gold = closest to teacher</sup>",
            font=dict(size=15), x=0.5, xanchor="center"),
        yaxis=dict(range=[0, 5.8], title="Gap to Teacher (%)",
                   showgrid=True, gridcolor="#f0f0f0"),
        xaxis=dict(tickfont=dict(size=9), tickangle=-30),
        showlegend=False, height=580,
    )
    save_fig(fig, "5_accuracy_gap.png")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 6 — Results table
# ══════════════════════════════════════════════════════════════════════════════
def plot_6_results_table():
    header = ["<b>Exp</b>", "<b>Model</b>", "<b>Method</b>",
              "<b>T</b>", "<b>Alpha</b>", "<b>Beta</b>",
              "<b>Epochs</b>", "<b>Best Ep</b>",
              "<b>Accuracy</b>", "<b>Gap</b>"]

    rows = [
        ["—", "Teacher (ResNet-110)", "—",   "—",   "—",   "—",   "200", "156",
         "<b>74.17%</b>", "—"],
        ["—", "Student Hard Labels",  "—",   "—",   "—",   "—",   "200", "155",
         "69.37%", "−4.80%"],
    ]
    fill_teacher = ["#D6E8F7"] * len(header)
    fill_hard    = ["#FDE8E8"] * len(header)
    row_fills    = [fill_teacher, fill_hard]

    for e in EXPERIMENTS:
        gap     = round(74.17 - e["acc"], 2)
        acc_str = f"<b>{e['acc']:.2f}%</b>" if e["acc"] == BEST_ACC else f"{e['acc']:.2f}%"
        beta_str = str(e["beta"]) if e["beta"] != "-" else "—"
        rows.append([
            str(e["exp"]),
            "ResNet-20",
            e["method"],
            f"T={e['T']}",
            f"α={e['alpha']}",
            beta_str,
            str(e["epochs"]),
            str(e["best_ep"]),
            acc_str,
            f"−{gap:.2f}%",
        ])
        if e["acc"] == BEST_ACC:
            row_fills.append(["#D4EDDA"] * len(header))
        elif len(rows) % 2 == 0:
            row_fills.append(["#FAFAFA"] * len(header))
        else:
            row_fills.append(["white"] * len(header))

    fill_by_col = []
    for j in range(len(header)):
        fill_by_col.append([row_fills[i][j] for i in range(len(rows))])

    fig = go.Figure(go.Table(
        columnwidth=[45, 160, 110, 60, 70, 60, 70, 75, 100, 80],
        header=dict(values=header, fill_color="#2196F3",
                    font=dict(color="white", size=11),
                    align="center", height=38, line_color="white"),
        cells=dict(values=list(zip(*rows)),
                   fill_color=fill_by_col,
                   font=dict(color="#2c3e50", size=10),
                   align=["center","left","center","center","center",
                           "center","center","center","center","center"],
                   height=33, line_color="#E0E0E0"),
    ))
    fig.update_layout(
        title=dict(
            text="<b>Knowledge Distillation — Complete Results Table</b><br>"
                 "<sup>ResNet-110 Teacher → ResNet-20 Student | CIFAR-100 | Green = best</sup>",
            font=dict(size=15), x=0.5, xanchor="center"),
        font=FONT,
        height=120 + len(rows) * 35,
        margin=dict(l=20, r=20, t=90, b=20),
        paper_bgcolor="white",
    )
    save_fig(fig, "6_results_table.png", w=1500, h=120 + len(rows) * 35)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 7 — Alpha comparison (bar chart — no overlap)
# ══════════════════════════════════════════════════════════════════════════════
def plot_7_alpha_comparison():
    # Verified best per alpha at T=4
    alpha_data = [
        {"alpha": "α=0.9",  "acc": 71.21, "exp": "Exp1",  "method": "Logit-KD",  "color": "#2ecc71"},
        {"alpha": "α=0.95", "acc": 71.82, "exp": "Exp7",  "method": "Attn-KD",   "color": "#9b59b6"},
        {"alpha": "α=1.0",  "acc": 72.62, "exp": "Exp10", "method": "DKD β=0.5", "color": "#f39c12"},
    ]

    best_acc = max(d["acc"] for d in alpha_data)
    labels   = [f"{d['alpha']}<br>{d['method']}<br>{d['exp']}" for d in alpha_data]
    accs     = [d["acc"] for d in alpha_data]
    colors   = [d["color"] for d in alpha_data]

    fig = go.Figure(go.Bar(
        x=labels, y=accs,
        marker_color=colors,
        marker_line_color=["#FFD700" if a == best_acc else c for a, c in zip(accs, colors)],
        marker_line_width=[3 if a == best_acc else 1 for a in accs],
        text=[f"<b>{a:.2f}%</b>" if a == best_acc else f"{a:.2f}%" for a in accs],
        textposition="outside",
        textfont=dict(size=13),
        width=0.45,
        hovertemplate="<b>%{x}</b><br>Accuracy: %{y:.2f}%<extra></extra>",
    ))

    # Gain annotations below each bar
    baseline = 69.37
    for i, d in enumerate(alpha_data):
        gain = d["acc"] - baseline
        fig.add_annotation(
            x=labels[i], y=baseline + 0.1,
            text=f"+{gain:.2f}% vs hard labels",
            showarrow=False,
            font=dict(size=9, color="#888888"),
            yref="y",
        )

    ref_lines(fig)
    fig.update_layout(**LAYOUT,
        title=dict(
            text="<b>Effect of Alpha on Best Achievable Accuracy</b><br>"
                 "<sup>T=4 | Best result per α value | α=1.0 with DKD achieves 72.62%</sup>",
            font=dict(size=15), x=0.5, xanchor="center"),
        xaxis=dict(title="Alpha α — Distillation Method — Experiment",
                   tickfont=dict(size=11)),
        yaxis=dict(title="Top-1 Accuracy (%)", range=[68.5, 74.5],
                   showgrid=True, gridcolor="#f0f0f0"),
        showlegend=False,
        height=560, width=900,
    )
    save_fig(fig, "7_alpha_comparison.png", w=900, h=560)




# ══════════════════════════════════════════════════════════════════════════════
# PLOT 8 — Research journey: accuracy improvement across experiments
# ══════════════════════════════════════════════════════════════════════════════
def plot_8_epochs_effect():
    # Best result per experiment number
    exp_best = {}
    for exp in EXPERIMENTS:
        k = exp["exp"]
        if k not in exp_best or exp["acc"] > exp_best[k]["acc"]:
            exp_best[k] = exp

    journey = sorted(exp_best.values(), key=lambda x: x["exp"])

    exp_nums = [f"Exp{e['exp']}" for e in journey]
    accs     = [e["acc"] for e in journey]
    methods  = [e["method"] for e in journey]
    colors   = [METHOD_COLORS[e["method"]] for e in journey]
    best_acc = max(accs)

    # Hover text with full details
    hover = [
        f"<b>Exp{e['exp']} {e['method']}</b><br>"
        f"T={e['T']} α={e['alpha']}" +
        (f" β={e['beta']}" if e['beta'] != '-' else "") +
        f"<br>Epochs: {e['epochs']} | Best: {e['best_ep']}<br>"
        f"<b>Accuracy: {e['acc']:.2f}%</b><br>"
        f"Gap to teacher: −{74.17-e['acc']:.2f}%"
        for e in journey
    ]

    fig = go.Figure()

    # Baseline reference
    fig.add_hline(y=69.37, line_dash="dot", line_color="#e74c3c",
                  line_width=1.5, opacity=0.6,
                  annotation_text="  Hard labels 69.37%",
                  annotation_position="right",
                  annotation_font=dict(size=10, color="#e74c3c"))
    fig.add_hline(y=74.17, line_dash="dash", line_color="#2c3e50",
                  line_width=1.5, opacity=0.6,
                  annotation_text="  Teacher 74.17%",
                  annotation_position="right",
                  annotation_font=dict(size=10, color="#2c3e50"))

    # Bars colored by method
    fig.add_trace(go.Bar(
        x=exp_nums, y=accs,
        marker_color=colors,
        marker_line_color=["#FFD700" if a == best_acc else c for a, c in zip(accs, colors)],
        marker_line_width=[3 if a == best_acc else 1 for a in accs],
        text=[f"<b>{a:.2f}%</b>" if a == best_acc else f"{a:.2f}%" for a in accs],
        textposition="outside",
        textfont=dict(size=9),
        customdata=methods,
        hovertemplate=[h + "<extra></extra>" for h in hover],
    ))



    fig.update_layout(
        font=FONT, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=70, r=200, t=110, b=60),
        title=dict(
            text="<b>Research Journey — Accuracy Improvement Across Experiments</b><br>"
                 "<sup>Each bar = best result per experiment | Color = method | Gold = best overall</sup>",
            font=dict(size=15), x=0.5, xanchor="center"),
        xaxis=dict(title="Experiment", tickfont=dict(size=10)),
        yaxis=dict(title="Top-1 Accuracy (%)", range=[68.5, 74.5],
                   showgrid=True, gridcolor="#f0f0f0"),
        showlegend=False,
        height=580, width=1200,
    )



    save_fig(fig, "8_epochs_effect.png", w=1200, h=580)










# ══════════════════════════════════════════════════════════════════════════════
# PLOT 9 — t-SNE
# ══════════════════════════════════════════════════════════════════════════════
def plot_9_tsne():
    try:
        from sklearn.manifold import TSNE
        import torchvision
        sys.path.insert(0, ".")
        from models import resnet20
        from utils  import get_cifar100_loaders, get_device
    except ImportError as e:
        print(f"  [skip] t-SNE: {e}"); return

    CKPT_DIR  = "checkpoints"
    hard_ckpt = os.path.join(CKPT_DIR, "student_hard.pth")
    best_ckpt = os.path.join(CKPT_DIR, "student_attn_layer2_T4_a0.95_b0.1_Exp7_e300.pth")

    if not os.path.isfile(hard_ckpt) or not os.path.isfile(best_ckpt):
        print("  [skip] t-SNE: checkpoints not found"); return

    print("  Running t-SNE (~2 min)...")
    device = get_device()
    _, loader = get_cifar100_loaders()
    N, N_CLS  = 1500, 20

    def get_features(path):
        m = resnet20(num_classes=100).to(device)
        m.load_state_dict(torch.load(path, map_location=device)["model_state"])
        m.eval()
        feats, lbls = [], []
        with torch.no_grad():
            for imgs, labels in loader:
                f = m(imgs.to(device))
                feats.append(f.cpu()); lbls.append(labels)
                if sum(len(x) for x in feats) >= N: break
        feats = torch.cat(feats)[:N]
        lbls  = torch.cat(lbls)[:N]
        mask  = lbls < N_CLS
        return feats[mask].numpy(), lbls[mask].numpy()

    hard_f, hard_l = get_features(hard_ckpt)
    best_f, best_l = get_features(best_ckpt)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    h2d  = tsne.fit_transform(hard_f)
    b2d  = tsne.fit_transform(best_f)

    dataset     = torchvision.datasets.CIFAR100(root="data", train=False, download=False)
    class_names = [dataset.classes[i] for i in range(N_CLS)]
    colors_20   = px.colors.qualitative.Tab20[:N_CLS]

    fig = make_subplots(rows=1, cols=2,
        subplot_titles=[
            "<b>Hard-Label Student (69.37%)</b><br><sup>scattered, overlapping clusters</sup>",
            "<b>Distilled Student Exp7 Attn-KD (71.82%)</b><br><sup>tighter, better-separated clusters</sup>",
        ], horizontal_spacing=0.06)

    for cls in range(N_CLS):
        c  = colors_20[cls]
        m1 = hard_l == cls
        m2 = best_l == cls
        fig.add_trace(go.Scatter(
            x=h2d[m1, 0], y=h2d[m1, 1], mode="markers",
            name=class_names[cls], marker=dict(color=c, size=5, opacity=0.7),
            legendgroup=class_names[cls], showlegend=True,
            hovertemplate=f"<b>{class_names[cls]}</b><extra></extra>",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=b2d[m2, 0], y=b2d[m2, 1], mode="markers",
            name=class_names[cls], marker=dict(color=c, size=5, opacity=0.7),
            legendgroup=class_names[cls], showlegend=False,
            hovertemplate=f"<b>{class_names[cls]}</b><extra></extra>",
        ), row=1, col=2)

    fig.update_layout(
        font=FONT, paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=40, r=40, t=110, b=40),
        title=dict(
            text="<b>t-SNE Feature Embeddings — First 20 CIFAR-100 Classes</b><br>"
                 "<sup>Knowledge distillation produces tighter, better-separated class representations</sup>",
            font=dict(size=15), x=0.5, xanchor="center"),
        height=640, width=1450,
        legend=dict(itemsizing="constant", font=dict(size=9), ncols=2),
    )
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
    save_fig(fig, "9_tsne.png", w=1450, h=640)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-slow", action="store_true", help="Skip t-SNE")
    args = parser.parse_args()

    print("\n" + "═"*62)
    print("  KD Visualizer Final — Verified Data, Clean Plots")
    print("═"*62)
    print(f"\n  Teacher     : {TEACHER['acc']:.2f}%")
    print(f"  Hard labels : {HARD['acc']:.2f}%")
    print(f"  Best result : Exp10 DKD {BEST_ACC:.2f}% (β=0.5)")
    print(f"  Gap closed  : {74.17-69.37:.2f}% → {74.17-BEST_ACC:.2f}%")
    print(f"  Experiments : {len(EXPERIMENTS)}\n")

    os.system("pip install kaleido plotly --quiet")

    print("  Generating plots...")
    plot_1_summary()
    plot_2_temperature_sweep()
    plot_3_training_curves()
    plot_4_method_comparison()
    plot_5_accuracy_gap()
    plot_6_results_table()
    plot_7_alpha_comparison()
    plot_8_epochs_effect()
    if not args.skip_slow:
        plot_9_tsne()
    else:
        print("  [skip] t-SNE (--skip-slow)")

    print(f"\n  All plots → {RESULTS_DIR}/")
    print("  Open .html files in browser for interactive versions!")
    print("═"*62 + "\n")


if __name__ == "__main__":
    main()