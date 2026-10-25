"""
Step 3: figures from the CSV outputs of step 2 (no numbers are typed by hand).

Run from the 5_resubmission folder:  python -I code/03_figures.py
"""
import ast
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

# Categorical palette (validated reference order) + markers as secondary encoding
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
MK = ["o", "s", "^", "D", "v", "P"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "savefig.dpi": 300,
    "lines.linewidth": 2, "lines.markersize": 5,
})
COH = {"dev": "Development\n(2011–2016, 20–69 y)", "ext_a": "External A\n(1999–2004, 20–69 y)",
       "ext_b": "External B\n(2005–2018, ≥70 y)"}
COH1 = {k: v.replace("\n", " ") for k, v in COH.items()}
MODELS = ["Copy", "Age-sex norm", "Ridge", "XGBoost"]
GROUPS = ["NH White", "NH Black", "NH Asian", "Mexican American", "Other Hispanic", "Other/Multiracial"]
parse = lambda s: ast.literal_eval(s) if isinstance(s, str) else s


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 1: cohort flow
flow = pd.read_csv(ROOT / "data_derived" / "cohort_flow.csv")
fig, axes = plt.subplots(1, 3, figsize=(10, 4.2))
for ax, (c, label) in zip(axes, COH1.items()):
    f = flow[flow.cohort == c]
    cyc = ", ".join(f.cycle)
    steps = [("Audiometry records\n" + cyc.replace(", ", "; "), f.aux_records.sum()),
             ("In age range", f.in_age_range.sum()),
             ("≥1 threshold recorded", f.any_threshold.sum()),
             ("Complete 14-threshold\naudiogram (analysed)", f.final.sum())]
    ax.set_xlim(0, 1); ax.set_ylim(0, len(steps)); ax.axis("off")
    ax.set_title(label, fontsize=9.5, color=INK, fontweight="bold")
    for i, (txt, n) in enumerate(steps):
        y = len(steps) - i - 0.5
        ax.add_patch(plt.Rectangle((0.06, y - 0.32), 0.88, 0.64, fc="#f3f2ef", ec=INK2, lw=0.8))
        ax.text(0.5, y, f"{txt}\nn = {n:,}", ha="center", va="center", fontsize=7.6, color=INK)
        if i:
            ax.annotate("", xy=(0.5, y + 0.33), xytext=(0.5, y + 0.67),
                        arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8))
save(fig, "fig1_cohort_flow")

# ---------------------------------------------------------------- Fig 2: per-frequency MAE by model
fq = pd.read_csv(OUT / "mae_by_freq.csv")
import json
res = json.load(open(OUT / "results.json"))
order = ["500", "1k", "3k", "6k", "8k"]
fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
for ax, c in zip(axes, COH):
    for i, m in enumerate(MODELS):
        s = fq[(fq.cohort == c) & (fq.model == m)].set_index("freq").loc[order]
        ax.plot(range(5), s.mae, marker=MK[i], color=C[i], label=m)
    rt = res["retest_1k"][c]["mean_abs_diff"]
    ax.axhline(rt, color=INK2, ls=":", lw=1.2)
    ax.text(4.1, rt, "1 kHz test–retest", fontsize=7, color=INK2, va="bottom", ha="right")
    ax.set_xticks(range(5)); ax.set_xticklabels(["0.5", "1", "3", "6", "8"])
    ax.set_xlabel("Predicted frequency (kHz)"); ax.set_title(COH[c], fontsize=9)
axes[0].set_ylabel("Mean absolute error (dB)")
axes[0].legend(loc="upper left", fontsize=7.5)
save(fig, "fig2_mae_by_frequency")

# ---------------------------------------------------------------- Fig 3: MAE by WHO severity
sev = pd.read_csv(OUT / "mae_by_severity.csv")
sev["ci"] = sev.ci.apply(parse)
strata = ["Normal", "Mild", "Moderate+"]
fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
w = 0.2
for ax, c in zip(axes, COH):
    for i, m in enumerate(MODELS):
        s = sev[(sev.cohort == c) & (sev.model == m)].set_index("severity").reindex(strata)
        x = np.arange(3) + (i - 1.5) * w
        lo = s.mae - s.ci.apply(lambda v: v[0] if isinstance(v, list) else np.nan)
        hi = s.ci.apply(lambda v: v[1] if isinstance(v, list) else np.nan) - s.mae
        ax.bar(x, s.mae, w * 0.9, color=C[i], label=m, zorder=2)
        ax.errorbar(x, s.mae, yerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=0.8, capsize=0, zorder=3)
    ns = sev[(sev.cohort == c) & (sev.model == "XGBoost")].set_index("severity").reindex(strata).n
    ax.set_xticks(range(3)); ax.set_xticklabels([f"{s}\n(n={int(n):,})" for s, n in zip(strata, ns)], fontsize=8)
    ax.set_title(COH[c], fontsize=9); ax.grid(axis="x", visible=False)
axes[0].set_ylabel("Per-participant MAE (dB)")
axes[0].legend(fontsize=7.5, loc="upper left")
axes[0].set_xlabel(""); fig.text(0.5, -0.04, "WHO 2021 grade (better-ear 0.5–4 kHz average)", ha="center")
save(fig, "fig3_mae_by_severity")

# ---------------------------------------------------------------- Fig 4: triage sensitivity/specificity by group
tg = pd.read_csv(OUT / "triage_by_group.csv")
for k in ["sens_ci", "spec_ci"]:
    tg[k] = tg[k].apply(parse)
scores = ["XGBoost 4FA", "Copy 4FA", "Measured HFA (2,4 kHz)"]
slabel = {"XGBoost 4FA": "XGBoost-completed 4FA", "Copy 4FA": "Copy-completed 4FA",
          "Measured HFA (2,4 kHz)": "Measured 2–4 kHz average"}
fig, axes = plt.subplots(2, 3, figsize=(10, 6.0), sharey=True)
for col, c in enumerate(COH):
    for row, (met, lab) in enumerate([("sens", "Sensitivity"), ("spec", "Specificity")]):
        ax = axes[row, col]
        sub = tg[(tg.cohort == c) & (tg.cut == 20)]
        for i, sc in enumerate(scores):
            s = sub[sub.score == sc].set_index("group").reindex(GROUPS)
            y = np.arange(len(GROUPS)) + (i - 1) * 0.24
            ok = s[met].notna().values
            lo = (s[met] - s[f"{met}_ci"].apply(lambda v: v[0] if isinstance(v, list) else np.nan))[ok]
            hi = (s[f"{met}_ci"].apply(lambda v: v[1] if isinstance(v, list) else np.nan) - s[met])[ok]
            ax.errorbar(s[met][ok], y[ok], xerr=[lo, hi], fmt=MK[i], color=C[[3, 0, 1][i]],
                        ms=4.5, elinewidth=1, label=slabel[sc])
        ax.set_yticks(range(len(GROUPS))); ax.set_yticklabels(GROUPS, fontsize=7.5)
        ax.set_ylim(len(GROUPS) - 0.5, -0.5)
        ax.set_xlim(0.15, 1.02); ax.set_xlabel(lab)
        if row == 0:
            ax.set_title(COH[c], fontsize=9)
fig.tight_layout(rect=(0, 0.05, 1, 1))
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, fontsize=8)
save(fig, "fig4_triage_by_group")

# ---------------------------------------------------------------- Fig 5: regression to the mean vs calibration
bn = pd.read_csv(OUT / "binned_error_calibration.csv")
fig, axes = plt.subplots(2, 3, figsize=(10, 6.2))
for col, c in enumerate(COH):
    for row, (x, ylab, xlab) in enumerate([("true", "Mean signed error, predicted − true (dB)", "True threshold (dB HL)"),
                                           ("pred", "Mean true threshold (dB HL)", "Predicted threshold (dB HL)")]):
        ax = axes[row, col]
        sub = bn[(bn.cohort == c) & (bn.x == x)]
        for i, g in enumerate(GROUPS):
            s = sub[sub.group == g]
            if len(s):
                ax.plot(s.mid, s.y, marker=MK[i], color=C[i], lw=1.4, ms=3.5, label=g)
        if row == 0:
            ax.axhline(0, color=INK2, lw=0.8)
            ax.set_title(COH[c], fontsize=9)
        else:
            lim = [sub.mid.min() - 5, sub.mid.max() + 5]
            ax.plot(lim, lim, color=INK2, lw=0.8, ls="--")
        ax.set_xlabel(xlab)
        if col == 0:
            ax.set_ylabel(ylab, fontsize=8)
axes[0, 0].legend(fontsize=6.8, loc="lower left")
save(fig, "fig5_rtm_vs_calibration")

# ---------------------------------------------------------------- Supplementary: SHAP
sh = pd.read_csv(OUT / "shap_mean_abs.csv", index_col=0)
names = {"R_2k": "Right 2 kHz", "R_4k": "Right 4 kHz", "L_2k": "Left 2 kHz", "L_4k": "Left 4 kHz",
         "age": "Age", "female": "Sex"}
fig, ax = plt.subplots(figsize=(7, 3.6))
im = ax.imshow(sh.values, cmap="Blues", aspect="auto")
ax.set_xticks(range(sh.shape[1])); ax.set_xticklabels([names[c] for c in sh.columns], rotation=30, ha="right")
ax.set_yticks(range(sh.shape[0])); ax.set_yticklabels([t.replace("_", " ").replace("500", "0.5 kHz").replace("k", " kHz") for t in sh.index])
for i in range(sh.shape[0]):
    for j in range(sh.shape[1]):
        v = sh.values[i, j]
        ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7, color="white" if v > sh.values.max() * 0.6 else INK)
ax.grid(False)
fig.colorbar(im, ax=ax, label="Mean |SHAP| (dB)")
save(fig, "figS1_shap")
print("figures written to", FIG)
