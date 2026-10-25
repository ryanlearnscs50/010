"""
Step 2b: additional triage analyses from saved predictions.
  - paired bootstrap differences in AUC between scores
  - sensitivity at matched specificity (cut-offs fixed in the development cohort, applied unchanged externally)
  - group differences in sensitivity, raw and conditional on true better-ear 4FA (logistic model)

Run from the 5_resubmission folder:  python -I code/02b_triage_extras.py
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
SEED, B = 2026, 1000
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
df = pd.read_csv(ROOT / "data_derived" / "cohorts.csv")
pr = pd.read_csv(OUT / "predictions.csv")
GROUPS = ["NH White", "NH Black", "NH Asian", "Mexican American", "Other Hispanic", "Other/Multiracial"]
RES = {}


def scores_for(c):
    d = df[df.cohort == c].reset_index(drop=True)
    out = {}
    for m in ["XGBoost", "Ridge", "Copy"]:
        p = pr[(pr.cohort == c) & (pr.model == m)].set_index("SEQN").loc[d.SEQN]
        ears = [(p[f"pred_{e}_500"].values + p[f"pred_{e}_1k"].values + d[f"{e}_2k"].values + d[f"{e}_4k"].values) / 4
                for e in ["R", "L"]]
        out[f"{m} 4FA"] = np.minimum(*ears)
    out["Measured HFA (2,4 kHz)"] = np.minimum((d.R_2k + d.R_4k).values / 2, (d.L_2k + d.L_4k).values / 2)
    return d, out


data = {c: scores_for(c) for c in ["dev", "ext_a", "ext_b"]}

# 1. paired bootstrap AUC differences
rows = []
for c, (d, S) in data.items():
    for cut in [20, 35]:
        y = d.be4fa.values >= cut
        r = np.random.default_rng(SEED)
        idx = [r.integers(0, len(y), len(y)) for _ in range(B)]
        for comp in ["Measured HFA (2,4 kHz)", "Copy 4FA", "Ridge 4FA"]:
            a = S["XGBoost 4FA"]; b = S[comp]
            diff = roc_auc_score(y, a) - roc_auc_score(y, b)
            boot = [roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]) for i in idx]
            rows.append(dict(cohort=c, cut=cut, comparison=f"XGBoost 4FA - {comp}", auc_xgb=roc_auc_score(y, a),
                             auc_comp=roc_auc_score(y, b), diff=diff, lo=np.percentile(boot, 2.5), hi=np.percentile(boot, 97.5)))
auc_df = pd.DataFrame(rows)
auc_df.to_csv(OUT / "triage_auc_differences.csv", index=False)

# 2. matched specificity: cut-off chosen in dev so that specificity = target, applied unchanged externally
rows = []
d0, S0 = data["dev"]
for cut, target in [(20, 0.95), (20, 0.90), (35, 0.95)]:
    neg = d0.be4fa.values < cut
    for sname, s in S0.items():
        thr = np.quantile(s[neg], target)  # spec = P(score < thr | negative)
        for c, (d, S) in data.items():
            y = d.be4fa.values >= cut
            yhat = S[sname] >= thr
            rows.append(dict(cut=cut, target_spec_dev=target, score=sname, threshold=thr, cohort=c,
                             sens=np.mean(yhat[y]), spec=np.mean(~yhat[~y])))
ms = pd.DataFrame(rows)
ms.to_csv(OUT / "triage_matched_specificity.csv", index=False)

# 3. group differences in sensitivity (XGBoost 4FA and measured HFA at the WHO cut-off of 20 dB HL)
rows, cond = [], []
for c, (d, S) in data.items():
    y = d.be4fa.values >= 20
    for sname in ["XGBoost 4FA", "Measured HFA (2,4 kHz)"]:
        det = S[sname] >= 20
        ref = (d.race == "NH White").values & y
        for g in GROUPS[1:]:
            m = (d.race == g).values & y
            if m.sum() < 30:
                continue
            r = np.random.default_rng(SEED)
            boot = [det[m][r.integers(0, m.sum(), m.sum())].mean() - det[ref][r.integers(0, ref.sum(), ref.sum())].mean()
                  for _ in range(B)]
            rows.append(dict(cohort=c, score=sname, group=g, n_pos=int(m.sum()), sens=det[m].mean(),
                             sens_white=det[ref].mean(), diff=det[m].mean() - det[ref].mean(),
                             lo=np.percentile(boot, 2.5), hi=np.percentile(boot, 97.5)))
        if sname != "XGBoost 4FA":  # measured-HFA detection is near 100% in some groups (separation)
            continue
        # among true positives: P(detected) ~ group + spline(true be4fa)
        pos = pd.DataFrame({"det": det[y].astype(int), "group": d.race.values[y], "true": d.be4fa.values[y]})
        keep = [g for g in GROUPS if (pos.group == g).sum() >= 30]
        pos = pos[pos.group.isin(keep)]
        pos["group"] = pd.Categorical(pos.group, categories=keep)
        if pos.det.mean() in (0, 1):
            continue
        for label, f in [("raw", "det ~ C(group)"), ("cond_true", "det ~ C(group) + bs(true, df=3)")]:
            try:
                fit = smf.logit(f, pos).fit(disp=0)
            except Exception as e:  # perfect separation in small groups
                print("logit failed", c, sname, label, e)
                continue
            for g in keep[1:]:
                k = f"C(group)[T.{g}]"
                lo, hi = fit.conf_int().loc[k]
                cond.append(dict(cohort=c, score=sname, model=label, group=g, OR=np.exp(fit.params[k]),
                                 lo=np.exp(lo), hi=np.exp(hi), p=fit.pvalues[k]))
pd.DataFrame(rows).to_csv(OUT / "triage_group_sens_differences.csv", index=False)
pd.DataFrame(cond).to_csv(OUT / "triage_group_detection_conditional.csv", index=False)

pd.set_option("display.width", 220)
print(auc_df.round(4).to_string())
print(ms.pivot_table(index=["cut", "target_spec_dev", "cohort"], columns="score", values=["sens", "spec"]).round(3).to_string())
print(pd.DataFrame(rows).round(3).to_string())
print(pd.DataFrame(cond).round(3).to_string())
