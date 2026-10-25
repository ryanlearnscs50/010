"""
Step 2: models, triage endpoint, and subgroup audit.

Inputs  : measured 2 and 4 kHz thresholds in both ears, age, sex
Targets : 0.5, 1, 3, 6, 8 kHz thresholds in both ears (10 targets)
Models  : Copy (nearest measured frequency), Age-sex norm, Ridge, XGBoost (nested-CV tuned)
Primary : classification of WHO 2021 hearing loss (better-ear 4FA >= 20 dB HL; secondary >= 35 dB HL)

Run from the 5_resubmission folder:  python -I code/02_models_and_audit.py
"""
import json
import time
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import xgboost as xgb
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
SEED = 2026
B = 1000  # bootstrap replicates
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(SEED)

df = pd.read_csv(ROOT / "data_derived" / "cohorts.csv")
FEATURES = ["R_2k", "R_4k", "L_2k", "L_4k", "age", "female"]
TFREQ = ["500", "1k", "3k", "6k", "8k"]
TARGETS = [f"{e}_{f}" for e in ["R", "L"] for f in TFREQ]
GROUPS = ["NH White", "NH Black", "NH Asian", "Mexican American", "Other Hispanic", "Other/Multiracial"]
COHORTS = {"dev": "Development (2011-2016, 20-69 y)",
           "ext_a": "External A (1999-2004, 20-69 y)",
           "ext_b": "External B (2005-2018, 70+ y)"}
MODELS = ["Copy", "Age-sex norm", "Ridge", "XGBoost"]
RES = {}


# ---------------------------------------------------------------- models
def predict_copy(d):
    p = pd.DataFrame(index=d.index)
    for e in ["R", "L"]:
        p[f"{e}_500"] = d[f"{e}_2k"]
        p[f"{e}_1k"] = d[f"{e}_2k"]
        p[f"{e}_3k"] = (d[f"{e}_2k"] + d[f"{e}_4k"]) / 2
        p[f"{e}_6k"] = d[f"{e}_4k"]
        p[f"{e}_8k"] = d[f"{e}_4k"]
    return p[TARGETS].values


def age_bin(a):
    return np.clip((np.asarray(a) // 10).astype(int), 2, 6)  # 20s..60s; 70+ mapped to 60s


def fit_norm(d):
    t = d.assign(ab=age_bin(d.age)).groupby(["ab", "female"])[TARGETS].median()
    return t


def predict_norm(t, d):
    key = pd.MultiIndex.from_arrays([age_bin(d.age), d.female.values])
    return t.reindex(key).values


def fit_ridge(X, y):
    m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
    return m.fit(X, y)


XGB_GRID = [dict(max_depth=d, n_estimators=n) for d, n in product([2, 3, 4], [150, 300, 600])]
XGB_FIXED = dict(learning_rate=0.05, min_child_weight=5, subsample=0.8, tree_method="hist",
                 random_state=SEED, n_jobs=-1)


def tune_xgb(X, y):
    kf = KFold(3, shuffle=True, random_state=SEED)
    scores = []
    for g in XGB_GRID:
        maes = []
        for tr, va in kf.split(X):
            m = xgb.XGBRegressor(**g, **XGB_FIXED).fit(X[tr], y[tr])
            maes.append(np.mean(np.abs(m.predict(X[va]) - y[va])))
        scores.append(np.mean(maes))
    return XGB_GRID[int(np.argmin(scores))]


def fit_all(dtr):
    X, y = dtr[FEATURES].values, dtr[TARGETS].values
    best = tune_xgb(X, y)
    return {"norm": fit_norm(dtr), "ridge": fit_ridge(X, y),
            "xgb": xgb.XGBRegressor(**best, **XGB_FIXED).fit(X, y), "xgb_params": best}


def predict_all(fits, d):
    X = d[FEATURES].values
    return {"Copy": predict_copy(d), "Age-sex norm": predict_norm(fits["norm"], d),
            "Ridge": fits["ridge"].predict(X), "XGBoost": fits["xgb"].predict(X)}


# ---------------------------------------------------------------- fit: CV on dev, then external
t0 = time.time()
dev = df[df.cohort == "dev"].reset_index(drop=True)
preds = {c: {} for c in COHORTS}
oof = {m: np.zeros((len(dev), len(TARGETS))) for m in MODELS}
fold_params = []
for k, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=SEED).split(dev)):
    fits = fit_all(dev.iloc[tr])
    fold_params.append(fits["xgb_params"])
    for m, p in predict_all(fits, dev.iloc[te]).items():
        oof[m][te] = p
    print(f"fold {k} done ({time.time() - t0:.0f}s) xgb={fits['xgb_params']}")
preds["dev"] = oof
final = fit_all(dev)
print("final xgb params:", final["xgb_params"])
data = {"dev": dev}
for c in ["ext_a", "ext_b"]:
    data[c] = df[df.cohort == c].reset_index(drop=True)
    preds[c] = predict_all(final, data[c])
RES["xgb_params"] = {"cv_folds": fold_params, "final": final["xgb_params"]}

# save per-participant predictions for reproducibility
rows = []
for c, d in data.items():
    for m in MODELS:
        p = pd.DataFrame(preds[c][m], columns=[f"pred_{t}" for t in TARGETS])
        p["SEQN"], p["cohort"], p["model"] = d.SEQN.values, c, m
        rows.append(p)
pd.concat(rows).to_csv(OUT / "predictions.csv", index=False)


# ---------------------------------------------------------------- helpers
def boot_ci(f, n, reps=B, seed=SEED):
    r = np.random.default_rng(seed)
    vals = [f(r.integers(0, n, n)) for _ in range(reps)]
    return np.nanpercentile(vals, [2.5, 97.5]).tolist()


def be4fa(d, p, src="pred"):
    """Better-ear 4FA with measured 2/4 kHz and predicted (or true) 0.5/1 kHz."""
    out = []
    for e in ["R", "L"]:
        i500, i1k = TARGETS.index(f"{e}_500"), TARGETS.index(f"{e}_1k")
        lo = (p[:, i500] + p[:, i1k]) if src == "pred" else (d[f"{e}_500"] + d[f"{e}_1k"]).values
        out.append((lo + d[f"{e}_2k"].values + d[f"{e}_4k"].values) / 4)
    return np.minimum(out[0], out[1])


def hfa(d):
    return np.minimum((d.R_2k + d.R_4k) / 2, (d.L_2k + d.L_4k) / 2).values


def cls_metrics(y, score, cut):
    yhat = score >= cut
    tp, fn = np.sum(yhat & y), np.sum(~yhat & y)
    tn, fp = np.sum(~yhat & ~y), np.sum(yhat & ~y)
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    auc = roc_auc_score(y, score) if 0 < y.sum() < len(y) else np.nan
    return dict(sens=sens, spec=spec, ppv=tp / (tp + fp) if tp + fp else np.nan,
                npv=tn / (tn + fn) if tn + fn else np.nan, auc=auc, n=len(y), n_pos=int(y.sum()))


# ---------------------------------------------------------------- 1. threshold accuracy
mae_rows, sev_rows = [], []
for c, d in data.items():
    y = d[TARGETS].values
    for m in MODELS:
        e = preds[c][m] - y
        for j, t in enumerate(TARGETS):
            mae_rows.append(dict(cohort=c, model=m, target=t, freq=t.split("_")[1], mae=np.mean(np.abs(e[:, j])),
                                 within10=np.mean(np.abs(e[:, j]) <= 10), bias=np.mean(e[:, j])))
        pmae = np.abs(e).mean(axis=1)
        for s in ["Normal", "Mild", "Moderate+"]:
            msk = (d.who3 == s).values
            if msk.sum():
                sev_rows.append(dict(cohort=c, model=m, severity=s, n=int(msk.sum()),
                                     mae=pmae[msk].mean(), ci=boot_ci(lambda i: pmae[msk][i].mean(), msk.sum(), 300)))
        sev_rows.append(dict(cohort=c, model=m, severity="All", n=len(d), mae=pmae.mean(),
                             ci=boot_ci(lambda i: pmae[i].mean(), len(d), 300)))
mae_df = pd.DataFrame(mae_rows)
mae_df.to_csv(OUT / "mae_by_target.csv", index=False)
sev_df = pd.DataFrame(sev_rows)
sev_df.to_csv(OUT / "mae_by_severity.csv", index=False)
freq_df = mae_df.groupby(["cohort", "model", "freq"])[["mae", "within10", "bias"]].mean().reset_index()
freq_df.to_csv(OUT / "mae_by_freq.csv", index=False)

# test-retest at 1 kHz (measurement noise floor)
RES["retest_1k"] = {}
for c, d in data.items():
    diffs = np.concatenate([(d[f"{e}_1k"] - d[f"{e}_1k_retest"]).dropna().values for e in ["R", "L"]])
    RES["retest_1k"][c] = dict(mean_abs_diff=float(np.mean(np.abs(diffs))), n_ears=int(len(diffs)),
                               within10=float(np.mean(np.abs(diffs) <= 10)))
    # same-ear model error at 1 kHz for comparison
    for m in ["Copy", "XGBoost"]:
        j = [TARGETS.index("R_1k"), TARGETS.index("L_1k")]
        RES["retest_1k"][c][f"{m}_mae_1k"] = float(np.mean(np.abs(preds[c][m][:, j] - d[["R_1k", "L_1k"]].values)))
print("threshold accuracy done", f"{time.time() - t0:.0f}s")

# ---------------------------------------------------------------- 2. triage endpoint
SCORES = {"XGBoost 4FA": lambda c: be4fa(data[c], preds[c]["XGBoost"]),
          "Ridge 4FA": lambda c: be4fa(data[c], preds[c]["Ridge"]),
          "Copy 4FA": lambda c: be4fa(data[c], preds[c]["Copy"]),
          "Measured HFA (2,4 kHz)": lambda c: hfa(data[c])}
tri_rows, tri_grp_rows = [], []
for c, d in data.items():
    true4 = be4fa(d, None, src="true")
    assert np.allclose(true4, d.be4fa.values)
    for cut in [20, 35]:
        y = true4 >= cut
        for sname, sf in SCORES.items():
            s = sf(c)
            met = cls_metrics(y, s, cut)
            for k in ["sens", "spec", "auc"]:
                met[f"{k}_ci"] = boot_ci(lambda i: cls_metrics(y[i], s[i], cut)[k], len(y), 500)
            tri_rows.append(dict(cohort=c, cut=cut, score=sname, **met))
            # by race group
            for g in GROUPS:
                msk = (d.race == g).values
                if msk.sum() < 30:
                    continue
                yg, sg = y[msk], s[msk]
                mg = cls_metrics(yg, sg, cut)
                for k in ["sens", "spec"]:
                    mg[f"{k}_ci"] = boot_ci(lambda i: cls_metrics(yg[i], sg[i], cut)[k], len(yg), B)
                tri_grp_rows.append(dict(cohort=c, cut=cut, score=sname, group=g, **mg))
tri_df = pd.DataFrame(tri_rows)
tri_df.to_csv(OUT / "triage_overall.csv", index=False)
tri_grp = pd.DataFrame(tri_grp_rows)
tri_grp.to_csv(OUT / "triage_by_group.csv", index=False)
print("triage done", f"{time.time() - t0:.0f}s")

# 4FA error attributable to reconstruction (only 0.5 and 1 kHz are predicted)
RES["be4fa_mae"] = {c: {m: float(np.mean(np.abs(be4fa(data[c], preds[c][m]) - data[c].be4fa.values)))
                        for m in ["Copy", "Ridge", "XGBoost"]} for c in data}


# ---------------------------------------------------------------- 3. subgroup error audit (XGBoost)
def group_audit(c, m="XGBoost"):
    d = data[c]
    e = preds[c][m] - d[TARGETS].values
    pmae, pse = np.abs(e).mean(axis=1), e.mean(axis=1)
    rows = []
    ref = (d.race == "NH White").values
    for g in GROUPS:
        msk = (d.race == g).values
        if msk.sum() < 30:
            continue
        r = dict(cohort=c, model=m, group=g, n=int(msk.sum()), mae=pmae[msk].mean(), mse=pse[msk].mean(),
                 mean_true=d.loc[msk, TARGETS].values.mean(),
                 mae_ci=boot_ci(lambda i: pmae[msk][i].mean(), msk.sum()),
                 mse_ci=boot_ci(lambda i: pse[msk][i].mean(), msk.sum()))
        if g != "NH White":
            a, b = pmae[msk], pmae[ref]
            rr = np.random.default_rng(SEED)
            diffs = [a[rr.integers(0, len(a), len(a))].mean() - b[rr.integers(0, len(b), len(b))].mean()
                     for _ in range(B)]
            r["mae_diff_vs_white"] = a.mean() - b.mean()
            r["mae_diff_ci"] = np.percentile(diffs, [2.5, 97.5]).tolist()
        rows.append(r)
    return rows


aud = pd.DataFrame([r for c in data for r in group_audit(c)])
aud.to_csv(OUT / "group_audit_xgb.csv", index=False)


# conditional analyses on long-format data (participant x target), cluster-robust SEs by participant
def long_format(c, m="XGBoost"):
    d = data[c]
    p = preds[c][m]
    L = []
    for j, t in enumerate(TARGETS):
        L.append(pd.DataFrame({"SEQN": d.SEQN.values, "group": d.race.values, "target": t,
                               "freq": t.split("_")[1], "true": d[t].values, "pred": p[:, j]}))
    L = pd.concat(L, ignore_index=True)
    L["err"] = L.pred - L.true
    L = L[L.group.isin(GROUPS)]
    L["group"] = pd.Categorical(L.group, categories=[g for g in GROUPS if g in set(L.group)])
    return L


cond = []
for c in data:
    L = long_format(c)
    fits = {
        "raw": smf.ols("err ~ C(group) + C(target)", L),
        "cond_true": smf.ols("err ~ C(group) + C(target) * bs(true, df=4)", L),
        "calib": smf.ols("true ~ C(group) + C(target) * bs(pred, df=4)", L),
    }
    for name, f in fits.items():
        r = f.fit(cov_type="cluster", cov_kwds={"groups": L.SEQN})
        for g in L.group.cat.categories[1:]:
            k = f"C(group)[T.{g}]"
            ci = r.conf_int().loc[k].tolist()
            cond.append(dict(cohort=c, analysis=name, group=g, coef=r.params[k], lo=ci[0], hi=ci[1], p=r.pvalues[k]))
cond_df = pd.DataFrame(cond)
cond_df.to_csv(OUT / "conditional_group_effects.csv", index=False)
print("audit done", f"{time.time() - t0:.0f}s")

# binned signed error vs true threshold, and calibration (true vs predicted) by group, for figures
bins = np.arange(-10, 130, 10)
bin_rows = []
for c in data:
    L = long_format(c)
    for g in L.group.cat.categories:
        s = L[L.group == g]
        for var, val in [("true", "err"), ("pred", "true")]:
            s2 = s.assign(b=pd.cut(s[var], bins, right=False))
            agg = s2.groupby("b", observed=True)[val].agg(["mean", "count"]).reset_index()
            agg["mid"] = agg.b.apply(lambda x: x.left + 5)
            for _, r in agg[agg["count"] >= 20].iterrows():
                bin_rows.append(dict(cohort=c, group=g, x=var, mid=r.mid, y=r["mean"], n=r["count"]))
pd.DataFrame(bin_rows).to_csv(OUT / "binned_error_calibration.csv", index=False)

# ---------------------------------------------------------------- 4. survey-weighted sensitivity (dev)
d = data["dev"]
w = (d.WTMEC2YR / 2).values  # two cycles combined
e = preds["dev"]["XGBoost"] - d[TARGETS].values
pmae = np.abs(e).mean(axis=1)
y20, s20 = d.be4fa.values >= 20, be4fa(d, preds["dev"]["XGBoost"]) >= 20
wres = {"overall_mae_weighted": float(np.average(pmae, weights=w)),
        "prev_who20_weighted": float(np.average(y20, weights=w)),
        "prev_who35_weighted": float(np.average(d.be4fa.values >= 35, weights=w)),
        "groups": {}}
for g in GROUPS:
    msk = (d.race == g).values & (w > 0)
    yy, ss, ww = y20[msk], s20[msk], w[msk]
    wres["groups"][g] = dict(
        mae_w=float(np.average(pmae[msk], weights=ww)),
        sens20_w=float(np.average(ss[yy], weights=ww[yy])) if yy.sum() else None,
        spec20_w=float(np.average(~ss[~yy], weights=ww[~yy])) if (~yy).sum() else None)
RES["survey_weighted_dev"] = wres

# ---------------------------------------------------------------- 5. sensitivity analyses (dev CV, XGBoost)
def cv_xgb(dd, feats, params=final["xgb_params"]):
    dd = dd.reset_index(drop=True)
    o = np.zeros((len(dd), len(TARGETS)))
    for tr, te in KFold(5, shuffle=True, random_state=SEED).split(dd):
        m = xgb.XGBRegressor(**params, **XGB_FIXED).fit(dd.loc[tr, feats].values, dd.loc[tr, TARGETS].values)
        o[te] = m.predict(dd.loc[te, feats].values)
    return o, dd


sens = {}
for name, dd, feats in [
    ("primary_features", dev, FEATURES),
    ("thresholds_only", dev, FEATURES[:4]),
    ("plus_pir_noise_cc", dev.dropna(subset=["pir", "occ_noise"]), FEATURES + ["pir", "occ_noise"]),
    ("primary_features_same_cc", dev.dropna(subset=["pir", "occ_noise"]), FEATURES),
    ("excl_recent_ear_illness", dev[dev.recent_ear_illness != 1], FEATURES),
    ("excl_any_no_response", dev[dev.any_nr == 0], FEATURES),
    ("with_race_feature", dev.assign(**{f"r_{i}": (dev.race == g).astype(int) for i, g in enumerate(GROUPS[1:])}),
     FEATURES + [f"r_{i}" for i in range(len(GROUPS) - 1)]),
]:
    o, dd = cv_xgb(dd, feats)
    err = o - dd[TARGETS].values
    pm = np.abs(err).mean(axis=1)
    yy = dd.be4fa.values >= 20
    met = cls_metrics(yy, be4fa(dd, o), 20)
    gm = {g: float(pm[(dd.race == g).values].mean()) for g in GROUPS}
    gs = {g: float(cls_metrics(yy[(dd.race == g).values], be4fa(dd, o)[(dd.race == g).values], 20)["sens"])
          for g in GROUPS}
    sens[name] = dict(n=len(dd), mae=float(pm.mean()), sens20=met["sens"], spec20=met["spec"], auc20=met["auc"],
                      group_mae=gm, group_sens20=gs)
RES["sensitivity"] = sens
print("sensitivity done", f"{time.time() - t0:.0f}s")

# ---------------------------------------------------------------- 6. SHAP (TreeSHAP via xgboost pred_contribs) on the final model
booster = final["xgb"].get_booster()
contrib = booster.predict(xgb.DMatrix(dev[FEATURES].values), pred_contribs=True)  # (n, targets, features+1)
shap_abs = np.abs(contrib[:, :, :-1]).mean(axis=0)  # targets x features
shap_df = pd.DataFrame(shap_abs, index=TARGETS, columns=FEATURES)
shap_df.to_csv(OUT / "shap_mean_abs.csv")
RES["shap_mean_over_targets"] = shap_df.mean(axis=0).to_dict()

# ---------------------------------------------------------------- 7. descriptives
desc = []
for c, d in data.items():
    for g in ["All"] + GROUPS:
        s = d if g == "All" else d[d.race == g]
        if len(s) == 0:
            continue
        desc.append(dict(cohort=c, group=g, n=len(s), age_mean=s.age.mean(), age_sd=s.age.std(),
                         female_pct=100 * s.female.mean(), be4fa_mean=s.be4fa.mean(), be4fa_sd=s.be4fa.std(),
                         who20_pct=100 * (s.be4fa >= 20).mean(), who35_pct=100 * (s.be4fa >= 35).mean(),
                         n_who20=int((s.be4fa >= 20).sum()), n_who35=int((s.be4fa >= 35).sum()),
                         any_nr_pct=100 * s.any_nr.mean(),
                         occ_noise_pct=100 * s.occ_noise.mean() if "occ_noise" in s and s.occ_noise.notna().any() else None,
                         pir_mean=s.pir.mean(), pir_missing_pct=100 * s.pir.isna().mean()))
pd.DataFrame(desc).to_csv(OUT / "descriptives.csv", index=False)

RES["versions"] = {"xgboost": xgb.__version__, "pandas": pd.__version__, "numpy": np.__version__}
RES["runtime_s"] = time.time() - t0
json.dump(RES, open(OUT / "results.json", "w"), indent=2, default=float)
print("ALL DONE", f"{time.time() - t0:.0f}s")
