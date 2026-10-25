"""
Step 4: write manuscript/numbers.tex (macros for every number quoted in the text) and manuscript/tables/*.tex,
directly from the analysis outputs. Nothing in the manuscript is typed by hand.

Run from the 5_resubmission folder:  python -I code/04_numbers_for_paper.py
"""
import ast
import re
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
MS = ROOT / "manuscript"
TAB = MS / "tables"
TAB.mkdir(parents=True, exist_ok=True)
P = lambda s: ast.literal_eval(s) if isinstance(s, str) else s

res = json.load(open(OUT / "results.json"))
flow = pd.read_csv(ROOT / "data_derived" / "cohort_flow.csv")
desc = pd.read_csv(OUT / "descriptives.csv")
fq = pd.read_csv(OUT / "mae_by_freq.csv")
sev = pd.read_csv(OUT / "mae_by_severity.csv")
tri = pd.read_csv(OUT / "triage_overall.csv")
tg = pd.read_csv(OUT / "triage_by_group.csv")
aucd = pd.read_csv(OUT / "triage_auc_differences.csv")
ms = pd.read_csv(OUT / "triage_matched_specificity.csv")
gsd = pd.read_csv(OUT / "triage_group_sens_differences.csv")
gdc = pd.read_csv(OUT / "triage_group_detection_conditional.csv")
aud = pd.read_csv(OUT / "group_audit_xgb.csv")
cond = pd.read_csv(OUT / "conditional_group_effects.csv")
shap = res["shap_mean_over_targets"]

M = {}
COH = {"dev": "Dev", "ext_a": "ExtA", "ext_b": "ExtB"}
GKEY = {"NH White": "White", "NH Black": "Black", "NH Asian": "Asian", "Mexican American": "MexAm",
        "Other Hispanic": "OthHisp", "Other/Multiracial": "Multi"}
SKEY = {"XGBoost 4FA": "Xgb", "Ridge 4FA": "Ridge", "Copy 4FA": "Copy", "Measured HFA (2,4 kHz)": "Hfa"}
MKEY = {"Copy": "Copy", "Age-sex norm": "Norm", "Ridge": "Ridge", "XGBoost": "Xgb"}
CUT = {20: "Twenty", 35: "ThirtyFive"}
f1 = lambda x: f"{x:.1f}"
f2 = lambda x: f"{x:.2f}"
f3 = lambda x: f"{x:.3f}"
n0 = lambda x: f"{int(round(x)):,}"
pct = lambda x: f"{100 * x:.1f}"


def ci(v, fmt=f2):
    return f"{fmt(v[0])}--{fmt(v[1])}"


# ---- cohorts
for c, k in COH.items():
    f = flow[flow.cohort == c]
    M[f"N{k}"] = n0(f.final.sum())
    M[f"NAux{k}"] = n0(f.aux_records.sum())
    d = desc[(desc.cohort == c) & (desc.group == "All")].iloc[0]
    M[f"Age{k}"] = f1(d.age_mean); M[f"AgeSd{k}"] = f1(d.age_sd)
    M[f"Fem{k}"] = f1(d.female_pct)
    M[f"PrevTwenty{k}"] = f1(d.who20_pct); M[f"PrevThirtyFive{k}"] = f1(d.who35_pct)
    M[f"NTwenty{k}"] = n0(d.n_who20); M[f"NThirtyFive{k}"] = n0(d.n_who35)
    M[f"NoResp{k}"] = f1(d.any_nr_pct)
M["NTotal"] = n0(flow.final.sum())
M["NoiseDevA"] = f1(100 * pd.read_csv(ROOT / "data_derived" / "cohorts.csv").query("cycle=='2011-2012'").occ_noise.mean())
M["NoiseDevB"] = f1(100 * pd.read_csv(ROOT / "data_derived" / "cohorts.csv").query("cycle=='2015-2016'").occ_noise.mean())
M["PrevTwentyDevW"] = pct(res["survey_weighted_dev"]["prev_who20_weighted"])
M["PrevThirtyFiveDevW"] = pct(res["survey_weighted_dev"]["prev_who35_weighted"])

# ---- accuracy
for c, k in COH.items():
    for m, mk in MKEY.items():
        s = sev[(sev.cohort == c) & (sev.model == m)].set_index("severity")
        M[f"Mae{mk}{k}"] = f2(s.loc["All", "mae"])
        M[f"MaeCi{mk}{k}"] = ci(P(s.loc["All", "ci"]))
        for sv, sk in [("Normal", "Norm"), ("Mild", "Mild"), ("Moderate+", "Mod")]:
            M[f"Mae{mk}{k}{sk}"] = f2(s.loc[sv, "mae"])
        for fr, fk in [("500", "Half"), ("1k", "One"), ("3k", "Three"), ("6k", "Six"), ("8k", "Eight")]:
            M[f"Mae{mk}{k}{fk}"] = f2(fq[(fq.cohort == c) & (fq.model == m) & (fq.freq == fr)].mae.iloc[0])
    M[f"Retest{k}"] = f2(res["retest_1k"][c]["mean_abs_diff"])
    M[f"RetestEars{k}"] = n0(res["retest_1k"][c]["n_ears"])
    for m, mk in [("Copy", "Copy"), ("Ridge", "Ridge"), ("XGBoost", "Xgb")]:
        M[f"FourFaMae{mk}{k}"] = f2(res["be4fa_mae"][c][m])
    w = lambda m: fq[(fq.cohort == c) & (fq.model == m)].within10.mean()
    M[f"WithinTen{k}"] = pct(w("XGBoost")); M[f"WithinTenCopy{k}"] = pct(w("Copy"))
M["MaeXgbDevW"] = f2(res["survey_weighted_dev"]["overall_mae_weighted"])
fp = res["xgb_params"]["final"]
M["XgbDepth"], M["XgbTrees"] = str(fp["max_depth"]), str(fp["n_estimators"])

# ---- triage
for c, k in COH.items():
    for cut, ck in CUT.items():
        for s, sk in SKEY.items():
            r = tri[(tri.cohort == c) & (tri.cut == cut) & (tri.score == s)].iloc[0]
            for met in ["sens", "spec", "auc", "ppv", "npv"]:
                M[f"{met.capitalize()}{sk}{ck}{k}"] = f3(r[met]) if met == "auc" else f2(r[met])
            for met in ["sens", "spec", "auc"]:
                M[f"{met.capitalize()}Ci{sk}{ck}{k}"] = ci(P(r[f"{met}_ci"]), f3 if met == "auc" else f2)
        for comp, cmpk in [("Measured HFA (2,4 kHz)", "Hfa"), ("Copy 4FA", "Copy"), ("Ridge 4FA", "Ridge")]:
            r = aucd[(aucd.cohort == c) & (aucd.cut == cut) & (aucd.comparison == f"XGBoost 4FA - {comp}")].iloc[0]
            M[f"AucDiff{cmpk}{ck}{k}"] = f3(r["diff"])
            M[f"AucDiffCi{cmpk}{ck}{k}"] = f"{r.lo:.3f} to {r.hi:.3f}"
    for s, sk in SKEY.items():
        r = ms[(ms.cut == 20) & (ms.target_spec_dev == 0.95) & (ms.score == s) & (ms.cohort == c)].iloc[0]
        M[f"MsSens{sk}{k}"] = f2(r.sens); M[f"MsSpec{sk}{k}"] = f2(r.spec)
for s, sk in SKEY.items():
    r = ms[(ms.cut == 20) & (ms.target_spec_dev == 0.95) & (ms.score == s)].iloc[0]
    M[f"MsThr{sk}"] = f1(r.threshold)
# group sensitivity differences and conditional detection ORs
for c, k in COH.items():
    for g, gk in GKEY.items():
        r = gsd[(gsd.cohort == c) & (gsd.score == "XGBoost 4FA") & (gsd.group == g)]
        if len(r):
            r = r.iloc[0]
            M[f"SensXgb{gk}{k}"] = f2(r.sens); M[f"SensDiff{gk}{k}"] = f2(r["diff"])
            M[f"SensDiffCi{gk}{k}"] = f"{r.lo:.2f} to {r.hi:.2f}"
            M[f"SensXgbWhite{k}"] = f2(r.sens_white)
        for mdl, mk in [("raw", "Raw"), ("cond_true", "Cond")]:
            r = gdc[(gdc.cohort == c) & (gdc.group == g) & (gdc.model == mdl)]
            if len(r):
                r = r.iloc[0]
                M[f"Or{mk}{gk}{k}"] = f2(r.OR); M[f"OrCi{mk}{gk}{k}"] = f"{r.lo:.2f}--{r.hi:.2f}"
# min / max group sensitivity (XGB, WHO>=20)
for c, k in COH.items():
    s = tg[(tg.cohort == c) & (tg.cut == 20) & (tg.score == "XGBoost 4FA")]
    M[f"SensGrpMin{k}"], M[f"SensGrpMax{k}"] = f2(s.sens.min()), f2(s.sens.max())

# ---- subgroup error audit
for c, k in COH.items():
    for g, gk in GKEY.items():
        r = aud[(aud.cohort == c) & (aud.group == g)]
        if not len(r):
            continue
        r = r.iloc[0]
        M[f"GMae{gk}{k}"] = f2(r.mae); M[f"GMse{gk}{k}"] = f2(r.mse); M[f"GN{gk}{k}"] = n0(r.n)
        if isinstance(r.mae_diff_ci, str):
            M[f"GMaeDiff{gk}{k}"] = f2(r.mae_diff_vs_white)
            M[f"GMaeDiffCi{gk}{k}"] = f"{P(r.mae_diff_ci)[0]:.2f} to {P(r.mae_diff_ci)[1]:.2f}"
        for a, ak in [("raw", "Raw"), ("cond_true", "Cond"), ("calib", "Cal")]:
            q = cond[(cond.cohort == c) & (cond.group == g) & (cond.analysis == a)]
            if len(q):
                q = q.iloc[0]
                M[f"Eff{ak}{gk}{k}"] = f2(q.coef)
                M[f"EffCi{ak}{gk}{k}"] = f"{q.lo:.2f} to {q.hi:.2f}"
    s = aud[aud.cohort == c]
    M[f"GMaeMin{k}"], M[f"GMaeMax{k}"] = f2(s.mae.min()), f2(s.mae.max())
    M[f"GMaeRange{k}"] = f2(s.mae.max() - s.mae.min())

# ---- sensitivity analyses
S = res["sensitivity"]
for name, k in [("primary_features", "Primary"), ("thresholds_only", "ThrOnly"), ("plus_pir_noise_cc", "PirNoise"),
                ("primary_features_same_cc", "SameCc"), ("excl_recent_ear_illness", "ExclIll"),
                ("excl_any_no_response", "ExclNr"), ("with_race_feature", "Race")]:
    M[f"Sa{k}Mae"] = f2(S[name]["mae"]); M[f"Sa{k}N"] = n0(S[name]["n"])
    M[f"Sa{k}Sens"] = f2(S[name]["sens20"]); M[f"Sa{k}Auc"] = f3(S[name]["auc20"])
for f, k in [("R_4k", "RFour"), ("L_4k", "LFour"), ("R_2k", "RTwo"), ("L_2k", "LTwo"), ("age", "Age"), ("female", "Sex")]:
    M[f"Shap{k}"] = f2(shap[f])
wg = res["survey_weighted_dev"]["groups"]
M["WSensMin"] = f2(min(v["sens20_w"] for v in wg.values()))
M["WSensMax"] = f2(max(v["sens20_w"] for v in wg.values()))
M["WMaeMin"] = f2(min(v["mae_w"] for v in wg.values()))
M["WMaeMax"] = f2(max(v["mae_w"] for v in wg.values()))

with open(MS / "numbers.tex", "w", encoding="utf8") as fh:
    fh.write("% AUTO-GENERATED by code/04_numbers_for_paper.py -- do not edit by hand\n")
    for k, v in sorted(M.items()):
        assert k.isalpha(), k
        v = re.sub(r"(^|[\s(])-(?=\d)", lambda m: m.group(1) + "\\textminus{}", v)  # true minus signs
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(len(M), "macros written")

# ================================================================ tables
GROUPS = list(GKEY)
# Table 1: descriptives
rows = []
for g in ["All"] + GROUPS:
    cells = []
    for c in COH:
        d = desc[(desc.cohort == c) & (desc.group == g)]
        if len(d):
            d = d.iloc[0]
            cells += [n0(d.n), f"{d.age_mean:.1f} ({d.age_sd:.1f})", f"{d.be4fa_mean:.1f} ({d.be4fa_sd:.1f})",
                      f"{d.who20_pct:.1f}", f"{d.who35_pct:.1f}"]
        else:
            cells += ["--"] * 5
    rows.append(("\\textbf{All}" if g == "All" else g) + " & " + " & ".join(cells) + r" \\")
t1 = r"""\begin{sidewaystable}
\centering
\caption{Participant characteristics by cohort and race/ethnicity (unweighted).}
\label{tab:desc}
\scriptsize
\setlength{\tabcolsep}{2.5pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l rrrrr rrrrr rrrrr}
\toprule
 & \multicolumn{5}{c}{Development (2011--2016, 20--69 y)} & \multicolumn{5}{c}{External A (1999--2004, 20--69 y)} & \multicolumn{5}{c}{External B (2005--2018, $\geq$70 y)} \\
\cmidrule(lr){2-6}\cmidrule(lr){7-11}\cmidrule(lr){12-16}
Group & $n$ & Age & BE4FA & $\geq$20 & $\geq$35 & $n$ & Age & BE4FA & $\geq$20 & $\geq$35 & $n$ & Age & BE4FA & $\geq$20 & $\geq$35 \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\par\smallskip\raggedright
Age and BE4FA (better-ear four-frequency average of 0.5, 1, 2 and 4~kHz, dB~HL) are mean (SD). $\geq$20 and $\geq$35 are the percentage with WHO 2021 hearing loss at those BE4FA cut-offs. NH, non-Hispanic. NHANES identifies non-Hispanic Asian participants only from 2011 onward; in earlier cycles they are included in Other/Multiracial. In External B, the NH Asian row is 2017--2018 only.
\end{sidewaystable}
"""
(TAB / "table1_descriptives.tex").write_text(t1, encoding="utf8")

# Table 2: MAE by frequency, cohort and model
rows = []
for c, lab in [("dev", "Development"), ("ext_a", "External A"), ("ext_b", "External B")]:
    rows.append(r"\multicolumn{8}{l}{\textit{" + lab + r"}} \\")
    for m in MKEY:
        f = fq[(fq.cohort == c) & (fq.model == m)].set_index("freq")
        s = sev[(sev.cohort == c) & (sev.model == m) & (sev.severity == "All")].iloc[0]
        rows.append(f"\\quad {m} & " + " & ".join(f"{f.loc[x, 'mae']:.2f}" for x in ["500", "1k", "3k", "6k", "8k"])
                    + f" & {s.mae:.2f} & {100 * f.within10.mean():.1f}" + r" \\")
t2 = r"""\begin{table}[p]
\centering
\caption{Mean absolute error (dB) at each predicted frequency (both ears pooled), overall per-participant MAE, and percentage of predicted thresholds within $\pm$10~dB.}
\label{tab:mae}
\small
\begin{tabular}{l rrrrr rr}
\toprule
Model & 0.5 kHz & 1 kHz & 3 kHz & 6 kHz & 8 kHz & Overall & $\pm$10 dB (\%) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\par\smallskip\raggedright
Development values are out-of-fold (5-fold cross-validation). External values come from models fitted once to the whole development cohort and applied unchanged. Copy: 0.5 and 1~kHz set to the measured 2~kHz threshold, 3~kHz to the mean of 2 and 4~kHz, and 6 and 8~kHz to the measured 4~kHz threshold, all in the same ear. Age-sex norm: median threshold by age decade and sex (participants aged 70 or over were assigned the 60--69 norm).
\end{table}
"""
(TAB / "table2_mae.tex").write_text(t2, encoding="utf8")

# Table 3: triage
lab = {"XGBoost 4FA": "XGBoost-completed BE4FA", "Ridge 4FA": "Ridge-completed BE4FA",
       "Copy 4FA": "Copy-completed BE4FA", "Measured HFA (2,4 kHz)": "Measured 2/4 kHz average"}
rows = []
for c, cl in [("dev", "Development"), ("ext_a", "External A"), ("ext_b", "External B")]:
    for cut in [20, 35]:
        npos = int(tri[(tri.cohort == c) & (tri.cut == cut)].n_pos.iloc[0])
        rows.append(r"\multicolumn{6}{l}{\textit{" + f"{cl}, BE4FA $\\geq${cut} dB HL (n = {npos:,} with hearing loss)" + r"}} \\")
        for s in SKEY:
            r = tri[(tri.cohort == c) & (tri.cut == cut) & (tri.score == s)].iloc[0]
            sc, pc = P(r.sens_ci), P(r.spec_ci)
            rows.append(f"\\quad {lab[s]} & {r.sens:.2f} ({sc[0]:.2f}--{sc[1]:.2f}) & {r.spec:.2f} ({pc[0]:.2f}--{pc[1]:.2f})"
                        f" & {r.ppv:.2f} & {r.npv:.2f} & {r.auc:.3f}" + r" \\")
t3 = r"""\begin{table}[p]
\centering
\caption{Classification of WHO 2021 hearing loss using each score at the WHO cut-off.}
\label{tab:triage}
\small
\begin{tabular}{l ccccc}
\toprule
Score & Sensitivity (95\% CI) & Specificity (95\% CI) & PPV & NPV & AUC \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\par\smallskip\raggedright
Each score is compared with the cut-off that defines the reference standard (20 or 35~dB~HL). Completed BE4FA scores use the measured 2 and 4~kHz thresholds and the predicted 0.5 and 1~kHz thresholds. The measured 2/4~kHz average is the better-ear mean of the two measured thresholds. CIs are from 500 participant-level bootstrap resamples. PPV and NPV depend on prevalence and apply to these samples only.
\end{table}
"""
(TAB / "table3_triage.tex").write_text(t3, encoding="utf8")

# Table 4: subgroup audit (XGBoost)
rows = []
for c, cl in [("dev", "Development"), ("ext_a", "External A"), ("ext_b", "External B")]:
    rows.append(r"\multicolumn{6}{l}{\textit{" + cl + r"}} \\")
    for g in GROUPS:
        r = aud[(aud.cohort == c) & (aud.group == g)]
        if not len(r):
            continue
        r = r.iloc[0]
        cells = [f"{r.mae:.2f}", f"{r.mse:+.2f}"]
        for a in ["raw", "cond_true", "calib"]:
            q = cond[(cond.cohort == c) & (cond.group == g) & (cond.analysis == a)]
            cells.append("ref." if g == "NH White" else (f"{q.coef.iloc[0]:+.2f} ({q.lo.iloc[0]:+.2f}, {q.hi.iloc[0]:+.2f})" if len(q) else "--"))
        rows.append(f"\\quad {g} ({int(r.n):,}) & " + " & ".join(cells) + r" \\")
t4 = r"""\begin{sidewaystable}
\centering
\caption{Subgroup audit of XGBoost threshold predictions: per-participant error, and group differences in signed error and calibration (reference: NH White).}
\label{tab:audit}
\scriptsize
\setlength{\tabcolsep}{3pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l rr ccc}
\toprule
 & & & \multicolumn{3}{c}{Group coefficient, dB (95\% CI)} \\
\cmidrule(lr){4-6}
Group ($n$) & MAE & Mean signed error & (a) Signed error, raw & (b) Signed error, same true level & (c) Calibration, same prediction \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\par\smallskip\raggedright
Signed error = predicted $-$ true; a positive value means hearing is predicted to be worse than measured. Models (a)--(c) are linear regressions on long-format data (one row per participant and target) with target fixed effects and cluster-robust standard errors by participant. (b) adds a cubic B-spline (4 df) in the true threshold, interacted with target. (c) regresses the true threshold on a spline of the predicted threshold (calibration within groups); a negative coefficient means that, at the same prediction, the group's measured thresholds are better than predicted. Groups with fewer than 30 participants in a cohort are omitted.
\end{sidewaystable}
"""
(TAB / "table4_audit.tex").write_text(t4, encoding="utf8")

# Supplementary table: sensitivity analyses
names = [("primary_features", "Primary model (2/4 kHz both ears, age, sex)"),
         ("thresholds_only", "Thresholds only (no age or sex)"),
         ("primary_features_same_cc", "Primary features, participants with PIR and noise data"),
         ("plus_pir_noise_cc", "Adding poverty-income ratio and occupational noise"),
         ("excl_recent_ear_illness", "Excluding cold/sinus/earache in past 24 h"),
         ("excl_any_no_response", "Excluding participants with any no-response code"),
         ("with_race_feature", "Adding race/ethnicity as a feature")]
rows = []
for k, lab_ in names:
    s = S[k]
    gm = s["group_mae"]
    rows.append(f"{lab_} & {s['n']:,} & {s['mae']:.2f} & {min(gm.values()):.2f}--{max(gm.values()):.2f} & "
                f"{s['sens20']:.2f} & {s['spec20']:.2f} & {s['auc20']:.3f}" + r" \\")
ts = r"""\begin{table}[p]
\centering
\caption{Sensitivity analyses in the development cohort (XGBoost, 5-fold cross-validation, primary hyperparameters).}
\label{tab:sens}
\scriptsize
\begin{tabular}{p{6cm} rrrrrr}
\toprule
Analysis & $n$ & MAE & Range of group MAE & Sens. & Spec. & AUC \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\par\smallskip\raggedright
Sensitivity, specificity and AUC refer to WHO hearing loss (BE4FA $\geq$20~dB~HL) using the completed BE4FA at the WHO cut-off. Occupational noise: 2011--2012 item AUQ330 and 2015--2016 item AUQ331 (same wording in both cycles: job exposure to loud noise ``for 4 or more hours a day, several days a week''; ``never worked'' treated as missing).
\end{table}
"""
(TAB / "tableS1_sensitivity.tex").write_text(ts, encoding="utf8")
print("tables written")
