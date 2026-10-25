"""
Step 1: build analysis cohorts from raw NHANES files.

Cohorts
  dev    : 2011-2012 + 2015-2016, ages 20-69 (development; 5-fold CV)
  ext_a  : 1999-2000 + 2001-2002 + 2003-2004, ages 20-69 (earlier-era external test)
  ext_b  : 2005-2006 + 2009-2010 + 2017-2018, ages 70+ (older, hearing-loss-enriched external test)

Threshold coding (NHANES AUX codebooks):
  666 = no response at maximum output -> censored; set to (max valid level at that frequency recorded with
        the same audiometer system + 5 dB). AD226 (1999-2016): manual limit 120 dB HL (110 at 8 kHz).
        ART (2017-2018): lower output limits (about 100 dB HL; 90 at 8 kHz in automated mode).
  888 = could not obtain -> missing
  No upper cap on valid values.
  The first 1 kHz measurement (AUXU1K1x) is used; the retest (AUXU1K2x) is kept for test-retest.

Run from the 5_resubmission folder:  python -I code/01_build_cohorts.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data_raw"
OUT = ROOT / "data_derived"
OUT.mkdir(exist_ok=True)

CYCLES = {
    # cycle: (file suffix, cohort, AUX file name, DEMO file name, AUQ file name)
    "1999-2000": ("A", "ext_a", "AUX1", "DEMO", "AUQ"),
    "2001-2002": ("B", "ext_a", "AUX_B", "DEMO_B", "AUQ_B"),
    "2003-2004": ("C", "ext_a", "AUX_C", "DEMO_C", "AUQ_C"),
    "2005-2006": ("D", "ext_b", "AUX_D", "DEMO_D", "AUQ_D"),
    "2009-2010": ("F", "ext_b", "AUX_F", "DEMO_F", "AUQ_F"),
    "2011-2012": ("G", "dev", "AUX_G", "DEMO_G", "AUQ_G"),
    "2015-2016": ("I", "dev", "AUX_I", "DEMO_I", "AUQ_I"),
    "2017-2018": ("J", "ext_b", "AUX_J", "DEMO_J", "AUQ_J"),
}
AGE_RANGE = {"dev": (20, 69), "ext_a": (20, 69), "ext_b": (70, 200)}

FREQS = ["500", "1k", "2k", "3k", "4k", "6k", "8k"]
SRC = {"500": "AUXU500", "1k": "AUXU1K1", "2k": "AUXU2K", "3k": "AUXU3K",
       "4k": "AUXU4K", "6k": "AUXU6K", "8k": "AUXU8K"}
EARS = ["R", "L"]
THR_COLS = [f"{e}_{f}" for e in EARS for f in FREQS]

# Audiometer system by cycle (NHANES AUX documentation): Interacoustics AD226 to 2016, ART system in 2017-2018
SYSTEM = {c: ("ART" if c == "2017-2018" else "AD226") for c in CYCLES}

# Race/ethnicity. RIDRETH3 (with NH Asian) exists from 2011; earlier cycles only RIDRETH1.
R3 = {1: "Mexican American", 2: "Other Hispanic", 3: "NH White", 4: "NH Black",
      6: "NH Asian", 7: "Other/Multiracial"}
R1 = {1: "Mexican American", 2: "Other Hispanic", 3: "NH White", 4: "NH Black",
      5: "Other/Multiracial"}


def read(name):
    return pd.read_sas(RAW / f"{name}.xpt", format="xport")


def max_valid(names):
    """Highest valid (non-666/888) threshold at each frequency, both ears, all ages, over the given AUX files."""
    m = {f: -np.inf for f in FREQS}
    for n in names:
        aux = read(n)
        for f in FREQS:
            for e in EARS:
                v = aux[f"{SRC[f]}{e}"]
                m[f] = max(m[f], v[~v.isin([666, 888])].max())
    return m


# Censoring level for 666 = highest valid value at that frequency with the same audiometer system + 5 dB
CENSOR = {s: {f: v + 5 for f, v in max_valid([CYCLES[c][2] for c in CYCLES if SYSTEM[c] == s]).items()}
          for s in ["AD226", "ART"]}
print("666 censoring levels (dB HL):", CENSOR)


def code_thresholds(aux, censor):
    out = pd.DataFrame({"SEQN": aux["SEQN"]})
    n666 = n888 = 0
    for e in EARS:
        for f in FREQS:
            v = aux[f"{SRC[f]}{e}"].copy()
            n666 += int((v == 666).sum())
            n888 += int((v == 888).sum())
            v = v.where(v != 888)
            v = v.where(v != 666, censor[f])
            out[f"{e}_{f}"] = v
            out[f"{e}_{f}_nr"] = (aux[f"{SRC[f]}{e}"] == 666).astype(int)
        # 1 kHz retest
        r = aux[f"AUXU1K2{e}"]
        out[f"{e}_1k_retest"] = r.where(~r.isin([666, 888]))
    return out, n666, n888


frames, flow = [], []
for cycle, (suf, cohort, auxf, demf, auqf) in CYCLES.items():
    aux = read(auxf)
    dem = read(demf)
    keep = ["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH1", "INDFMPIR",
            "WTMEC2YR", "SDMVSTRA", "SDMVPSU"]
    if "RIDRETH3" in dem.columns:
        keep.append("RIDRETH3")
    dem = dem[[c for c in keep if c in dem.columns]]

    thr, n666, n888 = code_thresholds(aux, CENSOR[SYSTEM[cycle]])
    extra = [c for c in ["AUQ020"] if c in aux.columns]
    thr = thr.merge(aux[["SEQN"] + extra], on="SEQN", how="left")
    d = dem.merge(thr, on="SEQN", how="inner")

    lo, hi = AGE_RANGE[cohort]
    step = {"cycle": cycle, "cohort": cohort, "aux_records": len(aux)}
    d = d[(d.RIDAGEYR >= lo) & (d.RIDAGEYR <= hi)]
    step["in_age_range"] = len(d)
    any_thr = d[THR_COLS].notna().any(axis=1)
    d = d[any_thr]
    step["any_threshold"] = len(d)
    d = d.dropna(subset=THR_COLS)
    step["complete_audiogram"] = len(d)
    d = d.dropna(subset=["RIAGENDR"])
    step["final"] = len(d)
    step["n666_codes_all_ages"] = n666
    step["n888_codes_all_ages"] = n888
    flow.append(step)

    # Occupational noise (development cycles only; item AUQ330 in 2011-12 was renamed AUQ331 in 2015-16, same wording)
    if cohort == "dev":
        auq = read(auqf)
        col = "AUQ330" if "AUQ330" in auq.columns else "AUQ331"
        q = auq[["SEQN", col]].rename(columns={col: "occ_noise_raw"})
        d = d.merge(q, on="SEQN", how="left")
        d["occ_noise"] = d["occ_noise_raw"].map({1: 1.0, 2: 0.0})  # 7/9/missing -> NaN

    d["cycle"] = cycle
    d["cohort"] = cohort
    frames.append(d)

df = pd.concat(frames, ignore_index=True)
df["age"] = df["RIDAGEYR"]
df["female"] = (df["RIAGENDR"] == 2).astype(int)
df["pir"] = df["INDFMPIR"]
df["race"] = np.where(df["RIDRETH3"].notna(), df["RIDRETH3"].map(R3), df["RIDRETH1"].map(R1))
# Coarse race variable comparable across all cycles (NH Asian folded into Other)
df["race5"] = df["RIDRETH1"].map(R1)
df["recent_ear_illness"] = (df["AUQ020"] == 1).astype(float).where(df["AUQ020"].isin([1, 2]))

# WHO 2021 grading: better-ear four-frequency average (0.5, 1, 2, 4 kHz)
for e in EARS:
    df[f"{e}_4fa"] = df[[f"{e}_500", f"{e}_1k", f"{e}_2k", f"{e}_4k"]].mean(axis=1)
df["be4fa"] = df[["R_4fa", "L_4fa"]].min(axis=1)
bins = [-np.inf, 20, 35, 50, 65, 80, 95, np.inf]
labels = ["Normal (<20)", "Mild (20-<35)", "Moderate (35-<50)", "Moderately severe (50-<65)",
          "Severe (65-<80)", "Profound (80-<95)", "Complete (>=95)"]
df["who_grade"] = pd.cut(df["be4fa"], bins=bins, labels=labels, right=False)
df["who3"] = pd.cut(df["be4fa"], bins=[-np.inf, 20, 35, np.inf],
                    labels=["Normal", "Mild", "Moderate+"], right=False)
df["any_nr"] = df[[c for c in df.columns if c.endswith("_nr")]].max(axis=1)

df.to_csv(OUT / "cohorts.csv", index=False)
flow_df = pd.DataFrame(flow)
flow_df.to_csv(OUT / "cohort_flow.csv", index=False)

print(flow_df.to_string(index=False))
print()
for c, g in df.groupby("cohort"):
    print(c, len(g), "| WHO>=20:", int((g.be4fa >= 20).sum()), "| WHO>=35:", int((g.be4fa >= 35).sum()),
          "| any 666:", int(g.any_nr.sum()))
    print(g["race"].value_counts().to_dict())
dev = df[df.cohort == "dev"]
print("dev occupational noise prevalence by cycle:", dev.groupby("cycle")["occ_noise"].mean().round(3).to_dict(),
      "missing:", int(dev["occ_noise"].isna().sum()))
print("dev PIR missing:", int(dev["pir"].isna().sum()))
