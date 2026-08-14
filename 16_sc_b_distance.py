"""
16_sc_b_distance.py — SC-b: School Choice × Escort with Continuous Distance

Extends 03_school_choice.py by adding continuous home-school distance
(car_dist_km) alongside the binary school_choice flag.

Original SC: logit(escort ~ school_choice) — binary in/out catchment only
SC-b adds:
  - Escort rate by distance quintile within each catchment group
  - Logit(escort ~ school_choice + car_dist_km_std + car_own)
  - Full model: escort ~ school_choice + car_dist_km_std + car_own +
                          school_choice × car_dist_km_std interaction

Motivation: Out-of-catchment trips may be further (by construction) but even
within each group, does actual distance predict escort beyond binary flag?
Under CIM framework, distance is a primary spatial barrier — farther trips
are more likely escorted regardless of catchment membership.

Data: data/school_choice_trips.csv
  (output of 03_school_choice.py — already has school_choice + car_dist_km + LOS)
  Variables used: escort, school_choice, car_dist_km, owned_num_cars, school_level

Outputs:
  figures/SC_b_escort_by_dist_catchment.png
  figures/SC_b_logit_comparison.png
  data/sc_b_logit_results.csv

Original 03_school_choice.py preserved.
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
DATA = BASE / "data"
FIG  = BASE / "figures"
FIG.mkdir(exist_ok=True)

SC_CSV = DATA / "school_choice_trips.csv"


def section(title):
    print(f"\n{'═'*65}\n  {title}\n{'═'*65}")


# ── Load ──────────────────────────────────────────────────────────────────────
section("Load school_choice_trips.csv")
df = pd.read_csv(SC_CSV, low_memory=False)
print(f"  Rows: {len(df):,}")
print(f"  Columns: {list(df.columns)}")

# Core variables
df["escort"]        = pd.to_numeric(df["escort"],        errors="coerce")
df["school_choice"] = pd.to_numeric(df["school_choice"], errors="coerce")
df["car_dist_km"]   = pd.to_numeric(df["car_dist_km"],   errors="coerce")
df["owned_num_cars"]= pd.to_numeric(df["owned_num_cars"],errors="coerce")
df["car_own"]       = (df["owned_num_cars"] > 0).astype(int)

# Drop rows missing critical variables
df_model = df.dropna(subset=["escort","school_choice","car_dist_km","car_own"]).copy()
df_model = df_model[df_model["car_dist_km"] > 0].copy()

print(f"\n  After filtering (non-null, dist>0): {len(df_model):,}")
print(f"  Escort rate overall:      {df_model['escort'].mean():.1%}")
print(f"  In-catchment (choice=0):  {(df_model['school_choice']==0).mean():.1%}")
print(f"  Out-of-catchment (1):     {(df_model['school_choice']==1).mean():.1%}")


# ── Standardize distance ──────────────────────────────────────────────────────
mu = df_model["car_dist_km"].mean()
sd = df_model["car_dist_km"].std()
df_model["car_dist_std"] = (df_model["car_dist_km"] - mu) / sd
print(f"\n  car_dist_km: mean={mu:.2f} km, sd={sd:.2f} km")


# ── Descriptive: distance distribution by catchment group ────────────────────
section("Descriptive: Distance by Catchment Group")

for label, val in [("In-catchment (0)", 0), ("Out-of-catchment (1)", 1)]:
    sub = df_model[df_model["school_choice"] == val]["car_dist_km"]
    print(f"  {label}: n={len(sub):,}, median={sub.median():.2f} km, "
          f"mean={sub.mean():.2f} km, p75={sub.quantile(0.75):.2f} km")


# ── Escort rate by distance quintile × catchment ─────────────────────────────
section("Escort rate by distance quintile × catchment")

df_model["dist_q"] = pd.qcut(df_model["car_dist_km"], q=5, labels=False, duplicates="drop")

qt_tbl = (df_model.groupby(["dist_q","school_choice"])
          .agg(n=("escort","count"),
               escort_rate=("escort","mean"),
               mean_dist=("car_dist_km","mean"))
          .reset_index())
print(qt_tbl.round(3).to_string(index=False))

# Overall correlation
r_val, r_p = pearsonr(df_model["car_dist_km"], df_model["escort"])
print(f"\n  Pearson r(car_dist, escort) = {r_val:.4f} (p={r_p:.4f})")


# ── Logit models ──────────────────────────────────────────────────────────────
section("Logit Models: escort ~ school_choice + distance + car_own")

models_def = {
    "SC_original":   "escort ~ school_choice + car_own",
    "SC_b_dist":     "escort ~ school_choice + car_dist_std + car_own",
    "SC_b_interact": "escort ~ school_choice * car_dist_std + car_own",
    "SC_b_full":     "escort ~ school_choice + car_dist_std + car_own + "
                     "school_choice:car_dist_std",
}

results = {}
for label, formula in models_def.items():
    try:
        m = smf.logit(formula, data=df_model).fit(disp=False, maxiter=300)
        results[label] = m
        print(f"\n  ── {label} ──")
        print(f"  n={int(m.nobs)}, McF-R²={m.prsquared:.4f}, AIC={m.aic:.1f}")
        tbl = m.summary2().tables[1][["Coef.","Std.Err.","z","P>|z|"]].round(4)
        print(tbl.to_string())
    except Exception as e:
        print(f"  ⚠️  {label}: {e}")

# Model comparison
if "SC_original" in results and "SC_b_dist" in results:
    m0 = results["SC_original"]
    m1 = results["SC_b_dist"]
    print(f"\n  SC_original vs SC_b_dist:")
    print(f"  ΔMcF-R²: {m1.prsquared - m0.prsquared:+.4f}")
    print(f"  ΔAIC:    {m1.aic - m0.aic:+.1f}  ({'better' if m1.aic < m0.aic else 'worse'})")


# ── Save ──────────────────────────────────────────────────────────────────────
rows = []
for label, m in results.items():
    tbl = m.summary2().tables[1].reset_index()
    tbl.columns = ["variable","coef","se","z","p","ci_lo","ci_hi"]
    tbl["model"]  = label
    tbl["nobs"]   = int(m.nobs)
    tbl["mcf_r2"] = round(m.prsquared, 4)
    tbl["aic"]    = round(m.aic, 2)
    rows.append(tbl)
pd.concat(rows, ignore_index=True).to_csv(DATA / "sc_b_logit_results.csv", index=False)
print(f"\n  Saved: sc_b_logit_results.csv")


# ── Figures ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: escort rate by distance quintile, split by catchment
ax = axes[0]
colors_catchment = {0: "#2980B9", 1: "#E74C3C"}
labels_catchment = {0: "In-catchment", 1: "Out-of-catchment"}
for val in [0, 1]:
    sub = qt_tbl[qt_tbl["school_choice"] == val]
    if len(sub) == 0:
        continue
    ax.plot(sub["dist_q"], sub["escort_rate"] * 100,
            "o-", color=colors_catchment[val], linewidth=2, markersize=7,
            label=labels_catchment[val])
    for _, r in sub.iterrows():
        ax.text(r["dist_q"] + 0.05, r["escort_rate"]*100 + 0.5,
                f"{r['escort_rate']*100:.0f}%", fontsize=8,
                color=colors_catchment[val])

ax.set_xlabel("Car distance quintile (1=short … 5=long)", fontsize=10)
ax.set_ylabel("Escort rate (%)", fontsize=10)
ax.set_title(f"SC-b: Escort Rate by Distance × Catchment\n"
             f"r(dist,escort)={r_val:.3f}", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)

# Right: coefficient comparison SC_original vs SC_b_dist
ax = axes[1]
if "SC_original" in results and "SC_b_dist" in results:
    orig_coef = results["SC_original"].summary2().tables[1]["Coef."].drop("Intercept")
    dist_coef = results["SC_b_dist"].summary2().tables[1]["Coef."].drop("Intercept")
    orig_p    = results["SC_original"].summary2().tables[1]["P>|z|"].drop("Intercept")
    dist_p    = results["SC_b_dist"].summary2().tables[1]["P>|z|"].drop("Intercept")
    all_vars  = list(orig_coef.index.union(dist_coef.index))
    label_map = {
        "school_choice": "Out-of-catchment",
        "car_own":       "Car ownership",
        "car_dist_std":  "Car distance (std) ★",
    }
    y = np.arange(len(all_vars))
    w = 0.35
    orig_vals = [orig_coef.get(v, np.nan) for v in all_vars]
    dist_vals = [dist_coef.get(v, np.nan) for v in all_vars]
    orig_ps   = [orig_p.get(v, 1.0) for v in all_vars]
    dist_ps   = [dist_p.get(v, 1.0) for v in all_vars]

    ax.barh(y + w/2, orig_vals, w, color="#B2BEB5", alpha=0.8, label="SC_original")
    ax.barh(y - w/2, dist_vals, w, color="#2980B9", alpha=0.8, label="SC_b_dist (+ distance)")
    ax.set_yticks(y)
    ax.set_yticklabels([label_map.get(v, v) for v in all_vars], fontsize=9)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Logit coefficient", fontsize=10)
    ax.set_title(f"SC-b Coefficients\nOriginal vs + Distance (★ = new)\n"
                 f"SC_b: McF-R²={results['SC_b_dist'].prsquared:.4f}", fontsize=11, fontweight="bold")
    for i, (v, p) in enumerate(zip(dist_vals, dist_ps)):
        if not np.isnan(v):
            sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ""
            ax.text(v + 0.05, i - w/2, sig, va="center", fontsize=10, color="#c0392b")
    ax.legend(fontsize=9)

plt.tight_layout()
fig.savefig(FIG / "SC_b_logit_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: SC_b_logit_comparison.png")

print(f"\n{'═'*65}")
print(f"  SC-b complete. Original 03_school_choice.py unchanged.")
print(f"{'═'*65}")
