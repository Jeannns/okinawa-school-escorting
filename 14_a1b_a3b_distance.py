"""
14_a1b_a3b_distance.py — SSD Retrofit: Direct Distance Measures

Replaces travel_time proxy (mode1_travel_time_min) with actual road network
distance (car_distance_m from SSD LOS data) for A1 and A3.

Why: mode1_travel_time_min conflates mode with distance — HS students are
car-driven so their time is SHORT despite travelling FAR. car_distance_m is a
purer spatial measure of home-school coverage.

A1b: Distribution of car_distance_m by school_level (cf. A1: travel_time)
     Kruskal-Wallis (car_distance is right-skewed → non-parametric)
     + boxplot + mean comparison table

A3b: OLS — car_distance_m ~ car_own + school_level (cf. A3: travel_time ~ car_own)
     Extended model also adds car_own × school_level interaction

Data: data/school_trips_los.csv  (requires car_distance_m column from SSD)
Outputs:
  figures/A1b_distance_by_level.png
  figures/A3b_ols_distance_car.png
  data/a1b_distance_summary.csv
  data/a3b_ols_results.csv

Original scripts preserved:
  analysis_rq.py — A1 (travel_time ANOVA), A3 (OLS travel_time ~ car_own)
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy.stats import kruskal, mannwhitneyu

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE  = Path(__file__).parent
DATA  = BASE / "data"
FIG   = BASE / "figures"
FIG.mkdir(exist_ok=True)

TRIPS_CSV = DATA / "school_trips_los.csv"

LEVEL_ORDER  = ["Elementary", "Middle", "High School"]
LEVEL_COLORS = ["#AED6F1", "#85C1E9", "#2980B9"]


def section(title):
    print(f"\n{'═'*65}\n  {title}\n{'═'*65}")


# ── Load data ─────────────────────────────────────────────────────────────────
section("Loading school_trips_los.csv")
df = pd.read_csv(TRIPS_CSV, low_memory=False)
print(f"  Total rows: {len(df):,}")

df["car_distance_m"] = pd.to_numeric(df["car_distance_m"], errors="coerce")
df["car_dist_km"]    = df["car_distance_m"] / 1000
df["mode1_travel_time_min"] = pd.to_numeric(df["mode1_travel_time_min"], errors="coerce")
df["owned_num_cars"] = pd.to_numeric(df["owned_num_cars"], errors="coerce")
df["has_car"]        = (df["owned_num_cars"] > 0).astype(int)

# Filter to rows with valid car_distance_m and school_level
df_dist = df.dropna(subset=["car_distance_m", "school_level"]).copy()
df_dist = df_dist[df_dist["car_distance_m"] > 0]          # exclude 0 (likely same zone)
df_dist = df_dist[df_dist["school_level"].isin(LEVEL_ORDER)]
print(f"  Rows with valid car_distance_m (>0): {len(df_dist):,}")
print(f"  Level breakdown:")
for lv in LEVEL_ORDER:
    n = (df_dist["school_level"] == lv).sum()
    print(f"    {lv:12s}: {n:,}")


# ═══════════════════════════════════════════════════════════════════════════════
# A1b — Distribution of car_distance_m by school_level
# ═══════════════════════════════════════════════════════════════════════════════
section("A1b — Car Distance by School Level")

groups = [df_dist[df_dist["school_level"] == lv]["car_dist_km"].values
          for lv in LEVEL_ORDER]

# Kruskal-Wallis (non-parametric — distance is right-skewed)
kw_stat, kw_p = kruskal(*groups)
print(f"\n  Kruskal-Wallis: H={kw_stat:.2f}, p={kw_p:.4f}")

# Pairwise Mann-Whitney
pairs = [("Elementary","Middle"), ("Elementary","High School"), ("Middle","High School")]
mw_results = []
for a, b in pairs:
    ga = df_dist[df_dist["school_level"] == a]["car_dist_km"].values
    gb = df_dist[df_dist["school_level"] == b]["car_dist_km"].values
    stat, p = mannwhitneyu(ga, gb, alternative="two-sided")
    mw_results.append({"pair": f"{a} vs {b}", "U": stat, "p": p,
                       "sig": "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "n.s."})
    print(f"  {a:12s} vs {b:12s}: U={stat:.0f}, p={p:.4f} {mw_results[-1]['sig']}")

# Summary table
summary = df_dist.groupby("school_level")["car_dist_km"].agg(
    n="count", mean="mean", median="median", std="std",
    q25=lambda x: x.quantile(0.25),
    q75=lambda x: x.quantile(0.75)
).reset_index()
summary["school_level"] = pd.Categorical(summary["school_level"],
                                          categories=LEVEL_ORDER, ordered=True)
summary = summary.sort_values("school_level")
print(f"\n  Summary (car_dist_km):")
print(summary.to_string(index=False, float_format="{:.2f}".format))
summary.to_csv(DATA / "a1b_distance_summary.csv", index=False)

# ── Comparison: travel_time vs car_distance proxy ──────────────────────────────
# Original A1 used mode1_travel_time_min — HS median 15 min ≈ Elementary 15 min
# A1b with car_distance_m should show clearer level-based pattern
df_time = df.dropna(subset=["mode1_travel_time_min","school_level"]).copy()
df_time = df_time[df_time["mode1_travel_time_min"] < 999]
df_time = df_time[df_time["school_level"].isin(LEVEL_ORDER)]

time_summ = df_time.groupby("school_level")["mode1_travel_time_min"].agg(
    n="count", median="median", mean="mean"
).reset_index()
print(f"\n  Comparison — original A1 (travel_time_min) medians:")
for _, r in time_summ.iterrows():
    print(f"    {r['school_level']:12s}: median={r['median']:.0f} min, mean={r['mean']:.1f} min")
print(f"  → travel_time cannot differentiate HS from Elementary (both ~15 min)")
print(f"\n  A1b (car_dist_km) medians:")
for _, r in summary.iterrows():
    print(f"    {r['school_level']:12s}: median={r['median']:.2f} km, mean={r['mean']:.2f} km")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: A1b boxplot (car_distance_m)
ax = axes[0]
data_for_box = [df_dist[df_dist["school_level"] == lv]["car_dist_km"].values
                for lv in LEVEL_ORDER]
bp = ax.boxplot(data_for_box, patch_artist=True, notch=False,
                medianprops={"color":"black","linewidth":2},
                flierprops={"marker":"o","markersize":2,"alpha":0.3})
for patch, color in zip(bp["boxes"], LEVEL_COLORS):
    patch.set_facecolor(color)
    patch.set_alpha(0.85)
ax.set_xticklabels(LEVEL_ORDER, fontsize=10)
ax.set_ylabel("Car road distance (km)", fontsize=10)
ax.set_title(f"A1b — School Trip Distance by Level\n"
             f"Kruskal-Wallis H={kw_stat:.1f}, p={kw_p:.4f}", fontsize=11, fontweight="bold")
for i, r in enumerate(summary.itertuples()):
    ax.text(i+1, r.q75 + 0.3, f"Med={r.median:.1f}km\nn={r.n:,}", ha="center",
            fontsize=8, color="#333")

# Right: comparison — travel_time vs distance (median side-by-side)
ax = axes[1]
x = np.arange(len(LEVEL_ORDER))
w = 0.35
time_med = [df_time[df_time["school_level"]==lv]["mode1_travel_time_min"].median()
            for lv in LEVEL_ORDER]
dist_med  = [df_dist[df_dist["school_level"]==lv]["car_dist_km"].median()
             for lv in LEVEL_ORDER]

ax2 = ax.twinx()
b1 = ax.bar(x - w/2, time_med, w, color="#B2BEB5", alpha=0.8, label="Travel time (min) [left]")
b2 = ax2.bar(x + w/2, dist_med, w, color="#2980B9", alpha=0.8, label="Car distance (km) [right]")
ax.set_xticks(x)
ax.set_xticklabels(LEVEL_ORDER, fontsize=10)
ax.set_ylabel("Median travel time (min)", fontsize=10, color="#666")
ax2.set_ylabel("Median car distance (km)", fontsize=10, color="#2980B9")
ax.set_title("Proxy Comparison:\nTravel Time vs Car Distance (medians)", fontsize=11, fontweight="bold")
lines = [b1, b2]
ax.legend(handles=lines, fontsize=8, loc="upper left")
for i, (tm, dm) in enumerate(zip(time_med, dist_med)):
    ax.text(i - w/2, tm + 0.3, f"{tm:.0f}", ha="center", fontsize=9, color="#444")
    ax2.text(i + w/2, dm + 0.05, f"{dm:.1f}", ha="center", fontsize=9, color="#1a5276")

plt.tight_layout()
fig.savefig(FIG / "A1b_distance_by_level.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  Saved: A1b_distance_by_level.png")


# ═══════════════════════════════════════════════════════════════════════════════
# A3b — OLS: car_distance_m ~ car_own + school_level
# ═══════════════════════════════════════════════════════════════════════════════
section("A3b — OLS: Car Distance ~ Car Ownership + School Level")

reg_df = df_dist[["car_dist_km", "has_car", "school_level"]].dropna().copy()
reg_df = reg_df[reg_df["school_level"].isin(LEVEL_ORDER)]

# Model 1: car_own + level (main effects)
formula_m1 = 'car_dist_km ~ has_car + C(school_level, Treatment("Elementary"))'
m1 = smf.ols(formula_m1, data=reg_df).fit()
print(f"\n  Model 1 — Main Effects (n={int(m1.nobs)})")
print(f"  R² = {m1.rsquared:.4f} | Adj-R² = {m1.rsquared_adj:.4f}")
print(m1.summary2().tables[1][["Coef.","Std.Err.","t","P>|t|"]].round(4).to_string())

# Model 2: add car_own × school_level interaction
formula_m2 = ('car_dist_km ~ has_car * C(school_level, Treatment("Elementary"))')
m2 = smf.ols(formula_m2, data=reg_df).fit()
print(f"\n  Model 2 — With Car×Level Interaction (n={int(m2.nobs)})")
print(f"  R² = {m2.rsquared:.4f} | Adj-R² = {m2.rsquared_adj:.4f}")
print(m2.summary2().tables[1][["Coef.","Std.Err.","t","P>|t|"]].round(4).to_string())

# Compare R²
print(f"\n  R² improvement M1→M2: {m2.rsquared - m1.rsquared:+.4f}")
print(f"  → Main effect model sufficient if interaction not significant")

# Save coefficients
coef_df = m1.summary2().tables[1][["Coef.","Std.Err.","t","P>|t|","[0.025","0.975]"]].reset_index()
coef_df.columns = ["variable","coef","se","t","p","ci_lo","ci_hi"]
coef_df["model"] = "M1_main"
coef_m2 = m2.summary2().tables[1][["Coef.","Std.Err.","t","P>|t|","[0.025","0.975]"]].reset_index()
coef_m2.columns = ["variable","coef","se","t","p","ci_lo","ci_hi"]
coef_m2["model"] = "M2_interaction"
out_coef = pd.concat([coef_df, coef_m2], ignore_index=True)
out_coef.to_csv(DATA / "a3b_ols_results.csv", index=False)
print(f"  Saved: a3b_ols_results.csv")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: OLS coefficients (M1)
ax = axes[0]
coef_plot = coef_df[coef_df["variable"] != "Intercept"].copy()
coef_plot["label"] = coef_plot["variable"].str.replace(
    r'C\(school_level, Treatment\("Elementary"\)\)\[T\.', "", regex=True
).str.replace(r'\]', "", regex=True)
colors_coef = ["#2980B9" if p < 0.05 else "#AAAAAA" for p in coef_plot["p"]]
ax.barh(range(len(coef_plot)), coef_plot["coef"], color=colors_coef, alpha=0.85,
        xerr=(coef_plot["ci_hi"] - coef_plot["ci_lo"]) / 2, error_kw={"linewidth":1.5})
ax.set_yticks(range(len(coef_plot)))
ax.set_yticklabels(coef_plot["label"], fontsize=9)
ax.axvline(0, color="black", lw=1)
ax.set_xlabel("Coefficient (km)", fontsize=10)
ax.set_title(f"A3b — OLS Coefficients\n(DV: car_distance_m, R²={m1.rsquared:.3f})",
             fontsize=11, fontweight="bold")
for i, row in enumerate(coef_plot.itertuples()):
    sig = "***" if row.p<0.001 else "**" if row.p<0.01 else "*" if row.p<0.05 else ""
    ax.text(row.coef + 0.05, i, sig, va="center", fontsize=10, color="#c0392b")

# Right: predicted means by group (car_own × school_level)
ax = axes[1]
pred_df = reg_df.groupby(["school_level","has_car"])["car_dist_km"].agg(
    mean="mean", se=lambda x: x.sem(), n="count"
).reset_index()
for i, lv in enumerate(LEVEL_ORDER):
    sub = pred_df[pred_df["school_level"] == lv]
    for _, row in sub.iterrows():
        label = "Car HH" if row["has_car"] else "No-car HH"
        x_pos = i + (0.2 if row["has_car"] else -0.2)
        color = "#2980B9" if row["has_car"] else "#E74C3C"
        ax.bar(x_pos, row["mean"], 0.35, color=color, alpha=0.8,
               label=label if i == 0 else "")
        ax.errorbar(x_pos, row["mean"], yerr=row["se"]*1.96,
                    fmt="none", color="#333", linewidth=1.5, capsize=4)
        ax.text(x_pos, row["mean"] + 0.1, f"{row['mean']:.1f}", ha="center", fontsize=8)
ax.set_xticks(range(len(LEVEL_ORDER)))
ax.set_xticklabels(LEVEL_ORDER, fontsize=10)
ax.set_ylabel("Mean car distance (km)", fontsize=10)
ax.set_title("A3b — Mean Distance by Car Ownership × School Level\n(bars = 95% CI)", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)

plt.tight_layout()
fig.savefig(FIG / "A3b_ols_distance_car.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: A3b_ols_distance_car.png")

# ── Comparison with original A3 ───────────────────────────────────────────────
print(f"\n{'─'*55}")
print(f"  Comparison with original A3 (DV = travel_time_min):")
print(f"  Original A3: travel_time ~ car_own + level  (R² not printed here)")
print(f"  A3b:         car_distance ~ car_own + level  R²={m1.rsquared:.4f}")
print(f"  → car_distance_m captures spatial coverage directly;")
print(f"    travel_time is endogenous to mode choice (car=fast even if far)")

print(f"\n{'═'*65}")
print(f"  A1b + A3b complete. Files saved to data/ and figures/")
print(f"  Original analysis_rq.py (A1, A3) unchanged.")
print(f"{'═'*65}")
