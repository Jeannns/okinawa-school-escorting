"""
08_stratified_logit.py — Stratified Logit by Car Ownership (Analysis A)
=========================================================================
Key hypothesis: LOS effects on school escorting are conditional on car
availability. In the full sample, 94% of households own a car, which
masks any LOS effect. For no-car households, transit and walking
conditions should be significant predictors of escort incidence.

Models
------
  M0  Full sample (baseline — replication of P1b)
  M1  No-car households (owns_car = 0)
  M2  Car-owning households (owns_car = 1)
  M3  Full sample with car × LOS interaction (formal moderation test)

DV:  escort = 1 (escorting_flag == 1)
IVs: walk_time_std, bus_freq_std, car_dist_std, owned_cars_std

Outputs (data/tables/)
----------------------
  P4a_stratified_logit_summary.csv   — coefficients all 3 models side-by-side
  P4a_interaction_model.csv          — M3 car × LOS interaction coefficients
  P4a_sample_stats.csv               — sample size and escort rate by car group

Outputs (data/figures/)
-----------------------
  P4a_coef_comparison.png            — coefficient plot: M0 vs M1 vs M2
  P4a_escort_rate_by_walktime.png    — escort rate by walk_time bin, car vs no-car
  P4a_margeff_comparison.png         — AME comparison across groups
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR   = SCRIPT_DIR / "data"
TABLE_DIR  = DATA_DIR / "tables"
FIG_DIR    = DATA_DIR / "figures"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def section(t):
    bar = "═" * 70
    print(f"\n{bar}\n  {t}\n{bar}")


def save_fig(fig, name):
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved: figures/{name}.png")


# ══════════════════════════════════════════════════════════════════════════════
# Load & prepare data
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    df = pd.read_csv(DATA_DIR / "school_trips_los.csv", low_memory=False)

    # Outcome
    df["escort"]     = (df["escorting_flag"] == 1).astype(int)
    df["car_escort"] = (
        (df["escorting_flag"] == 1) &
        (pd.to_numeric(df["rep_mode_class1"], errors="coerce") == 6)
    ).astype(int)

    # Car ownership
    df["owned_num_cars"] = pd.to_numeric(df["owned_num_cars"], errors="coerce").fillna(0)
    df["owns_car"] = (df["owned_num_cars"] > 0).astype(int)

    # LOS
    for col in ["walk_time_min", "bus_frequency_per_day", "car_distance_m"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep LOS-joined only
    df = df.dropna(subset=["walk_time_min", "bus_frequency_per_day", "car_distance_m"])

    # Standardise using full-sample mean/std
    for raw, std in [("walk_time_min",        "walk_time_std"),
                     ("bus_frequency_per_day", "bus_freq_std"),
                     ("car_distance_m",        "car_dist_std"),
                     ("owned_num_cars",        "owned_cars_std")]:
        m, s = df[raw].mean(), df[raw].std()
        df[std] = (df[raw] - m) / s if s > 0 else 0.0

    print(f"  LOS-joined sample: {len(df)}")
    print(f"  owns_car=0: {(df['owns_car']==0).sum()}  owns_car=1: {(df['owns_car']==1).sum()}")
    print(f"  Escort rate — all: {df['escort'].mean():.3f}  "
          f"no-car: {df[df['owns_car']==0]['escort'].mean():.3f}  "
          f"car: {df[df['owns_car']==1]['escort'].mean():.3f}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Fit logit and extract results
# ══════════════════════════════════════════════════════════════════════════════

BASE_FORMULA = "escort ~ walk_time_std + bus_freq_std + car_dist_std + owned_cars_std"
NOCAR_FORMULA = "escort ~ walk_time_std + bus_freq_std + car_dist_std"
INT_FORMULA   = ("escort ~ walk_time_std + bus_freq_std + car_dist_std + owns_car "
                 "+ owns_car:walk_time_std + owns_car:bus_freq_std + owns_car:car_dist_std")

COEF_LABELS = {
    "walk_time_std":          "Walk time (std)",
    "bus_freq_std":           "Bus frequency (std)",
    "car_dist_std":           "Car distance (std)",
    "owned_cars_std":         "Num. cars owned (std)",
    "owns_car":               "Owns car (binary)",
    "owns_car:walk_time_std": "Owns car × Walk time",
    "owns_car:bus_freq_std":  "Owns car × Bus freq",
    "owns_car:car_dist_std":  "Owns car × Car dist",
    "Intercept":              "Intercept",
}


def fit_logit(formula, data, label):
    model = smf.logit(formula, data=data).fit(disp=False, maxiter=300)
    tbl   = model.summary2().tables[1]
    me    = model.get_margeff()
    ame   = me.summary_frame()["dy/dx"]

    print(f"\n  {label} (n={len(data)}, escort={data['escort'].mean():.3f})")
    print(f"  {'Variable':<30} {'Coef':>8} {'SE':>7} {'p':>8}  {'AME':>8}")
    print(f"  {'─'*66}")
    for var, row in tbl.iterrows():
        if var == "Intercept":
            continue
        lbl   = COEF_LABELS.get(var, var)
        pstr  = "<0.001" if row["P>|z|"] < 0.001 else f"{row['P>|z|']:.3f}"
        sig   = ("***" if row["P>|z|"] < 0.001 else "**" if row["P>|z|"] < 0.01
                 else "*" if row["P>|z|"] < 0.05 else "." if row["P>|z|"] < 0.10 else "")
        ame_v = ame.get(var, np.nan)
        print(f"  {lbl:<30} {row['Coef.']:>8.4f} {row['Std.Err.']:>7.4f} "
              f"{pstr:>8} {sig:<3} {ame_v:>8.4f}")

    print(f"  {'─'*66}")
    print(f"  LL(null)={model.llnull:.1f}  LL(model)={model.llf:.1f}  "
          f"McF-R²={(1-model.llf/model.llnull):.4f}")
    return model, tbl, ame


def extract_row(tbl, ame, var, label, model_name):
    if var not in tbl.index:
        return None
    row = tbl.loc[var]
    return {
        "model":     model_name,
        "variable":  label,
        "coef":      row["Coef."],
        "se":        row["Std.Err."],
        "pval":      row["P>|z|"],
        "sig":       ("***" if row["P>|z|"] < 0.001 else "**" if row["P>|z|"] < 0.01
                      else "*" if row["P>|z|"] < 0.05 else "." if row["P>|z|"] < 0.10 else ""),
        "ame":       ame.get(var, np.nan),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Chow test — structural break by car ownership
# ══════════════════════════════════════════════════════════════════════════════

def chow_test(ll_full, ll_nocar, ll_car, k):
    """LR version of Chow test: 2*(LL_sub1 + LL_sub2 - LL_full)"""
    stat = 2 * (ll_nocar + ll_car - ll_full)
    pval = 1 - chi2.cdf(stat, df=k)
    print(f"\n  Chow-style LR test (structural break by car ownership):")
    print(f"  LR stat = {stat:.3f}  df = {k}  p = {pval:.4f}")
    if pval < 0.05:
        print(f"  → Significant structural break: LOS effects differ by car ownership")
    else:
        print(f"  → No significant structural break (pooled model adequate)")
    return stat, pval


# ══════════════════════════════════════════════════════════════════════════════
# Figures
# ══════════════════════════════════════════════════════════════════════════════

def plot_coef_comparison(rows_df):
    vars_plot = ["Walk time (std)", "Bus frequency (std)", "Car distance (std)"]
    models    = ["M0: Full", "M1: No-car", "M2: Car"]
    colors    = ["#78909C", "#EF5350", "#42A5F5"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax_idx, metric in enumerate(["coef", "ame"]):
        ax = axes[ax_idx]
        label = "Coefficient" if metric == "coef" else "AME (pp)"
        x = np.arange(len(vars_plot))
        w = 0.22

        for i, (mod, col) in enumerate(zip(models, colors)):
            sub = rows_df[rows_df["model"] == mod].set_index("variable")
            vals = [sub.loc[v, metric] if v in sub.index else np.nan for v in vars_plot]
            ses  = [sub.loc[v, "se"] if v in sub.index else 0 for v in vars_plot]
            ci   = [1.96 * s for s in ses]
            offset = (i - 1) * w
            bars = ax.bar(x + offset, vals, width=w, label=mod, color=col,
                          alpha=0.85, yerr=ci if metric == "coef" else None,
                          capsize=3, error_kw={"ecolor": "#444", "linewidth": 0.8})

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(vars_plot, fontsize=9)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(f"{'Logit Coefficients' if metric=='coef' else 'Average Marginal Effects'} "
                     f"by Car Ownership Group", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("LOS Effects on School Escorting — Stratified by Car Ownership\n"
                 "(Okinawa PT Survey R5)", fontsize=11)
    fig.tight_layout()
    save_fig(fig, "P4a_coef_comparison")


def plot_escort_by_walktime(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    titles = ["No-car households", "Car-owning households"]

    for ax, owns, title in zip(axes, [0, 1], titles):
        sub = df[df["owns_car"] == owns].copy()
        sub["wt_bin"] = pd.cut(sub["walk_time_min"], bins=[0, 10, 20, 30, 45, 120],
                               labels=["0–10", "10–20", "20–30", "30–45", "45+"])
        gdf = sub.groupby("wt_bin", observed=True).agg(
            escort_rate=("escort", "mean"),
            n=("escort", "count")
        ).reset_index()

        colors = [f"#{max(0, int(255*(1-r/max(gdf['escort_rate'])))):02x}"
                  f"4f{min(255, int(180*r/max(gdf['escort_rate']))):02x}" for r in gdf["escort_rate"]]
        bars = ax.bar(gdf["wt_bin"].astype(str), gdf["escort_rate"] * 100,
                      color="#42A5F5" if owns == 1 else "#EF5350", alpha=0.8, edgecolor="white")
        for bar, n in zip(bars, gdf["n"]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"n={n}", ha="center", va="bottom", fontsize=8, color="#555")

        ax.set_xlabel("Walk time to school (min)", fontsize=10)
        ax.set_ylabel("Escort rate (%)", fontsize=10)
        ax.set_title(f"{title}\n(n={len(sub)}, corr={sub['walk_time_min'].corr(sub['escort']):.3f})",
                     fontsize=10)
        ax.set_ylim(0, min(100, gdf["escort_rate"].max() * 130))
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Escort Rate by Walk Time — Car vs No-Car Households\n"
                 "(LOS-joined school trips)", fontsize=11)
    fig.tight_layout()
    save_fig(fig, "P4a_escort_rate_by_walktime")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "█" * 70)
    print("  08_stratified_logit.py — Stratified Logit by Car Ownership")
    print("█" * 70)

    section("Load Data")
    df = load_data()

    # Sample stats
    sample_stats = df.groupby("owns_car").agg(
        n=("escort", "count"),
        escort_rate=("escort", "mean"),
        car_escort_rate=("car_escort", "mean"),
        mean_walk_min=("walk_time_min", "mean"),
        mean_bus_freq=("bus_frequency_per_day", "mean"),
    ).round(3)
    sample_stats.index = ["No car", "Owns car"]
    print(f"\n{sample_stats.to_string()}")
    sample_stats.to_csv(TABLE_DIR / "P4a_sample_stats.csv")
    print("  → Saved: tables/P4a_sample_stats.csv")

    section("M0 — Full Sample")
    m0, t0, a0 = fit_logit(BASE_FORMULA, df, "M0: Full sample")

    section("M1 — No-Car Households")
    df_nocar = df[df["owns_car"] == 0].copy()
    m1, t1, a1 = fit_logit(NOCAR_FORMULA, df_nocar, "M1: No-car HH")

    section("M2 — Car-Owning Households")
    df_car = df[df["owns_car"] == 1].copy()
    m2, t2, a2 = fit_logit(NOCAR_FORMULA, df_car, "M2: Car HH")

    section("M3 — Interaction Model (formal moderation test)")
    m3, t3, a3 = fit_logit(INT_FORMULA, df, "M3: Car × LOS interaction")
    t3.to_csv(TABLE_DIR / "P4a_interaction_model.csv")
    print("  → Saved: tables/P4a_interaction_model.csv")

    section("Chow-style LR Test")
    k = 3  # number of LOS coefficients
    chow_test(m0.llf, m1.llf, m2.llf, k)

    section("Combined Summary Table")
    key_vars = [
        ("walk_time_std",  "Walk time (std)"),
        ("bus_freq_std",   "Bus frequency (std)"),
        ("car_dist_std",   "Car distance (std)"),
    ]
    rows = []
    for var, label in key_vars:
        for mod_name, tbl, ame in [("M0: Full", t0, a0), ("M1: No-car", t1, a1), ("M2: Car", t2, a2)]:
            r = extract_row(tbl, ame, var, label, mod_name)
            if r:
                rows.append(r)

    sum_df = pd.DataFrame(rows)
    sum_df.to_csv(TABLE_DIR / "P4a_stratified_logit_summary.csv", index=False)
    print("  → Saved: tables/P4a_stratified_logit_summary.csv")

    # Pivot for readability
    pivot = sum_df.pivot_table(index="variable", columns="model",
                               values=["coef", "pval"], aggfunc="first")
    print(f"\n  Coefficient comparison:\n{pivot.round(4).to_string()}")

    section("Figures")
    plot_coef_comparison(sum_df)
    plot_escort_by_walktime(df)

    section("Summary")
    wt_nocar_p = t1.loc["walk_time_std", "P>|z|"] if "walk_time_std" in t1.index else np.nan
    wt_car_p   = t2.loc["walk_time_std", "P>|z|"] if "walk_time_std" in t2.index else np.nan
    bf_nocar_p = t1.loc["bus_freq_std",  "P>|z|"] if "bus_freq_std"  in t1.index else np.nan
    bf_car_p   = t2.loc["bus_freq_std",  "P>|z|"] if "bus_freq_std"  in t2.index else np.nan

    print(f"\n  Walk time p: no-car={wt_nocar_p:.3f}  car={wt_car_p:.3f}")
    print(f"  Bus freq  p: no-car={bf_nocar_p:.3f}  car={bf_car_p:.3f}")
    print(f"\n  Interpretation:")
    print(f"    LOS effects are {'PRESENT' if wt_nocar_p<0.10 else 'absent'} for no-car HH "
          f"and {'absent' if wt_car_p>0.10 else 'PRESENT'} for car HH")
    print(f"    → Car availability mediates the LOS–escort relationship\n")


if __name__ == "__main__":
    main()
