"""
09_mnl_escort_mode.py — Multinomial Logit: Escort Mode Choice (Analysis B)
============================================================================
DV (3 categories):
  0 = No escort       (escorting_flag == 2)
  1 = Non-car escort  (escorting_flag == 1, rep_mode_class1 ≠ 6)
  2 = Car escort      (escorting_flag == 1, rep_mode_class1 == 6)

Hypothesis: LOS predicts the MODE of escort better than escort incidence.
  - Walk time ↑  →  car escort ↑  (walking route too long/unsafe)
  - Bus freq  ↑  →  non-car escort ↑  (transit alternative available)
  - Car dist  ↑  →  car escort ↑  (longer car distance = destination farther)

Models
------
  MNL-A  Base:    walk_time_std + bus_freq_std + car_dist_std + owned_cars_std
  MNL-B  Car-str: school_level × LOS (school-level heterogeneity)

Reference category: 0 = No escort

Outputs (data/tables/)
----------------------
  P4b_mnl_base_coef.csv          — MNL-A coefficients (both outcome equations)
  P4b_mnl_base_ame.csv           — Average Marginal Effects for each outcome
  P4b_predicted_probs.csv        — Predicted probability profile across LOS range

Outputs (data/figures/)
-----------------------
  P4b_mnl_coef.png               — coefficient plot for both outcome equations
  P4b_predicted_probs.png        — predicted P(mode) across walk_time & bus_freq range
  P4b_ame_heatmap.png            — AME heatmap: variable × outcome
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import statsmodels.api as sm

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
# Load & prepare
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    df = pd.read_csv(DATA_DIR / "school_trips_los.csv", low_memory=False)

    # Outcome: 3-category escort mode
    df["rep_mode"] = pd.to_numeric(df["rep_mode_class1"], errors="coerce")
    df["escort_mode"] = 0  # no escort
    df.loc[(df["escorting_flag"] == 1) & (df["rep_mode"] != 6), "escort_mode"] = 1  # non-car
    df.loc[(df["escorting_flag"] == 1) & (df["rep_mode"] == 6), "escort_mode"] = 2  # car

    # LOS
    for col in ["walk_time_min", "bus_frequency_per_day", "car_distance_m"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["owned_num_cars"] = pd.to_numeric(df["owned_num_cars"], errors="coerce").fillna(0)

    df = df.dropna(subset=["walk_time_min", "bus_frequency_per_day", "car_distance_m"])

    # Standardise
    std_map = {}
    for raw, std in [("walk_time_min",        "walk_time_std"),
                     ("bus_frequency_per_day", "bus_freq_std"),
                     ("car_distance_m",        "car_dist_std"),
                     ("owned_num_cars",        "owned_cars_std")]:
        m, s = df[raw].mean(), df[raw].std()
        df[std] = (df[raw] - m) / s if s > 0 else 0.0
        std_map[std] = (m, s)

    print(f"  LOS-joined sample: {len(df)}")
    print(f"  Outcome distribution:")
    vc = df["escort_mode"].value_counts().sort_index()
    labels = {0: "No escort", 1: "Non-car escort", 2: "Car escort"}
    for k, v in vc.items():
        print(f"    {k} ({labels[k]}): {v} ({v/len(df):.1%})")

    return df, std_map


# ══════════════════════════════════════════════════════════════════════════════
# Fit MNL
# ══════════════════════════════════════════════════════════════════════════════

FEATURES = ["walk_time_std", "bus_freq_std", "car_dist_std", "owned_cars_std"]
FEAT_LABELS = {
    "walk_time_std":  "Walk time",
    "bus_freq_std":   "Bus frequency",
    "car_dist_std":   "Car distance",
    "owned_cars_std": "Num. cars owned",
}
OUTCOME_LABELS = {1: "Non-car escort\n(vs no escort)",
                  2: "Car escort\n(vs no escort)"}


def fit_mnl(df, features, label="MNL"):
    X = sm.add_constant(df[features])
    y = df["escort_mode"]

    model = sm.MNLogit(y, X)
    result = model.fit(method="bfgs", maxiter=500, disp=False)

    ll_null  = result.llnull
    ll_model = result.llf
    mcf_r2   = 1 - ll_model / ll_null
    print(f"\n  {label}  LL(null)={ll_null:.1f}  LL(model)={ll_model:.1f}  "
          f"McF-R²={mcf_r2:.4f}")

    # Coefficients: result.params is (K, J-1) where J-1=2 outcomes
    params = result.params          # shape (K, 2)
    bse    = result.bse             # shape (K, 2)
    pval   = result.pvalues         # shape (K, 2)

    rows = []
    # params columns are 0-indexed: col 0 = outcome 1 (Non-car), col 1 = outcome 2 (Car)
    param_cols = params.columns.tolist()  # e.g. [0, 1]
    for j_idx, j_label in enumerate(["Non-car escort", "Car escort"]):
        col_key = param_cols[j_idx]
        print(f"\n  Equation: {j_label} vs No escort")
        print(f"  {'Variable':<22} {'Coef':>8} {'SE':>7} {'p':>8}")
        print(f"  {'─'*50}")
        for feat in ["const"] + features:
            coef = params.loc[feat, col_key]
            se   = bse.loc[feat, col_key]
            p    = pval.loc[feat, col_key]
            sig  = ("***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05
                    else "." if p<0.10 else "")
            pstr = "<0.001" if p<0.001 else f"{p:.3f}"
            lbl  = FEAT_LABELS.get(feat, feat)
            if feat != "const":
                print(f"  {lbl:<22} {coef:>8.4f} {se:>7.4f} {pstr:>8} {sig}")
                rows.append({"outcome": j_label, "variable": lbl,
                             "coef": coef, "se": se, "pval": p, "sig": sig})

    coef_df = pd.DataFrame(rows)
    return result, coef_df, mcf_r2


# ══════════════════════════════════════════════════════════════════════════════
# Average Marginal Effects (numerical)
# ══════════════════════════════════════════════════════════════════════════════

def compute_ame(result, df, features, delta=0.01):
    """Numerical AME: dP(j)/dx_k averaged over sample."""
    X_df = sm.add_constant(df[features])
    X0   = X_df.values                         # (N, K)
    P0   = result.predict(X_df)                # (N, J) — pass DataFrame to keep col names

    ame_rows = []
    col_names = list(X_df.columns)             # ['const', feat1, feat2, ...]
    for feat in features:
        k_idx = col_names.index(feat)
        X1_vals = X0.copy()
        X1_vals[:, k_idx] += delta
        X1_df = pd.DataFrame(X1_vals, columns=col_names)
        P1 = result.predict(X1_df)
        dP = (P1.values - P0.values) / delta   # (N, J)
        for j, j_label in enumerate(["No escort", "Non-car escort", "Car escort"]):
            ame_rows.append({
                "variable": FEAT_LABELS.get(feat, feat),
                "outcome":  j_label,
                "ame":      dP[:, j].mean(),
            })

    ame_df = pd.DataFrame(ame_rows)
    return ame_df


# ══════════════════════════════════════════════════════════════════════════════
# Predicted probabilities across LOS range
# ══════════════════════════════════════════════════════════════════════════════

def predicted_probs_curve(result, df, features, std_map):
    """Vary walk_time and bus_freq across their range; fix others at mean (0)."""
    records = []

    for vary_feat, vary_label, raw_col in [
        ("walk_time_std", "Walk time (min)", "walk_time_min"),
        ("bus_freq_std",  "Bus freq (trips/day)", "bus_frequency_per_day"),
    ]:
        m, s = std_map[vary_feat]
        raw_range = np.linspace(df[raw_col].quantile(0.05),
                                df[raw_col].quantile(0.95), 50)
        std_range = (raw_range - m) / s

        base_row = {f: 0.0 for f in features}  # all at mean
        base_row["const"] = 1.0

        for raw_v, std_v in zip(raw_range, std_range):
            row = base_row.copy()
            row[vary_feat] = std_v
            X_row = pd.DataFrame([[row["const"]] + [row[f] for f in features]],
                                 columns=["const"] + features)
            probs = result.predict(X_row).values[0]
            records.append({
                "vary_by": vary_label, "raw_value": raw_v,
                "P_no_escort": probs[0],
                "P_noncar_escort": probs[1],
                "P_car_escort": probs[2],
            })

    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════════
# Figures
# ══════════════════════════════════════════════════════════════════════════════

def plot_mnl_coef(coef_df):
    outcomes = ["Non-car escort", "Car escort"]
    features = list(FEAT_LABELS.values())
    colors   = {"Non-car escort": "#42A5F5", "Car escort": "#EF5350"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, outcome in zip(axes, outcomes):
        sub = coef_df[coef_df["outcome"] == outcome].set_index("variable")
        y   = np.arange(len(features))
        coefs = [sub.loc[f, "coef"] if f in sub.index else 0 for f in features]
        ses   = [sub.loc[f, "se"]   if f in sub.index else 0 for f in features]
        pvals = [sub.loc[f, "pval"] if f in sub.index else 1 for f in features]
        ci    = [1.96 * s for s in ses]

        bar_colors = [colors[outcome] if p < 0.10 else "#B0BEC5" for p in pvals]
        ax.barh(y, coefs, xerr=ci, color=bar_colors, height=0.5,
                alpha=0.85, ecolor="#444", capsize=4)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(features, fontsize=9)
        ax.set_xlabel("Coefficient (95% CI)", fontsize=9)
        ax.set_title(f"MNL: {outcome}\nvs No escort", fontsize=10)
        ax.grid(axis="x", alpha=0.3)

        # Significance stars
        for i, (c, p) in enumerate(zip(coefs, pvals)):
            sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "." if p<0.10 else ""
            if sig:
                ax.text(c + (ci[i]+0.05)*np.sign(c), i, sig,
                        ha="left" if c > 0 else "right", va="center", fontsize=9,
                        color="#333")

    fig.suptitle("MNL Escort Mode: Coefficients\n(ref: no escort; † / * / ** / *** = sig at 10/5/1/0.1%)",
                 fontsize=11)
    fig.tight_layout()
    save_fig(fig, "P4b_mnl_coef")


def plot_predicted_probs(pred_df):
    vary_groups = pred_df["vary_by"].unique()
    fig, axes = plt.subplots(1, len(vary_groups), figsize=(6 * len(vary_groups), 5))
    if len(vary_groups) == 1:
        axes = [axes]

    outcome_cols  = ["P_no_escort", "P_noncar_escort", "P_car_escort"]
    outcome_labels = ["No escort", "Non-car escort", "Car escort"]
    colors         = ["#90A4AE", "#42A5F5", "#EF5350"]

    for ax, vary in zip(axes, vary_groups):
        sub = pred_df[pred_df["vary_by"] == vary]
        for col, lbl, col_c in zip(outcome_cols, outcome_labels, colors):
            ax.plot(sub["raw_value"], sub[col] * 100, label=lbl, color=col_c, linewidth=2)
        ax.set_xlabel(vary, fontsize=10)
        ax.set_ylabel("Predicted probability (%)", fontsize=10)
        ax.set_title(f"Predicted escort mode\nvs {vary}", fontsize=10)
        ax.legend(fontsize=9)
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    fig.suptitle("MNL Predicted Mode Probabilities across LOS Range\n"
                 "(all other variables at mean)", fontsize=11)
    fig.tight_layout()
    save_fig(fig, "P4b_predicted_probs")


def plot_ame_heatmap(ame_df):
    pivot = ame_df.pivot_table(index="variable", columns="outcome",
                               values="ame", aggfunc="first")
    fig, ax = plt.subplots(figsize=(8, 4))
    vmax = max(abs(pivot.values.max()), abs(pivot.values.min()))
    im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=9, color="white" if abs(val) > vmax * 0.5 else "black")

    plt.colorbar(im, ax=ax, label="AME (probability)", shrink=0.8)
    ax.set_title("Average Marginal Effects: MNL Escort Mode\n"
                 "(numerical AME; all other variables at mean)", fontsize=10)
    fig.tight_layout()
    save_fig(fig, "P4b_ame_heatmap")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "█" * 70)
    print("  09_mnl_escort_mode.py — MNL Escort Mode Choice (Analysis B)")
    print("█" * 70)

    section("Load Data")
    df, std_map = load_data()

    section("MNL-A: Base Model")
    result, coef_df, mcf_r2 = fit_mnl(df, FEATURES, "MNL-A Base")
    coef_df.to_csv(TABLE_DIR / "P4b_mnl_base_coef.csv", index=False)
    print("  → Saved: tables/P4b_mnl_base_coef.csv")

    section("Average Marginal Effects")
    ame_df = compute_ame(result, df, FEATURES)
    ame_df.to_csv(TABLE_DIR / "P4b_mnl_base_ame.csv", index=False)
    print("  → Saved: tables/P4b_mnl_base_ame.csv")

    print(f"\n  AME summary (key variables):")
    for feat_lbl in ["Walk time", "Bus frequency", "Num. cars owned"]:
        sub = ame_df[ame_df["variable"] == feat_lbl]
        print(f"  {feat_lbl}:")
        for _, row in sub.iterrows():
            print(f"    → {row['outcome']}: {row['ame']:+.4f}")

    section("Predicted Probability Curves")
    pred_df = predicted_probs_curve(result, df, FEATURES, std_map)
    pred_df.to_csv(TABLE_DIR / "P4b_predicted_probs.csv", index=False)
    print("  → Saved: tables/P4b_predicted_probs.csv")

    section("Figures")
    plot_mnl_coef(coef_df)
    plot_predicted_probs(pred_df)
    plot_ame_heatmap(ame_df)

    section("Summary")
    print(f"  McFadden R² = {mcf_r2:.4f}")
    print(f"\n  Key coefficients:")
    for outcome in ["Non-car escort", "Car escort"]:
        sub = coef_df[coef_df["outcome"] == outcome].set_index("variable")
        for feat in ["Walk time", "Bus frequency", "Car distance", "Num. cars owned"]:
            if feat in sub.index:
                row = sub.loc[feat]
                print(f"    {outcome:<18} {feat:<20}: {row['coef']:+.4f} {row['sig']}")

    print(f"\n  Interpretation:")
    wt_car = coef_df[(coef_df["outcome"]=="Car escort")&
                     (coef_df["variable"]=="Walk time")]["coef"].values
    bf_nc  = coef_df[(coef_df["outcome"]=="Non-car escort")&
                     (coef_df["variable"]=="Bus frequency")]["coef"].values
    if len(wt_car):
        direction = "increases" if wt_car[0] > 0 else "decreases"
        print(f"    Walk time ↑ → car escort {direction} (coef={wt_car[0]:+.4f})")
    if len(bf_nc):
        direction = "increases" if bf_nc[0] > 0 else "decreases"
        print(f"    Bus freq  ↑ → non-car escort {direction} (coef={bf_nc[0]:+.4f})")
    print()


if __name__ == "__main__":
    main()
