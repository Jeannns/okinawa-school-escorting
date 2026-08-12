"""
10_school_choice_escort.py — Analysis D: School Trip Distance → Escort Mode
=============================================================================
Core question: Does the distance of the chosen school trip predict escort mode
choice, independent of built-environment LOS?

Logic:
  - In a car-dependent city, LOS (walk time, bus freq) barely predicts escort.
  - But TRIP DISTANCE (the realized travel distance, reflecting school choice)
    should predict car escort directly: longer trips → must drive.
  - This links school choice behavior to escorting mode.

Approach:
  1. Compute chosen_dist_km = haversine(origin C-zone centroid, dest C-zone centroid)
     using CZone.shp lat/lon centroids.
  2. Stratify escort_mode by distance quintile (descriptive).
  3. MNL-0 (baseline): escort_mode ~ walk_time_std + bus_freq_std + car_dist_std + owned_cars_std
  4. MNL-D (with distance): add chosen_dist_km_std
  5. LR test: does distance improve fit?
  6. In-catchment flag: is this a "regular" school trip (dest_czone == home school czone)?

Outputs (tables/)
-----------------
  D1_dist_by_quintile.csv    — escort mode breakdown by distance quintile
  D2_mnl_baseline.csv        — MNL-0 coefficients (no distance)
  D3_mnl_distance.csv        — MNL-D coefficients (with distance)
  D4_model_comparison.csv    — LL, McF-R², LR test

Outputs (figures/)
------------------
  D1_escort_by_distance.png  — bar chart: escort mode share by distance quintile
  D2_mnl_coef_compare.png    — coefficient comparison MNL-0 vs MNL-D
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR   = SCRIPT_DIR / "data"
TABLE_DIR  = DATA_DIR / "tables"
FIG_DIR    = DATA_DIR / "figures"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

SHP_PATH = Path("/sessions/awesome-beautiful-pascal/mnt/For Jehan 260721/"
                "Okinawa_PT_MasterData/01_PopulationFrame/shp/CZone.shp")
# Mac fallback
SHP_PATH_MAC = Path("/Volumes/Samsung_T5/For Jehan 260721/"
                    "Okinawa_PT_MasterData/01_PopulationFrame/shp/CZone.shp")


def section(t):
    bar = "═" * 68
    print(f"\n{bar}\n  {t}\n{bar}")


def save_fig(fig, name):
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved: figures/{name}.png")


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi  = np.radians(lat2 - lat1)
    dlam  = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


# ══════════════════════════════════════════════════════════════════════════════
# Load zone centroids
# ══════════════════════════════════════════════════════════════════════════════

def load_zone_centroids():
    shp_path = SHP_PATH if SHP_PATH.exists() else SHP_PATH_MAC
    gdf = gpd.read_file(shp_path)
    # Columns: Cゾーン, 緯度 (lat), 経度 (lon)
    gdf = gdf.rename(columns={"Cゾーン": "czone", "緯度": "lat", "経度": "lon"})
    centroids = gdf[["czone", "lat", "lon"]].copy()
    centroids["czone"] = pd.to_numeric(centroids["czone"], errors="coerce")
    centroids = centroids.dropna(subset=["czone"])
    centroids = centroids.set_index("czone")
    print(f"  Zone centroids loaded: {len(centroids)} zones")
    return centroids


# ══════════════════════════════════════════════════════════════════════════════
# Load data and compute chosen distance
# ══════════════════════════════════════════════════════════════════════════════

def load_and_compute(centroids):
    df = pd.read_csv(DATA_DIR / "school_trips_los.csv", low_memory=False)

    # Escort mode (3-category)
    df["rep_mode"] = pd.to_numeric(df["rep_mode_class1"], errors="coerce")
    df["escort_mode"] = 0
    df.loc[(df["escorting_flag"]==1) & (df["rep_mode"]!=6), "escort_mode"] = 1
    df.loc[(df["escorting_flag"]==1) & (df["rep_mode"]==6), "escort_mode"] = 2

    # Clean zone codes
    df["origin_czone"] = pd.to_numeric(df["origin_czone"], errors="coerce")
    df["dest_czone"]   = pd.to_numeric(df["dest_czone"],   errors="coerce")

    # Join origin centroid
    df = df.join(centroids[["lat","lon"]].rename(columns={"lat":"o_lat","lon":"o_lon"}),
                 on="origin_czone")
    # Join destination centroid
    df = df.join(centroids[["lat","lon"]].rename(columns={"lat":"d_lat","lon":"d_lon"}),
                 on="dest_czone")

    # Compute chosen distance
    valid = df["o_lat"].notna() & df["d_lat"].notna()
    df.loc[valid, "chosen_dist_km"] = haversine_km(
        df.loc[valid, "o_lat"], df.loc[valid, "o_lon"],
        df.loc[valid, "d_lat"], df.loc[valid, "d_lon"]
    )

    # Same-zone trips → set to small non-zero value (< 1 zone width ≈ 1km)
    same_zone = df["origin_czone"] == df["dest_czone"]
    df.loc[same_zone & valid, "chosen_dist_km"] = 0.5

    # LOS variables
    for col in ["walk_time_min", "bus_frequency_per_day", "car_distance_m"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["owned_num_cars"] = pd.to_numeric(df["owned_num_cars"], errors="coerce").fillna(0)

    # Drop missing LOS + distance
    df = df.dropna(subset=["walk_time_min","bus_frequency_per_day",
                            "car_distance_m","chosen_dist_km"])

    # Standardise (full-sample mean/std)
    std_map = {}
    for raw, std in [("walk_time_min",        "walk_time_std"),
                     ("bus_frequency_per_day", "bus_freq_std"),
                     ("car_distance_m",        "car_dist_std"),
                     ("owned_num_cars",        "owned_cars_std"),
                     ("chosen_dist_km",        "chosen_dist_std")]:
        m, s = df[raw].mean(), df[raw].std()
        df[std] = (df[raw] - m) / s if s > 0 else 0.0
        std_map[std] = (m, s)

    # Car ownership binary + interaction term
    df["owns_car_bin"] = (df["owned_num_cars"] > 0).astype(float)
    df["dist_x_car"]   = df["chosen_dist_std"] * df["owns_car_bin"]

    # Distance quintile
    df["dist_q"] = pd.qcut(df["chosen_dist_km"], q=5,
                            labels=["Q1\n(closest)","Q2","Q3","Q4","Q5\n(farthest)"])

    print(f"  Analysis D sample: {len(df)}")
    print(f"  chosen_dist_km — mean={df['chosen_dist_km'].mean():.2f} km, "
          f"median={df['chosen_dist_km'].median():.2f} km, "
          f"max={df['chosen_dist_km'].max():.1f} km")
    vc = df["escort_mode"].value_counts().sort_index()
    labels = {0:"No escort", 1:"Non-car escort", 2:"Car escort"}
    for k, v in vc.items():
        print(f"    {k} ({labels[k]}): {v} ({v/len(df):.1%})")

    return df, std_map


# ══════════════════════════════════════════════════════════════════════════════
# Descriptive: escort mode by distance quintile
# ══════════════════════════════════════════════════════════════════════════════

def desc_by_quintile(df):
    ct = pd.crosstab(df["dist_q"], df["escort_mode"], normalize="index") * 100
    ct.columns = ["No escort (%)", "Non-car escort (%)", "Car escort (%)"]

    # Add mean distance per quintile
    ct["mean_dist_km"] = df.groupby("dist_q", observed=True)["chosen_dist_km"].mean()
    ct["n"] = df.groupby("dist_q", observed=True).size()

    print("\n  Escort mode by distance quintile:")
    print(ct.round(1).to_string())
    ct.to_csv(TABLE_DIR / "D1_dist_by_quintile.csv")
    print("  → Saved: tables/D1_dist_by_quintile.csv")
    return ct


def plot_by_quintile(df):
    ct = pd.crosstab(df["dist_q"], df["escort_mode"], normalize="index") * 100
    ct.columns = ["No escort", "Non-car escort", "Car escort"]
    colors = ["#90A4AE", "#42A5F5", "#EF5350"]

    fig, ax = plt.subplots(figsize=(9, 5))
    bottom = np.zeros(len(ct))
    for col, color in zip(ct.columns, colors):
        ax.bar(ct.index, ct[col], bottom=bottom, label=col, color=color, alpha=0.87)
        for i, (val, bot) in enumerate(zip(ct[col], bottom)):
            if val > 5:
                ax.text(i, bot + val/2, f"{val:.0f}%", ha="center", va="center",
                        fontsize=8.5, color="white", fontweight="bold")
        bottom += ct[col].values

    ax.set_xlabel("School trip distance quintile", fontsize=10)
    ax.set_ylabel("Share (%)", fontsize=10)
    ax.set_title("Escort Mode Share by School Trip Distance Quintile\n"
                 "(Q1=shortest, Q5=farthest)", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    # Mean distance labels
    means = df.groupby("dist_q", observed=True)["chosen_dist_km"].mean()
    for i, (q, m) in enumerate(means.items()):
        ax.text(i, 101, f"{m:.1f}km", ha="center", va="bottom", fontsize=8, color="#555")

    fig.tight_layout()
    save_fig(fig, "D1_escort_by_distance")


# ══════════════════════════════════════════════════════════════════════════════
# MNL models
# ══════════════════════════════════════════════════════════════════════════════

BASE_FEATS = ["walk_time_std", "bus_freq_std", "car_dist_std", "owned_cars_std"]
DIST_FEATS = ["chosen_dist_std", "walk_time_std", "bus_freq_std",
              "car_dist_std", "owned_cars_std"]
INT_FEATS  = ["chosen_dist_std", "owns_car_bin", "dist_x_car",
              "walk_time_std", "bus_freq_std", "car_dist_std"]

FEAT_LABELS = {
    "chosen_dist_std": "School trip dist",
    "owns_car_bin":    "Has car (0/1)",
    "dist_x_car":      "Dist × Has car",
    "walk_time_std":   "Walk time",
    "bus_freq_std":    "Bus frequency",
    "car_dist_std":    "Car distance (LOS)",
    "owned_cars_std":  "Num. cars owned",
}


def fit_mnl(df, features, label="MNL"):
    X = sm.add_constant(df[features])
    y = df["escort_mode"]
    result = sm.MNLogit(y, X).fit(method="bfgs", maxiter=500, disp=False)

    ll_null  = result.llnull
    ll_model = result.llf
    mcf_r2   = 1 - ll_model / ll_null
    n        = int(result.nobs)
    print(f"\n  {label}  n={n}  LL={ll_model:.1f}  McF-R²={mcf_r2:.4f}")

    params     = result.params
    bse        = result.bse
    pval       = result.pvalues
    param_cols = params.columns.tolist()

    rows = []
    for j_idx, j_label in enumerate(["Non-car escort", "Car escort"]):
        col_key = param_cols[j_idx]
        for feat in features:
            coef = params.loc[feat, col_key]
            se   = bse.loc[feat, col_key]
            p    = pval.loc[feat, col_key]
            sig  = ("***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05
                    else "." if p<0.10 else "")
            rows.append({"model": label, "outcome": j_label,
                         "variable": FEAT_LABELS.get(feat, feat),
                         "coef": coef, "se": se, "pval": p, "sig": sig})

    return result, pd.DataFrame(rows), mcf_r2


def lr_test(result_restricted, result_full):
    """LR test: full vs restricted (nested)."""
    lr_stat = 2 * (result_full.llf - result_restricted.llf)
    df_diff = result_full.df_model - result_restricted.df_model
    p_val   = stats.chi2.sf(lr_stat, df_diff)
    return lr_stat, int(df_diff), p_val


# ══════════════════════════════════════════════════════════════════════════════
# Coefficient comparison plot
# ══════════════════════════════════════════════════════════════════════════════

def plot_interaction_effects(resultI, df):
    """Predicted P(escort mode) across distance range, by car ownership."""
    X_base = sm.add_constant(df[INT_FEATS])
    dist_range = np.linspace(df["chosen_dist_std"].min(),
                             df["chosen_dist_std"].max(), 60)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    outcome_cols  = [1, 2]
    outcome_lbls  = ["Non-car escort", "Car escort"]
    car_specs = [(0, "#42A5F5", "No car (owns_car=0)"),
                 (1, "#EF5350", "Has car (owns_car=1)")]

    param_cols = resultI.params.columns.tolist()

    for ax, j_col, j_lbl in zip(axes, outcome_cols, outcome_lbls):
        for car_val, color, lbl in car_specs:
            probs = []
            for d_std in dist_range:
                row = {f: 0.0 for f in INT_FEATS}
                row["chosen_dist_std"] = d_std
                row["owns_car_bin"]    = float(car_val)
                row["dist_x_car"]      = d_std * car_val
                row["const"] = 1.0
                X_row = pd.DataFrame([[row["const"]] + [row[f] for f in INT_FEATS]],
                                     columns=["const"] + INT_FEATS)
                p_all = resultI.predict(X_row).values[0]  # [P0, P1, P2]
                probs.append(p_all[j_col] * 100)

            # Convert std back to km for x-axis label
            m_d, s_d = df["chosen_dist_km"].mean(), df["chosen_dist_km"].std()
            dist_km = dist_range * s_d + m_d
            ax.plot(dist_km, probs, color=color, linewidth=2.2, label=lbl)

        ax.set_xlabel("School trip distance (km)", fontsize=10)
        ax.set_ylabel("Predicted probability (%)", fontsize=10)
        ax.set_title(f"P({j_lbl}) vs School Distance\nby Car Ownership", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, None)

    fig.suptitle("Interaction: School Trip Distance × Car Ownership → Escort Mode\n"
                 "(all other variables at mean)", fontsize=11)
    fig.tight_layout()
    save_fig(fig, "D3_interaction_dist_car")


def plot_coef_compare(coef0, coefD):
    outcomes = ["Non-car escort", "Car escort"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, outcome in zip(axes, outcomes):
        sub0 = coef0[(coef0["outcome"]==outcome)].set_index("variable")
        subD = coefD[(coefD["outcome"]==outcome)].set_index("variable")

        feats_base = [FEAT_LABELS[f] for f in BASE_FEATS]
        feats_dist = [FEAT_LABELS["chosen_dist_std"]]
        all_feats  = feats_dist + feats_base
        y = np.arange(len(all_feats))

        for i, feat in enumerate(all_feats):
            # MNL-D (always)
            if feat in subD.index:
                cd = subD.loc[feat, "coef"]
                sd = subD.loc[feat, "se"]
                pd_ = subD.loc[feat, "pval"]
                col = "#EF5350" if pd_ < 0.10 else "#BDBDBD"
                ax.barh(i - 0.18, cd, height=0.32, color=col, alpha=0.85,
                        xerr=1.96*sd, ecolor="#555", capsize=3, label="MNL-D" if i==0 else "")
            # MNL-0 (baseline, skip chosen_dist_std)
            if feat in sub0.index:
                c0 = sub0.loc[feat, "coef"]
                s0 = sub0.loc[feat, "se"]
                p0 = sub0.loc[feat, "pval"]
                col0 = "#42A5F5" if p0 < 0.10 else "#CFD8DC"
                ax.barh(i + 0.18, c0, height=0.32, color=col0, alpha=0.75,
                        xerr=1.96*s0, ecolor="#777", capsize=3, label="MNL-0" if i==0 else "")

        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(all_feats, fontsize=9)
        ax.set_xlabel("Coefficient (95% CI)", fontsize=9)
        ax.set_title(f"{outcome}\nvs No escort", fontsize=10)
        ax.grid(axis="x", alpha=0.3)
        if outcome == "Non-car escort":
            ax.legend(["MNL-D (with distance)", "MNL-0 (baseline)"],
                      fontsize=8, loc="lower right")

    fig.suptitle("MNL Escort Mode: Baseline vs + School Trip Distance\n"
                 "(colored = p<0.10; gray = n.s.)", fontsize=11)
    fig.tight_layout()
    save_fig(fig, "D2_mnl_coef_compare")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "█"*68)
    print("  10_school_choice_escort.py — Analysis D")
    print("  School Trip Distance → Escort Mode")
    print("█"*68)

    section("Load Zone Centroids")
    centroids = load_zone_centroids()

    section("Load Data + Compute Chosen Distance")
    df, std_map = load_and_compute(centroids)

    section("Descriptive: Escort Mode by Distance Quintile")
    desc_by_quintile(df)
    plot_by_quintile(df)

    section("MNL-0: Baseline (no distance)")
    result0, coef0, r2_0 = fit_mnl(df, BASE_FEATS, "MNL-0 Baseline")
    coef0.to_csv(TABLE_DIR / "D2_mnl_baseline.csv", index=False)
    print("  → Saved: tables/D2_mnl_baseline.csv")

    section("MNL-D: + School Trip Distance")
    resultD, coefD, r2_D = fit_mnl(df, DIST_FEATS, "MNL-D + Distance")
    coefD.to_csv(TABLE_DIR / "D3_mnl_distance.csv", index=False)
    print("  → Saved: tables/D3_mnl_distance.csv")

    section("Model Comparison + LR Test")
    lr_stat, df_diff, lr_p = lr_test(result0, resultD)
    print(f"  LR stat = {lr_stat:.2f}, df = {df_diff}, p = {lr_p:.4f}")
    sig = "***" if lr_p<0.001 else "**" if lr_p<0.01 else "*" if lr_p<0.05 else "n.s."
    print(f"  Distance improves fit: {sig}")

    comp = pd.DataFrame([
        {"model": "MNL-0 Baseline",      "n": int(result0.nobs),
         "ll": result0.llf, "mcf_r2": r2_0, "note": "walk+bus+car_dist+cars"},
        {"model": "MNL-D + Distance",    "n": int(resultD.nobs),
         "ll": resultD.llf, "mcf_r2": r2_D, "note": "adds chosen_dist_km"},
        {"model": "LR test",             "n": "",
         "ll": lr_stat, "mcf_r2": "",
         "note": f"df={df_diff}, p={lr_p:.4f} {sig}"},
    ])
    comp.to_csv(TABLE_DIR / "D4_model_comparison.csv", index=False)
    print("  → Saved: tables/D4_model_comparison.csv")

    section("Key Coefficients: School Trip Distance")
    dist_rows = coefD[coefD["variable"]=="School trip dist"]
    for _, row in dist_rows.iterrows():
        print(f"  {row['outcome']:<20}: coef={row['coef']:+.4f}  "
              f"se={row['se']:.4f}  p={row['pval']:.4f}  {row['sig']}")

    section("MNL-I: Distance × Car Ownership Interaction")
    resultI, coefI, r2_I = fit_mnl(df, INT_FEATS, "MNL-I Interaction")
    coefI.to_csv(TABLE_DIR / "D5_mnl_interaction.csv", index=False)
    print("  → Saved: tables/D5_mnl_interaction.csv")

    # LR test: MNL-D vs MNL-I
    lr_i_stat, df_i_diff, lr_i_p = lr_test(resultD, resultI)
    sig_i = ("***" if lr_i_p<0.001 else "**" if lr_i_p<0.01
             else "*" if lr_i_p<0.05 else "n.s.")
    print(f"  LR (MNL-D vs MNL-I): χ²={lr_i_stat:.2f}, df={df_i_diff}, "
          f"p={lr_i_p:.4f} {sig_i}")
    print(f"  McF-R²: {r2_D:.4f} (MNL-D) → {r2_I:.4f} (MNL-I)  Δ={r2_I-r2_D:.4f}")

    print("\n  Interaction term (Dist × Has car):")
    int_rows = coefI[coefI["variable"]=="Dist × Has car"]
    for _, row in int_rows.iterrows():
        print(f"    {row['outcome']:<20}: β={row['coef']:+.4f}  "
              f"p={row['pval']:.4f}  {row['sig']}")

    print("\n  Interpretation:")
    nc = int_rows[int_rows["outcome"]=="Non-car escort"]
    ce = int_rows[int_rows["outcome"]=="Car escort"]
    if not nc.empty:
        c = nc.iloc[0]["coef"]
        print(f"    No-car HH: dist ↑ → non-car escort shifts {'up' if c<0 else 'down'} "
              f"relative to car HH (interaction β={c:+.4f})")
    if not ce.empty:
        c = ce.iloc[0]["coef"]
        print(f"    Car HH: dist ↑ → additional car escort effect β={c:+.4f}")

    section("Figures")
    plot_coef_compare(coef0, coefD)
    plot_interaction_effects(resultI, df)

    section("Summary")
    dist_car = coefD[(coefD["outcome"]=="Car escort") &
                     (coefD["variable"]=="School trip dist")]
    if not dist_car.empty:
        c = dist_car.iloc[0]["coef"]
        p = dist_car.iloc[0]["pval"]
        direction = "increases" if c > 0 else "decreases"
        print(f"  School trip dist ↑ → car escort {direction} (β={c:+.4f}, p={p:.4f})")
    print(f"  McF-R²: {r2_0:.4f} (baseline) → {r2_D:.4f} (+ distance)  "
          f"Δ={r2_D-r2_0:.4f}")
    print(f"  LR test: χ²={lr_stat:.1f}, p={lr_p:.4f} {sig}")
    print()


if __name__ == "__main__":
    main()
