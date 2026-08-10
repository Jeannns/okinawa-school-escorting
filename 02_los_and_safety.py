"""
02_los_and_safety.py — LOS, Safety Concern, and Escort Mode Analyses
=====================================================================
Requires: data/school_trips_los.csv (from 01_data_prep.py)
          R05_Supplementary_EN.csv  (PT survey folder)
Outputs:  data/tables/  ← CSV tables for each analysis
          data/figures/ ← PNG figures

Analyses (in priority order):
  D1  (#60) — Escort rate by walk_time quartile
  D2  (#60) — Escort rate by bus_frequency group
  D3  (#61) — Car distance distribution: school_level × escort type
  D4  (#64) — Zone-level escort rate vs mean LOS
  P1a (#62) — 2×2 cross-tab: safety_concern × bus_freq → car escort rate
  P1b (#63) — Binary logit: escort ~ LOS variables
  P1c (#66) — Interaction logit: escort ~ safety_concern × bus_frequency

Notes on escort definition (from PT data):
  escorting_flag = 1 → child is escorted to school
  escorting_flag = 2 → child travels independently (no escort)

  car_escort = escorting_flag==1 AND rep_mode_class1==6

Notes on safety_concern (from SUPP):
  SUPP sample joined via composite HH key (municipality + b_zone + c_zone + hh_number).
  safety_concern = 1 if ANY child in HH has q6_child*_escort_reason == '02' (crime/safety).
  Non-matched trips kept as NaN; safety analyses use matched subset only.
  Join yields ~7–8% of school trips (only escorting SUPP households).
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─── optional statsmodels (for logit) ─────────────────────────────────────────
try:
    import statsmodels.formula.api as smf
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("⚠️  statsmodels not installed — P1b/P1c logit skipped.")
    print("   Run: pip install statsmodels --break-system-packages")

# ══════════════════════════════════════════════════════════════════════════════
# Paths
# ══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).resolve().parent

DATA_DIR = (SCRIPT_DIR.parents[1]
            / "Okinawa_PT survey"
            / "Ver20240430第4回沖縄PTマスターデータ"
            / "EN")

PREP_DIR   = SCRIPT_DIR / "data"
TABLE_DIR  = PREP_DIR / "tables"
FIG_DIR    = PREP_DIR / "figures"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Mode code → label (Okinawa PT classification)
MODE_LABELS = {
    1: "JR/Intercity Rail",
    4: "Monorail",
    6: "Private Car",
    7: "Walk",
    9: "Taxi",
    10: "Intercity Bus",
    11: "Regular Bus",
    12: "Bus",
    13: "Chartered Bus",
    15: "Ship/Ferry",
    16: "Airplane",
    19: "Other Public",
    22: "E-Bicycle",
    23: "Bicycle",
    99: "Unknown/Other",
}
CAR_MODE   = 6
WALK_MODE  = 7
BIKE_MODES = {22, 23}
BUS_MODES  = {10, 11, 12, 13}

# HH composite join key (shared between PT and SUPP)
HH_KEYS = [
    "id_municipality_code",
    "id_b_zone_code",
    "id_c_zone_code",
    "id_household_number",
]


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def section(title: str) -> None:
    bar = "═" * 70
    print(f"\n{bar}\n  {title}\n{bar}")


def save_table(df: pd.DataFrame, name: str) -> None:
    path = TABLE_DIR / f"{name}.csv"
    df.to_csv(path, index=True, encoding="utf-8-sig")
    print(f"  → Saved: tables/{name}.csv  ({df.shape[0]} rows × {df.shape[1]} cols)")


def save_fig(fig: plt.Figure, name: str) -> None:
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved: figures/{name}.png")


# ══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_school_trips() -> pd.DataFrame:
    """Load preprocessed school trips (output of 01_data_prep.py)."""
    path = PREP_DIR / "school_trips_los.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"  {path.name} not found.\n  Run 01_data_prep.py first."
        )
    df = pd.read_csv(path, low_memory=False)
    print(f"  school_trips_los.csv  : {len(df):,} rows × {df.shape[1]} cols")

    # Derived flags
    df["escort"]     = (df["escorting_flag"] == 1).astype(int)
    df["car_escort"] = ((df["escorting_flag"] == 1) & (df["rep_mode_class1"] == CAR_MODE)).astype(int)
    df["mode_label"] = df["rep_mode_class1"].map(MODE_LABELS).fillna("Other")

    # LOS-joined subset flag
    df["has_los"] = df["walk_time_min"].notna()

    n_los = df["has_los"].sum()
    print(f"  With LOS data         : {n_los:,}  ({n_los/len(df):.1%})")
    print(f"  Escort (flag=1)       : {df['escort'].sum():,}  ({df['escort'].mean():.1%})")
    print(f"  Car escort            : {df['car_escort'].sum():,}  ({df['car_escort'].mean():.1%})")
    return df


def join_safety_concern(df: pd.DataFrame) -> pd.DataFrame:
    """
    Join safety_concern from SUPP via composite household key.
    safety_concern = 1 if any child in HH has escort reason '02' (crime/safety).
    Rows without SUPP match keep NaN.
    """
    supp_path = DATA_DIR / "R05_Supplementary_EN.csv"
    if not supp_path.exists():
        print(f"  ⚠️  SUPP not found at {supp_path}")
        df["safety_concern"] = np.nan
        return df

    supp = pd.read_csv(supp_path, low_memory=False)
    supp.columns = supp.columns.str.strip()

    # Filter to escort households (q6_escorting_flag == '1')
    supp_esc = supp[supp["q6_escorting_flag"].astype(str).str.strip() == "1"].copy()

    def has_safety(row):
        for col in ["q6_child1_escort_reason",
                    "q6_child2_escort_reason",
                    "q6_child3_escort_reason"]:
            if col in row.index and str(row[col]).strip() == "02":
                return 1
        return 0

    supp_esc["safety_concern"] = supp_esc.apply(has_safety, axis=1)

    # Aggregate to HH level: safety_concern = max across children
    hh_safety = (
        supp_esc.groupby(HH_KEYS, as_index=False)["safety_concern"]
        .max()
    )

    # Ensure key types match for merge
    for k in HH_KEYS:
        if k in df.columns:
            df[k] = pd.to_numeric(df[k], errors="coerce")
        hh_safety[k] = pd.to_numeric(hh_safety[k], errors="coerce")

    df = df.merge(hh_safety, on=HH_KEYS, how="left")

    n_matched = df["safety_concern"].notna().sum()
    n_safe    = (df["safety_concern"] == 1).sum()
    print(f"  safety_concern matched: {n_matched:,} trips  ({n_matched/len(df):.1%})")
    print(f"    └─ safety concern=1 : {n_safe:,}  ({n_safe/n_matched:.1%} of matched)")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# D1 — Escort Rate by Walk-Time Quartile
# ══════════════════════════════════════════════════════════════════════════════

def analysis_d1(df: pd.DataFrame) -> None:
    section("D1  — Escort Rate by Walk-Time Quartile")

    sub = df.dropna(subset=["walk_time_min"]).copy()
    sub["walk_q"] = pd.qcut(
        sub["walk_time_min"], q=4,
        labels=["Q1\n(shortest)", "Q2", "Q3", "Q4\n(longest)"]
    )

    tbl = (
        sub.groupby("walk_q", observed=True)
        .agg(
            n           = ("escort", "count"),
            n_escort    = ("escort", "sum"),
            escort_rate = ("escort", "mean"),
            walk_min    = ("walk_time_min", "mean"),
        )
        .round(4)
    )
    tbl["escort_rate_pct"] = (tbl["escort_rate"] * 100).round(1)
    print(tbl.to_string())
    save_table(tbl, "D1_escort_rate_by_walk_quartile")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(tbl.index, tbl["escort_rate_pct"], color="#4C72B0", width=0.5)
    ax.set_xlabel("Walk-time quartile (min to school)")
    ax.set_ylabel("Escort rate (%)")
    ax.set_title("D1: Escort Rate by Walk-Time Quartile\n(Okinawa PT Survey R5, school trips)")
    ax.set_ylim(0, 100)
    for i, (idx, row) in enumerate(tbl.iterrows()):
        ax.text(i, row["escort_rate_pct"] + 1.5, f"{row['escort_rate_pct']:.1f}%",
                ha="center", fontsize=9)
    fig.tight_layout()
    save_fig(fig, "D1_escort_rate_walk_quartile")


# ══════════════════════════════════════════════════════════════════════════════
# D2 — Escort Rate by Bus Frequency Group
# ══════════════════════════════════════════════════════════════════════════════

def analysis_d2(df: pd.DataFrame) -> None:
    section("D2  — Escort Rate by Bus Frequency Group")

    sub = df.dropna(subset=["bus_frequency_per_day"]).copy()

    # Group: 0, 1–5, 6–15, 16+
    bins   = [-1, 0, 5, 15, sub["bus_frequency_per_day"].max() + 1]
    labels = ["0 (no bus)", "1–5", "6–15", "16+ (frequent)"]
    sub["bus_grp"] = pd.cut(sub["bus_frequency_per_day"], bins=bins, labels=labels)

    tbl = (
        sub.groupby("bus_grp", observed=True)
        .agg(
            n           = ("escort", "count"),
            n_escort    = ("escort", "sum"),
            escort_rate = ("escort", "mean"),
            bus_mean    = ("bus_frequency_per_day", "mean"),
        )
        .round(4)
    )
    tbl["escort_rate_pct"] = (tbl["escort_rate"] * 100).round(1)
    print(tbl.to_string())
    save_table(tbl, "D2_escort_rate_by_bus_frequency")

    # ── Figure ────────────────────────────────────────────────────────────────
    colors = ["#C44E52", "#DD8452", "#55A868", "#4C72B0"]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(tbl.index, tbl["escort_rate_pct"], color=colors, width=0.5)
    ax.set_xlabel("Bus frequency (trips/day on OD pair)")
    ax.set_ylabel("Escort rate (%)")
    ax.set_title("D2: Escort Rate by Bus Frequency Group\n(Okinawa PT Survey R5, school trips)")
    ax.set_ylim(0, 100)
    for i, (idx, row) in enumerate(tbl.iterrows()):
        ax.text(i, row["escort_rate_pct"] + 1.5, f"{row['escort_rate_pct']:.1f}%",
                ha="center", fontsize=9)
    fig.tight_layout()
    save_fig(fig, "D2_escort_rate_bus_frequency")


# ══════════════════════════════════════════════════════════════════════════════
# D3 — Car Distance Distribution: School Level × Escort Type
# ══════════════════════════════════════════════════════════════════════════════

def analysis_d3(df: pd.DataFrame) -> None:
    section("D3  — Car Distance Distribution: School Level × Escort Type")

    sub = df.dropna(subset=["car_distance_m"]).copy()
    sub["car_dist_km"] = sub["car_distance_m"] / 1000

    # Summary stats
    tbl = (
        sub.groupby(["school_level", "escort"], observed=True)["car_dist_km"]
        .agg(["count", "mean", "median", "std"])
        .round(3)
    )
    tbl.index.set_names(["school_level", "escort_flag"], inplace=True)
    escort_lbl = {0: "No escort", 1: "Escort"}
    tbl.index = tbl.index.set_levels(
        [tbl.index.get_level_values(0).unique(),
         [escort_lbl[x] for x in tbl.index.get_level_values(1).unique()]],
        level=[0, 1]
    )
    print(tbl.to_string())
    save_table(tbl, "D3_car_distance_school_level_escort")

    # ── Figure: boxplot ────────────────────────────────────────────────────────
    levels = sub["school_level"].dropna().unique()
    fig, axes = plt.subplots(1, len(levels), figsize=(5 * len(levels), 4), sharey=True)
    if len(levels) == 1:
        axes = [axes]

    for ax, lv in zip(axes, sorted(levels)):
        grp = sub[sub["school_level"] == lv]
        data_to_plot = [
            grp[grp["escort"] == 0]["car_dist_km"].dropna().values,
            grp[grp["escort"] == 1]["car_dist_km"].dropna().values,
        ]
        bp = ax.boxplot(data_to_plot, labels=["No escort", "Escort"],
                        patch_artist=True, medianprops={"color": "red", "linewidth": 2})
        bp["boxes"][0].set_facecolor("#A8C7E8")
        bp["boxes"][1].set_facecolor("#F4A460")
        ax.set_title(f"{lv} School")
        ax.set_ylabel("Car distance (km)" if ax == axes[0] else "")
        ax.set_ylim(0, min(grp["car_dist_km"].quantile(0.95) * 1.2, 30))

    fig.suptitle("D3: Car Distance Distribution by School Level × Escort Type\n(Okinawa PT Survey R5)",
                 y=1.01)
    fig.tight_layout()
    save_fig(fig, "D3_car_distance_school_escort")


# ══════════════════════════════════════════════════════════════════════════════
# D4 — Zone-Level: Escort Rate vs Mean LOS
# ══════════════════════════════════════════════════════════════════════════════

def analysis_d4(df: pd.DataFrame) -> None:
    section("D4  — Zone-Level: Escort Rate vs Mean LOS (C-zone aggregation)")

    sub = df.dropna(subset=["walk_time_min", "origin_czone"]).copy()

    zone_agg = (
        sub.groupby("origin_czone")
        .agg(
            n               = ("escort", "count"),
            escort_rate     = ("escort", "mean"),
            car_escort_rate = ("car_escort", "mean"),
            mean_walk_min   = ("walk_time_min", "mean"),
            mean_bus_freq   = ("bus_frequency_per_day", "mean"),
            mean_car_dist_m = ("car_distance_m", "mean"),
        )
        .round(4)
    )
    # Keep zones with ≥ 5 trips for stability
    zone_agg = zone_agg[zone_agg["n"] >= 5]
    print(f"  Zones with ≥5 trips: {len(zone_agg)}")
    print(zone_agg.describe().round(3).to_string())
    save_table(zone_agg, "D4_zone_level_escort_vs_los")

    # Correlation summary
    print("\n  Pearson correlations (zone-level, n≥5 trips per zone):")
    for los_var in ["mean_walk_min", "mean_bus_freq", "mean_car_dist_m"]:
        if zone_agg[los_var].notna().sum() > 5:
            r = zone_agg[["escort_rate", los_var]].dropna().corr().iloc[0, 1]
            print(f"    escort_rate ~ {los_var:<22}: r = {r:+.3f}")

    # ── Figure: scatter escort_rate vs bus_freq ────────────────────────────────
    sub_z = zone_agg.dropna(subset=["mean_bus_freq"])
    if len(sub_z) > 5:
        fig, ax = plt.subplots(figsize=(6, 4))
        sc = ax.scatter(sub_z["mean_bus_freq"], sub_z["escort_rate"] * 100,
                        s=sub_z["n"] * 0.5, alpha=0.6, c="#4C72B0")
        # Trend line
        z = np.polyfit(sub_z["mean_bus_freq"], sub_z["escort_rate"] * 100, 1)
        p = np.poly1d(z)
        xr = np.linspace(sub_z["mean_bus_freq"].min(), sub_z["mean_bus_freq"].max(), 100)
        ax.plot(xr, p(xr), "r--", lw=1.5, label="Linear trend")
        ax.set_xlabel("Mean bus frequency (trips/day, origin C-zone)")
        ax.set_ylabel("Zone escort rate (%)")
        ax.set_title("D4: Zone-Level Escort Rate vs Mean Bus Frequency\n(bubble size ∝ trip count)")
        ax.legend()
        fig.tight_layout()
        save_fig(fig, "D4_zone_escort_vs_bus_freq")


# ══════════════════════════════════════════════════════════════════════════════
# P1a — 2×2 Cross-Tab: Safety Concern × Bus Freq → Car Escort Rate
# ══════════════════════════════════════════════════════════════════════════════

def analysis_p1a(df: pd.DataFrame) -> None:
    section("P1a — 2×2 Cross-Tab: Safety Concern × Bus Frequency → Car Escort Rate")

    sub = df.dropna(subset=["safety_concern", "bus_frequency_per_day"]).copy()
    sub["safety_lbl"] = sub["safety_concern"].map({0: "No safety concern", 1: "Safety concern"})
    sub["bus_lbl"]    = (sub["bus_frequency_per_day"] > 0).map(
        {True: "Bus available (freq>0)", False: "No bus (freq=0)"}
    )

    # 2×2 table: car escort rate
    tbl = (
        sub.groupby(["safety_lbl", "bus_lbl"], observed=True)
        .agg(
            n            = ("car_escort", "count"),
            car_escort_n = ("car_escort", "sum"),
            car_escort_rate = ("car_escort", "mean"),
            escort_rate  = ("escort", "mean"),
        )
        .round(4)
    )
    tbl["car_escort_pct"] = (tbl["car_escort_rate"] * 100).round(1)
    tbl["escort_pct"]     = (tbl["escort_rate"] * 100).round(1)
    print(tbl.to_string())
    save_table(tbl, "P1a_safety_x_bus_car_escort_rate")

    # ── Figure: 2×2 heatmap ────────────────────────────────────────────────────
    pivot = tbl["car_escort_pct"].unstack("bus_lbl")
    fig, ax = plt.subplots(figsize=(5, 3.5))
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    plt.colorbar(im, ax=ax, label="Car escort rate (%)")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            n = tbl["n"].unstack("bus_lbl").values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}%\n(n={int(n)})",
                        ha="center", va="center", fontsize=9,
                        color="white" if v > 60 else "black")
    ax.set_title("P1a: Car Escort Rate (%) by Safety Concern × Bus Availability\n(SUPP-matched sample)")
    fig.tight_layout()
    save_fig(fig, "P1a_safety_x_bus_heatmap")

    # Also print sample sizes
    print(f"\n  Sample size (SUPP-matched): {len(sub):,} trips")
    print(f"  Cells: {tbl['n'].to_dict()}")


# ══════════════════════════════════════════════════════════════════════════════
# P1b — Binary Logit: Escort ~ LOS Variables
# ══════════════════════════════════════════════════════════════════════════════

def analysis_p1b(df: pd.DataFrame) -> None:
    section("P1b — Binary Logit: Escort ~ LOS Variables")

    if not HAS_STATSMODELS:
        print("  ⚠️  statsmodels not available — skipping P1b.")
        return

    import statsmodels.formula.api as smf

    # LOS-joined subset
    sub = df.dropna(subset=["walk_time_min", "bus_frequency_per_day", "car_distance_m"]).copy()

    # Note on school-level selection:
    # In the LOS-joined sample, 97.6% of High School trips are escorts (vs 1.9% Elementary).
    # This is a selection artifact: short elementary trips are within-zone (no LOS match);
    # long high-school trips are cross-zone (LOS matched) and predominantly car-escorted.
    # Including school-level dummies causes quasi-separation → drop them from this model.
    # Use school-level fixed effects in a separate stratified analysis instead.
    print("\n  LOS-joined escort rate by school level:")
    print(sub.groupby("school_level")["escort"].agg(["mean","count"]).round(3).to_string())

    # Standardize LOS variables for interpretability
    sub["walk_time_std"]  = (sub["walk_time_min"]         - sub["walk_time_min"].mean()        ) / sub["walk_time_min"].std()
    sub["bus_freq_std"]   = (sub["bus_frequency_per_day"] - sub["bus_frequency_per_day"].mean()) / sub["bus_frequency_per_day"].std()
    sub["car_dist_km"]    = sub["car_distance_m"] / 1000
    sub["car_dist_std"]   = (sub["car_dist_km"]   - sub["car_dist_km"].mean()   ) / sub["car_dist_km"].std()
    sub["owned_cars_std"] = (sub["owned_num_cars"] - sub["owned_num_cars"].mean()) / sub["owned_num_cars"].std()

    formula = ("escort ~ walk_time_std + bus_freq_std + car_dist_std + owned_cars_std")

    try:
        model = smf.logit(formula, data=sub).fit(disp=False)
        print(model.summary().as_text())

        # Average Marginal Effects
        me = model.get_margeff()
        print("\n  Average Marginal Effects:")
        print(me.summary().as_text())

        # Save coefficient + AME table (aligned on variable name)
        me_frame = me.summary_frame()
        coef_frame = pd.DataFrame({
            "coef":    model.params,
            "se":      model.bse,
            "z":       model.tvalues,
            "p_value": model.pvalues,
            "ci_low":  model.conf_int()[0],
            "ci_high": model.conf_int()[1],
        })
        # AME index excludes Intercept; merge on variable name
        coef_frame["ame"] = me_frame["dy/dx"].reindex(coef_frame.index)
        save_table(coef_frame.round(4), "P1b_logit_escort_LOS")

        print(f"\n  N = {int(model.nobs):,}   Pseudo R² = {model.prsquared:.4f}")
        print(f"  LLR p-value: {model.llr_pvalue:.4g}")

    except Exception as e:
        print(f"  ⚠️  Logit failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# P1c — Interaction Logit: Escort ~ Safety × Bus Frequency
# ══════════════════════════════════════════════════════════════════════════════

def analysis_p1c(df: pd.DataFrame) -> None:
    section("P1c — Interaction Logit: Escort ~ Safety_Concern × Bus_Frequency")

    if not HAS_STATSMODELS:
        print("  ⚠️  statsmodels not available — skipping P1c.")
        return

    import statsmodels.formula.api as smf

    # SUPP-matched + LOS-joined subset
    sub = df.dropna(subset=["safety_concern", "bus_frequency_per_day",
                             "walk_time_min", "car_distance_m"]).copy()
    sub["safety_concern"] = sub["safety_concern"].astype(int)

    sub["walk_time_std"]  = (sub["walk_time_min"]         - sub["walk_time_min"].mean()        ) / sub["walk_time_min"].std()
    sub["bus_freq_std"]   = (sub["bus_frequency_per_day"] - sub["bus_frequency_per_day"].mean()) / sub["bus_frequency_per_day"].std()
    sub["car_dist_km"]    = sub["car_distance_m"] / 1000
    sub["car_dist_std"]   = (sub["car_dist_km"]   - sub["car_dist_km"].mean()   ) / sub["car_dist_km"].std()
    sub["owned_cars_std"] = (sub["owned_num_cars"] - sub["owned_num_cars"].mean()) / sub["owned_num_cars"].std()

    sub["is_middle"]     = (sub["school_level"] == "Middle").astype(int)
    sub["is_highschool"] = (sub["school_level"] == "High School").astype(int)

    # Model 1: main effects only (same sample as P1c, with safety added)
    formula_main = ("escort ~ safety_concern + walk_time_std + bus_freq_std + "
                    "car_dist_std + owned_cars_std")

    # Model 2: with interaction safety × bus_freq
    formula_int  = ("escort ~ safety_concern * bus_freq_std + walk_time_std + "
                    "car_dist_std + owned_cars_std")

    print(f"\n  Sample: {len(sub):,} trips (SUPP-matched + LOS-joined)")
    print(f"  safety_concern=1: {sub['safety_concern'].sum():,}  ({sub['safety_concern'].mean():.1%})")

    models = {}
    for label, formula in [("Main effects", formula_main),
                            ("Interaction",  formula_int)]:
        print(f"\n  ── {label} ───────────────────────────")
        try:
            m = smf.logit(formula, data=sub).fit(disp=False)
            print(m.summary().as_text())
            me = m.get_margeff()
            print(f"\n  Average Marginal Effects:")
            print(me.summary().as_text())
            me_frame = me.summary_frame()
            coef_frame = pd.DataFrame({
                "coef":    m.params,
                "se":      m.bse,
                "p_value": m.pvalues,
            })
            coef_frame["ame"] = me_frame["dy/dx"].reindex(coef_frame.index)
            models[label] = coef_frame.round(4)
            print(f"\n  N={int(m.nobs):,}  Pseudo R²={m.prsquared:.4f}  LLR-p={m.llr_pvalue:.4g}")
        except Exception as e:
            print(f"  ⚠️  {label} model failed: {e}")

    # Save combined coefficient table
    if models:
        combined = pd.concat(models, axis=1)
        save_table(combined, "P1c_logit_safety_x_bus")

    # ── Figure: predicted probability by safety × bus_freq ────────────────────
    try:
        m_int = smf.logit(formula_int, data=sub).fit(disp=False)
        bus_grid = np.linspace(
            sub["bus_frequency_per_day"].min(),
            sub["bus_frequency_per_day"].max(), 100
        )
        fig, ax = plt.subplots(figsize=(6, 4))
        for sc, color, label in [(0, "#4C72B0", "No safety concern"),
                                  (1, "#C44E52", "Safety concern")]:
            pred_df = pd.DataFrame({
                "safety_concern": sc,
                "bus_freq_std":   (bus_grid - sub["bus_frequency_per_day"].mean())
                                   / sub["bus_frequency_per_day"].std(),
                "walk_time_std":  0.0,
                "car_dist_std":   0.0,
                "owned_cars_std": 0.0,
                "is_middle":      0,
                "is_highschool":  0,
            })
            prob = m_int.predict(pred_df)
            ax.plot(bus_grid, prob * 100, color=color, lw=2, label=label)
        ax.set_xlabel("Bus frequency (trips/day)")
        ax.set_ylabel("Predicted escort probability (%)")
        ax.set_title("P1c: Predicted Escort Probability\nby Bus Frequency × Safety Concern")
        ax.legend()
        ax.set_ylim(0, 100)
        fig.tight_layout()
        save_fig(fig, "P1c_predicted_escort_safety_bus")
    except Exception as e:
        print(f"  ⚠️  Predicted probability figure failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "█" * 70)
    print("  02_los_and_safety.py — LOS, Safety, and Escort Mode Analyses")
    print("█" * 70)

    section("Data Loading")
    df = load_school_trips()
    df = join_safety_concern(df)

    # ── Run analyses ───────────────────────────────────────────────────────────
    analysis_d1(df)   # D1: escort rate by walk-time quartile
    analysis_d2(df)   # D2: escort rate by bus frequency
    analysis_d3(df)   # D3: car distance by school level × escort
    analysis_d4(df)   # D4: zone-level escort rate vs LOS
    analysis_p1a(df)  # P1a: 2×2 safety × bus → car escort
    analysis_p1b(df)  # P1b: logit escort ~ LOS
    analysis_p1c(df)  # P1c: interaction logit safety × bus

    # ── Summary ────────────────────────────────────────────────────────────────
    section("Summary — Outputs")
    tables  = sorted(TABLE_DIR.glob("*.csv"))
    figures = sorted(FIG_DIR.glob("*.png"))
    print(f"  Tables  ({len(tables)}): {[t.name for t in tables]}")
    print(f"  Figures ({len(figures)}): {[f.name for f in figures]}")
    print(f"\n  Next: run 03_school_choice.py\n")


if __name__ == "__main__":
    main()
