"""
04_counterfactual.py — Counterfactual Simulations (P3b + P3c)
==============================================================
Requires: data/school_trips_los.csv (from 01_data_prep.py)
          SSD: /Volumes/Samsung_T5/For Jehan 260721  (for Future LOS)
Outputs:  data/tables/  ← counterfactual summary tables
          data/figures/ ← scenario comparison figures

Analyses:
  P3b (#65) — Hypothetical bus-frequency improvement scenarios (no SSD needed)
    P3b-1 — Escort incidence model: logit(escort ~ LOS)
    P3b-2 — Car escort model:       logit(car_escort ~ LOS)
    P3b-3 — Counterfactual: 4 bus-frequency policy scenarios
    P3b-4 — Dose–response curve: escort rate vs % bus improvement

  P3c (#71) — Counterfactual with actual Future LOS data from SSD
    #70 — Delta analysis: Current → FutureNI / FutureWI LOS changes
    P3c-1 — Join FutureNI + FutureWI LOS to school trips
    P3c-2 — Re-predict escort rates using P3b logit + future car_distance
    P3c-3 — New transit impact: new_transit_travel_time_min in extended model

Future LOS scenarios (SSD):
  FutureNI — No network improvement (population growth, same roads)
  FutureWI — With network improvement (new transit line + road upgrades)
             Extra cols: new_transit_travel_time_min, new_transit_fare_yen,
             origin/destination_transit_line, access walk/car times

Key LOS changes Current → Future:
  walk_time_min:          no change
  bus_frequency_per_day:  no change (transit service not in scope of improvements)
  car_distance_m:         +1,165m (NI) / +1,720m (WI) — longer routes due to network
  car_travel_time_min:    -11 min (NI) / -18 min (WI) — faster roads despite longer routes
  new transit:            available in WI for 40.4% of OD pairs
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import statsmodels.formula.api as smf
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("⚠️  statsmodels not installed. Run: pip install statsmodels --break-system-packages")

# ══════════════════════════════════════════════════════════════════════════════
# Paths
# ══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).resolve().parent
PREP_DIR   = SCRIPT_DIR / "data"
TABLE_DIR  = PREP_DIR / "tables"
FIG_DIR    = PREP_DIR / "figures"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

SSD_DIR = Path("/Volumes/Samsung_T5/For Jehan 260721")
if not SSD_DIR.exists():
    SSD_DIR = Path("/sessions/awesome-beautiful-pascal/mnt/For Jehan 260721")

LOS_DIR = SSD_DIR / "Okinawa_PT_MasterData" / "04_LOS_ModalSplitParameters"

# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def section(title: str) -> None:
    bar = "═" * 70
    print(f"\n{bar}\n  {title}\n{bar}")


def save_table(df: pd.DataFrame, name: str) -> None:
    path = TABLE_DIR / f"{name}.csv"
    df.to_csv(path, index=True, encoding="utf-8-sig")
    print(f"  → Saved: tables/{name}.csv")


def save_fig(fig: plt.Figure, name: str) -> None:
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved: figures/{name}.png")


# ══════════════════════════════════════════════════════════════════════════════
# Data Preparation
# ══════════════════════════════════════════════════════════════════════════════

def load_data() -> pd.DataFrame:
    path = PREP_DIR / "school_trips_los.csv"
    if not path.exists():
        raise FileNotFoundError("Run 01_data_prep.py first.")

    df = pd.read_csv(path, low_memory=False)
    df = df.dropna(subset=["walk_time_min", "bus_frequency_per_day", "car_distance_m"]).copy()

    df["escort"]     = (df["escorting_flag"] == 1).astype(int)
    df["car_escort"] = ((df["escorting_flag"] == 1) & (df["rep_mode_class1"] == 6)).astype(int)
    df["car_dist_km"] = df["car_distance_m"] / 1000

    print(f"  LOS-joined school trips : {len(df):,}")
    print(f"  Escort rate (baseline)  : {df['escort'].mean():.1%}")
    print(f"  Car escort rate         : {df['car_escort'].mean():.1%}")
    print(f"  Bus freq (mean/median)  : {df['bus_frequency_per_day'].mean():.1f} / "
          f"{df['bus_frequency_per_day'].median():.1f} trips/day")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# P3b-1 / P3b-2 — Fit Baseline Logit Models
# ══════════════════════════════════════════════════════════════════════════════

def fit_models(df: pd.DataFrame):
    """
    Fit two logit models on baseline data:
      Model 1: logit(escort ~ walk_time + bus_freq + car_dist + owned_cars)
      Model 2: logit(car_escort ~ walk_time + bus_freq + car_dist + owned_cars)

    Returns the fitted models and standardization parameters (μ, σ)
    for use in counterfactual prediction.
    """
    if not HAS_STATSMODELS:
        return None, None, None

    # Compute standardization parameters FROM BASELINE DATA (never from counterfactual)
    std_params = {}
    for var in ["walk_time_min", "bus_frequency_per_day", "car_dist_km", "owned_num_cars"]:
        std_params[var] = {"mean": df[var].mean(), "std": df[var].std()}

    def standardize(col, data=df):
        return (data[col] - std_params[col]["mean"]) / std_params[col]["std"]

    df = df.copy()
    df["walk_time_std"]  = standardize("walk_time_min")
    df["bus_freq_std"]   = standardize("bus_frequency_per_day")
    df["car_dist_std"]   = standardize("car_dist_km")
    df["owned_cars_std"] = standardize("owned_num_cars")

    formula = "~ walk_time_std + bus_freq_std + car_dist_std + owned_cars_std"

    models = {}
    for dv in ["escort", "car_escort"]:
        try:
            m = smf.logit(f"{dv} {formula}", data=df).fit(disp=False)
            models[dv] = m
            me = m.get_margeff()
            print(f"\n  ── Model: {dv} ────────────────────")
            print(m.summary().as_text())
            print("\n  Average Marginal Effects:")
            print(me.summary().as_text())

            # Save coefficient table
            me_frame = me.summary_frame()
            coef_df = pd.DataFrame({
                "coef":    m.params,
                "se":      m.bse,
                "z":       m.tvalues,
                "p_value": m.pvalues,
                "ame":     me_frame["dy/dx"].reindex(m.params.index),
            }).round(4)
            save_table(coef_df, f"P3b_logit_{dv}_LOS")
            print(f"  N={int(m.nobs):,}  Pseudo R²={m.prsquared:.4f}  LLR-p={m.llr_pvalue:.4g}")
        except Exception as e:
            print(f"  ⚠️  {dv} model failed: {e}")

    return models, std_params, df


# ══════════════════════════════════════════════════════════════════════════════
# P3b-3 — Counterfactual Scenarios
# ══════════════════════════════════════════════════════════════════════════════

SCENARIOS = {
    "S0_Baseline": {
        "label": "S0: Baseline (current)",
        "color": "#4C72B0",
        "func":  lambda x: x,
    },
    "S1_Incremental": {
        "label": "S1: +5 trips/day (freq < 20)",
        "color": "#55A868",
        "func":  lambda x: np.where(x < 20, x + 5, x),
    },
    "S2_Plus50pct": {
        "label": "S2: ×1.5 (all with bus > 0)",
        "color": "#DD8452",
        "func":  lambda x: np.where(x > 0, x * 1.5, x),
    },
    "S3_Plus100pct": {
        "label": "S3: ×2.0 (all with bus > 0)",
        "color": "#C44E52",
        "func":  lambda x: np.where(x > 0, x * 2.0, x),
    },
    "S4_Floor20": {
        "label": "S4: Floor = 20 trips/day",
        "color": "#8172B2",
        "func":  lambda x: np.maximum(x, 20),
    },
}


def run_counterfactuals(models, std_params, df_base: pd.DataFrame) -> pd.DataFrame:
    """
    For each scenario, apply bus_freq transformation,
    re-standardize using BASELINE mean/std, then predict escort and car_escort probability.
    """
    if models is None:
        return pd.DataFrame()

    results = []

    for scen_key, scen in SCENARIOS.items():
        cf = df_base.copy()

        # Apply bus_freq transformation
        cf["bus_freq_cf"] = scen["func"](cf["bus_frequency_per_day"].values)

        # Re-standardize using BASELINE parameters (not counterfactual distribution)
        cf["walk_time_std"]  = (cf["walk_time_min"]    - std_params["walk_time_min"]["mean"]    ) / std_params["walk_time_min"]["std"]
        cf["bus_freq_std"]   = (cf["bus_freq_cf"]       - std_params["bus_frequency_per_day"]["mean"]) / std_params["bus_frequency_per_day"]["std"]
        cf["car_dist_std"]   = (cf["car_dist_km"]       - std_params["car_dist_km"]["mean"]      ) / std_params["car_dist_km"]["std"]
        cf["owned_cars_std"] = (cf["owned_num_cars"]    - std_params["owned_num_cars"]["mean"]   ) / std_params["owned_num_cars"]["std"]

        row = {
            "scenario":       scen_key,
            "label":          scen["label"],
            "bus_freq_mean":  cf["bus_freq_cf"].mean(),
            "bus_freq_delta": cf["bus_freq_cf"].mean() - df_base["bus_frequency_per_day"].mean(),
        }

        for dv in ["escort", "car_escort"]:
            if dv not in models:
                continue
            prob = models[dv].predict(cf)
            row[f"pred_{dv}_rate"]  = prob.mean()
            row[f"pred_{dv}_delta"] = prob.mean() - models[dv].predict(df_base).mean()

            # By school level
            for lv in df_base["school_level"].dropna().unique():
                mask = cf["school_level"] == lv
                row[f"pred_{dv}_{lv.replace(' ','_').lower()}"] = prob[mask].mean()

        results.append(row)

    return pd.DataFrame(results).set_index("scenario")


def print_counterfactual_summary(cf_df: pd.DataFrame) -> None:
    if cf_df.empty:
        return
    print("\n  Scenario summary (aggregate):")
    cols = ["label", "bus_freq_mean", "bus_freq_delta",
            "pred_escort_rate", "pred_escort_delta",
            "pred_car_escort_rate", "pred_car_escort_delta"]
    cols = [c for c in cols if c in cf_df.columns]
    display = cf_df[cols].copy()
    # Format as percentages where applicable
    for c in display.columns:
        if "rate" in c:
            display[c] = (display[c] * 100).round(2).astype(str) + "%"
        elif "delta" in c and "bus" not in c:
            display[c] = (display[c] * 100).round(3).astype(str) + "pp"
    print(display.to_string())


# ══════════════════════════════════════════════════════════════════════════════
# P3b-4 — Dose–Response Curves
# ══════════════════════════════════════════════════════════════════════════════

def dose_response(models, std_params, df_base: pd.DataFrame) -> None:
    """
    Sweep bus_freq improvement from 0% to +200%.
    Plot predicted escort rate and car escort rate vs % improvement.
    """
    if models is None:
        return

    pct_range = np.arange(0, 201, 10)   # 0% to +200%
    records   = {"pct_improvement": pct_range}

    for dv in ["escort", "car_escort"]:
        if dv not in models:
            continue
        pred_rates = []
        for pct in pct_range:
            cf = df_base.copy()
            cf["bus_freq_cf"] = np.where(
                cf["bus_frequency_per_day"] > 0,
                cf["bus_frequency_per_day"] * (1 + pct / 100),
                cf["bus_frequency_per_day"]
            )
            cf["walk_time_std"]  = (cf["walk_time_min"] - std_params["walk_time_min"]["mean"]       ) / std_params["walk_time_min"]["std"]
            cf["bus_freq_std"]   = (cf["bus_freq_cf"]    - std_params["bus_frequency_per_day"]["mean"]) / std_params["bus_frequency_per_day"]["std"]
            cf["car_dist_std"]   = (cf["car_dist_km"]    - std_params["car_dist_km"]["mean"]         ) / std_params["car_dist_km"]["std"]
            cf["owned_cars_std"] = (cf["owned_num_cars"] - std_params["owned_num_cars"]["mean"]      ) / std_params["owned_num_cars"]["std"]
            pred_rates.append(models[dv].predict(cf).mean())
        records[dv] = pred_rates

    dr_df = pd.DataFrame(records).set_index("pct_improvement")
    save_table(dr_df.round(4), "P3b_dose_response_bus_freq")

    # ── Figure ────────────────────────────────────────────────────────────────
    baseline_escort     = dr_df["escort"].iloc[0]     if "escort"     in dr_df.columns else None
    baseline_car_escort = dr_df["car_escort"].iloc[0] if "car_escort" in dr_df.columns else None

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    titles = {"escort": "Escort probability", "car_escort": "Car escort probability"}
    baseline_vals = {"escort": baseline_escort, "car_escort": baseline_car_escort}

    for ax, dv in zip(axes, ["escort", "car_escort"]):
        if dv not in dr_df.columns:
            continue
        rates_pct = dr_df[dv] * 100
        ax.plot(dr_df.index, rates_pct, lw=2, color="#4C72B0")
        ax.axhline(baseline_vals[dv] * 100, color="gray", linestyle="--", lw=1,
                   label=f"Baseline: {baseline_vals[dv]*100:.1f}%")
        ax.fill_between(dr_df.index, baseline_vals[dv] * 100, rates_pct,
                        alpha=0.15, color="#4C72B0")
        ax.set_xlabel("Bus frequency improvement (%)")
        ax.set_ylabel(f"Predicted {titles[dv]} (%)")
        ax.set_title(f"P3b: {titles[dv]}\nvs bus frequency improvement")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 200)

    fig.suptitle("Dose–Response: Bus Frequency Improvement vs Escort Probability\n(Okinawa PT Survey R5, logit prediction)", y=1.01)
    fig.tight_layout()
    save_fig(fig, "P3b_dose_response_escort_vs_bus")


# ══════════════════════════════════════════════════════════════════════════════
# Counterfactual by Zone
# ══════════════════════════════════════════════════════════════════════════════

def counterfactual_by_zone(models, std_params, df_base: pd.DataFrame) -> None:
    """
    For each origin C-zone, compute baseline vs S3 (×2.0) escort rate delta.
    Save as zone-level CSV (for mapping in GIS or future spatial analysis).
    """
    if models is None or "escort" not in models or "car_escort" not in models:
        return

    zones = df_base["origin_czone"].dropna().unique()
    records = []

    # Counterfactual S3: ×2.0
    cf = df_base.copy()
    cf["bus_freq_cf"] = np.where(
        cf["bus_frequency_per_day"] > 0,
        cf["bus_frequency_per_day"] * 2.0,
        cf["bus_frequency_per_day"]
    )
    cf["walk_time_std"]  = (cf["walk_time_min"] - std_params["walk_time_min"]["mean"]       ) / std_params["walk_time_min"]["std"]
    cf["bus_freq_std"]   = (cf["bus_freq_cf"]    - std_params["bus_frequency_per_day"]["mean"]) / std_params["bus_frequency_per_day"]["std"]
    cf["car_dist_std"]   = (cf["car_dist_km"]    - std_params["car_dist_km"]["mean"]         ) / std_params["car_dist_km"]["std"]
    cf["owned_cars_std"] = (cf["owned_num_cars"] - std_params["owned_num_cars"]["mean"]      ) / std_params["owned_num_cars"]["std"]

    base_escort_pred     = models["escort"].predict(df_base)
    base_car_escort_pred = models["car_escort"].predict(df_base)
    cf_escort_pred       = models["escort"].predict(cf)
    cf_car_escort_pred   = models["car_escort"].predict(cf)

    df_base = df_base.copy()
    df_base["_base_escort"]     = base_escort_pred
    df_base["_base_car_escort"] = base_car_escort_pred
    df_base["_cf_escort"]       = cf_escort_pred
    df_base["_cf_car_escort"]   = cf_car_escort_pred

    zone_agg = (
        df_base.groupby("origin_czone")
        .agg(
            n                       = ("escort", "count"),
            actual_escort_rate      = ("escort", "mean"),
            actual_car_escort_rate  = ("car_escort", "mean"),
            baseline_escort_pred    = ("_base_escort", "mean"),
            cf_escort_pred          = ("_cf_escort", "mean"),
            baseline_car_escort_pred= ("_base_car_escort", "mean"),
            cf_car_escort_pred      = ("_cf_car_escort", "mean"),
            mean_bus_freq_current   = ("bus_frequency_per_day", "mean"),
        )
    )
    zone_agg["delta_escort_pp"]     = (zone_agg["cf_escort_pred"]     - zone_agg["baseline_escort_pred"])     * 100
    zone_agg["delta_car_escort_pp"] = (zone_agg["cf_car_escort_pred"] - zone_agg["baseline_car_escort_pred"]) * 100

    zone_agg = zone_agg[zone_agg["n"] >= 5].round(4)
    print(f"\n  Zone-level counterfactual (S3: ×2.0 bus freq), n_zones={len(zone_agg)}")
    print(f"  Δ escort rate  : mean={zone_agg['delta_escort_pp'].mean():+.3f} pp,  "
          f"range=[{zone_agg['delta_escort_pp'].min():+.3f}, {zone_agg['delta_escort_pp'].max():+.3f}]")
    print(f"  Δ car escort   : mean={zone_agg['delta_car_escort_pp'].mean():+.3f} pp,  "
          f"range=[{zone_agg['delta_car_escort_pp'].min():+.3f}, {zone_agg['delta_car_escort_pp'].max():+.3f}]")

    save_table(zone_agg, "P3b_zone_counterfactual_S3_bus2x")

    # ── Figure: histogram of zone-level deltas ─────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col, title in zip(axes,
                               ["delta_escort_pp", "delta_car_escort_pp"],
                               ["Escort rate Δ (pp)", "Car escort rate Δ (pp)"]):
        ax.hist(zone_agg[col], bins=20, color="#4C72B0", edgecolor="white", alpha=0.8)
        ax.axvline(zone_agg[col].mean(), color="red", lw=1.5, label=f"Mean: {zone_agg[col].mean():+.3f} pp")
        ax.axvline(0, color="black", lw=1, linestyle="--")
        ax.set_xlabel(title)
        ax.set_ylabel("Number of C-zones")
        ax.set_title(f"Distribution of zone-level {title}\n(S3: bus ×2.0)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, "P3b_zone_delta_histogram")


# ══════════════════════════════════════════════════════════════════════════════
# #70 — Future LOS Delta Analysis
# ══════════════════════════════════════════════════════════════════════════════

LOS_VARS_COMMON = [
    "walk_time_min", "bus_frequency_per_day",
    "bus_travel_time_min", "bus_fare_yen",
    "car_distance_m", "car_travel_time_min",
]
NEW_TRANSIT_VARS = [
    "new_transit_travel_time_min", "new_transit_fare_yen",
    "new_transit_access_walk_time_min", "new_transit_egress_walk_time_min",
    "destination_transit_line",
]
OD_KEY = ["origin_czone", "destination_czone"]


def load_all_los() -> tuple:
    """Load Current, FutureNI, FutureWI LOS CSVs from SSD."""
    if not LOS_DIR.exists():
        print(f"  ⚠️  LOS directory not found: {LOS_DIR}")
        return None, None, None

    cur     = pd.read_csv(LOS_DIR / "01_CurrentLOS.csv")
    fut_ni  = pd.read_csv(LOS_DIR / "02_FutureLOS_NoNetworkImprovement.csv")
    fut_wi  = pd.read_csv(LOS_DIR / "03_FutureLOS_WithNetworkImprovement.csv")
    print(f"  CurrentLOS   : {len(cur):,} OD pairs × {cur.shape[1]} cols")
    print(f"  FutureNI     : {len(fut_ni):,} OD pairs × {fut_ni.shape[1]} cols")
    print(f"  FutureWI     : {len(fut_wi):,} OD pairs × {fut_wi.shape[1]} cols")
    return cur, fut_ni, fut_wi


def analysis_los_delta(cur: pd.DataFrame, fut_ni: pd.DataFrame,
                       fut_wi: pd.DataFrame, df_trips: pd.DataFrame) -> None:
    """
    #70: For school-trip OD pairs, compute Δ LOS (Current → FutureNI, FutureWI).
    Reports aggregate stats and zone-level summary.
    """
    section("#70 — Future LOS Delta Analysis (school-trip OD pairs)")

    # Merge all 3 on the same OD pairs that appear in school trips
    trip_ods = df_trips[OD_KEY].dropna().drop_duplicates()
    print(f"  Unique school-trip OD pairs: {len(trip_ods):,}")

    # Merge current → FutureNI → FutureWI
    base = OD_KEY + [v for v in LOS_VARS_COMMON if v in cur.columns]
    m = (trip_ods
         .merge(cur[base], on=OD_KEY, how="left")
         .merge(fut_ni[[c for c in base if c in fut_ni.columns]], on=OD_KEY,
                how="left", suffixes=("", "_ni"))
         .merge(fut_wi[[c for c in base + NEW_TRANSIT_VARS if c in fut_wi.columns]],
                on=OD_KEY, how="left", suffixes=("", "_wi")))

    # Compute deltas
    rows = []
    for v in LOS_VARS_COMMON:
        if v not in cur.columns:
            continue
        cur_col = m[v]
        ni_col  = m.get(v + "_ni", None)
        wi_col  = m.get(v + "_wi", m.get(v))
        if ni_col is None:
            continue
        delta_ni = (ni_col - cur_col).dropna()
        delta_wi = (wi_col - cur_col).dropna()
        rows.append({
            "variable":       v,
            "current_mean":   cur_col.mean(),
            "futureNI_mean":  ni_col.mean(),
            "futureWI_mean":  wi_col.mean(),
            "delta_NI_mean":  delta_ni.mean(),
            "delta_WI_mean":  delta_wi.mean(),
            "pct_changed_NI": (delta_ni != 0).mean(),
            "pct_changed_WI": (delta_wi != 0).mean(),
        })

    delta_df = pd.DataFrame(rows).set_index("variable").round(3)
    print("\n  LOS delta summary (school-trip OD pairs):")
    print(delta_df.to_string())
    save_table(delta_df, "P3c_los_delta_summary")

    # New transit coverage in FutureWI for school trips
    if "new_transit_travel_time_min_wi" in m.columns or "new_transit_travel_time_min" in m.columns:
        nt_col = "new_transit_travel_time_min_wi" if "new_transit_travel_time_min_wi" in m.columns \
                 else "new_transit_travel_time_min"
        transit_line_col = "destination_transit_line_wi" if "destination_transit_line_wi" in m.columns \
                           else "destination_transit_line"
        n_with_transit = (m[nt_col] > 0).sum() if nt_col in m.columns else 0
        n_with_line    = (m[transit_line_col] > 0).sum() if transit_line_col in m.columns else 0
        print(f"\n  New transit (FutureWI):")
        print(f"    OD pairs with new transit time  : {n_with_transit:,} ({n_with_transit/len(m):.1%})")
        print(f"    OD pairs with transit line dest : {n_with_line:,} ({n_with_line/len(m):.1%})")

    # ── Figure: distribution of car_time delta ────────────────────────────────
    if "car_travel_time_min" in m.columns and "car_travel_time_min_ni" in m.columns:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, delta_col, label, color in [
            (axes[0], m["car_travel_time_min_ni"] - m["car_travel_time_min"],
             "FutureNI − Current", "#55A868"),
            (axes[1], m.get("car_travel_time_min_wi", m["car_travel_time_min_ni"]) - m["car_travel_time_min"],
             "FutureWI − Current", "#C44E52"),
        ]:
            d = delta_col.dropna()
            ax.hist(d, bins=40, color=color, edgecolor="white", alpha=0.8)
            ax.axvline(d.mean(), color="black", lw=1.5, linestyle="--",
                       label=f"Mean: {d.mean():+.1f} min")
            ax.set_xlabel("Δ car travel time (min)")
            ax.set_ylabel("Number of OD pairs")
            ax.set_title(f"#70: Δ Car Travel Time\n{label}")
            ax.legend(fontsize=8)
        fig.tight_layout()
        save_fig(fig, "P3c_car_time_delta_distribution")

    return m   # return merged for P3c


# ══════════════════════════════════════════════════════════════════════════════
# P3c — Counterfactual: Escort Rate Under Future LOS
# ══════════════════════════════════════════════════════════════════════════════

def analysis_p3c(models, std_params, df_base: pd.DataFrame,
                 cur: pd.DataFrame, fut_ni: pd.DataFrame, fut_wi: pd.DataFrame) -> None:
    """
    P3c (#71): Re-predict escort rates using Future LOS values for each trip's OD pair.

    Strategy:
      - For car_distance_m: use FutureNI/WI value (only variable that changes in the model)
      - For new transit: add new_transit_travel_time_min as extra predictor in extended model
      - Walk_time + bus_freq don't change → same standardized values
    """
    section("P3c (#71) — Counterfactual: Escort Rate Under Actual Future LOS")

    if models is None:
        print("  ⚠️  Requires logit models — run P3b first.")
        return

    # Join FutureNI and FutureWI car_distance to school trips
    ni_car  = fut_ni[OD_KEY + ["car_distance_m"]].rename(columns={"car_distance_m": "car_dist_ni"})
    wi_los  = fut_wi[OD_KEY + ["car_distance_m"] +
                     [c for c in NEW_TRANSIT_VARS if c in fut_wi.columns]].rename(
        columns={"car_distance_m": "car_dist_wi"})

    df = df_base.merge(ni_car, on=OD_KEY, how="left")
    df = df.merge(wi_los, on=OD_KEY, how="left")

    ni_join = df["car_dist_ni"].notna().mean()
    wi_join = df["car_dist_wi"].notna().mean()
    print(f"  FutureNI join rate: {ni_join:.1%}")
    print(f"  FutureWI join rate: {wi_join:.1%}")

    results = []
    for scenario, car_dist_col, label in [
        ("Current",   "car_dist_km",    "Current LOS"),
        ("FutureNI",  "car_dist_ni",    "Future (no improvement)"),
        ("FutureWI",  "car_dist_wi",    "Future (with new transit)"),
    ]:
        sub = df.dropna(subset=[car_dist_col]).copy()
        sub["car_dist_km_cf"] = sub[car_dist_col] / 1000 if "car_dist" in car_dist_col else sub[car_dist_col]
        if scenario == "Current":
            sub["car_dist_km_cf"] = sub["car_dist_km"]

        # Re-standardize car_dist using BASELINE mean/std
        sub["walk_time_std"]  = (sub["walk_time_min"]    - std_params["walk_time_min"]["mean"]    ) / std_params["walk_time_min"]["std"]
        sub["bus_freq_std"]   = (sub["bus_frequency_per_day"] - std_params["bus_frequency_per_day"]["mean"]) / std_params["bus_frequency_per_day"]["std"]
        sub["car_dist_std"]   = (sub["car_dist_km_cf"]   - std_params["car_dist_km"]["mean"]     ) / std_params["car_dist_km"]["std"]
        sub["owned_cars_std"] = (sub["owned_num_cars"]   - std_params["owned_num_cars"]["mean"]  ) / std_params["owned_num_cars"]["std"]

        row = {"scenario": scenario, "label": label, "n": len(sub)}
        for dv in ["escort", "car_escort"]:
            if dv not in models:
                continue
            prob = models[dv].predict(sub)
            row[f"pred_{dv}_rate"] = prob.mean()
        results.append(row)

    cf_df = pd.DataFrame(results).set_index("scenario")

    # Compute deltas relative to Current
    for dv in ["escort", "car_escort"]:
        col = f"pred_{dv}_rate"
        if col in cf_df.columns:
            baseline = cf_df.loc["Current", col]
            cf_df[f"delta_{dv}_pp"] = (cf_df[col] - baseline) * 100

    print("\n  Future LOS counterfactual results:")
    print(cf_df.round(4).to_string())
    save_table(cf_df.round(4), "P3c_future_los_counterfactual")

    # ── Extended model: add new transit travel time (FutureWI only) ───────────
    if HAS_STATSMODELS and "new_transit_travel_time_min" in df.columns:
        print("\n  ── Extended model (FutureWI): add new transit time ────────────")
        sub_wi = df.dropna(subset=["car_dist_wi","new_transit_travel_time_min"]).copy()
        sub_wi["car_dist_km_wi"] = sub_wi["car_dist_wi"] / 1000
        sub_wi["car_dist_std"]    = (sub_wi["car_dist_km_wi"] - std_params["car_dist_km"]["mean"]) / std_params["car_dist_km"]["std"]
        sub_wi["walk_time_std"]   = (sub_wi["walk_time_min"] - std_params["walk_time_min"]["mean"]) / std_params["walk_time_min"]["std"]
        sub_wi["bus_freq_std"]    = (sub_wi["bus_frequency_per_day"] - std_params["bus_frequency_per_day"]["mean"]) / std_params["bus_frequency_per_day"]["std"]
        sub_wi["owned_cars_std"]  = (sub_wi["owned_num_cars"] - std_params["owned_num_cars"]["mean"]) / std_params["owned_num_cars"]["std"]

        nt_mean = sub_wi["new_transit_travel_time_min"].mean()
        nt_std  = sub_wi["new_transit_travel_time_min"].std()
        sub_wi["new_transit_std"] = (sub_wi["new_transit_travel_time_min"] - nt_mean) / nt_std

        import statsmodels.formula.api as smf
        for dv in ["escort", "car_escort"]:
            try:
                formula = (f"{dv} ~ walk_time_std + bus_freq_std + car_dist_std + "
                           "owned_cars_std + new_transit_std")
                m = smf.logit(formula, data=sub_wi).fit(disp=False)
                me = m.get_margeff()
                print(f"\n  {dv} ~ LOS + new_transit_std  (N={int(m.nobs):,}):")
                print(m.summary().as_text())
                print("\n  AME:")
                print(me.summary().as_text())
                me_frame = me.summary_frame()
                coef_df = pd.DataFrame({
                    "coef":    m.params,
                    "se":      m.bse,
                    "p_value": m.pvalues,
                    "ame":     me_frame["dy/dx"].reindex(m.params.index),
                }).round(4)
                save_table(coef_df, f"P3c_logit_{dv}_with_new_transit")
            except Exception as e:
                print(f"  ⚠️  {dv} extended model failed: {e}")

    # ── Figure ─────────────────────────────────────────────────────────────────
    if "pred_escort_rate" in cf_df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        scenarios = cf_df.index.tolist()
        colors = ["#4C72B0", "#55A868", "#C44E52"]

        for ax, metric, ylabel in zip(
            axes,
            ["pred_escort_rate", "pred_car_escort_rate"],
            ["Predicted escort rate (%)", "Predicted car escort rate (%)"]
        ):
            if metric not in cf_df.columns:
                continue
            vals = cf_df[metric] * 100
            bars = ax.bar(range(len(vals)), vals, color=colors[:len(vals)], width=0.5)
            delta_col = metric.replace("_rate", "_pp").replace("pred_", "delta_")
            for i, (v, s) in enumerate(zip(vals, scenarios)):
                delta = cf_df[delta_col].iloc[i] if delta_col in cf_df.columns else 0
                sign = "+" if delta >= 0 else ""
                ax.text(i, v + 0.2, f"{v:.2f}%\n({sign}{delta:.3f}pp)",
                        ha="center", fontsize=8)
            ax.set_xticks(range(len(cf_df)))
            ax.set_xticklabels([cf_df.loc[s,"label"] for s in scenarios],
                               rotation=10, ha="right", fontsize=8)
            ax.set_ylabel(ylabel)
            ax.set_title(f"P3c: {ylabel}\nCurrent vs Future LOS Scenarios")
            ax.set_ylim(0, max(vals) * 1.2)
        fig.tight_layout()
        save_fig(fig, "P3c_future_los_escort_comparison")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "█" * 70)
    print("  04_counterfactual.py — Counterfactual: Bus Frequency Improvement")
    print("█" * 70)

    section("P3b-1/2 — Load Data + Fit Baseline Models")
    df = load_data()

    section("Baseline Logit Models (escort + car_escort ~ LOS)")
    models, std_params, df_std = fit_models(df)

    if models is None:
        print("  ⚠️  statsmodels required — exiting.")
        return

    section("P3b-3 — Counterfactual Scenarios")
    cf_df = run_counterfactuals(models, std_params, df_std)
    print_counterfactual_summary(cf_df)
    save_table(cf_df.round(4), "P3b_scenario_summary")

    # ── Figure: scenario bar chart ─────────────────────────────────────────────
    if not cf_df.empty and "pred_escort_rate" in cf_df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        scenario_labels = [cf_df.loc[s, "label"] for s in cf_df.index]
        colors = [SCENARIOS[s]["color"] for s in cf_df.index]

        for ax, metric, ylabel in zip(
            axes,
            ["pred_escort_rate", "pred_car_escort_rate"],
            ["Predicted escort rate (%)", "Predicted car escort rate (%)"]
        ):
            if metric not in cf_df.columns:
                continue
            vals = cf_df[metric] * 100
            bars = ax.bar(range(len(vals)), vals, color=colors, width=0.6)
            for i, (v, delta_col) in enumerate(
                zip(vals, cf_df.get(metric.replace("rate","delta"), [None]*len(vals)))
            ):
                delta_col = metric.replace("_rate","_delta")
                delta = cf_df[delta_col].iloc[i] * 100 if delta_col in cf_df.columns else 0
                sign = "+" if delta >= 0 else ""
                ax.text(i, v + 0.3, f"{v:.1f}%\n({sign}{delta:.2f}pp)",
                        ha="center", va="bottom", fontsize=7.5)
            ax.set_xticks(range(len(scenario_labels)))
            ax.set_xticklabels(scenario_labels, rotation=15, ha="right", fontsize=8)
            ax.set_ylabel(ylabel)
            ax.set_title(f"Counterfactual: {ylabel}\nby Bus Frequency Policy Scenario")
            ax.set_ylim(0, max(vals) * 1.15)
        fig.tight_layout()
        save_fig(fig, "P3b_scenario_comparison")

    section("P3b-4 — Dose–Response: % Bus Improvement vs Escort Rate")
    dose_response(models, std_params, df_std)

    section("P3b-5 — Zone-Level Counterfactual (S3: ×2.0)")
    counterfactual_by_zone(models, std_params, df_std)

    # ── P3c / #70: Future LOS counterfactual ──────────────────────────────────
    section("#70 + P3c — Load Future LOS (SSD required)")
    if SSD_DIR.exists():
        cur_los, fut_ni, fut_wi = load_all_los()
        if cur_los is not None:
            merged_los = analysis_los_delta(cur_los, fut_ni, fut_wi, df_std)
            analysis_p3c(models, std_params, df_std, cur_los, fut_ni, fut_wi)
    else:
        print(f"  ⚠️  SSD not found at {SSD_DIR} — skipping #70/P3c.")

    section("Summary")
    tables  = sorted([t.name for t in TABLE_DIR.glob("P3*.csv")])
    figures = sorted([f.name for f in FIG_DIR.glob("P3*.png")])
    print(f"  Tables  : {tables}")
    print(f"  Figures : {figures}")
    print(f"\n  Interpretation:")
    if not cf_df.empty and "pred_escort_delta" in cf_df.columns:
        for s in cf_df.index:
            row = cf_df.loc[s]
            label = row["label"]
            escort_delta = row.get("pred_escort_delta", 0) * 100
            car_delta    = row.get("pred_car_escort_delta", 0) * 100
            print(f"    {label:<40} escort Δ={escort_delta:+.3f}pp  car_escort Δ={car_delta:+.3f}pp")
    print(f"\n  Next: P1d (#67) — Spatial map of marginal effects by C-zone\n")


if __name__ == "__main__":
    main()
