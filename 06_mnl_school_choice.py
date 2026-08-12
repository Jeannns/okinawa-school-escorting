"""
06_mnl_school_choice.py — MNL School Choice Model (#72 P2c)
=============================================================
Conditional logit of school choice, where alternatives are all schools
of the same level within a 15-km radius of the student's home C-zone.

Model:
  V_ij = β₁·ln(dist_ij) + β₂·in_catchment_ij
  + [β₃·(owns_car·ln(dist_ij)) + β₄·(owns_car·in_catchment_ij)]

Identification: no school-specific constants (no individual-level covariates
in base conditional logit since they drop out without ASC interaction).
Extended model adds car-ownership interactions.

Counterfactual: assign every student to nearest school → how does
distance distribution shift?

Outputs (data/tables/):
  P2c_mnl_base.csv          — Model 1 coefficients + SE
  P2c_mnl_carinteract.csv   — Model 2 (car-ownership interaction)
  P2c_choiceset_stats.csv   — Choice set size distribution
  P2c_counterfactual.csv    — Nearest-school counterfactual distances

Outputs (data/figures/):
  P2c_mnl_coef.png          — Coefficient plot
  P2c_counterfactual.png    — Dist distribution: actual vs nearest-school
"""

import warnings
from math import atan2, cos, radians, sin, sqrt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2

warnings.filterwarnings("ignore")

try:
    import geopandas as gpd
    HAS_GPD = True
except ImportError:
    HAS_GPD = False

# ══════════════════════════════════════════════════════════════════════════════
# Paths
# ══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR   = SCRIPT_DIR / "data"
TABLE_DIR  = DATA_DIR / "tables"
FIG_DIR    = DATA_DIR / "figures"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

SSD_DIR = Path("/Volumes/Samsung_T5/For Jehan 260721")
if not SSD_DIR.exists():
    SSD_DIR = Path("/sessions/awesome-beautiful-pascal/mnt/For Jehan 260721")

CZONE_SHP = SSD_DIR / "Okinawa_PT_MasterData" / "01_PopulationFrame" / "shp" / "CZone.shp"

CHOICE_RADIUS_KM = 15.0   # search radius for choice set construction
MIN_ALTERNATIVES = 2      # minimum alternatives to include an observation


def section(t):
    bar = "═" * 70
    print(f"\n{bar}\n  {t}\n{bar}")


def save_fig(fig, name):
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved: figures/{name}.png")


# ══════════════════════════════════════════════════════════════════════════════
# Haversine
# ══════════════════════════════════════════════════════════════════════════════

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


# ══════════════════════════════════════════════════════════════════════════════
# Conditional Logit estimator (scipy)
# ══════════════════════════════════════════════════════════════════════════════

def clogit_negll(beta, X, choice_sets):
    """
    Parameters
    ----------
    beta       : (K,) array of coefficients
    X          : list of (J_i × K) numpy arrays — alternative attributes per person
    choice_sets: list of chosen-alternative indices (int) per person
    """
    ll = 0.0
    for xi, j_chosen in zip(X, choice_sets):
        v = xi @ beta                      # (J_i,)
        v = v - v.max()                    # numerical stability
        exp_v = np.exp(v)
        ll -= v[j_chosen] - np.log(exp_v.sum())
    return ll


def clogit_negll_grad(beta, X, choice_sets):
    ll = 0.0
    grad = np.zeros_like(beta)
    for xi, j_chosen in zip(X, choice_sets):
        v = xi @ beta
        v = v - v.max()
        exp_v = np.exp(v)
        p = exp_v / exp_v.sum()           # (J_i,)
        ll -= v[j_chosen] - np.log(exp_v.sum())
        grad -= xi[j_chosen] - p @ xi
    return ll, grad


def hessian_approx(beta, X, choice_sets):
    H = np.zeros((len(beta), len(beta)))
    for xi, _ in zip(X, choice_sets):
        v = xi @ beta
        v = v - v.max()
        exp_v = np.exp(v)
        p = exp_v / exp_v.sum()
        # weighted outer product
        xbar = p @ xi
        for j, pj in enumerate(p):
            diff = xi[j] - xbar
            H += pj * np.outer(diff, diff)
    return H


def fit_clogit(X, choice_sets, feature_names, label="Model"):
    """Fit conditional logit; return results dict."""
    n = len(X)
    K = X[0].shape[1]
    beta0 = np.zeros(K)

    res = minimize(
        lambda b: clogit_negll_grad(b, X, choice_sets),
        beta0,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 2000, "ftol": 1e-12}
    )
    beta_hat = res.x
    ll_opt = -res.fun

    H = hessian_approx(beta_hat, X, choice_sets)
    try:
        vcov = np.linalg.inv(H)
        se = np.sqrt(np.diag(vcov))
    except np.linalg.LinAlgError:
        se = np.full(K, np.nan)

    # Null log-likelihood: uniform choice
    ll_null = -sum(np.log(len(xi)) for xi in X)

    z = beta_hat / se
    pval = 2 * (1 - chi2.cdf(z ** 2, df=1))   # z-test
    mcfadden_r2 = 1 - ll_opt / ll_null

    print(f"\n  {label} (n={n})")
    print(f"  {'Variable':<22} {'Coef':>8} {'SE':>8} {'z':>8} {'p':>8}")
    print(f"  {'-'*56}")
    for name, b, s, zi, p in zip(feature_names, beta_hat, se, z, pval):
        print(f"  {name:<22} {b:>8.4f} {s:>8.4f} {zi:>8.3f} {p:>8.4f}")
    print(f"\n  LL(opt) = {ll_opt:.1f}  LL(null) = {ll_null:.1f}  McF-R² = {mcfadden_r2:.4f}")

    return {
        "features": feature_names,
        "coef": beta_hat,
        "se": se,
        "z": z,
        "pval": pval,
        "ll_opt": ll_opt,
        "ll_null": ll_null,
        "mcf_r2": mcfadden_r2,
        "n": n,
        "converged": res.success,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    # School trips
    df = pd.read_csv(DATA_DIR / "school_choice_trips.csv", low_memory=False)
    df = df[df["school_level"].isin(["Elementary", "Middle"])].copy()
    df = df[df["school_choice"].notna()].copy()

    # School centroids (deduplicate school names — keep first occurrence)
    sl = pd.read_csv(DATA_DIR / "school_locations.csv")
    sl_elem = (sl[sl["school_type"] == "Elementary"]
               .drop_duplicates(subset=["school_name"])
               .set_index("school_name"))
    sl_mid  = (sl[sl["school_type"] == "Middle"]
               .drop_duplicates(subset=["school_name"])
               .set_index("school_name"))

    # C-zone centroids from shapefile
    czone_latlon = None
    if HAS_GPD and CZONE_SHP.exists():
        cz = gpd.read_file(CZONE_SHP)
        cz = cz.rename(columns={"Cゾーン": "czone_str", "緯度": "lat", "経度": "lon"})
        cz["czone_int"] = cz["czone_str"].apply(lambda x: int(str(x)))
        czone_latlon = cz[["czone_int", "lat", "lon"]].set_index("czone_int")
        print(f"  CZone centroids loaded: {len(czone_latlon)} zones")
    else:
        print("  ⚠️  CZone.shp not available — cannot build choice sets")
        return None, None, None

    # Attach origin lat/lon
    df["origin_czone_int"] = df["origin_czone"].dropna().astype(int)
    df = df.merge(czone_latlon, left_on="origin_czone_int", right_index=True, how="left")
    df = df.rename(columns={"lat": "home_lat", "lon": "home_lon"})
    df = df[df["home_lat"].notna()].copy()

    # Owns car flag
    df["owns_car"] = (df["owned_num_cars"] > 0).astype(int)

    print(f"  School trips (with home lat/lon): {len(df)} rows")
    return df, sl_elem, sl_mid


# ══════════════════════════════════════════════════════════════════════════════
# Build long-format choice data
# ══════════════════════════════════════════════════════════════════════════════

def build_choice_data(df, sl_elem, sl_mid, radius_km):
    """
    For each student, build a choice set of schools within `radius_km` of home.
    Returns X (list of arrays), choice_sets (list of chosen indices), meta df.
    """
    X_base  = []   # [n_alts × 2] arrays: [log_dist, in_catchment]
    X_car   = []   # [n_alts × 4] arrays: + car interactions
    chosen  = []
    meta    = []   # (person_id, school_level, n_alts, chosen_dist, catchment_dist)

    records_dropped = {"no_dest": 0, "chosen_not_in_sl": 0, "too_few_alts": 0}

    for idx, row in df.iterrows():
        level = row["school_level"]
        sl_dict = sl_elem if level == "Elementary" else sl_mid
        home_col = "home_elem_dist" if level == "Elementary" else "home_mid_dist"
        dest_col = "dest_elem_dist" if level == "Elementary" else "dest_mid_dist"

        dest_school   = row.get(dest_col, np.nan)
        catchm_school = row.get(home_col, np.nan)
        home_lat = row["home_lat"]
        home_lon = row["home_lon"]
        owns_car  = row["owns_car"]

        if not isinstance(dest_school, str) or pd.isna(dest_school):
            records_dropped["no_dest"] += 1
            continue
        if dest_school not in sl_dict.index:
            records_dropped["chosen_not_in_sl"] += 1
            continue

        # Enumerate alternatives within radius
        alts = []
        for sc_name, sc_row in sl_dict.iterrows():
            d = haversine_km(home_lat, home_lon, sc_row["lat"], sc_row["lon"])
            if d <= radius_km:
                alts.append({
                    "school":      sc_name,
                    "dist_km":     d,
                    "in_catchment": int(sc_name == catchm_school),
                })

        # Ensure chosen school is in choice set (add it if just outside radius)
        chosen_names = {a["school"] for a in alts}
        if dest_school not in chosen_names:
            sc_row_chosen = sl_dict.loc[dest_school]
            ch_lat = float(sc_row_chosen["lat"].iloc[0]) if hasattr(sc_row_chosen["lat"], "iloc") else float(sc_row_chosen["lat"])
            ch_lon = float(sc_row_chosen["lon"].iloc[0]) if hasattr(sc_row_chosen["lon"], "iloc") else float(sc_row_chosen["lon"])
            d_chosen = haversine_km(home_lat, home_lon, ch_lat, ch_lon)
            alts.append({
                "school":       dest_school,
                "dist_km":      d_chosen,
                "in_catchment": int(dest_school == catchm_school),
            })

        if len(alts) < MIN_ALTERNATIVES:
            records_dropped["too_few_alts"] += 1
            continue

        # Sort so chosen school index is deterministic
        alts_df = pd.DataFrame(alts).sort_values("school").reset_index(drop=True)
        j_chosen = alts_df[alts_df["school"] == dest_school].index[0]

        # Catchment school distance (for counterfactual)
        catchm_dist = np.nan
        if isinstance(catchm_school, str) and catchm_school in sl_dict.index:
            c_row = sl_dict.loc[catchm_school]
            # handle duplicate index (take first row)
            c_lat = float(c_row["lat"].iloc[0]) if hasattr(c_row["lat"], "iloc") else float(c_row["lat"])
            c_lon = float(c_row["lon"].iloc[0]) if hasattr(c_row["lon"], "iloc") else float(c_row["lon"])
            catchm_dist = haversine_km(home_lat, home_lon, c_lat, c_lon)

        # Nearest school distance
        nearest_dist = alts_df["dist_km"].min()

        # Build X arrays
        log_d = np.log(alts_df["dist_km"].values + 0.01)
        catchm_vec = alts_df["in_catchment"].values.astype(float)

        xi_base = np.column_stack([log_d, catchm_vec])
        xi_car  = np.column_stack([
            log_d,
            catchm_vec,
            log_d * owns_car,       # car × log_dist
            catchm_vec * owns_car,  # car × in_catchment
        ])

        X_base.append(xi_base)
        X_car.append(xi_car)
        chosen.append(j_chosen)
        meta.append({
            "school_level":     level,
            "n_alts":           len(alts_df),
            "chosen_dist_km":   alts_df.loc[j_chosen, "dist_km"],
            "nearest_dist_km":  nearest_dist,
            "catchm_dist_km":   catchm_dist,
            "in_catchment":     int(dest_school == catchm_school),
            "owns_car":         owns_car,
        })

    print(f"  Dropped — no dest: {records_dropped['no_dest']}, "
          f"dest not in school list: {records_dropped['chosen_not_in_sl']}, "
          f"too few alts: {records_dropped['too_few_alts']}")
    print(f"  Observations for MNL: {len(X_base)}")

    meta_df = pd.DataFrame(meta)
    return X_base, X_car, chosen, meta_df


# ══════════════════════════════════════════════════════════════════════════════
# D-statistics (choice set descriptives)
# ══════════════════════════════════════════════════════════════════════════════

def choice_set_descriptives(meta_df):
    print("\n  Choice set statistics:")
    print(f"  {'Metric':<30} {'Elem':>10} {'Mid':>10} {'All':>10}")
    print(f"  {'-'*62}")

    for col, label in [
        ("n_alts",           "Alternatives per student"),
        ("chosen_dist_km",   "Chosen school dist (km)"),
        ("nearest_dist_km",  "Nearest school dist (km)"),
        ("catchm_dist_km",   "Catchment dist (km)"),
    ]:
        row_e = meta_df[meta_df["school_level"]=="Elementary"][col].mean()
        row_m = meta_df[meta_df["school_level"]=="Middle"][col].mean()
        row_a = meta_df[col].mean()
        print(f"  {label:<30} {row_e:>10.2f} {row_m:>10.2f} {row_a:>10.2f}")

    in_c = meta_df["in_catchment"].mean()
    print(f"\n  In-catchment rate (observed): {in_c:.1%}")

    stats = meta_df.groupby("school_level").agg(
        n=("chosen_dist_km", "count"),
        mean_alts=("n_alts", "mean"),
        mean_chosen_km=("chosen_dist_km", "mean"),
        pct_incatchment=("in_catchment", "mean"),
    ).round(2)
    print(f"\n{stats.to_string()}")
    stats.to_csv(TABLE_DIR / "P2c_choiceset_stats.csv")
    print("  → Saved: tables/P2c_choiceset_stats.csv")
    return meta_df


# ══════════════════════════════════════════════════════════════════════════════
# Save results
# ══════════════════════════════════════════════════════════════════════════════

def save_model(res, filename):
    df_out = pd.DataFrame({
        "variable": res["features"],
        "coef": res["coef"],
        "se": res["se"],
        "z": res["z"],
        "pval": res["pval"],
    })
    df_out["sig"] = df_out["pval"].apply(
        lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "."
    )
    df_out.loc[len(df_out)] = {
        "variable": "LL(opt)", "coef": res["ll_opt"], "se": np.nan,
        "z": np.nan, "pval": np.nan, "sig": ""
    }
    df_out.loc[len(df_out)] = {
        "variable": "LL(null)", "coef": res["ll_null"], "se": np.nan,
        "z": np.nan, "pval": np.nan, "sig": ""
    }
    df_out.loc[len(df_out)] = {
        "variable": "McFadden_R2", "coef": res["mcf_r2"], "se": np.nan,
        "z": np.nan, "pval": np.nan, "sig": ""
    }
    df_out.loc[len(df_out)] = {
        "variable": "N_obs", "coef": res["n"], "se": np.nan,
        "z": np.nan, "pval": np.nan, "sig": ""
    }
    path = TABLE_DIR / filename
    df_out.to_csv(path, index=False)
    print(f"  → Saved: tables/{filename}")
    return df_out


# ══════════════════════════════════════════════════════════════════════════════
# Counterfactual: assign all students to nearest school
# ══════════════════════════════════════════════════════════════════════════════

def counterfactual_nearest(meta_df):
    """Compare actual chosen distance vs nearest-school assignment."""
    section("Counterfactual: Nearest-School Assignment")

    delta = meta_df["nearest_dist_km"] - meta_df["chosen_dist_km"]
    print(f"  Δ dist (nearest - chosen): mean={delta.mean():.3f} km, "
          f"median={delta.median():.3f} km")
    print(f"  Pct already at nearest school: {(delta.abs()<0.01).mean():.1%}")
    print(f"  Pct would decrease distance: {(delta<-0.05).mean():.1%}")
    print(f"  Pct would increase distance: {(delta>0.05).mean():.1%}")

    cf_df = meta_df[["school_level", "chosen_dist_km", "nearest_dist_km",
                      "catchm_dist_km", "in_catchment", "owns_car"]].copy()
    cf_df["delta_km"] = delta
    cf_df.to_csv(TABLE_DIR / "P2c_counterfactual.csv", index=False)
    print("  → Saved: tables/P2c_counterfactual.csv")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Distribution comparison
    ax = axes[0]
    bins = np.linspace(0, 15, 31)
    ax.hist(meta_df["chosen_dist_km"].clip(upper=15), bins=bins, alpha=0.6,
            label="Actual", color="#2196F3")
    ax.hist(meta_df["nearest_dist_km"].clip(upper=15), bins=bins, alpha=0.6,
            label="Nearest-school CF", color="#FF5722")
    ax.set_xlabel("Distance to school (km)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Actual vs Nearest-School Distance Distribution", fontsize=10)
    ax.legend(fontsize=9)
    ax.annotate(f"Actual mean: {meta_df['chosen_dist_km'].mean():.2f} km\n"
                f"Nearest mean: {meta_df['nearest_dist_km'].mean():.2f} km",
                xy=(0.97, 0.97), xycoords="axes fraction", ha="right", va="top",
                fontsize=8, bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    # Delta by in-catchment
    ax = axes[1]
    groups = {
        "In-catchment": cf_df[cf_df["in_catchment"]==1]["delta_km"],
        "Out-of-catchment": cf_df[cf_df["in_catchment"]==0]["delta_km"],
    }
    colors = ["#4CAF50", "#E91E63"]
    for (label, vals), color in zip(groups.items(), colors):
        ax.hist(vals.clip(-10, 10), bins=30, alpha=0.6, label=label, color=color)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Δ distance (nearest − chosen, km)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Distance Change under Nearest-School Assignment", fontsize=10)
    ax.legend(fontsize=9)

    fig.suptitle("Counterfactual: All Students Attend Nearest School\n(Okinawa PT Survey R5)",
                 fontsize=11)
    fig.tight_layout()
    save_fig(fig, "P2c_counterfactual")

    return cf_df


# ══════════════════════════════════════════════════════════════════════════════
# Coefficient plot
# ══════════════════════════════════════════════════════════════════════════════

def plot_coefficients(res_base, res_car):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, res, title in zip(axes,
                               [res_base, res_car],
                               ["Model 1: Base", "Model 2: Car-Ownership Interactions"]):
        feats = res["features"]
        coefs = res["coef"]
        ses   = res["se"]
        ci95  = 1.96 * ses

        y = np.arange(len(feats))
        ax.barh(y, coefs, xerr=ci95, color=["#E53935" if c < 0 else "#1E88E5" for c in coefs],
                alpha=0.8, height=0.5, ecolor="#555", capsize=4)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(feats, fontsize=9)
        ax.set_xlabel("Coefficient (95% CI)", fontsize=9)
        ax.set_title(f"{title}\nMcF-R²={res['mcf_r2']:.4f}, N={res['n']}", fontsize=9)
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Conditional Logit: School Choice Coefficients", fontsize=11)
    fig.tight_layout()
    save_fig(fig, "P2c_mnl_coef")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "█" * 70)
    print("  06_mnl_school_choice.py — MNL School Choice Model (#72 P2c)")
    print("█" * 70)

    section("Load Data")
    df, sl_elem, sl_mid = load_data()
    if df is None:
        print("  ⚠️  Missing dependencies — exiting.")
        return

    section("Build Choice Sets")
    print(f"  Radius: {CHOICE_RADIUS_KM} km, min alternatives: {MIN_ALTERNATIVES}")
    X_base, X_car, chosen, meta_df = build_choice_data(df, sl_elem, sl_mid, CHOICE_RADIUS_KM)

    section("D-statistics: Choice Set Descriptives")
    choice_set_descriptives(meta_df)

    section("Model 1: Base Conditional Logit (log_dist + in_catchment)")
    feat_base = ["log_dist", "in_catchment"]
    res_base = fit_clogit(X_base, chosen, feat_base, "Base Model")
    save_model(res_base, "P2c_mnl_base.csv")

    section("Model 2: Car-Ownership Interactions")
    feat_car = ["log_dist", "in_catchment", "car×log_dist", "car×in_catchment"]
    res_car = fit_clogit(X_car, chosen, feat_car, "Car-Interaction Model")
    save_model(res_car, "P2c_mnl_carinteract.csv")

    section("LR Test: Base vs Car-Interaction")
    lr_stat = 2 * (res_car["ll_opt"] - res_base["ll_opt"])
    lr_p = 1 - chi2.cdf(lr_stat, df=2)
    print(f"  LR statistic = {lr_stat:.3f} (df=2), p = {lr_p:.4f}")
    if lr_p < 0.05:
        print("  → Car-interaction terms significantly improve fit.")
    else:
        print("  → Car-interaction terms do NOT improve fit significantly.")

    section("Counterfactual: Nearest-School Assignment")
    counterfactual_nearest(meta_df)

    section("Figures")
    plot_coefficients(res_base, res_car)

    section("Summary")
    print(f"  Model 1  coef(log_dist)={res_base['coef'][0]:.4f} "
          f"p={res_base['pval'][0]:.4f}")
    print(f"           coef(catchment)={res_base['coef'][1]:.4f} "
          f"p={res_base['pval'][1]:.4f}")
    print(f"           McF-R²={res_base['mcf_r2']:.4f}, N={res_base['n']}")
    print(f"\n  Interpretation:")
    beta_d = res_base["coef"][0]
    beta_c = res_base["coef"][1]
    dist_effect = (np.exp(beta_d * np.log(2)) - 1) * 100
    print(f"    • Doubling school distance → {dist_effect:+.1f}% change in utility")
    catchment_odds = np.exp(beta_c)
    print(f"    • In-catchment odds ratio = {catchment_odds:.2f} (vs out-of-catchment)")
    print(f"\n  Next: #15 Merge Gap 1 + Gap 2 into full document\n")


if __name__ == "__main__":
    main()
