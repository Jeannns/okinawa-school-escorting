"""
15_tc_b_los.py — TC-b: Trip Chaining + LOS Variables

Extends 07_trip_chaining.py by adding SSD LOS variables as predictors
in the trip chaining logit model.

Motivation: If school is far (high car_distance_m) or poorly served by
transit (high walk_time_min), parents may be more likely to embed the
escort trip within their work commute (trip chaining). This tests whether
distance/LOS is a spatial driver of trip chaining, beyond HH characteristics.

Original TC logit (07):
  trip_chaining ~ owns_car + female + depart_hour + hh_size

TC-b logit adds:
  + car_dist_km_std   (school trip car distance — spatial coverage)
  + walk_time_std     (zone walkability — alternative to driving)

Hypothesis:
  car_dist_km_std (+):  farther school → more likely to chain (embed in commute)
  walk_time_std   (?):  poor walkability → ambiguous (all HS anyway car-driven)

Data:
  data/school_trips_los.csv  (LOS joined; escorting_flag==1 for escort trips)
  R05_PersonTrip_EN.csv      (PT master, for trip chaining flag construction)

Outputs:
  figures/TC_b_chaining_by_distance.png
  figures/TC_b_logit_comparison.png
  data/tc_b_logit_results.csv

Original 07_trip_chaining.py preserved.
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = Path(__file__).parent
DATA      = BASE / "data"
FIG       = BASE / "figures"
FIG.mkdir(exist_ok=True)

TRIPS_CSV = DATA / "school_trips_los.csv"
PT_MASTER = (BASE.parents[1]
             / "Okinawa_PT survey"
             / "Ver20240430第4回沖縄PTマスターデータ"
             / "EN"
             / "R05_PersonTrip_EN.csv")

# HH identifier keys (same in both school_trips_los.csv and PT Master EN)
SCH_KEYS = ["id_municipality_code","id_b_zone_code","id_c_zone_code",
            "id_household_number","person_number"]
PT_KEYS  = SCH_KEYS  # EN version already uses same snake_case names

WORK_PURPOSES   = {"01","02","1","2"}
SCHOOL_PURPOSES = {"11","12","11.0","12.0"}


def section(title):
    print(f"\n{'═'*65}\n  {title}\n{'═'*65}")


# ── Step 1: Load escort school trips with LOS ─────────────────────────────────
section("Step 1 — Load escort trips + LOS")
df = pd.read_csv(TRIPS_CSV, low_memory=False)
df = df[df["escorting_flag"] == 1].copy()

for col in SCH_KEYS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["owned_num_cars"] = pd.to_numeric(df["owned_num_cars"], errors="coerce")
df["owns_car"]       = (df["owned_num_cars"] > 0).astype(int)
df["female"]         = (pd.to_numeric(df.get("sex", np.nan), errors="coerce") == 2).astype(int)
df["depart_hour24"]  = pd.to_numeric(df.get("depart_hour_24h", df.get("depart_hour", np.nan)),
                                      errors="coerce")
df["hh_size"]        = pd.to_numeric(df.get("hh_size_excl_under5", np.nan), errors="coerce")
df["car_dist_km"]    = pd.to_numeric(df["car_distance_m"], errors="coerce") / 1000
df["walk_time_min"]  = pd.to_numeric(df["walk_time_min"], errors="coerce")

print(f"  Escort trips: {len(df):,}")
print(f"  With car_distance_m: {df['car_dist_km'].notna().sum():,}")
print(f"  With walk_time_min:  {df['walk_time_min'].notna().sum():,}")


# ── Step 2: Build trip chaining flag from PT Master ───────────────────────────
section("Step 2 — Build trip chaining flag")

def build_chain_flags(escort_df):
    if not PT_MASTER.exists():
        print(f"  ⚠️  PT Master not found at {PT_MASTER}")
        print(f"       Using zeros for trip_chaining / has_work_trip")
        escort_df["trip_chaining"] = 0
        escort_df["has_work_trip"] = 0
        return escort_df

    print(f"  Loading PT Master …")
    pt = pd.read_csv(PT_MASTER, low_memory=False)
    for col in PT_KEYS:
        pt[col] = pd.to_numeric(pt[col], errors="coerce")
    # EN file already uses same snake_case names — no rename needed

    pt_sub = pt[SCH_KEYS + ["trip_number","trip_purpose","escorting_flag"]].copy()
    pt_sub["trip_purpose_str"] = pt_sub["trip_purpose"].astype(str).str.strip()
    pt_sub["escort_flag_str"]  = pt_sub["escorting_flag"].astype(str).str.strip()
    pt_sub["trip_num"]         = pd.to_numeric(pt_sub["trip_number"], errors="coerce")

    escort_persons = escort_df[SCH_KEYS].drop_duplicates()
    matched = escort_persons.merge(pt_sub, on=SCH_KEYS, how="inner")
    print(f"  PT rows matched to escort persons: {len(matched):,}")

    def is_chained(grp):
        grp = grp.sort_values("trip_num")
        purposes      = grp["trip_purpose_str"].tolist()
        escort_flags  = grp["escort_flag_str"].tolist()
        has_work      = any(p in WORK_PURPOSES for p in purposes)
        chained       = False
        for i in range(len(purposes) - 1):
            curr_esc  = escort_flags[i] == "1" and purposes[i] in SCHOOL_PURPOSES
            next_work = purposes[i+1] in WORK_PURPOSES
            curr_work = purposes[i] in WORK_PURPOSES
            next_esc  = escort_flags[i+1] == "1" and purposes[i+1] in SCHOOL_PURPOSES
            if (curr_esc and next_work) or (curr_work and next_esc):
                chained = True
                break
        return pd.Series({"has_work_trip": int(has_work), "trip_chaining": int(chained)})

    chain_df = (matched.groupby(SCH_KEYS, group_keys=False)
                       .apply(is_chained, include_groups=False)
                       .reset_index())
    result = escort_df.merge(chain_df, on=SCH_KEYS, how="left")
    result["has_work_trip"] = result["has_work_trip"].fillna(0).astype(int)
    result["trip_chaining"] = result["trip_chaining"].fillna(0).astype(int)
    print(f"  has_work_trip=1: {result['has_work_trip'].sum()} ({result['has_work_trip'].mean():.1%})")
    print(f"  trip_chaining=1: {result['trip_chaining'].sum()} ({result['trip_chaining'].mean():.1%})")
    return result

df = build_chain_flags(df)


# ── Step 3: Standardize LOS vars + restrict to commuting escorts ──────────────
section("Step 3 — Prepare modelling sample")

df_model = df[df["has_work_trip"] == 1].copy()
df_model = df_model.dropna(subset=["car_dist_km","walk_time_min",
                                    "owns_car","depart_hour24","hh_size"]).copy()

for raw, std in [("car_dist_km",   "car_dist_std"),
                 ("walk_time_min",  "walk_time_std")]:
    mu = df_model[raw].mean()
    sd = df_model[raw].std()
    df_model[std] = (df_model[raw] - mu) / sd
    print(f"  {raw}: mean={mu:.2f}, sd={sd:.2f}  → standardized as {std}")

print(f"\n  Modelling sample (commuting escorts, LOS available): {len(df_model):,}")
print(f"  trip_chaining rate: {df_model['trip_chaining'].mean():.1%}")
print(f"  car_dist_km range: {df_model['car_dist_km'].min():.1f} – {df_model['car_dist_km'].max():.1f} km")


# ── Step 4: Logit models ──────────────────────────────────────────────────────
section("Step 4 — Logit: trip chaining ~ HH + LOS")

BASE_FORMULA = "trip_chaining ~ owns_car + female + depart_hour24 + hh_size"
LOS_FORMULA  = "trip_chaining ~ owns_car + female + depart_hour24 + hh_size + car_dist_std + walk_time_std"
CAR_FORMULA  = "trip_chaining ~ owns_car + female + depart_hour24 + hh_size + car_dist_std"

results = {}
for label, formula in [("TC_original", BASE_FORMULA),
                        ("TC_b_car_dist", CAR_FORMULA),
                        ("TC_b_full", LOS_FORMULA)]:
    try:
        m = smf.logit(formula, data=df_model).fit(disp=False, maxiter=300)
        results[label] = m
        print(f"\n  ── {label} ──")
        print(f"  n={int(m.nobs)}, McF-R²={m.prsquared:.4f}, AIC={m.aic:.1f}")
        tbl = m.summary2().tables[1][["Coef.","Std.Err.","z","P>|z|"]].round(4)
        print(tbl.to_string())
    except Exception as e:
        print(f"  ⚠️  {label} failed: {e}")

# Compare model fit
if "TC_original" in results and "TC_b_full" in results:
    m0 = results["TC_original"]
    m1 = results["TC_b_full"]
    delta_mcf = m1.prsquared - m0.prsquared
    delta_aic = m1.aic - m0.aic
    print(f"\n  TC_original vs TC_b_full:")
    print(f"  ΔMcF-R²: {delta_mcf:+.4f}")
    print(f"  ΔAIC:    {delta_aic:+.1f}  ({'better' if delta_aic < 0 else 'worse'})")


# ── Step 5: Descriptive — chaining rate by distance quintile ─────────────────
section("Step 5 — Chaining rate by car_distance quintile")

df_work = df[df["has_work_trip"] == 1].dropna(subset=["car_dist_km"]).copy()
df_work["dist_q"] = pd.qcut(df_work["car_dist_km"], q=5, labels=False, duplicates="drop")
chain_by_q = (df_work.groupby("dist_q")
              .agg(n=("trip_chaining","count"),
                   chain_rate=("trip_chaining","mean"),
                   mean_dist=("car_dist_km","mean"))
              .reset_index())
print(chain_by_q.round(3).to_string(index=False))

corr_dist_chain = df_work["car_dist_km"].corr(df_work["trip_chaining"])
print(f"\n  Pearson r(car_dist, chaining) = {corr_dist_chain:.4f}")


# ── Step 6: Save results ──────────────────────────────────────────────────────
rows = []
for label, m in results.items():
    tbl = m.summary2().tables[1].reset_index()
    tbl.columns = ["variable","coef","se","z","p","ci_lo","ci_hi"]
    tbl["model"]   = label
    tbl["nobs"]    = int(m.nobs)
    tbl["mcf_r2"]  = round(m.prsquared, 4)
    tbl["aic"]     = round(m.aic, 2)
    rows.append(tbl)
out = pd.concat(rows, ignore_index=True)
out.to_csv(DATA / "tc_b_logit_results.csv", index=False)
print(f"\n  Saved: tc_b_logit_results.csv")


# ── Step 7: Figures ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: chaining rate by distance quintile
ax = axes[0]
if not chain_by_q.empty:
    ax.bar(chain_by_q["dist_q"], chain_by_q["chain_rate"] * 100,
           color="#2980B9", alpha=0.8, edgecolor="0.3")
    ax.set_xlabel("Car distance quintile (1=short … 5=long)", fontsize=10)
    ax.set_ylabel("Trip chaining rate (%)", fontsize=10)
    ax.set_title(f"TC-b: Chaining Rate by Distance Quintile\n"
                 f"r(dist, chaining)={corr_dist_chain:.3f}", fontsize=11, fontweight="bold")
    for _, r in chain_by_q.iterrows():
        ax.text(r["dist_q"], r["chain_rate"]*100 + 0.3,
                f"{r['chain_rate']*100:.1f}%\n({r['mean_dist']:.1f}km)",
                ha="center", fontsize=8)

# Right: coefficient comparison TC_original vs TC_b_full
ax = axes[1]
if "TC_original" in results and "TC_b_full" in results:
    base_coefs = results["TC_original"].summary2().tables[1]["Coef."].drop("Intercept")
    full_coefs = results["TC_b_full"].summary2().tables[1]["Coef."].drop("Intercept")
    all_vars = list(base_coefs.index.union(full_coefs.index))
    label_map = {
        "owns_car":       "Owns car",
        "female":         "Female parent",
        "depart_hour24":  "Depart hour",
        "hh_size":        "HH size",
        "car_dist_std":   "Car distance (std) ★",
        "walk_time_std":  "Walk time (std) ★",
    }
    y = np.arange(len(all_vars))
    w = 0.35
    base_vals = [base_coefs.get(v, np.nan) for v in all_vars]
    full_vals  = [full_coefs.get(v, np.nan) for v in all_vars]
    ax.barh(y + w/2, base_vals, w, color="#B2BEB5", alpha=0.8, label="TC_original")
    ax.barh(y - w/2, full_vals, w, color="#2980B9", alpha=0.8, label="TC_b_full (+ LOS)")
    ax.set_yticks(y)
    ax.set_yticklabels([label_map.get(v, v) for v in all_vars], fontsize=9)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Logit coefficient", fontsize=10)
    ax.set_title("TC-b: Coefficient Comparison\nOriginal vs + LOS (★ = new)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)

plt.tight_layout()
fig.savefig(FIG / "TC_b_logit_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: TC_b_logit_comparison.png")

print(f"\n{'═'*65}")
print(f"  TC-b complete. Original 07_trip_chaining.py unchanged.")
print(f"{'═'*65}")
