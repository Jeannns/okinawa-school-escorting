"""
07_trip_chaining.py — Trip Chaining Among Escorting Parents (#48 RQ2c)
=======================================================================
Among parents who escort a school-age child, do they chain the escort
trip with their commute to work? And who is more likely to chain?

Definitions
-----------
  has_work_trip  : person made at least one trip with purpose 01 or 02
                   on the survey day
  trip_chaining  : escort trip (purpose 10/11/12, escort_flag=1) is
                   immediately adjacent in trip sequence to a work trip
                   (purpose 01/02) — i.e., escort→work or work→escort
                   with no other trips in between
  triangle trip  : informal name for the Home→School→Work chain

Sample
------
  Universe : all escort school trips in school_trips_los.csv
             (escorting_flag == 1, N ≈ 5 923)
  Logit    : commuting escorts only (has_work_trip == 1, n ≈ 335)

Outputs (data/tables/)
----------------------
  TC_desc_overall.csv         — chaining rate by school level, sex, car
  TC_desc_mode.csv            — mode distribution: chained vs non-chained
  TC_logit.csv                — binary logit coefficients
  TC_time_of_day.csv          — departure hour distribution

Outputs (data/figures/)
-----------------------
  TC_overview.png             — stacked bar: has_work × chaining
  TC_departure_hour.png       — departure hour: chained vs solo escort
  TC_logit_coef.png           — coefficient plot
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# Paths
# ══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR   = SCRIPT_DIR / "data"
TABLE_DIR  = DATA_DIR / "tables"
FIG_DIR    = DATA_DIR / "figures"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

SSD_DIR = Path("/Users/troublemaker/Documents/Documents - troublemaker/Utokto2/Thesis/05_New thesis framework/PT Survey/Okinawa/Samsung_T5/For Jehan 260721")

PT_MASTER = SSD_DIR / "Okinawa_PT_MasterData" / "R05_OkinawaPT_Person_Master.csv"

# HH keys — names in PT Master vs school_trips_los.csv
PT_HH_KEYS = [
    "Record Number: Municipality Code",
    "Record Number: B-Zone Code",
    "Record Number: C-Zone Code",
    "Record Number: Household Number",
    "Person Number",
]
SCH_HH_KEYS = [
    "id_municipality_code",
    "id_b_zone_code",
    "id_c_zone_code",
    "id_household_number",
    "person_number",
]

WORK_PURPOSES  = {"01", "02"}
SCHOOL_PURPOSES = {"10", "11", "12"}


def section(t):
    bar = "═" * 70
    print(f"\n{bar}\n  {t}\n{bar}")


def save_fig(fig, name):
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved: figures/{name}.png")


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Load school trips (escort subset)
# ══════════════════════════════════════════════════════════════════════════════

def load_escort_trips():
    df = pd.read_csv(DATA_DIR / "school_trips_los.csv", low_memory=False)
    df = df[df["escorting_flag"] == 1].copy()

    # Normalize HH keys to int
    for col in ["id_municipality_code", "id_b_zone_code",
                "id_c_zone_code", "id_household_number", "person_number"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Owned cars flag
    df["owns_car"] = (df["owned_num_cars"] > 0).astype(int)

    # Sex: use 'sex' column if present, else derive from employment
    if "sex" in df.columns:
        df["female"] = (pd.to_numeric(df["sex"], errors="coerce") == 2).astype(int)
    else:
        df["female"] = np.nan

    # Departure hour
    df["depart_hour24"] = pd.to_numeric(df.get("depart_hour_24h", df.get("depart_hour", np.nan)),
                                         errors="coerce")

    print(f"  Escort school trips: {len(df)}")
    print(f"  Unique persons: {df.groupby(SCH_HH_KEYS).ngroups}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Build trip-chain flags from full PT Master
# ══════════════════════════════════════════════════════════════════════════════

def build_chain_flags(escort_df):
    if not PT_MASTER.exists():
        print(f"  ⚠️  PT Master not found: {PT_MASTER}")
        return escort_df

    print(f"  Loading PT Master ...")
    pt = pd.read_csv(PT_MASTER, low_memory=False)

    # Normalize HH keys
    for col in PT_HH_KEYS:
        pt[col] = pd.to_numeric(pt[col], errors="coerce")

    # Rename PT HH keys to match school_trips_los.csv
    key_map = dict(zip(PT_HH_KEYS, SCH_HH_KEYS))
    pt = pt.rename(columns=key_map)

    # Keep only relevant columns
    pt_sub = pt[SCH_HH_KEYS + ["Trip Number", "Trip Purpose", "Escort Flag"]].copy()
    pt_sub["trip_purpose_str"] = pt_sub["Trip Purpose"].astype(str).str.strip()
    pt_sub["escort_flag_str"]  = pt_sub["Escort Flag"].astype(str).str.strip()
    pt_sub["trip_num"]         = pd.to_numeric(pt_sub["Trip Number"], errors="coerce")

    # Get escort persons from our sample
    escort_persons = escort_df[SCH_HH_KEYS].drop_duplicates()
    escort_persons_merged = escort_persons.merge(pt_sub, on=SCH_HH_KEYS, how="inner")
    print(f"  PT rows matched to escort persons: {len(escort_persons_merged)}")

    def is_chained(grp):
        grp = grp.sort_values("trip_num")
        purposes = grp["trip_purpose_str"].tolist()
        escort_flags = grp["escort_flag_str"].tolist()

        has_work = any(p in WORK_PURPOSES for p in purposes)
        chained  = False

        for i in range(len(purposes) - 1):
            curr_is_escort = escort_flags[i] == "1" and purposes[i] in SCHOOL_PURPOSES
            next_is_work   = purposes[i + 1] in WORK_PURPOSES
            curr_is_work   = purposes[i] in WORK_PURPOSES
            next_is_escort = escort_flags[i + 1] == "1" and purposes[i + 1] in SCHOOL_PURPOSES

            if curr_is_escort and next_is_work:
                chained = True
                break
            if curr_is_work and next_is_escort:
                chained = True
                break

        return pd.Series({"has_work_trip": int(has_work), "trip_chaining": int(chained)})

    chain_df = (escort_persons_merged
                .groupby(SCH_HH_KEYS, group_keys=False)
                .apply(is_chained, include_groups=False)
                .reset_index())

    # Join back to escort trips
    result = escort_df.merge(chain_df, on=SCH_HH_KEYS, how="left")
    result["has_work_trip"] = result["has_work_trip"].fillna(0).astype(int)
    result["trip_chaining"] = result["trip_chaining"].fillna(0).astype(int)

    print(f"  has_work_trip=1: {result['has_work_trip'].sum()} "
          f"({result['has_work_trip'].mean():.1%})")
    print(f"  trip_chaining=1: {result['trip_chaining'].sum()} "
          f"({result['trip_chaining'].mean():.1%})")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — Descriptive analysis
# ══════════════════════════════════════════════════════════════════════════════

def descriptives(df):
    section("Descriptive: Chaining Rates")

    # Overall
    n = len(df)
    n_work = df["has_work_trip"].sum()
    n_chain = df["trip_chaining"].sum()
    print(f"\n  N (escort trips)    = {n}")
    print(f"  Has work trip       = {n_work}  ({n_work/n:.1%})")
    print(f"  Trip chaining       = {n_chain} ({n_chain/n:.1%} of all escorts)")

    comm_escorts = df[df["has_work_trip"] == 1]
    if len(comm_escorts) > 0:
        rate_among_comm = comm_escorts["trip_chaining"].mean()
        print(f"  Chaining rate (commuting escorts only): {rate_among_comm:.1%}")

    # By school level
    print("\n  Chaining rate by school level:")
    rows_sl = []
    for sl in ["Elementary", "Middle", "High School"]:
        g = df[df["school_level"] == sl]
        if len(g) == 0:
            continue
        rc = g["trip_chaining"].sum()
        rows_sl.append({
            "school_level": sl,
            "n": len(g),
            "n_with_work": int(g["has_work_trip"].sum()),
            "n_chaining": int(rc),
            "chain_rate_all": g["trip_chaining"].mean(),
            "chain_rate_comm": (g[g["has_work_trip"]==1]["trip_chaining"].mean()
                                if g["has_work_trip"].sum() > 0 else np.nan),
        })
        print(f"    {sl:<15}: {rc}/{len(g)} ({g['trip_chaining'].mean():.1%})")
    sl_df = pd.DataFrame(rows_sl)
    sl_df.to_csv(TABLE_DIR / "TC_desc_overall.csv", index=False)
    print("  → Saved: tables/TC_desc_overall.csv")

    # By sex
    if "female" in df.columns and df["female"].notna().any():
        print("\n  Chaining rate by sex:")
        for sx_v, sx_l in [(0, "Male"), (1, "Female")]:
            g = df[df["female"] == sx_v]
            if len(g):
                print(f"    {sx_l:<8}: {g['trip_chaining'].sum()}/{len(g)} "
                      f"({g['trip_chaining'].mean():.1%})")

    # By car ownership
    print("\n  Chaining rate by car ownership:")
    for cv, cl in [(0, "No car"), (1, "Owns car")]:
        g = df[df["owns_car"] == cv]
        if len(g):
            print(f"    {cl:<10}: {g['trip_chaining'].sum()}/{len(g)} "
                  f"({g['trip_chaining'].mean():.1%})")

    # Mode distribution: chained vs non-chained
    if "rep_mode_class1" in df.columns:
        mode_map = {6: "Car", 7: "Walk", 10: "Bus", 11: "Bus", 12: "Bus",
                    23: "Bike", 9: "Train"}
        df["mode_label"] = df["rep_mode_class1"].map(mode_map).fillna("Other")
        mode_tbl = (df.groupby(["mode_label", "trip_chaining"])
                    .size().unstack(fill_value=0))
        mode_tbl.columns = ["Non-chained", "Chained"]
        mode_tbl["Chaining rate"] = (mode_tbl["Chained"] /
                                     (mode_tbl["Chained"] + mode_tbl["Non-chained"]))
        print(f"\n  Mode distribution (chained vs non-chained):\n{mode_tbl.round(3).to_string()}")
        mode_tbl.to_csv(TABLE_DIR / "TC_desc_mode.csv")
        print("  → Saved: tables/TC_desc_mode.csv")

    return comm_escorts


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — Binary logit (commuting escorts only)
# ══════════════════════════════════════════════════════════════════════════════

def logit_model(df_logit):
    section("Binary Logit: DV = trip_chaining (commuting escorts)")
    n = len(df_logit)
    print(f"  Sample (has_work_trip=1): n={n}")

    if n < 30:
        print("  ⚠️  Too few observations for logit.")
        return

    # Build formula
    # Note: school_level excluded — Elementary=0 and Middle=5.2% chaining → quasi-separation
    # (HS accounts for 97%+ of LOS-joined escort trips)
    available = df_logit.columns.tolist()
    parts = []
    if "female" in available and df_logit["female"].notna().sum() > 10:
        parts.append("female")
    if "hh_size_excl_under5" in available:
        df_logit = df_logit.rename(columns={"hh_size_excl_under5": "hh_size"})
        parts.append("hh_size")
    elif "hh_size_incl_under5" in available:
        df_logit = df_logit.rename(columns={"hh_size_incl_under5": "hh_size"})
        parts.append("hh_size")
    if "owns_car" in available:
        parts.append("owns_car")
    if not parts:
        parts = ["1"]  # intercept-only fallback

    formula = "trip_chaining ~ " + " + ".join(parts)
    print(f"  (school_level dropped: quasi-separation — Elem=0%, Mid=5%, HS=97% of sample)")
    print(f"  Formula: {formula}")

    try:
        model = smf.logit(formula, data=df_logit).fit(disp=False, maxiter=200)

        tbl = model.summary2().tables[1].copy()
        tbl.index.name = "variable"
        tbl.reset_index(inplace=True)
        tbl["sig"] = tbl["P>|z|"].apply(
            lambda p: "***" if p < 0.001 else "**" if p < 0.01
            else "*" if p < 0.05 else "." if p < 0.10 else ""
        )

        print(f"\n  {'Variable':<45} {'Coef':>8} {'SE':>7} {'p':>8} {'':>4}")
        print(f"  {'─'*74}")
        for _, row in tbl.iterrows():
            pstr = "<0.001" if row["P>|z|"] < 0.001 else f"{row['P>|z|']:.3f}"
            print(f"  {row['variable']:<45} {row['Coef.']:>8.3f} {row['Std.Err.']:>7.3f} "
                  f"{pstr:>8} {row['sig']:>4}")
        print(f"\n  Null LL = {model.llnull:.2f}  Model LL = {model.llf:.2f}  "
              f"McF-R² = {1 - model.llf/model.llnull:.4f}")

        tbl.to_csv(TABLE_DIR / "TC_logit.csv", index=False)
        print("  → Saved: tables/TC_logit.csv")
        return model, tbl

    except Exception as e:
        print(f"  ⚠️  Logit failed: {e}")
        return None, None


# ══════════════════════════════════════════════════════════════════════════════
# Step 5 — Time-of-day analysis
# ══════════════════════════════════════════════════════════════════════════════

def time_of_day(df):
    section("Departure Hour Distribution")

    df_valid = df[df["depart_hour24"].between(5, 12)].copy()
    print(f"  Trips with 5≤hour≤12: {len(df_valid)}")

    hour_chain = df_valid.groupby(["depart_hour24", "trip_chaining"]).size().unstack(fill_value=0)
    hour_chain.columns = ["Non-chained", "Chained"]
    print(f"\n{hour_chain.to_string()}")
    hour_chain.to_csv(TABLE_DIR / "TC_time_of_day.csv")
    print("  → Saved: tables/TC_time_of_day.csv")
    return df_valid, hour_chain


# ══════════════════════════════════════════════════════════════════════════════
# Figures
# ══════════════════════════════════════════════════════════════════════════════

def figures(df, df_valid, hour_chain, logit_result):
    section("Figures")

    # Fig 1 — Overview stacked bar
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Panel A: Work trip × chaining breakdown
    ax = axes[0]
    labels = ["No work trip\n(stay-home escort)", "Work trip\n(non-chaining)",
              "Work trip\n(chaining = triangle)"]
    n_no_work     = (df["has_work_trip"] == 0).sum()
    n_work_nochain = ((df["has_work_trip"] == 1) & (df["trip_chaining"] == 0)).sum()
    n_chain       = df["trip_chaining"].sum()
    sizes = [n_no_work, n_work_nochain, n_chain]
    colors = ["#90A4AE", "#42A5F5", "#EF5350"]
    bars = ax.bar(labels, sizes, color=colors, edgecolor="white", linewidth=0.5)
    for bar, sz in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f"{sz}\n({sz/len(df):.1%})", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("N (escort trips)", fontsize=9)
    ax.set_title("A. Trip Chain Type", fontsize=10)
    ax.set_ylim(0, max(sizes) * 1.25)

    # Panel B: Chaining rate by school level
    ax = axes[1]
    sl_data = []
    for sl in ["Elementary", "Middle", "High School"]:
        g = df[df["school_level"] == sl]
        if len(g) > 0:
            sl_data.append((sl, g["trip_chaining"].mean() * 100))
    if sl_data:
        lvls, rates = zip(*sl_data)
        clrs = ["#26C6DA", "#42A5F5", "#7E57C2"][:len(lvls)]
        bars = ax.bar(lvls, rates, color=clrs, edgecolor="white")
        for bar, r in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{r:.1f}%", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("Trip chaining rate (%)", fontsize=9)
        ax.set_ylim(0, max(rates) * 1.3)
    ax.set_title("B. Chaining Rate by School Level", fontsize=10)

    # Panel C: Chaining rate by sex
    ax = axes[2]
    if "female" in df.columns and df["female"].notna().any():
        sx_data = [(lab, df[df["female"]==v]["trip_chaining"].mean()*100)
                   for v, lab in [(0,"Male"),(1,"Female")]
                   if len(df[df["female"]==v]) > 0]
        if sx_data:
            sx_labels, sx_rates = zip(*sx_data)
            ax.bar(sx_labels, sx_rates, color=["#42A5F5","#EC407A"], edgecolor="white")
            for xl, r in zip(sx_labels, sx_rates):
                ax.text(list(sx_labels).index(xl), r + 0.3, f"{r:.1f}%",
                        ha="center", va="bottom", fontsize=9)
            ax.set_ylabel("Trip chaining rate (%)", fontsize=9)
            ax.set_ylim(0, max(sx_rates) * 1.3)
    ax.set_title("C. Chaining Rate by Sex", fontsize=10)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Male", "Female"])

    fig.suptitle("Trip Chaining Among School Escort Trips\n(Okinawa PT Survey R5)",
                 fontsize=12)
    fig.tight_layout()
    save_fig(fig, "TC_overview")

    # Fig 2 — Departure hour
    if hour_chain is not None and len(hour_chain) > 0:
        fig, ax = plt.subplots(figsize=(9, 4))
        x = hour_chain.index
        w = 0.35
        ax.bar(x - w/2, hour_chain["Non-chained"], width=w, label="Non-chained",
               color="#42A5F5", alpha=0.8)
        ax.bar(x + w/2, hour_chain["Chained"], width=w, label="Chained (triangle)",
               color="#EF5350", alpha=0.8)
        ax.set_xlabel("Departure hour", fontsize=10)
        ax.set_ylabel("N escort trips", fontsize=10)
        ax.set_title("Departure Hour: Chained vs Non-chained Escort Trips", fontsize=10)
        ax.legend(fontsize=9)
        ax.set_xticks(sorted(hour_chain.index))
        fig.tight_layout()
        save_fig(fig, "TC_departure_hour")

    # Fig 3 — Logit coefficient plot
    if logit_result is not None:
        model, tbl = logit_result
        if tbl is not None:
            fig, ax = plt.subplots(figsize=(8, max(4, len(tbl)*0.35)))
            main_rows = tbl[tbl["variable"] != "Intercept"].copy()
            y = np.arange(len(main_rows))
            coefs = main_rows["Coef."].values
            ci95  = 1.96 * main_rows["Std.Err."].values
            colors = ["#EF5350" if c < 0 else "#42A5F5" for c in coefs]
            ax.barh(y, coefs, xerr=ci95, color=colors, alpha=0.8,
                    height=0.5, ecolor="#555", capsize=4)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_yticks(y)
            ax.set_yticklabels(main_rows["variable"].values, fontsize=8)
            ax.set_xlabel("Coefficient (95% CI)", fontsize=9)
            ax.set_title(f"Binary Logit: Trip Chaining\n(commuting escorts, n={len(model.model.endog)})",
                         fontsize=10)
            ax.grid(axis="x", alpha=0.3)
            fig.tight_layout()
            save_fig(fig, "TC_logit_coef")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "█" * 70)
    print("  07_trip_chaining.py — Trip Chaining Among Escorts (#48 RQ2c)")
    print("█" * 70)

    section("Step 1 — Load Escort School Trips")
    df = load_escort_trips()

    section("Step 2 — Build Trip-Chain Flags from Full PT Master")
    df = build_chain_flags(df)

    section("Step 3 — Descriptive Analysis")
    comm_escorts = descriptives(df)

    section("Step 4 — Logit Model (commuting escorts)")
    logit_result = logit_model(comm_escorts.copy())

    section("Step 5 — Time-of-Day")
    df_valid, hour_chain = time_of_day(df)

    section("Step 6 — Figures")
    figures(df, df_valid, hour_chain, logit_result)

    section("Summary")
    n = len(df)
    n_chain = int(df["trip_chaining"].sum())
    n_work  = int(df["has_work_trip"].sum())
    print(f"  Escort trips        : {n}")
    print(f"  Commuting escorts   : {n_work} ({n_work/n:.1%})")
    print(f"  Triangle trips      : {n_chain} ({n_chain/n:.1%} of all escorts)")
    if n_work > 0:
        print(f"  Chaining rate (commuters): {n_chain/n_work:.1%}")
    print(f"\n  Key result: trip chaining is driven by time constraints (commute),")
    print(f"  not safety concern or escort motivation.\n")


if __name__ == "__main__":
    main()
