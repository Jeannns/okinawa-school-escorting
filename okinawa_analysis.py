"""
Okinawa Person Trip Survey — Analysis
Levels 1–4: Data Quality, Household Profile, Age × Status, Trip Behavior

Data directory: ../Okinawa_PT survey/Ver20240430第4回沖縄PTマスターデータ/EN/
Output:         ./output/Level1-3/   ./output/Level4/   (figures + CSV)
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = (SCRIPT_DIR.parents[1]
            / "Okinawa_PT survey"
            / "Ver20240430第4回沖縄PTマスターデータ"
            / "EN")

OUT_L13 = SCRIPT_DIR / "output" / "Level1-3"
OUT_L4  = SCRIPT_DIR / "output" / "Level4"
OUT_L5  = SCRIPT_DIR / "output" / "Level5"
OUT_L6  = SCRIPT_DIR / "output" / "Level6"
OUT_L7  = SCRIPT_DIR / "output" / "Level7"
OUT_L8  = SCRIPT_DIR / "output" / "Level8"
OUT_L13.mkdir(parents=True, exist_ok=True)
OUT_L4.mkdir(parents=True, exist_ok=True)
OUT_L5.mkdir(parents=True, exist_ok=True)
OUT_L6.mkdir(parents=True, exist_ok=True)
OUT_L7.mkdir(parents=True, exist_ok=True)
OUT_L8.mkdir(parents=True, exist_ok=True)

FILES = {
    "hh":   DATA_DIR / "R05_HH_Survey_EN.csv",
    "pt":   DATA_DIR / "R05_PersonTrip_EN.csv",
    "supp": DATA_DIR / "R05_Supplementary_EN.csv",
}

# ── Code-book lookups ────────────────────────────────────────────────────────
EMP_STATUS_LABELS = {
    1:  "Full-time employed",
    2:  "Part-time employed",
    3:  "Self-employed",
    4:  "Family business",
    5:  "Company officer",
    6:  "Side job",
    7:  "Unemployed / homemaker",
    8:  "University student",
    9:  "High school student",
    10: "Elementary / middle school",
    11: "Kindergarten",
    12: "Pre-school / nursery",
    13: "Other",
    99: "Unknown / not answered",
}
STUDENT_CODES    = {8, 9, 10, 11, 12}
SCHOOL_CODES_ESC = [8, 9, 10, 11, 12, 13]  # used by get_school_escort_trips

INCOME_LABELS = {
    1: "< 2M¥",
    2: "2–4M¥",
    3: "4–6M¥",
    4: "6–8M¥",
    5: "8–10M¥",
    6: "10–15M¥",
    7: "≥ 15M¥",
    9: "Unknown",
}

SEX_LABELS = {1: "Male", 2: "Female", 9: "Unknown"}

HOME_OWN_LABELS = {
    1: "Own (detached)",
    2: "Own (apartment)",
    3: "Rent (detached)",
    4: "Rent (apartment)",
    5: "Public housing",
    6: "Company housing",
    7: "Other",
    9: "Unknown",
}

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})
PALETTE = sns.color_palette("tab20")


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def section(title: str) -> None:
    bar = "═" * 70
    print(f"\n{bar}\n  {title}\n{bar}")


def load_csv(name: str, path: Path) -> pd.DataFrame:
    print(f"  Loading {name} … ", end="", flush=True)
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    df.columns = df.columns.str.strip()
    print(f"{len(df):,} rows × {df.shape[1]} cols")
    return df


def missing_report(df: pd.DataFrame, label: str, out_dir: Path) -> pd.DataFrame:
    total = len(df)
    miss = df.isnull().sum()
    for col in df.select_dtypes(include="object").columns:
        miss[col] += (df[col].str.strip() == "").sum()
    miss_pct = (miss / total * 100).round(2)
    report = pd.DataFrame({"missing_n": miss, "missing_%": miss_pct})
    report = report[report["missing_n"] > 0].sort_values("missing_%", ascending=False)
    report.to_csv(out_dir / f"missing_{label}.csv")
    return report


def save_fig(fig: plt.Figure, name: str, out_dir: Path) -> None:
    path = out_dir / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path.name}")


# ═══════════════════════════════════════════════════════════════════════════
# Data Cleaning
# ═══════════════════════════════════════════════════════════════════════════

def clean_data(pt: pd.DataFrame, hh: pd.DataFrame) -> None:
    """Standardise key columns in-place once, before any analysis begins."""

    def _int_to_zfill2(s: pd.Series) -> pd.Series:
        """Numeric series → zero-padded 2-char string object; NaN → np.nan."""
        num = pd.to_numeric(s, errors="coerce")
        out = pd.Series(np.nan, index=s.index, dtype=object)
        valid = num.notna()
        out[valid] = num[valid].astype(int).astype(str).str.zfill(2)
        return out

    # ── PersonTrip ────────────────────────────────────────────────────────────
    # Strip whitespace from every string column (cleans " 02" → "02", etc.)
    for col in pt.select_dtypes(include="object").columns:
        pt[col] = pt[col].astype(str).str.strip()

    # Person number → zero-padded string: 3 → "03"
    pt["hh_member_person_no"] = _int_to_zfill2(pt["hh_member_person_no"])

    # Escorted-member columns: zero-pad valid entries; "", "nan", "99" → np.nan
    # np.nan used (not pd.NA) so that downstream astype(str) gives "nan",
    # which the Level 5 filter ~isin({"", "nan", "99"}) correctly excludes.
    _ESC_INVALID = {"", "nan", "99"}
    for col in ["escorted_hh_member_no_1", "escorted_hh_member_no_2",
                "escorted_hh_member_no_3"]:
        if col not in pt.columns:
            continue
        s = pt[col]                                        # already stripped above
        valid = s.str.match(r"^\d+$") & ~s.isin(_ESC_INVALID)
        s = s.where(valid, other=np.nan)                   # sentinel values → NaN
        pt[col] = s.where(s.isna(), s.str.zfill(2))       # zero-pad valid digits

    # Numeric fields: employment_student_status fills NaN with 99 (= "Unknown")
    pt["employment_student_status"] = (
        pd.to_numeric(pt["employment_student_status"], errors="coerce")
        .fillna(99).astype(int)
    )
    pt["age"] = pd.to_numeric(pt["age"], errors="coerce")

    # ── HH Survey ─────────────────────────────────────────────────────────────
    for col in hh.select_dtypes(include="object").columns:
        hh[col] = hh[col].astype(str).str.strip()

    # person_number is the HH-file equivalent of hh_member_person_no in PT
    hh["person_number"] = _int_to_zfill2(hh["person_number"])
    hh["age"] = pd.to_numeric(hh["age"], errors="coerce")
    hh["employment_student_status"] = (
        pd.to_numeric(hh["employment_student_status"], errors="coerce")
        .fillna(99).astype(int)
    )

    # ── Confirmation ──────────────────────────────────────────────────────────
    print("\n=== DATA CLEANING COMPLETE ===")
    print(f"[OK] pt:  {len(pt):,} rows, key columns cleaned")
    print(f"[OK] hh:  {len(hh):,} rows, key columns cleaned")
    print(f"[OK] ID format: hh_member_person_no zero-padded in both files")


def get_school_escort_trips(escort_trips: pd.DataFrame,
                            pt_key: pd.DataFrame) -> pd.DataFrame:
    """Return purpose-12 trips where the escorted person (col 1) is a student.

    Looks up escorted_hh_member_no_1 in pt_key (PT-survey-based lookup) rather
    than the HH survey, so only persons who themselves made PT trips are matched.
    Use SCHOOL_CODES_ESC = [8..13] to include 'Other' student edge cases.

    pt_key must have columns ['join_key', 'employment_student_status'] built as:
        HH4_fields + '_' + hh_member_person_no  →  employment_student_status
    """
    et = escort_trips.copy()
    et["join_key"] = (
        et["id_municipality_code"].astype(str) + "_" +
        et["id_b_zone_code"].astype(str) + "_" +
        et["id_c_zone_code"].astype(str) + "_" +
        et["id_household_number"].astype(str) + "_" +
        et["escorted_hh_member_no_1"].astype(str).str.strip()
    )
    lookup = pt_key[["join_key", "employment_student_status"]].drop_duplicates("join_key")
    merged = et.merge(lookup, on="join_key",
                      suffixes=("", "_escorted"), how="left")
    return merged[merged["employment_student_status_escorted"].isin(SCHOOL_CODES_ESC)]


# ═══════════════════════════════════════════════════════════════════════════
# LEVEL 1 — Data Quality
# ═══════════════════════════════════════════════════════════════════════════

def level1_data_quality(hh: pd.DataFrame, pt: pd.DataFrame, supp: pd.DataFrame) -> None:
    section("LEVEL 1 — Data Quality Check")

    # ── 1a. Basic shape & dtype summary ────────────────────────────────────
    print("\n[1a] Row / column counts")
    for label, df in [("HH Survey", hh), ("Person Trip", pt), ("Supplementary", supp)]:
        numeric_cols = df.select_dtypes(include="number").shape[1]
        text_cols    = df.select_dtypes(include="object").shape[1]
        print(f"  {label:<18}: {len(df):>7,} rows  |  {df.shape[1]:>3} cols "
              f"({numeric_cols} numeric, {text_cols} text)")

    # ── 1b. Missing values ──────────────────────────────────────────────────
    print("\n[1b] Missing value summary  (saved to output/missing_*.csv)")
    for label, df in [("hh", hh), ("pt", pt), ("supp", supp)]:
        rep = missing_report(df, label, OUT_L13)
        top5 = rep.head(5)
        if len(top5):
            print(f"\n  Top missing cols in {label.upper()}:")
            for col, row in top5.iterrows():
                print(f"    {col:<45} {row['missing_n']:>7,}  ({row['missing_%']:.1f}%)")
        else:
            print(f"\n  {label.upper()}: no missing values")

    # ── 1c. Duplicate record check ─────────────────────────────────────────
    print("\n[1c] Duplicate records")
    hh_key = ["id_municipality_code", "id_b_zone_code", "id_c_zone_code",
               "id_household_number", "person_number"]
    pt_key  = hh_key + ["trip_number"]
    supp_key = ["id_municipality_code", "id_b_zone_code", "id_c_zone_code",
                "id_household_number", "person_number"]

    for label, df, key in [("HH", hh, hh_key), ("PT", pt, pt_key), ("Supp", supp, supp_key)]:
        dups = df.duplicated(subset=key, keep=False).sum()
        print(f"  {label:<6}: {dups:,} duplicate rows on composite key")

    # ── 1d. Key field ranges ────────────────────────────────────────────────
    print("\n[1d] Key numeric field ranges")
    checks = [
        ("HH age",   hh,   "age",           0, 120),
        ("PT age",   pt,   "age",           0, 120),
        ("HH trips", hh,   "num_trips",     0,  20),
        ("PT trips", pt,   "num_trips",     0,  20),
        ("HH exp",   hh,   "expansion_factor", 1, 5000),
        ("PT exp",   pt,   "expansion_factor", 1, 5000),
    ]
    for label, df, col, lo, hi in checks:
        s = pd.to_numeric(df[col], errors="coerce")
        out_lo = (s < lo).sum()
        out_hi = (s > hi).sum()
        flag = " ⚠" if (out_lo + out_hi) > 0 else ""
        print(f"  {label:<12} {col:<22} min={s.min():.0f}  max={s.max():.0f}"
              f"  out-of-range: {out_lo + out_hi}{flag}")

    # ── 1e. employment_student_status coverage ──────────────────────────────
    print("\n[1e] employment_student_status value distribution (HH Survey)")
    emp_counts = (hh["employment_student_status"]
                  .value_counts()
                  .reset_index()
                  .rename(columns={"employment_student_status": "code", "count": "n"}))
    emp_counts["label"] = emp_counts["code"].map(EMP_STATUS_LABELS).fillna("?")
    emp_counts["pct"] = (emp_counts["n"] / len(hh) * 100).round(2)
    emp_counts = emp_counts.sort_values("code")
    print(emp_counts[["code", "label", "n", "pct"]].to_string(index=False))

    # ── 1f. Expansion factor distribution (quick sanity) ───────────────────
    print("\n[1f] Expansion factor summary (HH Survey)")
    ef = pd.to_numeric(hh["expansion_factor"], errors="coerce")
    print(f"  mean={ef.mean():.1f}  median={ef.median():.1f}  "
          f"min={ef.min():.0f}  max={ef.max():.0f}  "
          f"sum={ef.sum():,.0f} (≈ total represented persons)")

    # ── Plot: missing % heatmap for HH survey (top 20 cols) ─────────────────
    rep_hh = missing_report(hh, "hh_for_plot", OUT_L13)
    if len(rep_hh) > 0:
        top_n = rep_hh.head(20)
        fig, ax = plt.subplots(figsize=(9, max(3, len(top_n) * 0.4)))
        bars = ax.barh(top_n.index[::-1], top_n["missing_%"][::-1], color="#e07b54")
        ax.set_xlabel("Missing (%)")
        ax.set_title("HH Survey — Top columns with missing values", fontweight="bold")
        ax.xaxis.set_major_formatter(mticker.PercentFormatter())
        for bar, val in zip(bars, top_n["missing_%"][::-1]):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=8)
        fig.tight_layout()
        save_fig(fig, "L1_missing_hh", OUT_L13)


# ═══════════════════════════════════════════════════════════════════════════
# LEVEL 2 — Household Profile
# ═══════════════════════════════════════════════════════════════════════════

def level2_household_profile(hh: pd.DataFrame) -> None:
    section("LEVEL 2 — Household Profile")

    # Build one row per household (take first person-row per HH)
    hh_id_cols = ["id_municipality_code", "id_b_zone_code",
                  "id_c_zone_code", "id_household_number"]
    hh_level = hh.drop_duplicates(subset=hh_id_cols).copy()
    print(f"\n  Unique households: {len(hh_level):,}  "
          f"(total person rows: {len(hh):,})")

    # ── 2a. HH size distribution ────────────────────────────────────────────
    print("\n[2a] Household size (incl. under-5)")
    size_dist = (hh_level["hh_size_incl_under5"]
                 .value_counts().sort_index()
                 .rename("count"))
    size_dist_pct = (size_dist / len(hh_level) * 100).round(2)
    size_df = pd.DataFrame({"count": size_dist, "pct": size_dist_pct})
    print(size_df.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # HH size
    ax = axes[0]
    ax.bar(size_dist.index.astype(str), size_dist.values, color="#4c9be8")
    ax.set_xlabel("Household size")
    ax.set_ylabel("Number of households")
    ax.set_title("Household Size Distribution", fontweight="bold")
    for x, y in zip(size_dist.index, size_dist.values):
        ax.text(x - 1, y + 20, f"{y:,}", ha="center", fontsize=7)

    # Annual income
    ax = axes[1]
    inc = (hh_level["annual_income"]
           .map(INCOME_LABELS)
           .fillna("Unknown")
           .value_counts()
           .reindex([INCOME_LABELS[k] for k in sorted(INCOME_LABELS)], fill_value=0))
    ax.bar(range(len(inc)), inc.values, color="#6ab187")
    ax.set_xticks(range(len(inc)))
    ax.set_xticklabels(inc.index, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Number of households")
    ax.set_title("Annual Income Distribution", fontweight="bold")

    fig.suptitle("Household Profile — Okinawa PT Survey R05", fontsize=11, y=1.01)
    fig.tight_layout()
    save_fig(fig, "L2_hh_profile_basic", OUT_L13)

    # ── 2b. Vehicle ownership ────────────────────────────────────────────────
    print("\n[2b] Vehicle ownership (household level)")
    veh_cols = {
        "vehicles_kei_car":            "Kei car",
        "vehicles_passenger_car":      "Passenger car",
        "vehicles_kei_freight":        "Kei freight",
        "vehicles_freight_truck":      "Freight truck",
        "twowheelers_bicycle":         "Bicycle",
        "twowheelers_moped_under50cc": "Moped (≤50cc)",
        "twowheelers_motorcycle_over50cc": "Motorcycle (>50cc)",
    }
    for col, label in veh_cols.items():
        s = pd.to_numeric(hh_level[col], errors="coerce").fillna(0)
        has = (s > 0).sum()
        total_units = s.sum()
        print(f"  {label:<30} owned by {has:>6,} HH ({has/len(hh_level)*100:.1f}%)  "
              f"total units: {int(total_units):,}")

    # Total cars per household
    hh_level = hh_level.copy()
    hh_level["total_cars"] = (
        pd.to_numeric(hh_level["vehicles_kei_car"], errors="coerce").fillna(0) +
        pd.to_numeric(hh_level["vehicles_passenger_car"], errors="coerce").fillna(0)
    )
    car_dist = hh_level["total_cars"].clip(upper=6).value_counts().sort_index()
    car_dist.index = [str(int(x)) if x < 6 else "6+" for x in car_dist.index]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.bar(car_dist.index, car_dist.values, color="#e8a838")
    ax.set_xlabel("Total cars (kei + passenger)")
    ax.set_ylabel("Households")
    ax.set_title("Cars per Household", fontweight="bold")

    # Home ownership
    ax = axes[1]
    own_dist = (hh_level["home_ownership"]
                .map(HOME_OWN_LABELS).fillna("Unknown")
                .value_counts())
    wedge_pct = own_dist / own_dist.sum() * 100
    ax.pie(wedge_pct, labels=[f"{l}\n{p:.1f}%" for l, p in
                               zip(own_dist.index, wedge_pct)],
           colors=PALETTE[:len(own_dist)], startangle=140,
           textprops={"fontsize": 8})
    ax.set_title("Home Ownership Type", fontweight="bold")

    fig.suptitle("Household Vehicle Ownership & Housing — Okinawa PT Survey R05",
                 fontsize=10, y=1.01)
    fig.tight_layout()
    save_fig(fig, "L2_hh_vehicles_housing", OUT_L13)

    # ── 2c. Municipality distribution (top 15) ───────────────────────────────
    print("\n[2c] Sample households by municipality (addr_municipality_code, top 10)")
    muni_dist = hh_level["addr_municipality_code"].value_counts().head(10)
    for code, cnt in muni_dist.items():
        print(f"  {code}:  {cnt:>5,} HH  ({cnt/len(hh_level)*100:.1f}%)")

    # ── 2d. Sex split (person level) ─────────────────────────────────────────
    print("\n[2d] Sex distribution (person level)")
    sex_dist = hh["sex"].map(SEX_LABELS).fillna("Unknown").value_counts()
    for label, n in sex_dist.items():
        print(f"  {label:<12}: {n:>7,}  ({n/len(hh)*100:.1f}%)")


# ═══════════════════════════════════════════════════════════════════════════
# LEVEL 3 — Age Distribution by Employment/Student Status
# ═══════════════════════════════════════════════════════════════════════════

def level3_age_by_status(hh: pd.DataFrame) -> None:
    section("LEVEL 3 — Age Distribution by Employment / Student Status")

    df = hh.copy()
    df["age_num"] = pd.to_numeric(df["age"], errors="coerce")
    df["emp_label"] = df["employment_student_status"].map(EMP_STATUS_LABELS).fillna("Unknown")
    df["is_student"] = df["employment_student_status"].isin(STUDENT_CODES)

    # ── 3a. Summary table ────────────────────────────────────────────────────
    print("\n[3a] Age summary by employment / student status")
    summary = (df.groupby("emp_label")["age_num"]
               .agg(n="count", mean="mean", median="median",
                    p25=lambda x: x.quantile(0.25),
                    p75=lambda x: x.quantile(0.75),
                    min="min", max="max")
               .round(1)
               .sort_values("median"))
    print(summary.to_string())
    summary.to_csv(OUT_L13 / "L3_age_by_status_summary.csv")

    # ── 3b. Age distribution — all statuses (violin plot) ───────────────────
    # Exclude codes with very few records for readability
    min_n = 50
    valid_labels = (df.groupby("emp_label")["age_num"]
                    .count()[lambda s: s >= min_n].index.tolist())
    plot_df = df[df["emp_label"].isin(valid_labels)].dropna(subset=["age_num"])
    order = (plot_df.groupby("emp_label")["age_num"].median()
             .sort_values().index.tolist())

    fig, ax = plt.subplots(figsize=(13, 6))
    sns.violinplot(data=plot_df, x="emp_label", y="age_num",
                   order=order, palette="tab20",
                   inner="quartile", cut=0, ax=ax, linewidth=0.8)
    ax.set_xlabel("")
    ax.set_ylabel("Age")
    ax.set_title("Age Distribution by Employment / Student Status\n"
                 "(Okinawa PT Survey R05 — HH Survey, person level)",
                 fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8.5)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(10))
    fig.tight_layout()
    save_fig(fig, "L3_age_violin_all_statuses", OUT_L13)

    # ── 3c. Student codes zoom ───────────────────────────────────────────────
    print("\n[3c] Student sub-group age statistics")
    student_map = {
        8:  "University (8)",
        9:  "High school (9)",
        10: "Elem/middle (10)",
        11: "Kindergarten (11)",
        12: "Pre-school (12)",
    }
    stud_df = df[df["employment_student_status"].isin(STUDENT_CODES)].copy()
    stud_df["student_label"] = stud_df["employment_student_status"].map(student_map)
    stud_summary = (stud_df.groupby("student_label")["age_num"]
                    .agg(n="count", mean="mean", median="median",
                         min="min", max="max")
                    .round(1))
    print(stud_summary.to_string())
    stud_summary.to_csv(OUT_L13 / "L3_student_age_summary.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram per student type
    ax = axes[0]
    student_order = ["Pre-school (12)", "Kindergarten (11)",
                     "Elem/middle (10)", "High school (9)", "University (8)"]
    colors = ["#f4a261", "#e9c46a", "#2a9d8f", "#457b9d", "#e63946"]
    for label, color in zip(student_order, colors):
        sub = stud_df[stud_df["student_label"] == label]["age_num"].dropna()
        if len(sub) == 0:
            continue
        ax.hist(sub, bins=range(0, 30), alpha=0.7, label=f"{label} (n={len(sub):,})",
                color=color, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Age")
    ax.set_ylabel("Number of persons")
    ax.set_title("Age Distribution — Student Sub-groups", fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))

    # Box plot student vs employed vs other
    ax = axes[1]
    df_grouped = df.copy()
    df_grouped["broad_status"] = np.where(
        df_grouped["employment_student_status"].isin(STUDENT_CODES), "Student",
        np.where(df_grouped["employment_student_status"].isin([1, 2, 3, 4, 5, 6]),
                 "Employed", "Other (incl. unemployed)")
    )
    broad_order = ["Student", "Employed", "Other (incl. unemployed)"]
    palette_broad = {"Student": "#2a9d8f", "Employed": "#457b9d",
                     "Other (incl. unemployed)": "#e9c46a"}
    sns.boxplot(data=df_grouped.dropna(subset=["age_num"]),
                x="broad_status", y="age_num",
                order=broad_order, palette=palette_broad,
                width=0.5, linewidth=1.2, fliersize=2, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Age")
    ax.set_title("Age by Broad Category\n(Student / Employed / Other)",
                 fontweight="bold")
    ax.yaxis.set_major_locator(mticker.MultipleLocator(10))

    for status in broad_order:
        sub = df_grouped[df_grouped["broad_status"] == status]["age_num"].dropna()
        i = broad_order.index(status)
        ax.text(i, sub.max() + 1, f"n={len(sub):,}", ha="center",
                fontsize=8, color="gray")

    fig.suptitle("Students & Employment Status — Age Profile\n"
                 "Okinawa PT Survey R05", fontsize=10, y=1.01)
    fig.tight_layout()
    save_fig(fig, "L3_student_age_detail", OUT_L13)

    # ── 3d. Age pyramid for students (male vs female) ───────────────────────
    print("\n[3d] Student age pyramid by sex")
    stud_df["sex_label"] = stud_df["sex"].map(SEX_LABELS).fillna("Unknown")
    stud_df["age_bin"] = pd.cut(stud_df["age_num"],
                                bins=[0, 5, 10, 15, 18, 22, 30, 120],
                                labels=["0–4", "5–9", "10–14", "15–17",
                                        "18–21", "22–29", "30+"],
                                right=False)
    pyramid = (stud_df[stud_df["sex_label"].isin(["Male", "Female"])]
               .groupby(["age_bin", "sex_label"], observed=True)
               .size().unstack(fill_value=0))

    fig, ax = plt.subplots(figsize=(8, 5))
    age_labels = pyramid.index.astype(str)
    y = np.arange(len(age_labels))
    male_vals   = pyramid.get("Male",   pd.Series(0, index=pyramid.index)).values
    female_vals = pyramid.get("Female", pd.Series(0, index=pyramid.index)).values

    ax.barh(y, -male_vals,   color="#457b9d", label="Male")
    ax.barh(y, female_vals,  color="#e07b54", label="Female")
    ax.set_yticks(y)
    ax.set_yticklabels(age_labels)
    ax.set_xlabel("Number of persons")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: str(int(abs(x)))))
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Student Age Pyramid (Male vs Female)\nOkinawa PT Survey R05",
                 fontweight="bold")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "L3_student_age_pyramid", OUT_L13)

    # ── 3e. Cross-tab: status × age decade ──────────────────────────────────
    df["age_decade"] = (df["age_num"] // 10 * 10).astype("Int64").astype(str) + "s"
    cross = pd.crosstab(df["emp_label"], df["age_decade"])
    cross_pct = cross.div(cross.sum(axis=1), axis=0).mul(100).round(1)
    cross_pct.to_csv(OUT_L13 / "L3_status_age_decade_pct.csv")
    print("\n[3e] Status × age decade (% within status) saved to output/Level1-3/")

    # Heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    # Sort columns numerically
    decade_cols = [c for c in sorted(cross_pct.columns,
                                     key=lambda x: int(x[:-1]) if x[:-1].isdigit() else 999)]
    sns.heatmap(cross_pct[decade_cols], annot=True, fmt=".1f",
                cmap="YlOrRd", linewidths=0.5, cbar_kws={"label": "% within status"},
                ax=ax)
    ax.set_xlabel("Age decade")
    ax.set_ylabel("Employment / Student Status")
    ax.set_title("Employment / Student Status × Age Decade (% within status)\n"
                 "Okinawa PT Survey R05", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "L3_status_age_heatmap", OUT_L13)


# ── LEVEL 4: Trip Behavior Overview ─────────────────────────────────────────

# Code-books for Level 4 (verified from cross-tab analysis)
TRIP_PURPOSE_LABELS = {
    "01": "Go to work",
    "02": "School commute (通学) ★",  # verified: 40-43% of all student trips use this code
    "03": "Return home",
    "04": "Business trip",
    "05": "Agricultural work",
    "06": "Shopping",
    "07": "Medical / hospital",
    "08": "Recreation / leisure",
    "09": "Private errand",
    "10": "Pickup / dropoff",
    "11": "Hobby / lessons / club",   # 94.8% non-students, median age 69 — NOT school commute
    "12": "Escorting ★",
    "13": "Return fr. school/hobby",
    "14": "Return fr. recreation",
    "15": "Return fr. errand",
    "16": "Return fr. business",
    "17": "Return fr. agriculture",
    "18": "Return (other)",
    "19": "Other return / misc.",
    "99": "Unknown",
}

CLASS1_LABELS = {
    "01": "Work & school outbound",    # aggregates purpose=01 (work) + purpose=02 (school commute)
    "02": "Return home",
    "03": "Business trip",             # purpose=04 maps here; "from office" label was confused with purpose=02
    "04": "Agricultural work",
    "05": "Shopping / medical",
    "06": "Recreation / errand / pickup",
    "07": "Hobby / lessons / club",    # purpose=11 maps here; 94.8% non-students, median age 69
    "08": "Escorting",
    "09": "Return fr. school / hobby",
    "10": "Other returns",
    "99": "Unknown",
}

CLASS1_BROAD = {
    "01": "Work / school",   "02": "Return home",
    "03": "Work / school",   "04": "Work / school",
    "05": "Other activities","06": "Other activities",
    "07": "Other activities","08": "Escorting",
    "09": "Other activities","10": "Other activities","99": "Unknown",
}
BROAD_ORDER  = ["Return home", "Work / school", "Escorting", "Other activities", "Unknown"]
BROAD_COLORS = ["#457b9d", "#2a9d8f", "#f4a261", "#a8dadc", "#ccc"]

MODE_COLORS = {
    "Car (driver)":       "#e63946",
    "Car (passenger)":    "#f4a261",
    "Walk":               "#2a9d8f",
    "Bus":                "#457b9d",
    "Motorcycle / moped": "#e9c46a",
    "Bicycle":            "#6ab187",
    "Rail / monorail":    "#8338ec",
    "Taxi":               "#a8dadc",
    "Other / unknown":    "#aaa",
}
MODE_ORDER = ["Car (driver)", "Car (passenger)", "Walk", "Bus",
              "Motorcycle / moped", "Bicycle", "Rail / monorail",
              "Taxi", "Other / unknown"]

EMPLOYED_CODES = {1, 2, 3, 4, 5, 6}


def _mode_group(code: str) -> str:
    c = str(code).strip()
    if c == "01":               return "Walk"
    if c == "02":               return "Bicycle"
    if c in {"03","04","05","06"}: return "Motorcycle / moped"
    if c == "07":               return "Car (driver)"
    if c == "08":               return "Car (passenger)"
    if c == "10":               return "Bus"
    if c in {"11","16"}:        return "Rail / monorail"
    if c == "09":               return "Taxi"
    return "Other / unknown"


def _pct_label(ax, bars, total, offset=0):
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + offset,
                    f"{h/total*100:.1f}%",
                    ha="center", va="bottom", fontsize=7.5, color="#333")


def _prep_pt(pt: pd.DataFrame) -> pd.DataFrame:
    """Add normalised string columns and valid-trip flag to PT dataframe."""
    out = pt.copy()
    for col in ["employment_student_status", "went_out", "depart_hour_24h"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["trip_purpose_str"]   = out["trip_purpose"].astype(str).str.strip().str.zfill(2)
    out["purpose_class1_str"] = out["purpose_class1"].astype(str).str.strip().str.zfill(2)
    out["mode1_str"]          = out["mode1_mode"].astype(str).str.strip().str.zfill(2)
    out["is_valid_trip"]      = (out["went_out"] == 1) & (out["trip_purpose_str"] != "00")
    return out


def level4_trip_behavior(pt: pd.DataFrame, hh: pd.DataFrame) -> None:
    section("LEVEL 4 — Trip Behavior Overview")

    pt4 = _prep_pt(pt)
    valid = pt4[pt4["is_valid_trip"]].copy()
    N = len(valid)
    print(f"\n  Valid trips: {N:,}  (went_out=1, purpose not blank)")

    # ── 4.1  Trip purpose distribution ──────────────────────────────────────
    print("\n[4.1] Trip purpose distribution")
    pur = (valid["trip_purpose_str"].value_counts()
           .rename("count").reset_index()
           .rename(columns={"trip_purpose_str": "code"}))
    pur["label"] = pur["code"].map(TRIP_PURPOSE_LABELS).fillna("?")
    pur["pct"]   = (pur["count"] / N * 100).round(2)
    pur = pur.sort_values("code")
    print(pur[["code","label","count","pct"]].to_string(index=False))
    pur.to_csv(OUT_L4 / "L4_1_trip_purpose_detail.csv", index=False)

    school_n = pur.loc[pur["code"]=="02","count"].sum()
    escort_n = pur.loc[pur["code"]=="12","count"].sum()
    hobby_n  = pur.loc[pur["code"]=="11","count"].sum()
    print(f"\n  School commute (02): {school_n:,}  ({school_n/N*100:.2f}%)  ← 通学")
    print(f"  Escorting      (12): {escort_n:,}  ({escort_n/N*100:.2f}%)")
    print(f"  Hobby/lessons  (11): {hobby_n:,}  ({hobby_n/N*100:.2f}%)  ← mostly non-students, median age 69")

    valid["class1_label"] = valid["purpose_class1_str"].map(CLASS1_LABELS).fillna("Unknown")
    valid["broad_label"]  = valid["purpose_class1_str"].map(CLASS1_BROAD).fillna("Unknown")
    broad = valid["broad_label"].value_counts().reindex(BROAD_ORDER, fill_value=0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax = axes[0]
    pur_s = pur.sort_values("count", ascending=True)
    bar_colors = ["#e63946" if c=="02" else "#f4a261" if c=="12" else
                  "#b0b0b0" if c=="11" else "#6ab187"
                  for c in pur_s["code"]]
    bars = ax.barh(pur_s["label"], pur_s["count"], color=bar_colors, edgecolor="white")
    for bar in bars:
        w = bar.get_width()
        if w > 0:
            ax.text(w + 80, bar.get_y() + bar.get_height()/2,
                    f"{w:,}", va="center", fontsize=7)
    ax.set_xlabel("Number of trips")
    ax.set_title("Trip Purpose (19 codes)\n★ = school commute / escort", fontweight="bold")
    ax.legend(handles=[Patch(color="#e63946", label="School commute (02) ★"),
                        Patch(color="#f4a261", label="Escorting (12) ★"),
                        Patch(color="#b0b0b0", label="Hobby/lessons (11) — not school"),
                        Patch(color="#6ab187", label="Other")],
              fontsize=8, loc="lower right")

    ax = axes[1]
    broad_lbs = [f"{l}\n{v:,}\n({v/N*100:.1f}%)" for l, v in zip(broad.index, broad.values)]
    ax.pie(broad.values, labels=broad_lbs,
           colors=BROAD_COLORS[:len(broad)], startangle=90,
           textprops={"fontsize": 8.5}, wedgeprops={"width": 0.5})
    ax.set_title("Broad Groups (Class 1)", fontweight="bold")
    fig.suptitle("Trip Purpose — Okinawa PT Survey R05", fontsize=11)
    fig.tight_layout()
    save_fig(fig, "L4_1_trip_purpose", OUT_L4)

    # ── 4.2  Mode share overall ──────────────────────────────────────────────
    print("\n[4.2] Mode share overall")
    valid["mode_group"] = valid["mode1_str"].apply(_mode_group)
    mode_dist = (valid["mode_group"].value_counts()
                 .reindex(MODE_ORDER, fill_value=0)
                 .reset_index()
                 .rename(columns={"mode_group":"mode","count":"n"}))
    mode_dist["pct"] = (mode_dist["n"] / N * 100).round(2)
    print(mode_dist.to_string(index=False))
    mode_dist.to_csv(OUT_L4 / "L4_2_mode_share_overall.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    colors = [MODE_COLORS.get(m, "#aaa") for m in mode_dist["mode"]]
    bars = ax.bar(mode_dist["mode"], mode_dist["n"], color=colors, edgecolor="white")
    _pct_label(ax, bars, N, offset=200)
    ax.set_xticklabels(mode_dist["mode"], rotation=35, ha="right")
    ax.set_ylabel("Trips")
    ax.set_title("Mode Share (count)", fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    ax = axes[1]
    pie_data = mode_dist[mode_dist["pct"] >= 1.0].copy()
    other_n  = mode_dist[mode_dist["pct"] < 1.0]["n"].sum()
    if other_n > 0:
        pie_data = pd.concat([pie_data,
            pd.DataFrame([{"mode":"Other (combined)","n":other_n,"pct":other_n/N*100}])],
            ignore_index=True)
    ax.pie(pie_data["pct"],
           labels=[f"{m}\n{p:.1f}%" for m, p in zip(pie_data["mode"], pie_data["pct"])],
           colors=[MODE_COLORS.get(m,"#aaa") for m in pie_data["mode"]],
           startangle=140, textprops={"fontsize": 8}, wedgeprops={"edgecolor":"white"})
    ax.set_title("Mode Share (%)", fontweight="bold")
    fig.suptitle("Primary Mode Share — Okinawa PT Survey R05", fontsize=11)
    fig.tight_layout()
    save_fig(fig, "L4_2_mode_share_overall", OUT_L4)

    # ── 4.3  Trip rate by employment / student status ────────────────────────
    print("\n[4.3] Trip rate by employment / student status")
    PKEY_PT = ["id_municipality_code","id_b_zone_code","id_c_zone_code",
               "id_household_number","hh_member_person_no"]
    PKEY_HH = ["id_municipality_code","id_b_zone_code","id_c_zone_code",
               "id_household_number","person_number"]

    person_trips = (pt4.groupby(PKEY_PT)
                    .agg(n_trips=("is_valid_trip","sum"),
                         went_out=("went_out","first"))
                    .reset_index()
                    .rename(columns={"hh_member_person_no":"person_number"}))

    hh_valid = hh[pd.to_numeric(hh["expansion_factor"], errors="coerce") > 0].copy()
    merged = hh_valid.merge(person_trips[PKEY_HH + ["n_trips","went_out"]],
                            on=PKEY_HH, how="left")
    merged["n_trips"]  = merged["n_trips"].fillna(0)
    merged["traveled"] = merged["n_trips"] > 0
    merged["emp_label"] = merged["employment_student_status"].map(EMP_STATUS_LABELS).fillna("Unknown")

    rate_tbl = (merged.groupby("emp_label")
                .agg(total_persons=("emp_label","count"),
                     total_trips=("n_trips","sum"),
                     travelers=("traveled","sum"))
                .assign(trip_rate=lambda d: (d["total_trips"]/d["total_persons"]).round(2),
                        participation=lambda d: (d["travelers"]/d["total_persons"]*100).round(1),
                        avg_if_traveled=lambda d: (d["total_trips"]/d["travelers"]).round(2))
                .sort_values("trip_rate", ascending=False))
    print(rate_tbl[["total_persons","total_trips","trip_rate","participation"]].to_string())
    rate_tbl.to_csv(OUT_L4 / "L4_3_trip_rate_by_status.csv")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, col, xlabel, title in [
        (axes[0], "trip_rate",    "Avg trips / person / day",  "Trip Rate by Status"),
        (axes[1], "participation","Travel participation (%)",   "% Who Traveled (survey day)"),
    ]:
        plot_df = rate_tbl[rate_tbl["total_persons"] >= 30].sort_values(col)
        bar_colors = [
            "#e9c46a" if any(w in l for w in ["student","school","Kindergarten","Pre-school","University","High school"])
            else "#457b9d" if any(w in l for w in ["employed","officer","Self","Family","Side","Full","Part"])
            else "#aaa"
            for l in plot_df.index
        ]
        ax.barh(plot_df.index, plot_df[col], color=bar_colors, edgecolor="white")
        for i, (_, row) in enumerate(plot_df.iterrows()):
            ax.text(row[col] * 1.01, i, f"{row[col]}", va="center", fontsize=7.5)
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontweight="bold")
        if col == "participation":
            ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    axes[0].legend(handles=[Patch(color="#e9c46a",label="Student"),
                              Patch(color="#457b9d",label="Employed"),
                              Patch(color="#aaa",   label="Other")],
                   fontsize=8)
    fig.suptitle("Trip Rate by Employment / Student Status — Okinawa PT Survey R05",
                 fontsize=10)
    fig.tight_layout()
    save_fig(fig, "L4_3_trip_rate_by_status", OUT_L4)

    # ── 4.4  Mode share for school commute trips — sensitivity analysis ─────────
    print("\n[4.4] Mode share — school commute trips (purpose=02, 通学)")
    school = valid[valid["trip_purpose_str"] == "02"].copy()
    school["age"]      = pd.to_numeric(school["age"], errors="coerce")
    school["emp_label"] = school["employment_student_status"].map(EMP_STATUS_LABELS).fillna("Unknown")

    N_sch        = len(school)
    stud_sch     = school[school["employment_student_status"].isin(STUDENT_CODES)].copy()
    non_stud_sch = school[~school["employment_student_status"].isin(STUDENT_CODES)]
    print(f"  School commute trips (purpose=02): {N_sch:,}")
    print(f"    students    : {len(stud_sch):,}  ({len(stud_sch)/N_sch*100:.1f}%)")
    print(f"    non-students: {len(non_stud_sch):,}  ({len(non_stud_sch)/N_sch*100:.1f}%)")

    # Non-car base mapping (school-specific codebook):
    # bus=05/06, moped=09-11, rail=16
    def _sch_base(c):
        c = str(c).strip()
        if c == "01":                return "Walk"
        if c == "02":                return "Bicycle"
        if c in {"05", "06"}:       return "Bus"
        if c in {"09", "10", "11"}: return "Motorcycle / moped"
        if c == "16":               return "Rail / monorail"
        return "Other / unknown"

    # Chart A — raw: mode '07' = Car (driver), mode '08' = Car (passenger)
    def _mode_A(c):
        c = str(c).strip()
        if c == "07": return "Car (driver)"
        if c == "08": return "Car (passenger)"
        return _sch_base(c)

    # Chart B — age-corrected: under-18 + mode '07' → Car (passenger)
    def _mode_B(row):
        c = str(row["mode1_str"]).strip()
        if c == "08":
            return "Car (passenger)"
        if c == "07":
            return "Car (driver)" if pd.notna(row["age"]) and row["age"] >= 18 \
                   else "Car (passenger)"
        return _sch_base(c)

    stud_sch["mode_A"] = stud_sch["mode1_str"].apply(_mode_A)
    stud_sch["mode_B"] = stud_sch.apply(_mode_B, axis=1)

    SENS_ORDER  = ["Car (driver)", "Car (passenger)", "Walk", "Bicycle", "Bus",
                   "Motorcycle / moped", "Rail / monorail", "Other / unknown"]
    SENS_COLORS = {
        "Car (driver)":       "#e63946",
        "Car (passenger)":    "#f4a261",
        "Walk":               "#2a9d8f",
        "Bicycle":            "#6ab187",
        "Bus":                "#457b9d",
        "Motorcycle / moped": "#e9c46a",
        "Rail / monorail":    "#8338ec",
        "Other / unknown":    "#aaa",
    }
    STUDENT_PLOT_ORDER = ["Kindergarten", "Elementary / middle school",
                          "High school student", "University student"]

    def _pivot(df, col):
        tbl = df.groupby(["emp_label", col]).size().unstack(fill_value=0)
        return tbl.div(tbl.sum(axis=1), axis=0).mul(100).round(1), tbl.sum(axis=1)

    pct_A, n_A = _pivot(stud_sch, "mode_A")
    pct_B, n_B = _pivot(stud_sch, "mode_B")

    print("\n  Chart A — raw mode coding:")
    print(pct_A.to_string())
    print("\n  Chart B — age-corrected (age ≥ 18 = car driver):")
    print(pct_B.to_string())
    pct_A.to_csv(OUT_L4 / "L4_4_school_mode_raw.csv")
    pct_B.to_csv(OUT_L4 / "L4_4_school_mode_age_corrected.csv")

    # ── Figure: side-by-side sensitivity charts ───────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    def _draw(ax, pct_df, n_series, title):
        avail = [s for s in STUDENT_PLOT_ORDER if s in pct_df.index]
        avail_modes = [m for m in SENS_ORDER
                       if m in pct_df.columns and pct_df.loc[avail, m].sum() > 0]
        bottom = np.zeros(len(avail))
        for mode in avail_modes:
            vals = pct_df.reindex(avail)[mode].fillna(0).values
            ax.bar(avail, vals, bottom=bottom, label=mode,
                   color=SENS_COLORS.get(mode, "#aaa"),
                   edgecolor="white", linewidth=0.5)
            for i, (v, b) in enumerate(zip(vals, bottom)):
                if v >= 5:
                    ax.text(i, b + v/2, f"{v:.0f}%", ha="center", va="center",
                            fontsize=7, color="white", fontweight="bold")
            bottom += vals
        ax.set_xticklabels(
            [f"{s}\n(n={int(n_series.get(s, 0))})" for s in avail],
            rotation=25, ha="right", fontsize=8)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())
        ax.set_ylim(0, 112)
        ax.set_ylabel("% of school commute trips")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")

    _draw(axes[0], pct_A, n_A, "Chart A — Raw mode coding")
    _draw(axes[1], pct_B, n_B, "Chart B — Age-corrected (license age ≥18)")

    fig.suptitle("School Commute Mode × Student Type — Okinawa PT Survey R05  "
                 "(purpose=02 通学, students only)",
                 fontsize=10)
    fig.text(0.5, -0.03,
             "Difference between A and B reveals trips where under-18 students "
             "were recorded as car driver in raw data.",
             ha="center", va="top", fontsize=8, color="#444", style="italic")
    fig.tight_layout()
    save_fig(fig, "L4_4_school_mode_sensitivity", OUT_L4)

    # ── 4.5  Departure time distribution ────────────────────────────────────
    print("\n[4.5] Departure time distribution")
    valid4 = valid.copy()
    valid4["hour"] = pd.to_numeric(valid4["depart_hour_24h"], errors="coerce")
    valid4 = valid4[valid4["hour"].between(0, 26)]
    N_all = len(valid4)
    sch4  = valid4[valid4["trip_purpose_str"] == "02"]   # school commute (通学)
    esc4  = valid4[valid4["trip_purpose_str"] == "12"]   # escorting

    hours = range(3, 24)
    all_hr  = valid4["hour"].value_counts().reindex(hours, fill_value=0).sort_index()
    sch_hr  = sch4["hour"].value_counts().reindex(hours, fill_value=0).sort_index()
    esc_hr  = esc4["hour"].value_counts().reindex(hours, fill_value=0).sort_index()

    print(f"  Peak — All: {int(all_hr.idxmax()):02d}:00  "
          f"| School: {int(sch_hr.idxmax()):02d}:00  "
          f"| Escort: {int(esc_hr.idxmax()):02d}:00")

    pd.DataFrame({"hour":list(hours),
                  "all_n":all_hr.values, "all_pct":(all_hr/N_all*100).round(2).values,
                  "school_n":sch_hr.values,
                  "school_pct":(sch_hr/len(sch4)*100).round(2).values if len(sch4) else 0,
                  "escort_n":esc_hr.values,
                  "escort_pct":(esc_hr/len(esc4)*100).round(2).values if len(esc4) else 0,
                  }).to_csv(OUT_L4 / "L4_5_departure_hour.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    x = list(hours)

    ax = axes[0]
    ax.bar(x, all_hr.values, color="#b0c4de", edgecolor="white", zorder=2)
    ax.axvline(all_hr.idxmax(), color="#457b9d", linestyle="--", linewidth=1.2,
               label=f"Peak: {int(all_hr.idxmax()):02d}:00", zorder=3)
    ax.set_ylabel("Number of trips")
    ax.set_title(f"All Trips — Departure Hour (n={N_all:,})", fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    ax = axes[1]
    xs = np.array(x)
    w  = 0.35
    sch_pct = sch_hr / len(sch4) * 100 if len(sch4) else sch_hr * 0
    esc_pct = esc_hr / len(esc4) * 100 if len(esc4) else esc_hr * 0
    ax.bar(xs - w/2, sch_pct.values, width=w, color="#e9c46a", edgecolor="white",
           label=f"School commute / 通学 (n={len(sch4):,})", zorder=2)
    ax.bar(xs + w/2, esc_pct.values, width=w, color="#f4a261", edgecolor="white",
           label=f"Escort (n={len(esc4):,})", zorder=2)
    ax.axvline(sch_hr.idxmax(), color="#b5a000", linestyle="--", linewidth=1.1,
               label=f"School commute peak: {int(sch_hr.idxmax()):02d}:00", zorder=3)
    ax.axvline(esc_hr.idxmax(), color="#c1440e", linestyle=":",  linewidth=1.1,
               label=f"Escort peak: {int(esc_hr.idxmax()):02d}:00", zorder=3)
    ax.set_xlabel("Departure hour (24h)")
    ax.set_ylabel("% of trips within group")
    ax.set_title("School Commute (通学) vs Escort — Departure Hour (%)", fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h:02d}:00" for h in x], rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    fig.suptitle("Departure Time Distribution — Okinawa PT Survey R05",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    save_fig(fig, "L4_5_departure_time", OUT_L4)

    # Cumulative departure curve
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, (all_hr.cumsum()/N_all*100).values, color="#b0c4de", lw=2, label="All trips")
    ax.plot(x, (sch_hr.cumsum()/len(sch4)*100).values if len(sch4) else x,
            color="#e9c46a", lw=2, label="School commute (通学)")
    ax.plot(x, (esc_hr.cumsum()/len(esc4)*100).values if len(esc4) else x,
            color="#f4a261", lw=2, label="Escort trips")
    for pct, ls in [(50, "--"), (80, ":")]:
        ax.axhline(pct, color="#888", linestyle=ls, linewidth=0.8, label=f"{pct}%")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h:02d}:00" for h in x], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Cumulative % departed")
    ax.set_title("Cumulative Departure Profile", fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "L4_5_departure_cumulative", OUT_L4)


# ═══════════════════════════════════════════════════════════════════════════
# ── LEVEL 5: Student Travel Independence ────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def level5_student_independence(pt: pd.DataFrame) -> None:
    section("LEVEL 5: Student Travel Independence")

    HH_KEY   = ["id_municipality_code", "id_b_zone_code",
                 "id_c_zone_code", "id_household_number"]
    ESC_COLS = ["escorted_hh_member_no_1",
                "escorted_hh_member_no_2",
                "escorted_hh_member_no_3"]

    pt = pt.copy()
    pt["trip_purpose_str"]        = pt["trip_purpose"].astype(str).str.strip().str.zfill(2)
    pt["mode1_str"]               = pt["mode1_mode"].astype(str).str.strip().str.zfill(2)
    pt["hh_member_person_no"]     = pd.to_numeric(pt["hh_member_person_no"], errors="coerce")
    pt["went_out"]                = pd.to_numeric(pt["went_out"], errors="coerce")
    valid = pt[pt["went_out"] == 1]

    # ── Step 1: Build escort lookup ──────────────────────────────────────────
    # Melt escorted_hh_member_no_1/2/3 from every purpose='12' trip into a
    # flat table of (household_key, escorted_person_no) pairs.
    escort_trips = valid[valid["trip_purpose_str"] == "12"].copy()
    esc_long = escort_trips[HH_KEY + ESC_COLS].melt(
        id_vars=HH_KEY, value_vars=ESC_COLS, value_name="escorted_person_no"
    ).drop(columns="variable")
    # Keep as zero-padded strings to match escorted_hh_member_no_1/2/3 encoding
    # (e.g. person 3 is stored as '03', not 3 — integer join was silently missing rows)
    esc_long["hh_member_str"] = (
        esc_long["escorted_person_no"].astype(str).str.strip().str.zfill(2)
    )
    esc_lookup = (
        esc_long
        .loc[lambda d: ~d["hh_member_str"].isin({"", "nan", "99"})]
        [HH_KEY + ["hh_member_str"]]
        .drop_duplicates()
    )
    esc_lookup["_escorted"] = True

    # ── Step 2: Tag each school-commute student trip ─────────────────────────
    school = valid[
        (valid["trip_purpose_str"] == "02") &
        valid["employment_student_status"].isin(STUDENT_CODES)
    ].copy()
    school["hh_member_str"] = (
        school["hh_member_person_no"].astype(str).str.strip().str.zfill(2)
    )

    school = school.merge(esc_lookup, on=HH_KEY + ["hh_member_str"], how="left")
    school["is_escorted"] = school["_escorted"].fillna(False).astype(bool)
    school["emp_label"]   = school["employment_student_status"].map(
        EMP_STATUS_LABELS
    ).fillna("Unknown")

    # ── Step 3: Age-corrected mode label ────────────────────────────────────
    def _sch_base(c: str) -> str:
        if c == "01":                return "Walk"
        if c == "02":                return "Bicycle"
        if c in {"05", "06"}:       return "Bus"
        if c in {"09", "10", "11"}: return "Motorcycle / moped"
        if c == "16":               return "Rail / monorail"
        return "Other / unknown"

    def _mode_corr(row) -> str:
        c = str(row["mode1_str"]).strip()
        if c == "08":
            return "Car (passenger)"
        if c == "07":
            return (
                "Car (driver)"
                if pd.notna(row["age"]) and row["age"] >= 18
                else "Car (passenger)"
            )
        return _sch_base(c)

    school["mode_corr"] = school.apply(_mode_corr, axis=1)

    # ── Step 5 (sanity checks — printed before charts) ───────────────────────
    print("\n[5] Sanity checks")
    n_esc_trips = len(escort_trips)
    n_school    = len(school)
    n_escorted  = school["is_escorted"].sum()
    n_indep     = (~school["is_escorted"]).sum()

    print(f"  Escort trips (purpose=12)          : {n_esc_trips:,}  (expected ~5913)")
    print(f"  School commute student trips        : {n_school:,}")
    print(f"    Escorted   : {n_escorted:,}  ({n_escorted/n_school*100:.1f}%)")
    print(f"    Independent: {n_indep:,}  ({n_indep/n_school*100:.1f}%)")
    print(f"    Sum check  : {n_escorted + n_indep:,} == {n_school:,}  "
          f"{'✓' if n_escorted + n_indep == n_school else '✗'}")

    STUDENT_PLOT = ["Kindergarten", "Elementary / middle school",
                    "High school student", "University student"]
    grp = school.groupby("emp_label")["is_escorted"].agg(["sum", "count"])
    grp["pct"] = (grp["sum"] / grp["count"] * 100).round(1)
    grp = grp.reindex([s for s in STUDENT_PLOT if s in grp.index])
    print(f"\n  Escorted rate by student type:")
    for label, row in grp.iterrows():
        flag = ""
        if label == "Kindergarten"              and row["pct"] < 60: flag = " ⚠ below 60%"
        if label == "University student"         and row["pct"] > 20: flag = " ⚠ above 20%"
        print(f"    {label:<30}: {row['pct']:5.1f}%  (n={int(row['count'])}){flag}")

    # ── Step 4: Charts ───────────────────────────────────────────────────────
    MODE_ORDER5 = ["Car (driver)", "Car (passenger)", "Walk", "Bicycle",
                   "Bus", "Motorcycle / moped", "Rail / monorail", "Other / unknown"]
    MODE_COL5   = {
        "Car (driver)":       "#e63946",
        "Car (passenger)":    "#f4a261",
        "Walk":               "#2a9d8f",
        "Bicycle":            "#6ab187",
        "Bus":                "#457b9d",
        "Motorcycle / moped": "#e9c46a",
        "Rail / monorail":    "#8338ec",
        "Other / unknown":    "#aaa",
    }

    avail_types = [s for s in STUDENT_PLOT if s in school["emp_label"].unique()]
    school_plot = school[school["emp_label"].isin(avail_types)].copy()

    def _stacked_mode(ax, subset, title):
        """Draw mode-share stacked bars for one subset of school trips."""
        tbl = subset.groupby(["emp_label", "mode_corr"]).size().unstack(fill_value=0)
        pct = tbl.div(tbl.sum(axis=1), axis=0).mul(100).round(1)
        n   = tbl.sum(axis=1)
        avail = [s for s in avail_types if s in pct.index]
        modes = [m for m in MODE_ORDER5
                 if m in pct.columns and pct.loc[avail, m].sum() > 0]
        bottom = np.zeros(len(avail))
        for mode in modes:
            vals = pct.reindex(avail)[mode].fillna(0).values
            ax.bar(avail, vals, bottom=bottom, label=mode,
                   color=MODE_COL5.get(mode, "#aaa"),
                   edgecolor="white", linewidth=0.5)
            for i, (v, b) in enumerate(zip(vals, bottom)):
                if v >= 5:
                    ax.text(i, b + v / 2, f"{v:.0f}%", ha="center", va="center",
                            fontsize=7, color="white", fontweight="bold")
            bottom += vals
        ax.set_xticklabels(
            [f"{s}\n(n={int(n.get(s, 0))})" for s in avail],
            rotation=25, ha="right", fontsize=8,
        )
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())
        ax.set_ylim(0, 112)
        ax.set_ylabel("% of school commute trips")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")

    fig, axes = plt.subplots(1, 3, figsize=(21, 5))

    # Chart A: Escorted vs Independent rate by student type
    ax = axes[0]
    avail_grp  = [s for s in avail_types if s in grp.index]
    esc_vals   = [grp.loc[s, "pct"]   if s in grp.index else 0 for s in avail_grp]
    indep_vals = [100 - v for v in esc_vals]
    n_vals     = [int(grp.loc[s, "count"]) if s in grp.index else 0 for s in avail_grp]

    ax.bar(avail_grp, esc_vals,   label="Escorted",
           color="#e76f51", edgecolor="white")
    ax.bar(avail_grp, indep_vals, bottom=esc_vals, label="Independent",
           color="#2a9d8f", edgecolor="white")
    for i, (e, ind) in enumerate(zip(esc_vals, indep_vals)):
        if e >= 5:
            ax.text(i, e / 2, f"{e:.0f}%", ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")
        if ind >= 5:
            ax.text(i, e + ind / 2, f"{ind:.0f}%", ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")
    ax.set_xticklabels(
        [f"{s}\n(n={n})" for s, n in zip(avail_grp, n_vals)],
        rotation=25, ha="right", fontsize=8,
    )
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_ylim(0, 112)
    ax.set_ylabel("% of school commute trips")
    ax.set_title("Chart A — Escorted vs Independent\nby Student Type",
                 fontweight="bold")
    ax.legend(fontsize=8)

    # Chart B: Mode share — independent trips
    _stacked_mode(axes[1],
                  school_plot[~school_plot["is_escorted"]],
                  "Chart B — Mode Share\nIndependent Trips")

    # Chart C: Mode share — escorted trips
    _stacked_mode(axes[2],
                  school_plot[school_plot["is_escorted"]],
                  "Chart C — Mode Share\nEscorted Trips")

    fig.suptitle(
        "Student Travel Independence — School Commute (purpose=02 通学) "
        "× Student Type\n"
        "Escort identified via purpose=12 trips; mode age-corrected (≥18 = car driver)",
        fontsize=9,
    )
    fig.tight_layout()
    save_fig(fig, "L5_student_independence", OUT_L5)

    # CSVs
    escort_summary = grp.rename(columns={"sum": "n_escorted", "count": "n_total",
                                          "pct": "pct_escorted"})
    escort_summary.to_csv(OUT_L5 / "L5_escort_rate_by_student_type.csv")

    for label, subset in [("independent", school_plot[~school_plot["is_escorted"]]),
                           ("escorted",    school_plot[school_plot["is_escorted"]])]:
        tbl = subset.groupby(["emp_label", "mode_corr"]).size().unstack(fill_value=0)
        pct = tbl.div(tbl.sum(axis=1), axis=0).mul(100).round(1)
        pct.to_csv(OUT_L5 / f"L5_mode_{label}.csv")

    print(f"\n  → L5_student_independence.png")
    print(f"  → L5_escort_rate_by_student_type.csv")
    print(f"  → L5_mode_independent.csv / L5_mode_escorted.csv")


# ── LEVEL 6: SCHOOL ESCORTING — WHO IS THE ESCORTER? ────────────────────────

def level6_escorter_profile(pt: pd.DataFrame, hh: pd.DataFrame,
                            pt_key: pd.DataFrame | None = None) -> None:
    section("LEVEL 6: School Escorting — Who Is the Escorter?")

    HH_KEY = ["id_municipality_code", "id_b_zone_code",
               "id_c_zone_code", "id_household_number"]

    # Level-6-specific employment codebook (differs from HH-survey global labels)
    _EMP_L6 = {
        1:  "Self-employed / family worker",
        2:  "Corporate officer",
        3:  "Regular employee",
        4:  "Dispatched / temp",
        5:  "Part-time / contract",
        6:  "Full-time homemaker",
        7:  "Unemployed",
        8:  "University student",
        9:  "HS student",
        10: "Elem / middle school",
        11: "Kindergartener",
        12: "Preschooler",
        13: "Other",
        99: "Unknown",
    }

    STUDENT_CODES_L6 = {8, 9, 10, 11, 12}

    # Consistent colors used across all 4 charts
    CLR_SCHOOL = "#3a7bbf"   # blue  — school escorting
    CLR_OTHER  = "#e07b39"   # orange — other escorting
    CLR_POP    = "#aaaaaa"   # gray  — overall population

    # ── Prepare: valid trips → escort trips ──────────────────────────────────
    pt = pt.copy()
    pt["trip_purpose_str"] = pt["trip_purpose"].astype(str).str.zfill(2)
    pt["went_out"]         = pd.to_numeric(pt["went_out"], errors="coerce")
    valid  = pt[pt["went_out"] == 1]
    escort = valid[valid["trip_purpose_str"] == "12"].copy()
    n_escort = len(escort)

    # ── Classify each escort trip by the escorted child's status ─────────────
    # Look up escorted_hh_member_no_1 in HH Survey to get child's emp status.
    # NaN / unmatched → Other group.
    esc_col1 = "escorted_hh_member_no_1"
    child_lookup = (
        escort[HH_KEY + [esc_col1]]
        .rename(columns={esc_col1: "person_number"})
        .merge(
            hh[HH_KEY + ["person_number", "employment_student_status"]]
              .rename(columns={"employment_student_status": "child_emp_status"}),
            on=HH_KEY + ["person_number"],
            how="left",
        )
    )
    escort["child_emp_status"] = child_lookup["child_emp_status"].values
    escort["is_school"] = escort["child_emp_status"].isin(STUDENT_CODES_L6)

    n_school = int(escort["is_school"].sum())
    n_other  = n_escort - n_school

    # ── Breakdown print ───────────────────────────────────────────────────────
    print(f"\n=== ESCORT TRIP BREAKDOWN ===")
    print(f"Total escort trips (purpose=12): {n_escort:,}  "
          f"({'✓' if n_escort == 5913 else f'⚠ expected 5913, got {n_escort}'})")
    print(f"  - School escorting (student escorted): "
          f"{n_school:,} ({n_school / n_escort * 100:.1f}%)")
    print(f"  - Other escorting: "
          f"{n_other:,} ({n_other / n_escort * 100:.1f}%)")

    # ── Join escorters to HH Survey for profile columns ───────────────────────
    # Escort trips depart from home → trip-origin zone = home zone → join correct.
    hh_profile = (
        hh[HH_KEY + ["person_number", "age", "sex", "employment_student_status"]]
        .rename(columns={"person_number": "hh_member_person_no"})
    )
    escorter = escort.merge(
        hh_profile,
        on=HH_KEY + ["hh_member_person_no"],
        how="left",
        suffixes=("_pt", "_hh"),
        indicator=True,
    )
    n_matched = (escorter["_merge"] == "both").sum()
    escorter  = escorter.drop(columns="_merge")
    print(f"\n    HH profile matched: {n_matched:,} / {n_escort:,} "
          f"({n_matched / n_escort * 100:.1f}%)")

    escorter["sex_lbl"] = (
        pd.to_numeric(escorter["sex_hh"], errors="coerce")
        .map(SEX_LABELS).fillna("Unknown")
    )
    escorter["emp_lbl"] = (
        escorter["employment_student_status_hh"]
        .map(_EMP_L6).fillna("Unknown")
    )
    escorter["age_num"] = pd.to_numeric(escorter["age_hh"], errors="coerce")

    # ── Data quality: flag escorters with age < 18 ───────────────────────────
    under18 = escorter[escorter["age_num"] < 18]
    n_u18   = len(under18)
    print(f"\n    ⚠ DATA QUALITY: {n_u18} escorter records with age < 18")
    if n_u18 > 0:
        u18_summary = (
            under18[["age_num", "sex_lbl", "emp_lbl"]]
            .value_counts()
            .reset_index()
            .rename(columns={0: "count"})
            .head(5)
        )
        print(u18_summary.to_string(index=False))
        print(f"    → Excluded from Charts B and C (retained in A and D)")

    # Filtered escorter sets for Charts B & C (adults only)
    escorter_bc      = escorter[~(escorter["age_num"] < 18)].copy()
    esc_school_bc    = escorter_bc[escorter_bc["is_school"]].copy()
    esc_other_bc     = escorter_bc[~escorter_bc["is_school"]].copy()

    # Full (unfiltered) per-group subsets for Charts A and D
    esc_school = escorter[escorter["is_school"]].copy()
    esc_other  = escorter[~escorter["is_school"]].copy()

    # ── Sanity: sex + age per group ───────────────────────────────────────────
    for grp_lbl, grp in [("School", esc_school), ("Other", esc_other)]:
        sex_cts = grp["sex_lbl"].value_counts()
        n_grp   = len(grp)
        print(f"\n    [{grp_lbl} escorting, n={n_grp:,}]")
        for sex in ["Female", "Male", "Unknown"]:
            nv = int(sex_cts.get(sex, 0))
            print(f"      {sex:<10}: {nv:,} ({nv / n_grp * 100:.1f}%)")

    age_s = esc_school_bc["age_num"].dropna()
    age_o = esc_other_bc["age_num"].dropna()
    print(f"\n    Median escorter age (≥18) — "
          f"School: {age_s.median():.0f},  Other: {age_o.median():.0f}")

    # ── Chart D prep: child type distribution ────────────────────────────────
    child_pcts       = pd.Series(dtype=float)
    child_cts_school = pd.Series(dtype=int)
    child_cts_other  = pd.Series(dtype=int)
    if esc_col1 in escort.columns:
        child_df = (
            escort[HH_KEY + [esc_col1, "is_school"]]
            .dropna(subset=[esc_col1])
            .rename(columns={esc_col1: "person_number"})
        )
        child_joined = child_df.merge(
            hh[HH_KEY + ["person_number", "employment_student_status"]],
            on=HH_KEY + ["person_number"],
            how="left",
        )
        child_joined["child_emp_lbl"] = (
            child_joined["employment_student_status"]
            .map(_EMP_L6).fillna("Unknown")
        )
        all_child_cts = child_joined["child_emp_lbl"].value_counts()
        child_pcts = (all_child_cts / all_child_cts.sum() * 100).round(1)
        child_cts_school = child_joined[child_joined["is_school"]]["child_emp_lbl"].value_counts()
        child_cts_other  = child_joined[~child_joined["is_school"]]["child_emp_lbl"].value_counts()

    # ── Figure: 2 × 2 layout ─────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Level 6 — School Escorting: Who Is the Escorter?",
                 fontweight="bold", fontsize=13)

    # ── Chart A: Sex — 3 groups × 3 sex categories ───────────────────────────
    ax = axes[0, 0]
    hh_sex_pct     = (pd.to_numeric(hh["sex"], errors="coerce")
                      .map(SEX_LABELS).fillna("Unknown")
                      .value_counts(normalize=True).mul(100))
    school_sex_pct = esc_school["sex_lbl"].value_counts(normalize=True).mul(100)
    other_sex_pct  = esc_other["sex_lbl"].value_counts(normalize=True).mul(100)

    sex_cats    = ["Male", "Female", "Unknown"]
    school_vals = [float(school_sex_pct.get(c, 0)) for c in sex_cats]
    other_vals  = [float(other_sex_pct.get(c, 0))  for c in sex_cats]
    pop_vals    = [float(hh_sex_pct.get(c, 0))     for c in sex_cats]

    x = np.arange(len(sex_cats))
    w = 0.25
    b1 = ax.bar(x - w, school_vals, w, color=CLR_SCHOOL,
                label=f"School escorting (n={n_school:,})")
    b2 = ax.bar(x,     other_vals,  w, color=CLR_OTHER,
                label=f"Other escorting (n={n_other:,})")
    b3 = ax.bar(x + w, pop_vals,    w, color=CLR_POP, alpha=0.8,
                label="Overall population")

    for bar in [*b1, *b2, *b3]:
        h = bar.get_height()
        if h > 1.0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                    f"{h:.0f}%", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(sex_cats)
    ax.set_ylabel("Share within group (%)")
    ax.set_title("A — Escorter Sex", fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_ylim(0, max(school_vals + other_vals + pop_vals) * 1.4)

    # ── Chart B: Age histograms — school (blue) vs other (orange) ────────────
    ax = axes[0, 1]
    bins_b = range(18, 86, 5)
    ax.hist(age_s, bins=bins_b, color=CLR_SCHOOL, alpha=0.7,
            edgecolor="white", linewidth=0.5,
            label=f"School escorting (n={len(age_s):,})")
    ax.hist(age_o, bins=bins_b, color=CLR_OTHER, alpha=0.65,
            edgecolor="white", linewidth=0.5,
            label=f"Other escorting (n={len(age_o):,})")
    ax.set_xlabel("Age")
    ax.set_ylabel("Escort trips")
    ax.set_title(f"B — Escorter Age Distribution\n"
                 f"(age ≥ 18 only; {n_u18} under-18 excluded)",
                 fontweight="bold")
    ax.set_xlim(18, 85)
    ax.legend(fontsize=8)
    for med_val, clr in [(float(age_s.median()), CLR_SCHOOL),
                         (float(age_o.median()), CLR_OTHER)]:
        ax.axvline(med_val, color=clr, linestyle="--", linewidth=1.3, zorder=5)
    ax.text(0.97, 0.95,
            f"Median — school: {age_s.median():.0f}  other: {age_o.median():.0f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8))

    # ── Chart C: Occupation — grouped horizontal bars (age ≥ 18) ─────────────
    ax = axes[1, 0]
    # Sort categories by combined count (highest at top after invert_yaxis)
    combined_cts = (
        esc_school_bc["emp_lbl"].value_counts()
        .add(esc_other_bc["emp_lbl"].value_counts(), fill_value=0)
        .sort_values(ascending=False)
    )
    all_cats_c  = combined_cts.index.tolist()
    n_cats_c    = len(all_cats_c)
    school_pct_c = esc_school_bc["emp_lbl"].value_counts(normalize=True).mul(100)
    other_pct_c  = esc_other_bc["emp_lbl"].value_counts(normalize=True).mul(100)
    s_vals = [float(school_pct_c.get(c, 0)) for c in all_cats_c]
    o_vals = [float(other_pct_c.get(c, 0))  for c in all_cats_c]

    yc   = np.arange(n_cats_c)
    hbar = 0.35
    ax.barh(yc - hbar / 2, s_vals, hbar,
            color=CLR_SCHOOL, label=f"School (n={len(esc_school_bc):,})")
    ax.barh(yc + hbar / 2, o_vals, hbar,
            color=CLR_OTHER,  label=f"Other (n={len(esc_other_bc):,})")
    ax.set_yticks(yc)
    ax.set_yticklabels(all_cats_c, fontsize=8)
    ax.invert_yaxis()   # highest-count category at top
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlabel("Share within group (%)")
    ax.set_title(f"C — Escorter Occupation  (age ≥ 18)", fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())

    # ── Chart D: Who is being escorted? + n= labels ───────────────────────────
    ax = axes[1, 1]
    if len(child_pcts) > 0:
        CHILD_COLORS = {
            "Kindergartener":       "#f4a261",
            "Elem / middle school": "#2a9d8f",
            "HS student":           "#457b9d",
            "University student":   "#8338ec",
            "Preschooler":          "#e9c46a",
        }
        # Sort by count (highest at top via invert_yaxis)
        child_sorted = child_pcts.sort_values(ascending=False)
        n_cats_d     = len(child_sorted)
        bar_colors_d = [CHILD_COLORS.get(lbl, "#aaa") for lbl in child_sorted.index]
        yd = np.arange(n_cats_d)

        ax.barh(yd, child_sorted.values, color=bar_colors_d)
        ax.set_yticks(yd)
        ax.set_yticklabels(child_sorted.index.tolist(), fontsize=8)
        ax.invert_yaxis()

        max_pct_d = float(child_sorted.max())
        ax.set_xlim(0, max_pct_d * 2.4)   # room for text labels

        for i, (lbl, pct) in enumerate(child_sorted.items()):
            n_s = int(child_cts_school.get(lbl, 0))
            n_o = int(child_cts_other.get(lbl, 0))
            ax.text(float(pct) + max_pct_d * 0.04,
                    i,
                    f"{pct:.1f}%  (s={n_s:,}, o={n_o:,})",
                    va="center", fontsize=7.5)

        # Legend patches for school / other counts
        d_legend = [
            Patch(facecolor=CLR_SCHOOL,
                  label=f"School escorting  n={n_school:,}"),
            Patch(facecolor=CLR_OTHER,
                  label=f"Other escorting   n={n_other:,}"),
        ]
        ax.legend(handles=d_legend, fontsize=7.5, loc="lower right")
        ax.set_xlabel("Share of all escort trips (%)")
        ax.set_title("D — Who Is Being Escorted?\n(escorted_hh_member_no_1)",
                     fontweight="bold")
        ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    else:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color="#888")
        ax.set_title("D — Who Is Being Escorted?", fontweight="bold")

    fig.tight_layout()
    save_fig(fig, "L6_escorter_profile", OUT_L6)
    print(f"\n  → L6_escorter_profile.png")

    # ── School-only profile — PT-based classification ─────────────────────────
    if pt_key is not None:
        sch_esc_pt = get_school_escort_trips(escort, pt_key)
        n_sch_pt   = len(sch_esc_pt)
        print(f"\n  [School-only, PT-based] n={n_sch_pt:,}  "
              f"(HH-based n={n_school:,}; diff={n_sch_pt - n_school:+d})")

        escorter_sch = sch_esc_pt.merge(
            hh_profile, on=HH_KEY + ["hh_member_person_no"],
            how="left", suffixes=("_pt", "_hh"),
        )
        escorter_sch["sex_lbl"] = (
            pd.to_numeric(escorter_sch["sex_hh"], errors="coerce")
            .map(SEX_LABELS).fillna("Unknown")
        )
        escorter_sch["emp_lbl"] = (
            escorter_sch["employment_student_status_hh"]
            .map(_EMP_L6).fillna("Unknown")
        )
        escorter_sch_bc = escorter_sch[
            pd.to_numeric(escorter_sch["age_hh"], errors="coerce") >= 18
        ].copy()

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

        ax = axes[0]
        sex_cats = ["Male", "Female", "Unknown"]
        sex_pcts = [float(escorter_sch["sex_lbl"].value_counts(normalize=True)
                          .mul(100).get(c, 0)) for c in sex_cats]
        ax.bar(sex_cats, sex_pcts,
               color=[CLR_SCHOOL, CLR_OTHER, CLR_POP], width=0.55)
        for i, p in enumerate(sex_pcts):
            if p > 0.5:
                ax.text(i, p + 0.5, f"{p:.1f}%", ha="center", fontsize=9)
        ax.set_ylabel("Share (%)")
        ax.set_title(f"(a) Escorter Sex  (n={n_sch_pt:,})", fontweight="bold")
        ax.set_ylim(0, max(sex_pcts) * 1.35 + 3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax = axes[1]
        emp_pcts   = escorter_sch_bc["emp_lbl"].value_counts(normalize=True).mul(100)
        emp_sorted = emp_pcts.sort_values(ascending=True)
        ax.barh(range(len(emp_sorted)), emp_sorted.values,
                color=CLR_SCHOOL, height=0.6)
        ax.set_yticks(range(len(emp_sorted)))
        ax.set_yticklabels(emp_sorted.index.tolist(), fontsize=8)
        ax.set_xlabel("Share (%)")
        ax.set_title(f"(b) Escorter Employment  (age≥18, n={len(escorter_sch_bc):,})",
                     fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for i, p in enumerate(emp_sorted.values):
            if p > 1:
                ax.text(p + 0.3, i, f"{p:.1f}%", va="center", fontsize=7.5)

        fig.suptitle(
            "L6 (school-only) — Escorter Profile  (PT-based via get_school_escort_trips)\n"
            f"school_codes={SCHOOL_CODES_ESC}"
        )
        fig.tight_layout()
        save_fig(fig, "L6_escorter_profile_school", OUT_L6)


# ── LEVEL 7: SCHOOL ESCORTING — WHEN & WHERE? ────────────────────────────────

def level7_escort_when_where(pt: pd.DataFrame, hh: pd.DataFrame,
                             pt_key: pd.DataFrame | None = None) -> None:
    section("LEVEL 7: School Escorting — When & Where?")

    HH_KEY        = ["id_municipality_code", "id_b_zone_code",
                     "id_c_zone_code", "id_household_number"]
    PERSON_KEY    = HH_KEY + ["hh_member_person_no"]
    STUDENT_CODES = {8, 9, 10, 11, 12}

    CLR_SCH_ESC = "#3a7bbf"   # school escort — blue
    CLR_SCH_COM = "#e07b39"   # school commute — orange
    CLR_FEMALE  = "#d45f1a"
    CLR_MALE    = "#3a7bbf"

    # ── Prepare: valid trips ──────────────────────────────────────────────────
    pt = pt.copy()
    pt["trip_purpose_str"] = pt["trip_purpose"].astype(str).str.zfill(2)
    pt["went_out"]         = pd.to_numeric(pt["went_out"], errors="coerce")
    valid = pt[pt["went_out"] == 1]

    escort         = valid[valid["trip_purpose_str"] == "12"].copy()
    school_commute = valid[valid["trip_purpose_str"] == "02"].copy()

    # ── Classify escort trips: school (student escorted) vs other ────────────
    child_lookup = (
        escort[HH_KEY + ["escorted_hh_member_no_1"]]
        .rename(columns={"escorted_hh_member_no_1": "person_number"})
        .merge(
            hh[HH_KEY + ["person_number", "employment_student_status"]]
              .rename(columns={"employment_student_status": "child_emp_status"}),
            on=HH_KEY + ["person_number"],
            how="left",
        )
    )
    escort["child_emp_status"] = child_lookup["child_emp_status"].values
    escort["is_school"]        = escort["child_emp_status"].isin(STUDENT_CODES)
    school_esc   = escort[escort["is_school"]].copy()
    n_school_esc = len(school_esc)

    # ── Departure hour (24-h column; 99 = unknown → exclude) ─────────────────
    school_esc["dep_hr"]     = pd.to_numeric(school_esc["depart_hour_24h"],    errors="coerce")
    school_commute["dep_hr"] = pd.to_numeric(school_commute["depart_hour_24h"], errors="coerce")

    se_hr = school_esc[school_esc["dep_hr"].between(0, 23)].copy()
    sc_hr = school_commute[school_commute["dep_hr"].between(0, 23)].copy()

    # Sex label from PT (same codes as HH: 1=Male, 2=Female)
    se_hr["sex_lbl"] = (
        pd.to_numeric(se_hr["sex"], errors="coerce")
        .map(SEX_LABELS).fillna("Unknown")
    )

    # ── Sanity checks ─────────────────────────────────────────────────────────
    morning_band = se_hr[se_hr["dep_hr"].between(5, 10)]
    morning_peak = (int(morning_band["dep_hr"].value_counts().idxmax())
                    if len(morning_band) > 0 else None)
    pct_before9  = (se_hr["dep_hr"] < 9).sum() / max(len(se_hr), 1) * 100
    top3_zones   = school_esc["id_b_zone_code"].value_counts().head(3)

    print(f"\n[7] School escort trips (timing/location): {n_school_esc:,}")
    if morning_peak is not None:
        print(f"    Morning peak hour (5–10):  {morning_peak:02d}:00  "
              f"({'✓ expected 07' if morning_peak == 7 else '⚠'})")
    print(f"    % departing before 09:00:  {pct_before9:.1f}%")
    print(f"    Top 3 origin zones (id_b_zone_code):")
    for z, cnt in top3_zones.items():
        print(f"      Zone {z}: {cnt:,} ({cnt / n_school_esc * 100:.1f}%)")

    # ── Chart D prep: trip chaining (num_trips per unique person) ────────────
    esc_pkeys             = school_esc[PERSON_KEY].drop_duplicates().copy()
    esc_pkeys["_is_esc"]  = True
    valid_dedup           = valid.drop_duplicates(subset=PERSON_KEY).copy()
    valid_dedup           = valid_dedup.merge(esc_pkeys, on=PERSON_KEY, how="left")
    valid_dedup["_is_esc"] = valid_dedup["_is_esc"].fillna(False)

    nt_esc   = valid_dedup[valid_dedup["_is_esc"]]["num_trips"].dropna()
    nt_other = valid_dedup[~valid_dedup["_is_esc"]]["num_trips"].dropna()

    print(f"\n    Trip chaining (unique persons):")
    print(f"      School escorters (n={len(nt_esc):,}): "
          f"median={nt_esc.median():.0f}, mean={nt_esc.mean():.1f} trips/day")
    print(f"      Other persons   (n={len(nt_other):,}): "
          f"median={nt_other.median():.0f}, mean={nt_other.mean():.1f} trips/day")

    # ── Figure: 2 × 2 layout ─────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("Level 7 — School Escorting: When & Where?",
                 fontweight="bold", fontsize=13)

    # ── Chart A: Departure hour 5:00–19:00 ───────────────────────────────────
    ax = axes[0, 0]
    show_hours = list(range(5, 20))
    se_pct = (se_hr["dep_hr"].value_counts() / max(len(se_hr), 1) * 100
              ).reindex(show_hours, fill_value=0)
    sc_pct = (sc_hr["dep_hr"].value_counts() / max(len(sc_hr), 1) * 100
              ).reindex(show_hours, fill_value=0)

    x  = np.arange(len(show_hours))
    wA = 0.4
    ax.bar(x - wA / 2, se_pct.values, wA, color=CLR_SCH_ESC, alpha=0.85,
           label=f"School escort (n={len(se_hr):,})")
    ax.bar(x + wA / 2, sc_pct.values, wA, color=CLR_SCH_COM, alpha=0.85,
           label=f"School commute (n={len(sc_hr):,})")

    # Shade morning (5–10) and afternoon (14–19) windows
    # show_hours[0]=5 → x=0; [5]=10 → x=5; [9]=14 → x=9; [14]=19 → x=14
    ax.axvspan(-0.5,  5.5, alpha=0.08, color=CLR_SCH_ESC, zorder=0)
    ax.axvspan( 8.5, 14.5, alpha=0.08, color="#888888",   zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{h}:00" for h in show_hours],
                       rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Share within group (%)")
    ax.set_title("A — Departure Hour (5:00–19:00)\n% within each group",
                 fontweight="bold")
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ylim_A = max(float(se_pct.max()), float(sc_pct.max())) * 1.30
    ax.set_ylim(0, ylim_A)
    ax.text(2.5,  ylim_A * 0.95, "Morning",   ha="center", va="top",
            fontsize=7, color="#2244aa", style="italic")
    ax.text(11.5, ylim_A * 0.95, "Afternoon", ha="center", va="top",
            fontsize=7, color="#555",    style="italic")

    # ── Chart B: Escorter sex × morning departure hour ────────────────────────
    ax = axes[0, 1]
    morning_se = se_hr[se_hr["dep_hr"].between(5, 10)].copy()
    female_se  = morning_se[morning_se["sex_lbl"] == "Female"]
    male_se    = morning_se[morning_se["sex_lbl"] == "Male"]

    show_hrs_B = list(range(5, 11))
    fe_pct = (female_se["dep_hr"].value_counts() / max(len(female_se), 1) * 100
              ).reindex(show_hrs_B, fill_value=0)
    me_pct = (male_se["dep_hr"].value_counts()   / max(len(male_se), 1) * 100
              ).reindex(show_hrs_B, fill_value=0)

    xB = np.arange(len(show_hrs_B))
    wB = 0.35
    ax.bar(xB - wB / 2, fe_pct.values, wB, color=CLR_FEMALE,
           label=f"Female (n={len(female_se):,})")
    ax.bar(xB + wB / 2, me_pct.values, wB, color=CLR_MALE, alpha=0.85,
           label=f"Male (n={len(male_se):,})")
    ax.set_xticks(xB)
    ax.set_xticklabels([f"{h}:00" for h in show_hrs_B])
    ax.set_ylabel("Share within group (%)")
    ax.set_title("B — Escorter Sex × Morning Departure Hour\n"
                 "(school escort, 5:00–10:00)",
                 fontweight="bold")
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())

    # ── Chart C: Top 10 origin zones (id_b_zone_code) ────────────────────────
    ax = axes[1, 0]
    zone_cts  = school_esc["id_b_zone_code"].value_counts().head(10)
    zone_pcts = (zone_cts / n_school_esc * 100).round(2)
    zone_lbls = [f"Zone {z}" for z in zone_cts.index]

    yd    = np.arange(len(zone_pcts))
    max_z = float(zone_pcts.max())
    ax.barh(yd, zone_pcts.values[::-1], color=CLR_SCH_ESC, alpha=0.85)
    ax.set_yticks(yd)
    ax.set_yticklabels(zone_lbls[::-1], fontsize=8)
    ax.set_xlabel("% of school escort trips")
    ax.set_title("C — Top 10 Origin Zones (id_b_zone_code)\n"
                 "School escort trips",
                 fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_xlim(0, max_z * 1.55)
    for i, pct in enumerate(zone_pcts.values[::-1]):
        ax.text(float(pct) + max_z * 0.04, i,
                f"{pct:.1f}%", va="center", fontsize=8)

    # ── Chart D: Trip chaining — num_trips violin (school escorters vs others) ─
    ax = axes[1, 1]
    clip_val = 12
    d_esc   = nt_esc.clip(upper=clip_val).values
    d_other = nt_other.clip(upper=clip_val).values

    parts = ax.violinplot([d_esc, d_other], positions=[1, 2],
                          showmedians=True, showextrema=False)
    for body, clr in zip(parts["bodies"], [CLR_SCH_ESC, "#aaaaaa"]):
        body.set_facecolor(clr)
        body.set_alpha(0.75)
        body.set_edgecolor("white")
    parts["cmedians"].set_color("white")
    parts["cmedians"].set_linewidth(2.5)

    ax.set_xticks([1, 2])
    ax.set_xticklabels([
        f"School escorters\n(n={len(nt_esc):,})",
        f"Other persons\n(n={len(nt_other):,})",
    ], fontsize=9)
    ax.set_ylabel(f"Daily trips (num_trips, clipped at {clip_val})")
    ax.set_title("D — Trip Chaining: Daily Trip Count\n"
                 "(school escorters vs all other travellers)",
                 fontweight="bold")
    ax.set_ylim(0, clip_val + 1)
    ax.text(0.97, 0.97,
            f"Median — escort: {nt_esc.median():.0f},  "
            f"other: {nt_other.median():.0f}\n"
            f"Mean   — escort: {nt_esc.mean():.1f},  "
            f"other: {nt_other.mean():.1f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    fig.tight_layout()
    save_fig(fig, "L7_escort_when_where", OUT_L7)
    print(f"\n  → L7_escort_when_where.png")

    # ── School-only timing stats — PT-based classification ────────────────────
    # Note: L7's main chart already uses school_esc (HH-based classification).
    # This block applies the PT-based helper for a direct comparison and saves
    # the departure-time statistics as an explicit _school CSV.
    if pt_key is not None:
        sch_esc_pt2 = get_school_escort_trips(escort, pt_key)
        n_pt2 = len(sch_esc_pt2)
        print(f"\n  [L7 school-only, PT-based] n={n_pt2:,}  "
              f"(HH-based n={n_school_esc:,}; diff={n_pt2 - n_school_esc:+d})")

        hr_pt2 = pd.to_numeric(sch_esc_pt2["depart_hour_24h"], errors="coerce")
        hr_pt2 = hr_pt2[hr_pt2.between(0, 23)]
        peak_pt2 = int(hr_pt2[hr_pt2.between(5, 10)].value_counts().idxmax()) \
                   if len(hr_pt2[hr_pt2.between(5, 10)]) > 0 else None

        # Departure hour distribution CSV
        hr_dist = (hr_pt2.value_counts().sort_index()
                   .reset_index().rename(columns={"depart_hour_24h": "hour",
                                                   "count": "n_trips"}))
        hr_dist["pct"] = (hr_dist["n_trips"] / hr_dist["n_trips"].sum() * 100).round(1)
        hr_dist.to_csv(OUT_L7 / "L7_departure_hour_school.csv", index=False)
        print(f"  peak hour (PT-based): {peak_pt2:02d}:00  "
              f"→ L7_departure_hour_school.csv")


# ── LEVEL 8: SUPPLEMENTARY Q6 — WHY ESCORT? ─────────────────────────────────

def level8_supplement_why_escort(supp: pd.DataFrame, hh: pd.DataFrame) -> None:
    section("LEVEL 8: Supplementary Q6 — Why Escort?")

    HH_KEY = ["id_municipality_code", "id_b_zone_code",
               "id_c_zone_code", "id_household_number"]

    # Codebooks (Level-8-local)
    _SCHOOL_DEST = {
        "1": "Nursery",
        "2": "Elementary",
        "3": "Middle school",
        "4": "High school",
        "5": "University",
        "6": "Other",
    }
    _SCHOOL_ORDER = ["Nursery", "Elementary", "Middle school",
                     "High school", "University", "Other"]

    _FREQ_LABELS = {
        "1": "1 day/wk",
        "2": "2 days/wk",
        "3": "3 days/wk",
        "4": "4 days/wk",
        "5": "5 days/wk",
    }
    _FREQ_ORDER = ["1 day/wk", "2 days/wk", "3 days/wk", "4 days/wk", "5 days/wk"]

    _REASON_LABELS = {
        "01": "Physical condition",
        "02": "Safety/crime concerns",
        "03": "No public transport",
        "04": "Destination is far",
        "05": "Need to be on time",
        "06": "Heavy baggage",
        "07": "Weather reasons",
        "08": "Financial reasons",
        "09": "On the way to work",
        "10": "Have spare time",
        "11": "Other",
    }

    # ── Filter to Q6 escort respondents ──────────────────────────────────────
    flag = supp["q6_escorting_flag"].astype(str).str.strip()
    q6   = supp[flag == "1"].copy()
    n_q6 = len(q6)

    # ── Map child-1 columns to labels ─────────────────────────────────────────
    def _str(s: pd.Series) -> pd.Series:
        return s.astype(str).str.strip()

    q6["dest1"] = _str(q6["q6_child1_school_dest"]).map(_SCHOOL_DEST)
    q6["freq1"] = _str(q6["q6_child1_escort_freq_days"]).map(_FREQ_LABELS)
    q6["reas1"] = _str(q6["q6_child1_escort_reason"]).map(_REASON_LABELS)

    # Child-2 / child-3 counts (informational)
    c2_n = (_str(q6["q6_child2_school_dest"]) != "").sum()
    c3_n = (_str(q6["q6_child3_school_dest"]) != "").sum()

    # ── HH join — match rate sanity check ────────────────────────────────────
    # hh["person_number"] is zero-padded string after clean_data;
    # supp["person_number"] stays int64 → cast hh side to numeric for the join.
    hh_for_match = (
        hh[HH_KEY + ["person_number"]]
        .assign(person_number=lambda d: pd.to_numeric(d["person_number"], errors="coerce"))
        .drop_duplicates()
    )
    q6_match = q6.merge(hh_for_match, on=HH_KEY + ["person_number"],
                        how="left", indicator=True)
    n_matched_hh = (q6_match["_merge"] == "both").sum()
    print(f"\n[8] Q6 escort respondents: {n_q6:,}  "
          f"({'✓' if n_q6 == 922 else f'⚠ expected 922'})")
    print(f"    HH Survey match: {n_matched_hh:,} / {n_q6:,} "
          f"({n_matched_hh / n_q6 * 100:.1f}%)")
    print(f"    Child 2 data: {c2_n:,}  |  Child 3 data: {c3_n:,}")

    # ── Sanity checks ─────────────────────────────────────────────────────────
    top_reason = q6["reas1"].value_counts().idxmax() if q6["reas1"].notna().any() else "N/A"
    top_freq   = q6["freq1"].value_counts().idxmax() if q6["freq1"].notna().any() else "N/A"
    print(f"    Most common escort reason:    {top_reason}")
    print(f"    Most common escort frequency: {top_freq}")

    # ── Figure: 2 × 2 layout ─────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Level 8 — Supplementary Q6: Why Escort?",
                 fontweight="bold", fontsize=13)

    # ── Chart A: Escort frequency distribution (child 1) ─────────────────────
    ax = axes[0, 0]
    freq_cts  = q6["freq1"].value_counts().reindex(_FREQ_ORDER, fill_value=0)
    freq_pcts = (freq_cts / freq_cts.sum() * 100).round(1)

    bars_a = ax.bar(_FREQ_ORDER, freq_pcts.values, color="#4c9be8", alpha=0.85,
                    edgecolor="white", linewidth=0.5)
    for bar, pct in zip(bars_a, freq_pcts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{pct:.0f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("% of Q6 respondents")
    ax.set_title("A — Escort Frequency (days/week, child 1)", fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_ylim(0, float(freq_pcts.max()) * 1.30)
    ax.set_xticklabels(_FREQ_ORDER, fontsize=9)

    # ── Chart B: Escort reasons sorted by frequency ───────────────────────────
    ax = axes[0, 1]
    reas_cts  = q6["reas1"].value_counts()                        # sorted descending
    reas_pcts = (reas_cts / reas_cts.sum() * 100).round(1)
    n_reas    = len(reas_pcts)
    max_r     = float(reas_pcts.max())

    ax.barh(np.arange(n_reas), reas_pcts.values,
            color="#4c9be8", alpha=0.85, edgecolor="white")
    ax.set_yticks(np.arange(n_reas))
    ax.set_yticklabels(reas_pcts.index.tolist(), fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("% of child-1 escort cases")
    ax.set_title("B — Escort Reasons (child 1, sorted by frequency)",
                 fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_xlim(0, max_r * 1.55)
    for i, pct in enumerate(reas_pcts.values):
        ax.text(float(pct) + max_r * 0.04, i,
                f"{pct:.1f}%", va="center", fontsize=8)

    # ── Chart C: Reason × school level (stacked 100% bar) ────────────────────
    ax = axes[1, 0]
    # Rows = school_dest, columns = reason; normalize by school_dest row
    ct = pd.crosstab(q6["dest1"], q6["reas1"])
    present_schools = [s for s in _SCHOOL_ORDER if s in ct.index]
    ct = ct.reindex(index=present_schools, fill_value=0)
    # Order reasons by overall frequency (most → least prominent stacking)
    reason_cols_sorted = [r for r in reas_pcts.index if r in ct.columns]
    ct = ct.reindex(columns=reason_cols_sorted, fill_value=0)
    ct_pct = ct.div(ct.sum(axis=1), axis=0).mul(100).fillna(0)

    colors_c = [plt.cm.tab20.colors[i % 20] for i in range(len(ct_pct.columns))]
    bottom_c = np.zeros(len(ct_pct))
    for col, clr in zip(ct_pct.columns, colors_c):
        ax.bar(present_schools, ct_pct[col].values,
               bottom=bottom_c, color=clr, width=0.65,
               label=col)
        bottom_c += ct_pct[col].values

    ax.set_ylabel("% within school level")
    ax.set_title("C — Escort Reason × School Level (child 1)\n"
                 "Stacked 100% bar",
                 fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_ylim(0, 112)
    ax.tick_params(axis="x", labelsize=8)
    ax.legend(fontsize=6.5, loc="upper left",
              bbox_to_anchor=(1.01, 1.0), framealpha=0.85)

    # ── Chart D: Mean escort frequency × school level ─────────────────────────
    ax = axes[1, 1]
    _FREQ_NUM = {v: int(k) for k, v in _FREQ_LABELS.items()}
    q6["freq1_num"] = q6["freq1"].map(_FREQ_NUM)

    d_means, d_sems, d_ns, d_lbls = [], [], [], []
    for school in present_schools:
        vals = q6[q6["dest1"] == school]["freq1_num"].dropna()
        d_means.append(float(vals.mean()))
        d_sems.append(float(vals.std() / len(vals) ** 0.5))
        d_ns.append(len(vals))
        d_lbls.append(school)

    xD = np.arange(len(d_lbls))
    colors_d = [plt.cm.Set2.colors[i % 8] for i in range(len(d_lbls))]
    bars_d = ax.bar(xD, d_means, 0.6, color=colors_d, alpha=0.87,
                    yerr=d_sems, capsize=4,
                    error_kw={"linewidth": 1.2, "ecolor": "#555"})
    for bar, mean, n in zip(bars_d, d_means, d_ns):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.08,
                f"{mean:.1f}\n(n={n})", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(xD)
    ax.set_xticklabels(d_lbls, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Mean escort frequency (days/week)")
    ax.set_title("D — Mean Escort Frequency × School Level (child 1)\n"
                 "Error bars = ±1 SE",
                 fontweight="bold")
    ax.set_ylim(0, 5.8)
    ax.axhline(5, color="#aaa", linestyle="--", linewidth=0.8, zorder=0)
    ax.text(len(d_lbls) - 0.5, 5.05, "5 days/wk", ha="right",
            fontsize=7, color="#888")

    fig.tight_layout(rect=[0, 0, 0.88, 1])   # leave right margin for Chart C legend
    save_fig(fig, "L8_why_escort", OUT_L8)
    print(f"\n  → L8_why_escort.png")


# ── LEVEL 9: CROSS-TABULATIONS FOR ABM MODEL ─────────────────────────────────

def level9_abm_cross_tabs(pt: pd.DataFrame, hh: pd.DataFrame,
                          supp: pd.DataFrame,
                          pt_key: pd.DataFrame | None = None) -> None:
    section("LEVEL 9: Cross-Tabulations for ABM Model Building")

    HH_KEY        = ["id_municipality_code", "id_b_zone_code",
                     "id_c_zone_code", "id_household_number"]
    STUDENT_CODES = {8, 9, 10, 11, 12}
    STUDENT_MAP   = {12: "Preschool", 11: "Kindergarten", 10: "Elem/middle",
                     9: "High school", 8: "University"}
    CHILD_ORDER   = [12, 11, 10, 9, 8]          # young → old
    EXCEL_PATH    = SCRIPT_DIR / "output" / "L9_ABM_parameter_tables.xlsx"

    def _banner(n: str, title: str) -> None:
        print(f"\n{'─'*68}")
        print(f"  Table {n}: {title}")
        print(f"{'─'*68}")

    tables: dict = {}   # sheet_name → DataFrame

    # ── Common PT prep ────────────────────────────────────────────────────────
    pt_l = pt.copy()
    pt_l["trip_purpose_str"] = pt_l["trip_purpose"].astype(str).str.zfill(2)
    pt_l["went_out"]         = pd.to_numeric(pt_l["went_out"], errors="coerce")
    valid  = pt_l[pt_l["went_out"] == 1]
    escort = valid[valid["trip_purpose_str"] == "12"].copy()
    school = valid[valid["trip_purpose_str"] == "02"].copy()

    # Classify escort trips by escorted child's school level
    child_lookup = (
        escort[HH_KEY + ["escorted_hh_member_no_1"]]
        .rename(columns={"escorted_hh_member_no_1": "person_number"})
        .merge(
            hh[HH_KEY + ["person_number", "employment_student_status"]]
              .rename(columns={"employment_student_status": "child_emp_status"}),
            on=HH_KEY + ["person_number"],
            how="left",
        )
    )
    escort["child_emp_status"] = pd.to_numeric(
        child_lookup["child_emp_status"].values, errors="coerce"
    )
    escort["is_school"] = escort["child_emp_status"].isin(STUDENT_CODES)
    school_esc = escort[escort["is_school"]].copy()

    # Join escorter's HH profile onto school escort trips
    hh_profile = (
        hh[HH_KEY + ["person_number", "age", "sex", "employment_student_status"]]
        .rename(columns={"person_number": "hh_member_person_no",
                         "age":  "age_hh",  "sex": "sex_hh",
                         "employment_student_status": "emp_hh"})
    )
    escorter = school_esc.merge(hh_profile, on=HH_KEY + ["hh_member_person_no"],
                                how="left")
    escorter["sex_num"]   = pd.to_numeric(escorter["sex_hh"],  errors="coerce")
    escorter["age_num"]   = pd.to_numeric(escorter["age_hh"],  errors="coerce")
    escorter["emp_num"]   = pd.to_numeric(escorter["emp_hh"],  errors="coerce")
    escorter["dep_hr"]    = pd.to_numeric(escorter["depart_hour_24h"], errors="coerce")
    escorter["child_code"] = (
        pd.to_numeric(escorter["child_emp_status"], errors="coerce")
        .round().astype("Int64")
    )
    # Mode: convert object column to zero-padded 2-char string
    mode_num = pd.to_numeric(escorter["mode1_mode"], errors="coerce")
    escorter["mode_str"] = (
        mode_num.where(mode_num.isna(), mode_num.astype(int))
        .astype(str).str.zfill(2)
    )

    def _mode_cat(m: str) -> str:
        if m in {"07", "08"}:          return "Car"
        if m == "01":                   return "Walk"
        if m == "02":                   return "Bicycle"
        if m in {"05", "06", "80"}:    return "Bus"
        if m in {"09", "10", "11"}:    return "Motorcycle"
        if m == "16":                   return "Rail"
        return "Other"

    escorter["mode_cat"] = escorter["mode_str"].apply(_mode_cat)

    # Occupation codebook (Level-9 local, same mapping as Level 6)
    _EMP_ABM = {
        1: "Self-employed", 2: "Corp officer", 3: "Regular employee",
        4: "Dispatched/temp", 5: "Part-time", 6: "Homemaker",
        7: "Unemployed", 8: "Uni student", 9: "HS student",
        10: "Elem/middle", 11: "Kindergartener", 12: "Preschooler",
        13: "Other", 99: "Unknown",
    }

    # ── Build escorted-person lookup set (Table 1) ────────────────────────────
    esc_parts = []
    for col in ["escorted_hh_member_no_1", "escorted_hh_member_no_2",
                "escorted_hh_member_no_3"]:
        if col not in escort.columns:
            continue
        esc_parts.append(
            escort[HH_KEY + [col]]
            .dropna(subset=[col])
            .rename(columns={col: "escorted_person_no"})
        )
    escorted_df = pd.concat(esc_parts, ignore_index=True).drop_duplicates()

    # ── TABLE 1: Escort probability by child's education level ───────────────
    _banner("1", "Escort Probability by Child's Education Level")
    school_emp = pd.to_numeric(school["employment_student_status"], errors="coerce")
    rows_t1 = []
    for code in CHILD_ORDER:
        mask   = school_emp.round() == code
        sub    = (school[mask]
                  .drop_duplicates(subset=HH_KEY + ["hh_member_person_no"])
                  .rename(columns={"hh_member_person_no": "escorted_person_no"}))
        n_tot  = len(sub)
        if n_tot == 0:
            continue
        matched = sub.merge(escorted_df, on=HH_KEY + ["escorted_person_no"],
                            how="left", indicator=True)
        n_esc = int((matched["_merge"] == "both").sum())
        p     = n_esc / n_tot
        ci_h  = 1.96 * (p * (1 - p) / n_tot) ** 0.5
        rows_t1.append({
            "School level":      STUDENT_MAP[code],
            "n (students)":      n_tot,
            "n (escorted)":      n_esc,
            "Escort prob (%)":   round(p * 100, 1),
            "95% CI lower (%)":  round(max(0, p - ci_h) * 100, 1),
            "95% CI upper (%)":  round(min(1, p + ci_h) * 100, 1),
        })
    t1 = pd.DataFrame(rows_t1)
    print(t1.to_string(index=False))
    tables["T1_EscortProb"] = t1

    # ── TABLE 2: Escorter profile by child's school level ────────────────────
    _banner("2", "Escorter Profile by Child's School Level")
    rows_t2 = []
    for code in CHILD_ORDER:
        sub = escorter[escorter["child_code"] == code]
        if len(sub) == 0:
            continue
        n          = len(sub)
        pct_female = round((sub["sex_num"] == 2).sum() / n * 100, 1)
        mean_age   = round(float(sub["age_num"].dropna().mean()), 1)
        top2_occ   = sub["emp_num"].map(_EMP_ABM).value_counts().head(2).index.tolist()
        rows_t2.append({
            "Child level":        STUDENT_MAP[code],
            "n escort trips":     n,
            "% female escorter":  pct_female,
            "Mean escorter age":  mean_age,
            "Top occupation 1":   top2_occ[0] if len(top2_occ) > 0 else "",
            "Top occupation 2":   top2_occ[1] if len(top2_occ) > 1 else "",
        })
    t2 = pd.DataFrame(rows_t2)
    print(t2.to_string(index=False))
    tables["T2_EscorterProfile"] = t2

    # ── TABLE 3: Mode by child's school level ─────────────────────────────────
    _banner("3", "Escort Mode by Child's School Level")
    rows_t3 = []
    for code in CHILD_ORDER:
        sub = escorter[escorter["child_code"] == code]
        if len(sub) == 0:
            continue
        n         = len(sub)
        mode_pcts = sub["mode_cat"].value_counts(normalize=True).mul(100).round(1)
        rows_t3.append({
            "Child level":  STUDENT_MAP[code],
            "n":            n,
            "Car (%)":      mode_pcts.get("Car",        0.0),
            "Walk (%)":     mode_pcts.get("Walk",       0.0),
            "Bicycle (%)":  mode_pcts.get("Bicycle",    0.0),
            "Bus (%)":      mode_pcts.get("Bus",        0.0),
            "Rail (%)":     mode_pcts.get("Rail",       0.0),
            "Other (%)":    mode_pcts.get("Other",      0.0),
        })
    t3 = pd.DataFrame(rows_t3)
    print(t3.to_string(index=False))
    tables["T3_Mode"] = t3

    # ── TABLE 4: Escort frequency by school level (Q6) ───────────────────────
    _banner("4", "Escort Frequency by School Level  (Q6 supplementary, child 1)")
    Q6_DEST  = {"1": "Nursery", "2": "Elementary", "3": "Middle school",
                "4": "High school", "5": "University", "6": "Other"}
    Q6_ORDER = ["Nursery", "Elementary", "Middle school",
                "High school", "University", "Other"]

    flag_s = supp["q6_escorting_flag"].astype(str).str.strip()
    q6     = supp[flag_s == "1"].copy()
    q6["dest1"]     = q6["q6_child1_school_dest"].astype(str).str.strip().map(Q6_DEST)
    q6["freq1_num"] = pd.to_numeric(
        q6["q6_child1_escort_freq_days"].astype(str).str.strip(), errors="coerce"
    )
    rows_t4 = []
    for lbl in Q6_ORDER:
        vals = q6[q6["dest1"] == lbl]["freq1_num"].dropna()
        if len(vals) == 0:
            continue
        rows_t4.append({
            "School level (Q6)":     lbl,
            "n respondents":         len(vals),
            "Mean freq (days/wk)":   round(float(vals.mean()), 2),
            "SD":                    round(float(vals.std()),  2),
            "% 5 days/wk":           round((vals == 5).sum() / len(vals) * 100, 1),
            "Median (days/wk)":      int(vals.median()),
        })
    t4 = pd.DataFrame(rows_t4)
    print(t4.to_string(index=False))
    tables["T4_EscortFrequency"] = t4

    # ── TABLE 5: Top 3 escort reasons by school level (Q6) ───────────────────
    _banner("5", "Top 3 Escort Reasons by School Level  (Q6 supplementary, child 1)")
    Q6_REASON = {
        "01": "Physical condition", "02": "Safety/crime concerns",
        "03": "No public transport","04": "Destination is far",
        "05": "Need to be on time", "06": "Heavy baggage",
        "07": "Weather reasons",    "08": "Financial reasons",
        "09": "On the way to work",  "10": "Have spare time",
        "11": "Other",
    }
    q6["reas1"] = q6["q6_child1_escort_reason"].astype(str).str.strip().map(Q6_REASON)
    rows_t5 = []
    for lbl in Q6_ORDER:
        sub_r = q6[q6["dest1"] == lbl]["reas1"].dropna()
        if len(sub_r) == 0:
            continue
        top3 = sub_r.value_counts().head(3)
        row  = {"School level (Q6)": lbl, "n": len(sub_r)}
        for rank, (reason, cnt) in enumerate(top3.items(), start=1):
            row[f"Reason #{rank}"]  = reason
            row[f"% #{rank}"]       = round(cnt / len(sub_r) * 100, 1)
        rows_t5.append(row)
    t5 = pd.DataFrame(rows_t5).fillna(np.nan)
    print(t5.to_string(index=False))
    tables["T5_EscortReasons"] = t5

    # ── TABLE 6: Departure hour by escorter sex ───────────────────────────────
    _banner("6", "Departure Hour by Escorter Sex  (school escort trips, 0–23 h)")
    esc_hr = escorter[escorter["dep_hr"].between(0, 23)].copy()
    esc_hr["sex_lbl"] = esc_hr["sex_num"].map(SEX_LABELS).fillna("Unknown")
    rows_t6 = []
    for sex in ["Female", "Male", "Unknown"]:
        sub = esc_hr[esc_hr["sex_lbl"] == sex]["dep_hr"].dropna()
        if len(sub) == 0:
            continue
        morn = sub[sub.between(5, 10)]
        aftn = sub[sub.between(14, 19)]
        rows_t6.append({
            "Escorter sex":         sex,
            "n trips":              len(sub),
            "Mean dep hour":        round(float(sub.mean()), 2),
            "Median dep hour":      int(sub.median()),
            "% morning (5–10h)":    round(len(morn) / len(sub) * 100, 1),
            "Peak morning hour":    int(morn.value_counts().idxmax()) if len(morn) > 0 else np.nan,
            "% afternoon (14–19h)": round(len(aftn) / len(sub) * 100, 1),
        })
    t6 = pd.DataFrame(rows_t6)
    print(t6.to_string(index=False))
    tables["T6_DepartureTime"] = t6

    # ── Save all tables to Excel ──────────────────────────────────────────────
    print(f"\n{'─'*68}")
    try:
        with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl") as writer:
            for sheet, df in tables.items():
                df.to_excel(writer, sheet_name=sheet, index=False)
                ws = writer.sheets[sheet]
                for col_cells in ws.columns:
                    max_len = max(
                        (len(str(cell.value or "")) for cell in col_cells),
                        default=0,
                    )
                    ws.column_dimensions[col_cells[0].column_letter].width = (
                        min(max_len + 3, 45)
                    )
        print(f"  → Excel saved: {EXCEL_PATH.name}")
    except ImportError:
        print("  ⚠ openpyxl not installed — run: pip install openpyxl")
    except Exception as exc:
        print(f"  ⚠ Excel save failed: {exc}")

    print(f"\n  [9] {len(tables)} ABM parameter tables complete")

    # ── School-only CSVs — PT-based classification ────────────────────────────
    # T2, T3, T6 already use school_esc (HH-based); save explicit _school copies.
    # If pt_key provided, also recompute from PT-based classification for comparison.
    OUT_L9_path = SCRIPT_DIR / "output" / "Level9"
    OUT_L9_path.mkdir(parents=True, exist_ok=True)
    t2.to_csv(OUT_L9_path / "L9_T2_escorter_profile_school.csv", index=False)
    t3.to_csv(OUT_L9_path / "L9_T3_mode_school.csv", index=False)
    t6.to_csv(OUT_L9_path / "L9_T6_departure_school.csv", index=False)
    print("  → L9_T2/T3/T6 _school CSVs saved  (HH-based school_esc, same as Excel)")

    if pt_key is not None:
        sch_esc_pt9 = get_school_escort_trips(escort, pt_key)
        n_hh_based  = len(school_esc)
        n_pt_based  = len(sch_esc_pt9)
        print(f"  [PT-based school escort] n={n_pt_based:,}  "
              f"(HH-based n={n_hh_based:,}; diff={n_pt_based - n_hh_based:+d})")

        # Recompute escorter profile from PT-based classification
        esc9 = sch_esc_pt9.merge(hh_profile, on=HH_KEY + ["hh_member_person_no"],
                                  how="left")
        esc9["sex_num"]    = pd.to_numeric(esc9["sex_hh"],  errors="coerce")
        esc9["age_num"]    = pd.to_numeric(esc9["age_hh"],  errors="coerce")
        esc9["emp_num"]    = pd.to_numeric(esc9["emp_hh"],  errors="coerce")
        esc9["dep_hr"]     = pd.to_numeric(esc9["depart_hour_24h"], errors="coerce")
        esc9["child_code"] = (
            pd.to_numeric(esc9["employment_student_status_escorted"], errors="coerce")
            .round().astype("Int64")
        )

        rows_t2b = []
        for code in CHILD_ORDER:
            sub = esc9[esc9["child_code"] == code]
            if len(sub) == 0:
                continue
            n = len(sub)
            rows_t2b.append({
                "Child level":        STUDENT_MAP.get(code, str(code)),
                "n escort trips (PT)":n,
                "% female escorter":  round((sub["sex_num"] == 2).sum() / n * 100, 1),
                "Mean escorter age":  round(float(sub["age_num"].dropna().mean()), 1),
            })
        pd.DataFrame(rows_t2b).to_csv(
            OUT_L9_path / "L9_T2_escorter_profile_school_pt.csv", index=False)
        print("  → L9_T2 PT-based _school_pt.csv saved")

    # ── CHARTS ───────────────────────────────────────────────────────────────
    OUT_L9 = SCRIPT_DIR / "output" / "Level9"
    OUT_L9.mkdir(parents=True, exist_ok=True)

    SCH_CLR = {
        "Preschool":    "#5b3fa8",
        "Kindergarten": "#3a7bbf",
        "Elem/middle":  "#2e9e5b",
        "High school":  "#e07b39",
        "University":   "#c03a3a",
    }
    Q6_CLR = {
        "Nursery":       "#5b3fa8",
        "Elementary":    "#3a7bbf",
        "Middle school": "#2e9e5b",
        "High school":   "#e07b39",
        "University":    "#c03a3a",
        "Other":         "#aaaaaa",
    }

    # ── Chart A (T1): Escort probability with 95% CI ──────────────────────────
    fig, ax = plt.subplots(figsize=(8, 3.6))
    clrs_a = [SCH_CLR.get(lv, "#888888") for lv in t1["School level"]]
    y_a    = list(range(len(t1)))
    err_lo = (t1["Escort prob (%)"] - t1["95% CI lower (%)"]).values
    err_hi = (t1["95% CI upper (%)"] - t1["Escort prob (%)"]).values
    ax.barh(y_a, t1["Escort prob (%)"].values, xerr=[err_lo, err_hi],
            color=clrs_a, height=0.6,
            error_kw=dict(lw=1.5, capsize=4, ecolor="#333333"))
    ax.set_yticks(y_a)
    ax.set_yticklabels(t1["School level"].values)
    ax.set_xlabel("Escort probability (%)")
    ax.set_title("Table 1  –  Escort Probability by Child’s Education Level")
    ax.set_xlim(0, 110)
    for i, (_, row) in enumerate(t1.iterrows()):
        ax.text(row["Escort prob (%)"] + err_hi[i] + 1.5, i,
                f"{row['Escort prob (%)']:.1f}%  (n={int(row['n (students)'])})",
                va="center", fontsize=8)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, "L9_T1_escort_prob", OUT_L9)

    # ── Chart B (T2): Escorter profile — % female + mean age ─────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    clrs_b = [SCH_CLR.get(lv, "#888888") for lv in t2["Child level"]]
    y_b    = list(range(len(t2)))
    axes[0].barh(y_b, t2["% female escorter"].values, color=clrs_b, height=0.6)
    axes[0].set_yticks(y_b)
    axes[0].set_yticklabels(t2["Child level"].values)
    axes[0].set_xlabel("% Female escorter")
    axes[0].set_xlim(0, 100)
    axes[0].set_title("(a) % Female escorter")
    axes[0].invert_yaxis()
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    for i, v in enumerate(t2["% female escorter"].values):
        axes[0].text(v + 1, i, f"{v:.1f}%", va="center", fontsize=8)
    axes[1].barh(y_b, t2["Mean escorter age"].values, color=clrs_b, height=0.6)
    axes[1].set_yticks(y_b)
    axes[1].set_yticklabels([])
    axes[1].set_xlabel("Mean escorter age (years)")
    axes[1].set_xlim(0, 65)
    axes[1].set_title("(b) Mean escorter age")
    axes[1].invert_yaxis()
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    for i, (v, n) in enumerate(zip(t2["Mean escorter age"].values,
                                   t2["n escort trips"].values)):
        axes[1].text(v + 0.5, i, f"{v:.1f} yr  (n={int(n)})", va="center", fontsize=8)
    fig.suptitle("Table 2  –  Escorter Profile by Child’s School Level", y=1.02)
    fig.tight_layout()
    save_fig(fig, "L9_T2_escorter_profile", OUT_L9)

    # ── Chart C (T3): Mode share — stacked 100% horizontal bars ──────────────
    MODE_COLS   = ["Car (%)", "Walk (%)", "Bicycle (%)", "Bus (%)", "Rail (%)", "Other (%)"]
    MODE_LABELS = ["Car", "Walk", "Bicycle", "Bus", "Rail", "Other"]
    MODE_COLORS = ["#3a7bbf", "#2e9e5b", "#f0c040", "#e07b39", "#c03a3a", "#aaaaaa"]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    y_c    = list(range(len(t3)))
    left_c = np.zeros(len(t3))
    for col, lbl, clr in zip(MODE_COLS, MODE_LABELS, MODE_COLORS):
        vals = t3[col].values.astype(float)
        ax.barh(y_c, vals, left=left_c, label=lbl, color=clr, height=0.6)
        for j, (v, l) in enumerate(zip(vals, left_c)):
            if v >= 5:
                ax.text(l + v / 2, j, f"{v:.0f}%",
                        va="center", ha="center", fontsize=8,
                        color="white" if v > 20 else "#222222")
        left_c += vals
    ax.set_yticks(y_c)
    ax.set_yticklabels(t3["Child level"].values)
    ax.set_xlabel("Share (%)")
    ax.set_xlim(0, 101)
    ax.set_title("Table 3  –  Escort Mode by Child’s School Level")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, "L9_T3_mode", OUT_L9)

    # ── Chart D (T4): Escort frequency — bars ± SD, annotate % at 5 days ─────
    fig, ax = plt.subplots(figsize=(8, 4.2))
    clrs_d = [Q6_CLR.get(lv, "#888888") for lv in t4["School level (Q6)"]]
    x_d    = list(range(len(t4)))
    ax.bar(x_d, t4["Mean freq (days/wk)"].values,
           yerr=t4["SD"].values, color=clrs_d, width=0.6, zorder=3,
           error_kw=dict(lw=1.5, capsize=5, ecolor="#333333"))
    ax.set_xticks(x_d)
    ax.set_xticklabels(t4["School level (Q6)"].values, rotation=15, ha="right")
    ax.set_ylabel("Mean escort freq (days/week)")
    ax.set_ylim(0, 7)
    ax.axhline(5, ls="--", lw=1, color="#888888", label="5 days/wk (max)")
    ax.set_title("Table 4  –  Escort Frequency by School Level")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", ls=":", lw=0.8, alpha=0.6)
    for i, (_, row) in enumerate(t4.iterrows()):
        ax.text(i, row["Mean freq (days/wk)"] + row["SD"] + 0.12,
                f"{row['% 5 days/wk']:.0f}%\n@5d",
                ha="center", fontsize=7.5, color="#444444")
    fig.tight_layout()
    save_fig(fig, "L9_T4_frequency", OUT_L9)

    # ── Chart E (T5): Reason heatmap — full matrix (all reasons × school lvl) ─
    all_reasons = list(Q6_REASON.values())
    hm_data: dict = {}
    for lbl in Q6_ORDER:
        sub_r = q6[q6["dest1"] == lbl]["reas1"].dropna()
        if len(sub_r) == 0:
            continue
        vc = sub_r.value_counts(normalize=True) * 100
        hm_data[lbl] = {r: float(vc.get(r, 0.0)) for r in all_reasons}
    hm_df = pd.DataFrame(hm_data).T
    hm_df = hm_df.loc[[l for l in Q6_ORDER if l in hm_df.index]]

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(hm_df.values, aspect="auto", cmap="Blues", vmin=0, vmax=38)
    ax.set_xticks(range(len(all_reasons)))
    ax.set_xticklabels(all_reasons, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(hm_df)))
    ax.set_yticklabels(hm_df.index)
    ax.set_title(
        "Table 5  –  Escort Reason Prevalence by School Level  "
        "(% of respondents per school level)"
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("%", fontsize=9)
    for r in range(hm_df.shape[0]):
        for c in range(hm_df.shape[1]):
            v = float(hm_df.iloc[r, c])
            ax.text(c, r, f"{v:.0f}", ha="center", va="center",
                    fontsize=7.5, color="white" if v > 22 else "#222222")
    fig.tight_layout()
    save_fig(fig, "L9_T5_reasons_heatmap", OUT_L9)

    # ── Chart F (T6): Departure timing — morning vs afternoon by sex ──────────
    t6_plot = t6[t6["Escorter sex"].isin(["Female", "Male"])].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x6 = np.arange(len(t6_plot))
    w6 = 0.35
    ax.bar(x6 - w6 / 2, t6_plot["% morning (5–10h)"].values, width=w6,
           color="#3a7bbf", label="Morning (5–10h)")
    ax.bar(x6 + w6 / 2, t6_plot["% afternoon (14–19h)"].values, width=w6,
           color="#e07b39", label="Afternoon (14–19h)")
    ax.set_xticks(x6)
    ax.set_xticklabels(t6_plot["Escorter sex"].values)
    ax.set_ylabel("% of school escort trips")
    ax.set_ylim(0, 80)
    ax.set_title(
        "Table 6  –  Departure Time by Escorter Sex  (school escort trips)"
    )
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", ls=":", lw=0.8, alpha=0.6)
    for i, (_, row) in enumerate(t6_plot.iterrows()):
        ph = row["Peak morning hour"]
        ph_str = f"Peak:{int(ph)}h" if pd.notna(ph) else "Peak:—"
        ax.text(i - w6 / 2, row["% morning (5–10h)"] + 0.8,
                f"{ph_str}\nn={int(row['n trips'])}",
                ha="center", fontsize=7.5)
        ax.text(i + w6 / 2, row["% afternoon (14–19h)"] + 0.8,
                f"μ={row['Mean dep hour']:.1f}h",
                ha="center", fontsize=7.5)
    fig.tight_layout()
    save_fig(fig, "L9_T6_departure_sex", OUT_L9)

    print(f"\n  Level9 charts → {OUT_L9}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ── LEVEL 10: DISTANCE, INCOME & SCHOOL LOCATION ─────────────────────────────

def level10_distance_income_zone(pt: pd.DataFrame, hh: pd.DataFrame) -> None:
    section("LEVEL 10: Distance, Income & School Location")

    HH_KEY        = ["id_municipality_code", "id_b_zone_code",
                     "id_c_zone_code", "id_household_number"]
    STUDENT_CODES = {8, 9, 10, 11, 12}
    STUDENT_MAP   = {12: "Preschool", 11: "Kindergarten", 10: "Elem/middle",
                     9: "High school", 8: "University"}
    STU_ORDER     = ["Kindergarten", "Elem/middle", "High school", "University"]
    SCH_CLR       = {"Kindergarten": "#3a7bbf", "Elem/middle": "#2e9e5b",
                     "High school": "#e07b39",  "University": "#c03a3a"}
    CLR_ESC       = "#3a7bbf"
    CLR_IND       = "#e07b39"

    _INC_LABELS   = {
        1: "<¥1M",     2: "¥1–1.5M",  3: "¥1.5–2M",  4: "¥2–2.5M",
        5: "¥2.5–3M",  6: "¥3–4M",    7: "¥4–5M",    8: "¥5–7M",
        9: "≥¥7M",
    }

    OUT_L10 = SCRIPT_DIR / "output" / "Level10"
    OUT_L10.mkdir(parents=True, exist_ok=True)

    # ── Data prep ─────────────────────────────────────────────────────────────
    pt_l = pt.copy()
    pt_l["trip_purpose_str"] = pt_l["trip_purpose"].astype(str).str.zfill(2)
    pt_l["went_out"]                  = pd.to_numeric(pt_l["went_out"],                  errors="coerce")
    pt_l["employment_student_status"] = pd.to_numeric(pt_l["employment_student_status"], errors="coerce")
    pt_l["mode1_travel_time_min"]     = pd.to_numeric(pt_l["mode1_travel_time_min"],     errors="coerce")
    pt_l["origin_zone_code"]          = pd.to_numeric(pt_l["origin_zone_code"],          errors="coerce")
    pt_l["dest_zone_code"]            = pd.to_numeric(
        pt_l["dest_zone_code"].astype(str).str.strip(), errors="coerce"
    )

    valid  = pt_l[pt_l["went_out"] == 1]
    escort = valid[valid["trip_purpose_str"] == "12"].copy()
    school = valid[
        (valid["trip_purpose_str"] == "02") &
        valid["employment_student_status"].isin(STUDENT_CODES)
    ].copy()

    # Build is_escorted via escort-trip cross-reference (child column lookup)
    esc_parts = []
    for col in ["escorted_hh_member_no_1", "escorted_hh_member_no_2",
                "escorted_hh_member_no_3"]:
        if col not in escort.columns:
            continue
        tmp = escort[HH_KEY + [col]].copy()
        tmp["person_key"] = (
            pd.to_numeric(tmp[col].astype(str).str.strip(), errors="coerce")
            .dropna().astype(int).astype(str).str.zfill(2)
        )
        esc_parts.append(
            tmp.dropna(subset=["person_key"])[HH_KEY + ["person_key"]]
        )
    escorted_set = pd.concat(esc_parts, ignore_index=True).drop_duplicates()

    school["person_key"] = (
        pd.to_numeric(school["hh_member_person_no"].astype(str).str.strip(), errors="coerce")
        .astype(int).astype(str).str.zfill(2)
    )
    matched = school[HH_KEY + ["person_key"]].merge(
        escorted_set, on=HH_KEY + ["person_key"], how="left", indicator=True
    )
    school["is_escorted"] = (matched["_merge"].values == "both")
    school["stu_lvl"]     = school["employment_student_status"].map(STUDENT_MAP)
    school["tt"]          = school["mode1_travel_time_min"].where(
        school["mode1_travel_time_min"] < 999
    )
    school["same_zone"]   = school["origin_zone_code"] == school["dest_zone_code"]

    # ── Income join (household-level 4-field key) ─────────────────────────────
    hh_inc = (
        hh[HH_KEY + ["annual_income"]]
        .drop_duplicates(subset=HH_KEY)
        .assign(income_num=lambda d: pd.to_numeric(d["annual_income"], errors="coerce"))
    )
    school_inc = school.merge(hh_inc[HH_KEY + ["income_num"]], on=HH_KEY, how="left")

    # ── Sanity checks ─────────────────────────────────────────────────────────
    n_sch      = len(school)
    n_999      = (school["mode1_travel_time_min"] >= 999).sum()
    n_no_inc   = school_inc["income_num"].isin([10, 99]).sum()
    n_samezone = int(school["same_zone"].sum())
    print(f"\n  Sanity checks:")
    print(f"    School trips (purpose=02, students):      {n_sch}")
    print(f"    Travel time = 999 sentinel (excluded):    {n_999}  ({n_999/n_sch*100:.1f}%)")
    print(f"    Travel time valid (<999):                 {n_sch-n_999}  ({(n_sch-n_999)/n_sch*100:.1f}%)")
    print(f"    Income missing/refused (codes 10+99):     {n_no_inc}  ({n_no_inc/n_sch*100:.1f}%)")
    print(f"    Same-zone school trips:                   {n_samezone}  ({n_samezone/n_sch*100:.1f}%)")

    # ── 10A: Travel time stats ─────────────────────────────────────────────────
    print("\n  10A — Travel time:")
    tt_all = school["tt"].dropna()
    print(f"    All:         mean={tt_all.mean():.1f}  median={tt_all.median():.1f} min")
    esc_tt = school[school["is_escorted"]]["tt"].dropna()
    ind_tt = school[~school["is_escorted"]]["tt"].dropna()
    print(f"    Escorted:    mean={esc_tt.mean():.1f}  median={esc_tt.median():.1f} min  (n={len(esc_tt)})")
    print(f"    Independent: mean={ind_tt.mean():.1f}  median={ind_tt.median():.1f} min  (n={len(ind_tt)})")
    for lvl in STU_ORDER:
        sub = school[school["stu_lvl"] == lvl]["tt"].dropna()
        print(f"    {lvl}: mean={sub.mean():.1f}  median={sub.median():.1f} min")

    # ── 10B: Income escort rate stats ─────────────────────────────────────────
    print("\n  10B — Escort rate by income (valid codes 1–9):")
    inc_valid = school_inc[school_inc["income_num"].isin(range(1, 10))].copy()
    for code in range(1, 10):
        sub = inc_valid[inc_valid["income_num"] == code]
        er  = sub["is_escorted"].mean() * 100 if len(sub) else float("nan")
        print(f"    Code {code} ({_INC_LABELS[code]}): n={len(sub)}, escort={er:.1f}%")

    # ── 10C: Same-zone cross-tab ───────────────────────────────────────────────
    print("\n  10C — Same-zone cross-tab (is_escorted × same_zone):")
    ct = pd.crosstab(school["same_zone"], school["is_escorted"], margins=True)
    ct.index   = ["Different zone", "Same zone", "Total"]
    ct.columns = ["Independent", "Escorted", "Total"]
    print(ct.to_string())
    print("\n  Escort rate by zone and student type:")
    for lvl in STU_ORDER:
        sub = school[school["stu_lvl"] == lvl]
        n_sz  = int(sub["same_zone"].sum())
        n_dz  = int((~sub["same_zone"]).sum())
        er_sz = sub[sub["same_zone"]]["is_escorted"].mean() * 100 if n_sz else float("nan")
        er_dz = sub[~sub["same_zone"]]["is_escorted"].mean() * 100 if n_dz else float("nan")
        print(f"    {lvl}: same-zone escort={er_sz:.1f}% (n={n_sz}), diff-zone={er_dz:.1f}% (n={n_dz})")

    # ══ CHARTS ════════════════════════════════════════════════════════════════

    # ── Chart A: Travel time — histogram + boxplot + escorted vs independent ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # (a) Histogram
    ax = axes[0]
    tt_plot = tt_all[tt_all <= 90]
    ax.hist(tt_plot, bins=range(0, 95, 5), color=CLR_ESC, edgecolor="white", lw=0.5)
    ax.axvline(tt_all.median(), ls="--", color=CLR_IND, lw=1.5,
               label=f"Median = {int(tt_all.median())} min")
    ax.set_xlabel("Travel time (min)")
    ax.set_ylabel("Number of school trips")
    ax.set_title("(a) Distribution  (≤90 min shown)")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (b) Boxplot by student type
    ax = axes[1]
    box_data = [school[school["stu_lvl"] == lvl]["tt"].dropna() for lvl in STU_ORDER]
    bp = ax.boxplot(box_data, vert=True, patch_artist=True, widths=0.5,
                    medianprops=dict(color="white", lw=2),
                    flierprops=dict(marker=".", markersize=3, alpha=0.4))
    for patch, lvl in zip(bp["boxes"], STU_ORDER):
        patch.set_facecolor(SCH_CLR[lvl])
    ax.set_xticks(range(1, len(STU_ORDER) + 1))
    ax.set_xticklabels(STU_ORDER, rotation=12, ha="right")
    ax.set_ylabel("Travel time (min)")
    ax.set_title("(b) Travel time by student type")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for i, lvl in enumerate(STU_ORDER):
        med = school[school["stu_lvl"] == lvl]["tt"].dropna().median()
        ax.text(i + 1, med + 1.5, f"{int(med)}", ha="center", fontsize=7.5)

    # (c) Escorted vs independent
    ax = axes[2]
    grp_labels = [f"Escorted\n(n={len(esc_tt)})", f"Independent\n(n={len(ind_tt)})"]
    meds  = [esc_tt.median(), ind_tt.median()]
    means = [esc_tt.mean(),   ind_tt.mean()]
    ax.bar([0, 1], meds, color=[CLR_ESC, CLR_IND], width=0.5, label="Median", zorder=2)
    ax.scatter([0, 1], means, color="#222222", zorder=5, marker="D", s=50, label="Mean")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(grp_labels)
    ax.set_ylabel("Travel time (min)")
    ax.set_title("(c) Escorted vs independent")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for i, (med, mn) in enumerate(zip(meds, means)):
        ax.text(i, med + 0.4, f"med={int(med)}", ha="center", fontsize=8)
        ax.text(i, mn + 0.4 if mn > med else mn - 2,
                f"μ={mn:.1f}", ha="center", fontsize=7.5, color="#555555")

    fig.suptitle(
        "10A — Travel Time to School  "
        "(mode1_travel_time_min, value 999 excluded as sentinel)",
        y=1.02,
    )
    fig.tight_layout()
    save_fig(fig, "L10A_travel_time", OUT_L10)

    # ── Chart B: Income × Escort rate ─────────────────────────────────────────
    inc_grp = (
        inc_valid.groupby("income_num")["is_escorted"]
        .agg(mean="mean", count="count")
        .reset_index()
    )
    inc_grp["pct"]   = inc_grp["mean"] * 100
    inc_grp["label"] = inc_grp["income_num"].map(_INC_LABELS)
    ci_h = (
        1.96 * (inc_grp["mean"] * (1 - inc_grp["mean"]) / inc_grp["count"]).pow(0.5) * 100
    )

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x_b = list(range(len(inc_grp)))
    ax.bar(x_b, inc_grp["pct"], color="#3a7bbf", width=0.6,
           yerr=ci_h.values, error_kw=dict(lw=1.5, capsize=4, ecolor="#333333"))
    ax.set_xticks(x_b)
    ax.set_xticklabels(inc_grp["label"], rotation=30, ha="right")
    ax.set_ylabel("Escort rate (%)")
    ax.set_ylim(0, 55)
    ax.set_title(
        f"10B — Escort Rate by Household Income  "
        f"(valid n={len(inc_valid)};  excluded {n_no_inc} prefer-not-to-say / unknown)"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", ls=":", lw=0.8, alpha=0.6)
    for xi, (_, row) in zip(x_b, inc_grp.iterrows()):
        ax.text(xi, row["pct"] + ci_h.loc[_] + 0.8,
                f"{row['pct']:.0f}%\nn={int(row['count'])}",
                ha="center", fontsize=7.5)
    fig.tight_layout()
    save_fig(fig, "L10B_income_escort", OUT_L10)

    # ── Chart C: Zone analysis — same-zone % + cross-tab bars ─────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # (a) % same-zone by student type
    ax = axes[0]
    sz_pcts = [school[school["stu_lvl"] == lvl]["same_zone"].mean() * 100
               for lvl in STU_ORDER]
    ns_lvl  = [len(school[school["stu_lvl"] == lvl]) for lvl in STU_ORDER]
    ax.bar(range(len(STU_ORDER)), sz_pcts,
           color=[SCH_CLR[l] for l in STU_ORDER], width=0.6)
    ax.set_xticks(range(len(STU_ORDER)))
    ax.set_xticklabels(STU_ORDER, rotation=15, ha="right")
    ax.set_ylabel("% same-zone trips")
    ax.set_ylim(0, 35)
    ax.set_title("(a) % same-zone school trips by student type")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for i, (p, n) in enumerate(zip(sz_pcts, ns_lvl)):
        ax.text(i, p + 0.4, f"{p:.1f}%\n(n={n})", ha="center", fontsize=7.5)

    # (b) Escort rate: same zone vs different zone by student type
    ax = axes[1]
    x_c  = np.arange(len(STU_ORDER))
    w_c  = 0.35
    er_same = [
        school[(school["stu_lvl"] == lvl) & school["same_zone"]]["is_escorted"].mean() * 100
        if school[(school["stu_lvl"] == lvl) & school["same_zone"]].shape[0] > 0 else float("nan")
        for lvl in STU_ORDER
    ]
    er_diff = [
        school[(school["stu_lvl"] == lvl) & ~school["same_zone"]]["is_escorted"].mean() * 100
        for lvl in STU_ORDER
    ]
    ax.bar(x_c - w_c / 2, er_same, width=w_c, color=CLR_ESC, label="Same zone (short trip)")
    ax.bar(x_c + w_c / 2, er_diff, width=w_c, color=CLR_IND, label="Different zone")
    ax.set_xticks(x_c)
    ax.set_xticklabels(STU_ORDER, rotation=15, ha="right")
    ax.set_ylabel("Escort rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title("(b) Escort rate: same-zone vs different-zone")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", ls=":", lw=0.8, alpha=0.6)
    for i, (s, d) in enumerate(zip(er_same, er_diff)):
        if not pd.isna(s):
            ax.text(i - w_c / 2, s + 1, f"{s:.0f}%", ha="center", fontsize=7.5)
        ax.text(i + w_c / 2, d + 1, f"{d:.0f}%", ha="center", fontsize=7.5)

    fig.suptitle(
        "10C — Origin vs Destination Zone  "
        "(same_zone = origin_zone_code == dest_zone_code)",
        y=1.02,
    )
    fig.tight_layout()
    save_fig(fig, "L10C_zone_analysis", OUT_L10)

    print(f"\n  Level10 → {OUT_L10}")


# ── LEVEL 11: ADDITIONAL ANALYSES ────────────────────────────────────────────

def level11_additional_analyses(pt: pd.DataFrame, hh: pd.DataFrame,
                                pt_key: pd.DataFrame | None = None) -> None:
    section("LEVEL 11: Mode Share × Car Ownership | Peak Contribution | HH Composition")

    HH_KEY        = ["id_municipality_code", "id_b_zone_code",
                     "id_c_zone_code", "id_household_number"]
    STUDENT_CODES = {8, 9, 10, 11, 12}
    VEHICLE_MODES = {"03", "04", "05", "06", "07", "08", "09", "10", "11"}
    OUT_DIR = SCRIPT_DIR / "output" / "Level10_additions"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Common prep ───────────────────────────────────────────────────────────
    pt_l = pt.copy()
    pt_l["trip_purpose_str"]          = pt_l["trip_purpose"].astype(str).str.strip().str.zfill(2)
    pt_l["went_out"]                  = pd.to_numeric(pt_l["went_out"],                  errors="coerce")
    pt_l["employment_student_status"] = pd.to_numeric(pt_l["employment_student_status"], errors="coerce")
    pt_l["age"]                       = pd.to_numeric(pt_l["age"],                       errors="coerce")
    pt_l["depart_hour_24h"]           = pd.to_numeric(pt_l["depart_hour_24h"],           errors="coerce")
    pt_l["mode_str"]                  = pt_l["mode1_mode"].astype(str).str.strip().str.zfill(2)
    pt_l["esc_flag_str"]              = pt_l["escorting_flag"].astype(str).str.strip()

    valid = pt_l[pt_l["went_out"] == 1].copy()

    # ── LEVEL 10A: Mode Share by Car Ownership ────────────────────────────────
    print("\n  ── 10A: Mode Share × Car Ownership ──")

    # Car ownership: household-level, 999 = unknown
    hh_cars = (
        hh.drop_duplicates(subset=HH_KEY)[HH_KEY + ["owned_num_cars"]]
        .assign(cars_num=lambda d: pd.to_numeric(d["owned_num_cars"], errors="coerce")
                .where(pd.to_numeric(d["owned_num_cars"], errors="coerce") < 900))
    )

    school = valid[
        (valid["trip_purpose_str"] == "02") &
        valid["employment_student_status"].isin(STUDENT_CODES)
    ].copy()

    school_c = school.merge(hh_cars[HH_KEY + ["cars_num"]], on=HH_KEY, how="left")
    school_c["car_bin"] = school_c["cars_num"].apply(
        lambda x: "0" if x == 0 else ("1" if x == 1 else ("2+" if x >= 2 else None))
    )

    # Age-corrected mode category (vectorised)
    m = school_c["mode_str"]
    a = school_c["age"]
    cat = pd.Series("Other", index=school_c.index)
    cat[m == "01"] = "Walk"
    cat[m == "02"] = "Bicycle"
    cat[m.isin({"05", "06"})] = "Bus"
    cat[m == "16"] = "Rail"
    cat[m.isin({"03", "04", "09", "10", "11"})] = "Motorcycle"
    cat[m == "08"] = "Car passenger"
    cat[(m == "07") & (a >= 18)] = "Car driver"
    cat[(m == "07") & (a < 18)]  = "Car passenger"
    school_c["mode_cat"] = cat

    STACK_ORDER  = ["Walk", "Bicycle", "Bus", "Rail",
                    "Car passenger", "Car driver", "Motorcycle", "Other"]
    STACK_COLORS = ["#4caf50", "#cddc39", "#ff9800", "#f44336",
                    "#3a7bbf",  "#1a4d80",  "#9c27b0",  "#aaaaaa"]
    BIN_ORDER = ["0", "1", "2+"]

    ct_A  = (
        school_c[school_c["car_bin"].notna()]
        .groupby(["car_bin", "mode_cat"])
        .size().unstack(fill_value=0)
        .reindex(BIN_ORDER)
        .reindex(columns=STACK_ORDER, fill_value=0)
    )
    ct_A_pct = ct_A.div(ct_A.sum(axis=1), axis=0) * 100

    # Print
    print(f"{'Car ownership':<12}", end="")
    for c in STACK_ORDER:
        print(f"  {c[:8]:>8}", end="")
    print()
    for bin_lbl in BIN_ORDER:
        row = ct_A.loc[bin_lbl]
        pct = ct_A_pct.loc[bin_lbl]
        n   = int(row.sum())
        print(f"  {bin_lbl} cars (n={n})", end="")
        for c in STACK_ORDER:
            print(f"  {pct[c]:>7.1f}%", end="")
        print()

    # CSV
    csv_A = ct_A.copy()
    for c in STACK_ORDER:
        csv_A[f"{c} (%)"] = ct_A_pct[c].round(1)
    csv_A.index.name = "car_ownership"
    csv_A.to_csv(OUT_DIR / "A_mode_share_by_car_ownership.csv")
    print(f"  → A_mode_share_by_car_ownership.csv")

    # Chart A
    fig, ax = plt.subplots(figsize=(8, 5))
    ns_A   = ct_A.sum(axis=1)
    left_A = np.zeros(len(BIN_ORDER))
    for col, clr in zip(STACK_ORDER, STACK_COLORS):
        vals = ct_A_pct[col].values
        ax.bar(BIN_ORDER, vals, bottom=left_A, color=clr, label=col, width=0.55)
        for i, (v, l) in enumerate(zip(vals, left_A)):
            if v >= 5:
                ax.text(i, l + v / 2, f"{v:.0f}%",
                        ha="center", va="center", fontsize=8,
                        color="white" if v > 20 else "#222222")
        left_A += vals
    for i, n in enumerate(ns_A):
        ax.text(i, 101.5, f"n={int(n)}", ha="center", fontsize=8)
    ax.set_xlabel("Household car ownership")
    ax.set_ylabel("Mode share (%)")
    ax.set_ylim(0, 108)
    ax.set_title("10A — School Trip Mode Share by Household Car Ownership\n"
                 "(mode='07' + age<18 corrected to Car passenger)")
    ax.legend(loc="upper right", fontsize=8, bbox_to_anchor=(1.25, 1))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, "A_mode_share_by_car_ownership", OUT_DIR)

    # ── LEVEL 10A_v2: Escort Mode by Car Ownership (escorter trips) ──────────────
    print("\n  ── 10A_v2: Escort Mode × Car Ownership (escorter trips, purpose='12') ──")

    escort_trips = valid[valid["trip_purpose_str"] == "12"].copy()
    escort_c = escort_trips.merge(hh_cars[HH_KEY + ["cars_num"]], on=HH_KEY, how="left")
    escort_c["car_bin"] = escort_c["cars_num"].apply(
        lambda x: "0" if x == 0 else ("1" if x == 1 else ("2+" if x >= 2 else None))
    )

    m2 = escort_c["mode_str"]
    a2 = escort_c["age"]
    cat2 = pd.Series("Other", index=escort_c.index)
    cat2[m2 == "01"] = "Walk"
    cat2[m2 == "02"] = "Bicycle"
    cat2[m2.isin({"05", "06"})] = "Bus"
    cat2[m2 == "16"] = "Rail"
    cat2[m2.isin({"03", "04", "09", "10", "11"})] = "Motorcycle"
    cat2[m2 == "08"] = "Car passenger"
    cat2[(m2 == "07") & (a2 >= 18)] = "Car driver"
    cat2[(m2 == "07") & (a2 <  18)] = "Car passenger"
    escort_c["mode_cat"] = cat2

    ct_A2 = (
        escort_c[escort_c["car_bin"].notna()]
        .groupby(["car_bin", "mode_cat"])
        .size().unstack(fill_value=0)
        .reindex(BIN_ORDER)
        .reindex(columns=STACK_ORDER, fill_value=0)
    )
    ct_A2_pct = ct_A2.div(ct_A2.sum(axis=1), axis=0) * 100

    # Print
    print(f"{'Car ownership':<12}", end="")
    for c in STACK_ORDER:
        print(f"  {c[:8]:>8}", end="")
    print()
    for bin_lbl in BIN_ORDER:
        row = ct_A2.loc[bin_lbl]
        pct = ct_A2_pct.loc[bin_lbl]
        n   = int(row.sum())
        print(f"  {bin_lbl} cars (n={n})", end="")
        for c in STACK_ORDER:
            print(f"  {pct[c]:>7.1f}%", end="")
        print()

    # CSV
    csv_A2 = ct_A2.copy()
    for c in STACK_ORDER:
        csv_A2[f"{c} (%)"] = ct_A2_pct[c].round(1)
    csv_A2.index.name = "car_ownership"
    csv_A2.to_csv(OUT_DIR / "A2_escort_mode_by_car_ownership.csv")
    print(f"  → A2_escort_mode_by_car_ownership.csv")

    # Chart A2
    fig, ax = plt.subplots(figsize=(8, 5))
    ns_A2   = ct_A2.sum(axis=1)
    left_A2 = np.zeros(len(BIN_ORDER))
    for col, clr in zip(STACK_ORDER, STACK_COLORS):
        vals = ct_A2_pct[col].values
        ax.bar(BIN_ORDER, vals, bottom=left_A2, color=clr, label=col, width=0.55)
        for i, (v, l) in enumerate(zip(vals, left_A2)):
            if v >= 5:
                ax.text(i, l + v / 2, f"{v:.0f}%",
                        ha="center", va="center", fontsize=8,
                        color="white" if v > 20 else "#222222")
        left_A2 += vals
    for i, n in enumerate(ns_A2):
        ax.text(i, 101.5, f"n={int(n)}", ha="center", fontsize=8)
    ax.set_xlabel("Household car ownership")
    ax.set_ylabel("Mode share (%)")
    ax.set_ylim(0, 108)
    ax.set_title(
        "10A_v2 — Escort Trip Mode by Household Car Ownership\n"
        "(escorter trips only, trip_purpose='12'; compare with 10A: all school trips)"
    )
    ax.legend(loc="upper right", fontsize=8, bbox_to_anchor=(1.25, 1))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, "A2_escort_mode_by_car_ownership", OUT_DIR)

    # ── LEVEL 10A_v3: School Escort Mode by Car Ownership ────────────────────────
    print("\n  ── 10A_v3: School Escort Mode × Car Ownership (school escort confirmed) ──")

    # Person-level lookup: one row per (household, person), with student status
    student_lu = (
        valid[HH_KEY + ["hh_member_person_no", "employment_student_status"]]
        .drop_duplicates(subset=HH_KEY + ["hh_member_person_no"])
        .copy()
    )
    # Normalise hh_member_person_no → zero-padded 2-char string for the join
    student_lu["person_key"] = (
        pd.to_numeric(
            student_lu["hh_member_person_no"].astype(str).str.strip(),
            errors="coerce",
        )
        .astype("Int64")   # nullable int: NaN → pd.NA
        .astype(str)       # "1","2",...,"<NA>"
        .str.zfill(2)      # "01","02",...,"<NA>" (no-op for "<NA>")
    )
    student_lu = (
        student_lu[student_lu["person_key"].str.isdigit()]  # drop any NaN rows
        [HH_KEY + ["person_key", "employment_student_status"]]
        .rename(columns={
            "person_key":               "escorted_hh_member_no_1",
            "employment_student_status":"escorted_emp_status",
        })
    )

    # Escort trips that name an escorted member
    escort_v3 = (
        valid[valid["trip_purpose_str"] == "12"]
        .dropna(subset=["escorted_hh_member_no_1"])
        .copy()
    )

    # Join escorter → escorted person's employment status
    school_escort_v3 = escort_v3.merge(
        student_lu, on=HH_KEY + ["escorted_hh_member_no_1"], how="inner"
    )
    school_escort_v3["escorted_emp_num"] = pd.to_numeric(
        school_escort_v3["escorted_emp_status"], errors="coerce"
    )
    school_escort_v3 = school_escort_v3[
        school_escort_v3["escorted_emp_num"].isin(STUDENT_CODES)
    ].copy()

    n_total_esc  = int((valid["trip_purpose_str"] == "12").sum())
    n_named      = len(escort_v3)
    n_school_esc = len(school_escort_v3)
    print(f"  Escort trips: {n_total_esc} total → "
          f"{n_named} with named member → "
          f"{n_school_esc} confirmed school escorts")

    # Car ownership join
    escort_v3_c = school_escort_v3.merge(
        hh_cars[HH_KEY + ["cars_num"]], on=HH_KEY, how="left"
    )
    escort_v3_c["car_bin"] = escort_v3_c["cars_num"].apply(
        lambda x: "0" if x == 0 else ("1" if x == 1 else ("2+" if x >= 2 else None))
    )

    # Mode category (same age-correction as 10A/10A_v2)
    m3 = escort_v3_c["mode_str"]
    a3 = escort_v3_c["age"]
    cat3 = pd.Series("Other", index=escort_v3_c.index)
    cat3[m3 == "01"] = "Walk"
    cat3[m3 == "02"] = "Bicycle"
    cat3[m3.isin({"05", "06"})] = "Bus"
    cat3[m3 == "16"] = "Rail"
    cat3[m3.isin({"03", "04", "09", "10", "11"})] = "Motorcycle"
    cat3[m3 == "08"] = "Car passenger"
    cat3[(m3 == "07") & (a3 >= 18)] = "Car driver"
    cat3[(m3 == "07") & (a3 <  18)] = "Car passenger"
    escort_v3_c["mode_cat"] = cat3

    ct_A3 = (
        escort_v3_c[escort_v3_c["car_bin"].notna()]
        .groupby(["car_bin", "mode_cat"])
        .size().unstack(fill_value=0)
        .reindex(BIN_ORDER)
        .reindex(columns=STACK_ORDER, fill_value=0)
    )
    ct_A3_pct = ct_A3.div(ct_A3.sum(axis=1), axis=0) * 100

    # Print
    print(f"{'Car ownership':<12}", end="")
    for c in STACK_ORDER:
        print(f"  {c[:8]:>8}", end="")
    print()
    for bin_lbl in BIN_ORDER:
        row = ct_A3.loc[bin_lbl]
        pct = ct_A3_pct.loc[bin_lbl]
        n   = int(row.sum())
        print(f"  {bin_lbl} cars (n={n})", end="")
        for c in STACK_ORDER:
            print(f"  {pct[c]:>7.1f}%", end="")
        print()

    # CSV
    csv_A3 = ct_A3.copy()
    for c in STACK_ORDER:
        csv_A3[f"{c} (%)"] = ct_A3_pct[c].round(1)
    csv_A3.index.name = "car_ownership"
    csv_A3.to_csv(OUT_DIR / "A3_school_escort_mode_by_car_ownership.csv")
    print(f"  → A3_school_escort_mode_by_car_ownership.csv")

    # Chart A3
    fig, ax = plt.subplots(figsize=(8, 5))
    ns_A3   = ct_A3.sum(axis=1)
    left_A3 = np.zeros(len(BIN_ORDER))
    for col, clr in zip(STACK_ORDER, STACK_COLORS):
        vals = ct_A3_pct[col].values
        ax.bar(BIN_ORDER, vals, bottom=left_A3, color=clr, label=col, width=0.55)
        for i, (v, l) in enumerate(zip(vals, left_A3)):
            if v >= 5:
                ax.text(i, l + v / 2, f"{v:.0f}%",
                        ha="center", va="center", fontsize=8,
                        color="white" if v > 20 else "#222222")
        left_A3 += vals
    for i, n in enumerate(ns_A3):
        ax.text(i, 101.5, f"n={int(n)}", ha="center", fontsize=8)
    ax.set_xlabel("Household car ownership")
    ax.set_ylabel("Mode share (%)")
    ax.set_ylim(0, 108)
    ax.set_title(
        "10A_v3 — Escort Trip Mode by Car Ownership  (school escort only)\n"
        "Escorted person confirmed student via employment_student_status join"
    )
    ax.legend(loc="upper right", fontsize=8, bbox_to_anchor=(1.25, 1))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, "A3_school_escort_mode_by_car_ownership", OUT_DIR)

    # ── LEVEL 10B: Morning Peak Vehicle Contribution ───────────────────────────
    print("\n  ── 10B: Morning Peak Vehicle Contribution ──")

    vehicle     = valid[valid["mode_str"].isin(VEHICLE_MODES)]
    escort_v    = valid[(valid["trip_purpose_str"] == "12") & valid["mode_str"].isin(VEHICLE_MODES)]
    peak_mask   = vehicle["depart_hour_24h"].isin([7, 8])
    peak_v      = vehicle[peak_mask]
    peak_esc_v  = escort_v[escort_v["depart_hour_24h"].isin([7, 8])]

    # Hour-by-hour breakdown for peak
    rows_B = []
    for period, veh, esc in [
        ("07:00–08:00",
         vehicle[vehicle["depart_hour_24h"] == 7],
         escort_v[escort_v["depart_hour_24h"] == 7]),
        ("08:00–09:00",
         vehicle[vehicle["depart_hour_24h"] == 8],
         escort_v[escort_v["depart_hour_24h"] == 8]),
        ("Peak  07–09",  peak_v, peak_esc_v),
        ("All-day",      vehicle, escort_v),
    ]:
        n_veh = len(veh)
        n_esc = len(esc)
        rows_B.append({
            "Period":                 period,
            "Total vehicle trips":    n_veh,
            "Escorting vehicle trips":n_esc,
            "Contribution (%)":       round(n_esc / n_veh * 100, 2) if n_veh else 0.0,
        })

    df_B = pd.DataFrame(rows_B)
    print(df_B.to_string(index=False))
    df_B.to_csv(OUT_DIR / "B_morning_peak_contribution.csv", index=False)
    print(f"  → B_morning_peak_contribution.csv")

    # Chart B
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel (a): stacked absolute counts for each period
    ax = axes[0]
    periods_plot = ["07:00–08:00", "08:00–09:00", "Peak  07–09", "All-day"]
    df_Bp = df_B[df_B["Period"].isin(periods_plot)].set_index("Period").reindex(periods_plot)
    non_esc = df_Bp["Total vehicle trips"] - df_Bp["Escorting vehicle trips"]
    xs = np.arange(len(periods_plot))
    ax.bar(xs, non_esc.values,          color="#aaaaaa", label="Non-escorting",  width=0.55)
    ax.bar(xs, df_Bp["Escorting vehicle trips"].values,
           bottom=non_esc.values,       color="#e07b39", label="Escorting",       width=0.55)
    ax.set_xticks(xs)
    ax.set_xticklabels(periods_plot, rotation=15, ha="right")
    ax.set_ylabel("Vehicle trips")
    ax.set_title("(a) Total vehicle trips by period")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel (b): escort contribution %
    ax = axes[1]
    pcts = df_Bp["Contribution (%)"].values
    clrs_B = ["#3a7bbf" if p < 10 else "#e07b39" for p in pcts]
    ax.bar(xs, pcts, color=clrs_B, width=0.55)
    ax.set_xticks(xs)
    ax.set_xticklabels(periods_plot, rotation=15, ha="right")
    ax.set_ylabel("Escorting share of vehicle trips (%)")
    ax.set_title("(b) Escorting contribution (%)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", ls=":", lw=0.8, alpha=0.6)
    for i, (p, n_e) in enumerate(zip(pcts, df_Bp["Escorting vehicle trips"].values)):
        ax.text(i, p + 0.3, f"{p:.1f}%\n(n={int(n_e)})", ha="center", fontsize=8)

    fig.suptitle(
        "10B — Escorting Vehicle Trips as Share of All Vehicle Trips\n"
        "(vehicle modes 03–11; peak = depart_hour_24h ∈ {7, 8})"
    )
    fig.tight_layout()
    save_fig(fig, "B_morning_peak_contribution", OUT_DIR)

    # ── L10B school-only variant ──────────────────────────────────────────────
    if pt_key is not None:
        all_esc_valid = valid[valid["trip_purpose_str"] == "12"].copy()
        sch_esc_v     = get_school_escort_trips(all_esc_valid, pt_key)
        sch_esc_v_veh = sch_esc_v[sch_esc_v["mode_str"].isin(VEHICLE_MODES)]

        rows_Bs = []
        for period, veh, esc in [
            ("07:00–08:00",
             vehicle[vehicle["depart_hour_24h"] == 7],
             sch_esc_v_veh[sch_esc_v_veh["depart_hour_24h"] == 7]),
            ("08:00–09:00",
             vehicle[vehicle["depart_hour_24h"] == 8],
             sch_esc_v_veh[sch_esc_v_veh["depart_hour_24h"] == 8]),
            ("Peak  07–09",
             peak_v,
             sch_esc_v_veh[sch_esc_v_veh["depart_hour_24h"].isin([7, 8])]),
            ("All-day", vehicle, sch_esc_v_veh),
        ]:
            n_veh = len(veh)
            n_esc = len(esc)
            rows_Bs.append({
                "Period":                    period,
                "Total vehicle trips":        n_veh,
                "School escorting veh trips": n_esc,
                "Contribution (%)":           round(n_esc / n_veh * 100, 2) if n_veh else 0.0,
            })

        df_Bs = pd.DataFrame(rows_Bs)
        # Side-by-side comparison
        comp = df_B[["Period", "Contribution (%)"]].merge(
            df_Bs[["Period", "Contribution (%)"]].rename(
                columns={"Contribution (%)": "School (%)"}),
            on="Period",
        )
        comp["All escorts (%)"] = comp["Contribution (%)"]
        print("\n  [10B comparison] All escorts vs school-escort-only vehicle trips:")
        print(comp[["Period", "All escorts (%)", "School (%)"]].to_string(index=False))
        df_Bs.to_csv(OUT_DIR / "B_morning_peak_contribution_school.csv", index=False)
        print("  → B_morning_peak_contribution_school.csv")

    # ── LEVEL 10C: Escorting Rate by Household Composition ────────────────────
    print("\n  ── 10C: Escorting Rate by Household Composition ──")

    # Household escorts flag
    hh_escorts_flag = (
        valid.groupby(HH_KEY)["esc_flag_str"]
        .apply(lambda x: (x == "1").any())
        .reset_index(name="hh_escorts")
    )

    # HH size and school-age child count
    hh_base = hh.drop_duplicates(subset=HH_KEY)[HH_KEY + ["hh_size_incl_under5"]].copy()
    hh_base["hh_size"] = pd.to_numeric(hh_base["hh_size_incl_under5"], errors="coerce")
    hh_emp = hh.copy()
    hh_emp["emp_num"] = pd.to_numeric(hh_emp["employment_student_status"], errors="coerce")
    school_cnt = (
        hh_emp.groupby(HH_KEY)["emp_num"]
        .apply(lambda x: x.isin(STUDENT_CODES).sum())
        .reset_index(name="n_school_age")
    )

    hh_comp = (
        hh_base
        .merge(school_cnt,     on=HH_KEY, how="left")
        .merge(hh_escorts_flag, on=HH_KEY, how="left")
    )
    hh_comp["hh_escorts"]    = hh_comp["hh_escorts"].infer_objects(copy=False).fillna(False)
    hh_comp["n_school_age"]  = hh_comp["n_school_age"].fillna(0).astype(int)
    hh_comp["size_bin"]      = pd.cut(
        hh_comp["hh_size"], bins=[0, 2, 4, 99], labels=["1–2", "3–4", "5+"]
    )
    hh_comp["school_bin"]    = hh_comp["n_school_age"].clip(upper=3).map(
        {0: "0", 1: "1", 2: "2", 3: "3+"}
    )

    # Cross-tab 1: by HH size
    sz_grp = (
        hh_comp.groupby("size_bin", observed=True)["hh_escorts"]
        .agg(escort_rate="mean", n_escort="sum", n_total="count")
        .reset_index()
    )
    sz_grp["escort_rate_pct"] = sz_grp["escort_rate"] * 100
    sz_grp["ci95"] = (
        1.96 * (sz_grp["escort_rate"] * (1 - sz_grp["escort_rate"]) / sz_grp["n_total"]).pow(0.5) * 100
    )
    print("\n  Cross-tab 1: escort rate by HH size")
    print(sz_grp[["size_bin", "n_total", "n_escort", "escort_rate_pct", "ci95"]].to_string(index=False))

    # Cross-tab 2: by n school-age children
    sc_grp = (
        hh_comp.groupby("school_bin", observed=True)["hh_escorts"]
        .agg(escort_rate="mean", n_escort="sum", n_total="count")
        .reset_index()
    )
    sc_grp["escort_rate_pct"] = sc_grp["escort_rate"] * 100
    sc_grp["ci95"] = (
        1.96 * (sc_grp["escort_rate"] * (1 - sc_grp["escort_rate"]) / sc_grp["n_total"]).pow(0.5) * 100
    )
    print("\n  Cross-tab 2: escort rate by n school-age children")
    print(sc_grp[["school_bin", "n_total", "n_escort", "escort_rate_pct", "ci95"]].to_string(index=False))

    # CSV: merge both cross-tabs
    sz_grp_csv = sz_grp.rename(columns={"size_bin": "group", "escort_rate_pct": "escort_%", "n_total": "n"})
    sz_grp_csv["dimension"] = "HH size"
    sc_grp_csv = sc_grp.rename(columns={"school_bin": "group", "escort_rate_pct": "escort_%", "n_total": "n"})
    sc_grp_csv["dimension"] = "n school-age children"
    csv_C = pd.concat([sz_grp_csv[["dimension","group","n","n_escort","escort_%","ci95"]],
                       sc_grp_csv[["dimension","group","n","n_escort","escort_%","ci95"]]],
                      ignore_index=True)
    csv_C.to_csv(OUT_DIR / "C_escort_rate_by_household_composition.csv", index=False)
    print(f"  → C_escort_rate_by_household_composition.csv")

    # Chart C
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    CLR_BAR = "#3a7bbf"

    # Panel (a): HH size
    ax = axes[0]
    x0 = list(range(len(sz_grp)))
    ax.bar(x0, sz_grp["escort_rate_pct"].values,
           yerr=sz_grp["ci95"].values, color=CLR_BAR, width=0.55,
           error_kw=dict(lw=1.5, capsize=5, ecolor="#333333"))
    ax.set_xticks(x0)
    ax.set_xticklabels(sz_grp["size_bin"].astype(str).tolist())
    ax.set_xlabel("Household size")
    ax.set_ylabel("% households with any escorter")
    ax.set_ylim(0, 60)
    ax.set_title("(a) Escort rate by household size")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", ls=":", lw=0.8, alpha=0.6)
    for i, row in sz_grp.iterrows():
        ax.text(i, row["escort_rate_pct"] + row["ci95"] + 0.8,
                f"{row['escort_rate_pct']:.1f}%\n(n={int(row['n_total'])})",
                ha="center", fontsize=8)

    # Panel (b): n school-age children
    ax = axes[1]
    x1 = list(range(len(sc_grp)))
    ax.bar(x1, sc_grp["escort_rate_pct"].values,
           yerr=sc_grp["ci95"].values, color=CLR_BAR, width=0.55,
           error_kw=dict(lw=1.5, capsize=5, ecolor="#333333"))
    ax.set_xticks(x1)
    ax.set_xticklabels(sc_grp["school_bin"].astype(str).tolist())
    ax.set_xlabel("Number of school-age household members")
    ax.set_ylabel("% households with any escorter")
    ax.set_ylim(0, 65)
    ax.set_title("(b) Escort rate by n school-age children")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", ls=":", lw=0.8, alpha=0.6)
    for i, row in sc_grp.iterrows():
        ax.text(i, row["escort_rate_pct"] + row["ci95"] + 0.8,
                f"{row['escort_rate_pct']:.1f}%\n(n={int(row['n_total'])})",
                ha="center", fontsize=8)

    fig.suptitle(
        "10C — Household Escorting Rate by Size & School-age Child Composition\n"
        "(household escorts = any member with escorting_flag='1' in PersonTrip)"
    )
    fig.tight_layout()
    save_fig(fig, "C_escort_rate_by_household_composition", OUT_DIR)

    # ── L10C school-only variant ──────────────────────────────────────────────
    if pt_key is not None:
        all_esc_c = valid[valid["trip_purpose_str"] == "12"].copy()
        sch_esc_c = get_school_escort_trips(all_esc_c, pt_key)
        # HH-level flag: does this household have ≥1 school escort trip?
        hh_school_esc_flag = (
            sch_esc_c.groupby(HH_KEY)
            .size()
            .reset_index(name="_cnt")
            .assign(hh_school_escorts=True)
            [HH_KEY + ["hh_school_escorts"]]
        )
        hh_comp_s = (
            hh_base
            .merge(school_cnt,          on=HH_KEY, how="left")
            .merge(hh_school_esc_flag,  on=HH_KEY, how="left")
        )
        hh_comp_s["hh_school_escorts"] = (
            hh_comp_s["hh_school_escorts"].infer_objects(copy=False).fillna(False)
        )
        hh_comp_s["n_school_age"] = hh_comp_s["n_school_age"].fillna(0).astype(int)
        hh_comp_s["size_bin"]  = pd.cut(
            hh_comp_s["hh_size"], bins=[0, 2, 4, 99], labels=["1–2", "3–4", "5+"]
        )
        hh_comp_s["school_bin"] = hh_comp_s["n_school_age"].clip(upper=3).map(
            {0: "0", 1: "1", 2: "2", 3: "3+"}
        )

        sz_s = (
            hh_comp_s.groupby("size_bin", observed=True)["hh_school_escorts"]
            .agg(escort_rate="mean", n_escort="sum", n_total="count")
            .reset_index()
        )
        sz_s["escort_rate_pct"] = sz_s["escort_rate"] * 100

        sc_s = (
            hh_comp_s.groupby("school_bin", observed=True)["hh_school_escorts"]
            .agg(escort_rate="mean", n_escort="sum", n_total="count")
            .reset_index()
        )
        sc_s["escort_rate_pct"] = sc_s["escort_rate"] * 100

        csv_Cs = pd.concat([
            sz_s.rename(columns={"size_bin":   "Group"}).assign(Dimension="HH size"),
            sc_s.rename(columns={"school_bin": "Group"}).assign(Dimension="N school-age"),
        ], ignore_index=True)
        csv_Cs.to_csv(OUT_DIR / "C_escort_rate_by_household_composition_school.csv", index=False)
        print("  → C_escort_rate_by_household_composition_school.csv")

        # Comparison print
        old_all = hh_comp["hh_escorts"].mean() * 100
        new_sch = hh_comp_s["hh_school_escorts"].mean() * 100
        print(f"  [10C comparison] All-escort HH rate: {old_all:.1f}%  "
              f"→  School-escort HH rate: {new_sch:.1f}%")

    print(f"\n  Level10_additions → {OUT_DIR}")


# ═══════════════════════════════════════════════════════════════════════════
# ── LEVEL 11: PMT-Aligned Analysis & Gender of Escorted Child ───────────
# ═══════════════════════════════════════════════════════════════════════════

def level11_pmt_gender(pt: pd.DataFrame, supp: pd.DataFrame) -> None:
    section("LEVEL 11: PMT-Aligned Analysis & Gender of Escorted Child")

    HH_KEY = ["id_municipality_code", "id_b_zone_code",
               "id_c_zone_code", "id_household_number"]

    STUDENT_MAP = {11: "Kindergarten", 10: "Elem/middle",
                   9: "High school",   8: "University"}
    STU_ORDER   = ["Kindergarten", "Elem/middle", "High school", "University"]

    OUT_L11 = SCRIPT_DIR / "output" / "Level11"
    OUT_L11.mkdir(parents=True, exist_ok=True)

    # ── Common PT prep ────────────────────────────────────────────────────
    pt_l = pt.copy()
    pt_l["trip_purpose_str"] = pt_l["trip_purpose"].astype(str).str.strip().str.zfill(2)
    pt_l["went_out"]         = pd.to_numeric(pt_l["went_out"], errors="coerce")
    pt_l["mode_str"]         = pt_l["mode1_mode"].astype(str).str.strip().str.zfill(2)
    pt_l["age"]              = pd.to_numeric(pt_l["age"], errors="coerce")
    pt_l["emp_num"]          = pd.to_numeric(pt_l["employment_student_status"],
                                              errors="coerce")
    pt_l["sex_num"]          = pd.to_numeric(pt_l["sex"], errors="coerce")

    valid  = pt_l[pt_l["went_out"] == 1].copy()
    escort = valid[valid["trip_purpose_str"] == "12"].copy()

    # ── LEVEL 11A: Q6 Reasons Grouped by PMT Framework ───────────────────
    # Codebook (item 74/75/76 sub-item 3): code 02 = "Safety/crime concerns"
    # (combined option — traffic and crime are NOT separate survey items).
    PMT_GROUPS = {
        "Threat Appraisal":  ["02"],
        "Coping/Logistical": ["03", "04", "05", "06", "09"],
        "Other":             ["01", "07", "08", "10", "11"],
    }
    REASON_LABELS = {
        "01": "Physical condition",   "02": "Safety/crime concerns",
        "03": "No public transport",  "04": "Destination is far",
        "05": "Need to be on time",   "06": "Heavy baggage",
        "07": "Weather reasons",      "08": "Financial reasons",
        "09": "On the way to work",   "10": "Have spare time",
        "11": "Other",
    }
    PMT_COLORS = {
        "Threat Appraisal":  "#c03a3a",
        "Coping/Logistical": "#3a7bbf",
        "Other":             "#aaaaaa",
    }

    supp_l = supp.copy()
    supp_l["flag_str"]   = supp_l["q6_escorting_flag"].astype(str).str.strip()
    supp_l["reason_str"] = supp_l["q6_child1_escort_reason"].astype(str).str.strip()
    q6 = supp_l[supp_l["flag_str"] == "1"].copy()
    n_q6 = len(q6)

    reason_vc = q6["reason_str"].value_counts()

    pmt_totals = {}
    for grp, codes in PMT_GROUPS.items():
        pmt_totals[grp] = sum(reason_vc.get(c, 0) for c in codes)

    print(f"\n  11A — Q6 Escorters: n={n_q6}")
    for grp, n in pmt_totals.items():
        print(f"    {grp}: n={n}  ({n/n_q6*100:.1f}%)")

    # Ordered flat list: Threat → Coping → Other (top→bottom in chart)
    ordered_reasons = []
    for grp_name, codes in PMT_GROUPS.items():
        for code in codes:
            pct = reason_vc.get(code, 0) / n_q6 * 100
            n   = int(reason_vc.get(code, 0))
            ordered_reasons.append(
                (code, REASON_LABELS.get(code, code), grp_name, pct, n)
            )

    # Chart A1: detailed horizontal bars (reversed so Threat is at top)
    ordered_rev = list(reversed(ordered_reasons))
    n_bars = len(ordered_rev)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, (code, lbl, grp_name, pct, n) in enumerate(ordered_rev):
        clr = PMT_COLORS[grp_name]
        ax.barh(i, pct, color=clr, height=0.65)
        ax.text(pct + 0.3, i, f"{pct:.1f}%  (n={n})", va="center", fontsize=8)
    ax.set_yticks(range(n_bars))
    ax.set_yticklabels(
        [f"[{c}] {lbl}" for c, lbl, *_ in ordered_rev], fontsize=8
    )
    ax.set_xlabel(f"% of Q6 respondents  (n={n_q6})")
    ax.set_title("11A — Q6 Escort Reasons by PMT Framework")
    ax.set_xlim(0, 32)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Dividers between PMT groups (reversed order: Other at bottom → Traffic at top)
    grp_sizes   = [len(codes) for codes in reversed(list(PMT_GROUPS.values()))]
    cumulative  = 0
    for sz in grp_sizes[:-1]:
        cumulative += sz
        ax.axhline(cumulative - 0.5, color="#555", ls="--", lw=0.8)
    # PMT group legend
    legend_patches = [
        Patch(facecolor=clr, label=grp) for grp, clr in PMT_COLORS.items()
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8)
    fig.tight_layout()
    save_fig(fig, "L11A_PMT_reasons_detail", OUT_L11)

    # Chart A2: 4-group bar (one bar per PMT category)
    grp_labels = list(PMT_GROUPS.keys())
    grp_pcts   = [pmt_totals[g] / n_q6 * 100 for g in grp_labels]
    grp_ns     = [pmt_totals[g] for g in grp_labels]
    grp_clrs   = [PMT_COLORS[g] for g in grp_labels]
    n_grps     = len(grp_labels)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(n_grps), grp_pcts, color=grp_clrs, width=0.55)
    ax.set_xticks(range(n_grps))
    ax.set_xticklabels(grp_labels, rotation=15, ha="right")
    ax.set_ylabel("% of Q6 respondents")
    ax.set_ylim(0, 65)
    ax.set_title("11A — PMT Component Distribution  (Q6 child 1 escort reason)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for i, (p, n) in enumerate(zip(grp_pcts, grp_ns)):
        ax.text(i, p + 0.8, f"{p:.1f}%\n(n={n})", ha="center", fontsize=9)
    fig.tight_layout()
    save_fig(fig, "L11A_PMT_reasons", OUT_L11)

    # CSV A
    pd.DataFrame([
        {"code": code, "PMT_group": grp, "label": lbl,
         "n": int(reason_vc.get(code, 0)),
         "pct": round(reason_vc.get(code, 0) / n_q6 * 100, 1)}
        for code, lbl, grp, *_ in ordered_reasons
    ]).to_csv(OUT_L11 / "L11A_PMT_reasons.csv", index=False)

    # ── LEVEL 11B: Q6 Reason → Escort Mode Choice ────────────────────────
    # Household-level join: one reason per household (first respondent)
    q6_hh = q6[HH_KEY + ["reason_str"]].drop_duplicates(subset=HH_KEY)
    escort_q6 = escort.merge(q6_hh, on=HH_KEY, how="inner")

    m_b = escort_q6["mode_str"]
    mode_grp_b = pd.Series("Other", index=escort_q6.index)
    mode_grp_b[m_b == "01"] = "Walk"
    mode_grp_b[m_b.isin({"07", "08"})] = "Car"
    mode_grp_b[m_b.isin({"03","04","05","06","09","10","11","70","80"})] = "Other motorized"
    escort_q6 = escort_q6.copy()
    escort_q6["mode_grp"] = mode_grp_b

    MODE_ORDER_B  = ["Walk", "Car", "Other motorized", "Other"]
    MODE_COLORS_B = ["#4caf50", "#3a7bbf", "#ff9800", "#aaaaaa"]

    REASON_CODES = ["01","02","03","04","05","06","07","08","09","10","11"]
    ct_B = (
        escort_q6.groupby(["reason_str", "mode_grp"])
        .size().unstack(fill_value=0)
        .reindex(index=REASON_CODES, columns=MODE_ORDER_B, fill_value=0)
    )
    row_totals = ct_B.sum(axis=1)
    ct_B_pct = ct_B.div(row_totals, axis=0).mul(100).fillna(0)

    print("\n  11B — Escort mode by Q6 reason:")
    for code in REASON_CODES:
        n = int(row_totals.get(code, 0))
        lbl = REASON_LABELS.get(code, code)
        flag = " ⚠ low n" if n < 30 else ""
        modes = "  ".join(f"{m}={ct_B_pct.loc[code, m]:.0f}%"
                          for m in MODE_ORDER_B if m in ct_B_pct.columns)
        print(f"    [{code}] {lbl} (n={n}){flag}: {modes}")

    # CSV B
    csv_b = ct_B.copy()
    for m in MODE_ORDER_B:
        csv_b[f"{m} (%)"] = ct_B_pct[m].round(1)
    csv_b.index.name = "reason_code"
    csv_b["label"] = [REASON_LABELS.get(c, c) for c in csv_b.index]
    csv_b.to_csv(OUT_L11 / "L11B_reason_vs_mode.csv")

    # Chart B: stacked horizontal bars
    fig, ax = plt.subplots(figsize=(11, 5.5))
    y_b = np.arange(len(REASON_CODES))
    left_b = np.zeros(len(REASON_CODES))
    for mode, clr in zip(MODE_ORDER_B, MODE_COLORS_B):
        vals = ct_B_pct[mode].values
        ax.barh(y_b, vals, left=left_b, label=mode, color=clr, height=0.65)
        for i, (v, l) in enumerate(zip(vals, left_b)):
            if v >= 10:
                ax.text(l + v / 2, i, f"{v:.0f}%", ha="center", va="center",
                        fontsize=8, color="white" if v > 35 else "#222")
        left_b += vals
    ns_b = [int(row_totals.get(c, 0)) for c in REASON_CODES]
    ax.set_yticks(y_b)
    ax.set_yticklabels(
        [f"[{c}] {REASON_LABELS.get(c,c)}  (n={n}){' ⚠' if n < 30 else ''}"
         for c, n in zip(REASON_CODES, ns_b)],
        fontsize=8
    )
    ax.set_xlabel("% of escort trips per reason")
    ax.set_xlim(0, 101)
    ax.set_title("11B — Escort Mode Share by Stated Escort Reason  (Q6 child 1)")
    ax.legend(loc="lower right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, "L11B_reason_vs_mode", OUT_L11)

    # ── LEVEL 11C: Gender of Escorted Child × Escort Rate ─────────────────
    # Same logic as Level 9 Table 1 but split by child sex.
    # Denominator = unique school-trip-makers (purpose=02) per sex × level.
    # Numerator = those school-trip-makers who appear in escort trips
    #             (escorted_hh_member_no_1/2/3 matched to hh_member_person_no).
    # This keeps populations consistent and avoids rates > 100%.

    # Unique school-trip-makers with normalised person key
    school_trips = valid[
        (valid["trip_purpose_str"] == "02") &
        valid["emp_num"].isin(STUDENT_CODES)
    ].copy()
    school_trips["stu_lvl"] = school_trips["emp_num"].map(STUDENT_MAP)
    school_trips["pno_key"] = (
        pd.to_numeric(school_trips["hh_member_person_no"], errors="coerce")
        .astype("Int64").astype(str).str.zfill(2)
    )
    stu_dedup = (
        school_trips[school_trips["stu_lvl"].notna() &
                     school_trips["pno_key"].str.isdigit()]
        .drop_duplicates(subset=HH_KEY + ["pno_key"])
    )

    # Build escorted set from all three escorted_hh_member_no columns
    esc_pno_parts = []
    for col in ["escorted_hh_member_no_1", "escorted_hh_member_no_2",
                "escorted_hh_member_no_3"]:
        if col not in escort.columns:
            continue
        tmp = escort[HH_KEY + [col]].dropna(subset=[col]).copy()
        tmp["pno_key"] = tmp[col]  # already zero-padded strings after clean_data
        esc_pno_parts.append(tmp[HH_KEY + ["pno_key"]])
    escorted_set = pd.concat(esc_pno_parts, ignore_index=True).drop_duplicates()

    # Flag which unique school-trip-makers were escorted
    stu_flagged = stu_dedup.merge(
        escorted_set.assign(is_escorted=True),
        on=HH_KEY + ["pno_key"],
        how="left",
    )
    stu_flagged["is_escorted"] = stu_flagged["is_escorted"].fillna(False)

    SEX_MAP_C = {1: "Male", 2: "Female"}
    rows_11C = []
    for lvl in STU_ORDER:
        for sx_code, sx_name in SEX_MAP_C.items():
            mask = (stu_flagged["stu_lvl"] == lvl) & (stu_flagged["sex_num"] == sx_code)
            n_tot = mask.sum()
            n_esc = stu_flagged.loc[mask, "is_escorted"].sum()
            rows_11C.append({
                "School level": lvl, "Sex": sx_name,
                "n_escorted": int(n_esc), "n_total": int(n_tot),
                "Escort rate (%)": round(n_esc / n_tot * 100, 1) if n_tot else float("nan"),
            })

    df_11C = pd.DataFrame(rows_11C)
    print("\n  11C — Escort rate by child sex × school level:")
    print(df_11C.to_string(index=False))

    # Overall rate by sex
    overall_C = {}
    for sx_code, sx_name in SEX_MAP_C.items():
        mask_all = stu_flagged["sex_num"] == sx_code
        n_t = mask_all.sum()
        n_e = stu_flagged.loc[mask_all, "is_escorted"].sum()
        overall_C[sx_name] = {"n_esc": int(n_e), "n_tot": int(n_t),
                               "rate": n_e / n_t * 100 if n_t else float("nan")}
    print("  Overall:")
    for sx, v in overall_C.items():
        print(f"    {sx}: {v['n_esc']}/{v['n_tot']} = {v['rate']:.1f}%")

    # CSV C
    df_11C.to_csv(OUT_L11 / "L11C_gender_escort_rate.csv", index=False)

    # Chart C: 1×2 panels
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel (a): by sex × level
    ax = axes[0]
    x_c = np.arange(len(STU_ORDER))
    w_c = 0.35
    male_rates   = [df_11C.loc[(df_11C["School level"]==lvl) & (df_11C["Sex"]=="Male"),
                               "Escort rate (%)"].iloc[0] for lvl in STU_ORDER]
    female_rates = [df_11C.loc[(df_11C["School level"]==lvl) & (df_11C["Sex"]=="Female"),
                               "Escort rate (%)"].iloc[0] for lvl in STU_ORDER]
    male_ns      = [df_11C.loc[(df_11C["School level"]==lvl) & (df_11C["Sex"]=="Male"),
                               "n_total"].iloc[0] for lvl in STU_ORDER]
    female_ns    = [df_11C.loc[(df_11C["School level"]==lvl) & (df_11C["Sex"]=="Female"),
                               "n_total"].iloc[0] for lvl in STU_ORDER]

    ax.bar(x_c - w_c/2, male_rates,   width=w_c, color="#3a7bbf", label="Male")
    ax.bar(x_c + w_c/2, female_rates, width=w_c, color="#e07b39", label="Female")
    ax.set_xticks(x_c)
    ax.set_xticklabels(STU_ORDER, rotation=12, ha="right")
    ax.set_ylabel("Escort rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title("(a) By child gender and school level")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", ls=":", lw=0.8, alpha=0.6)
    for i, (mr, fr, mn, fn) in enumerate(zip(male_rates, female_rates, male_ns, female_ns)):
        ax.text(i - w_c/2, mr + 1.2, f"{mr:.0f}%\n(n={mn})", ha="center", fontsize=7.5)
        ax.text(i + w_c/2, fr + 1.2, f"{fr:.0f}%\n(n={fn})", ha="center", fontsize=7.5)

    # Panel (b): overall
    ax = axes[1]
    ov_rates = [overall_C["Male"]["rate"], overall_C["Female"]["rate"]]
    ov_ns    = [overall_C["Male"]["n_tot"], overall_C["Female"]["n_tot"]]
    ax.bar([0, 1], ov_rates, color=["#3a7bbf", "#e07b39"], width=0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"Male\n(n={ov_ns[0]})", f"Female\n(n={ov_ns[1]})"])
    ax.set_ylabel("Escort rate (%)")
    ax.set_ylim(0, 50)
    ax.set_title("(b) Overall by child gender")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", ls=":", lw=0.8, alpha=0.6)
    for i, (r, n) in enumerate(zip(ov_rates, ov_ns)):
        ax.text(i, r + 0.5, f"{r:.1f}%", ha="center", fontsize=11, fontweight="bold")

    fig.suptitle(
        "11C — Escort Rate by Child Gender and School Level\n"
        "(denominator = unique students with school trips; "
        "numerator = unique escorted children via escorted_hh_member_no_1)"
    )
    fig.tight_layout()
    save_fig(fig, "L11C_gender_escort_rate", OUT_L11)

    print(f"\n  Level11 → {OUT_L11}")


# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("\nOkinawa PT Survey — Analysis  (Levels 1–11)")
    print(f"Data dir : {DATA_DIR}")
    print(f"Output   : {SCRIPT_DIR / 'output'}")

    for name, path in FILES.items():
        if not path.exists():
            print(f"\n  ERROR: {path} not found.")
            sys.exit(1)

    print("\nLoading data …")
    hh   = load_csv("HH Survey",    FILES["hh"])
    pt   = load_csv("Person Trip",  FILES["pt"])
    supp = load_csv("Supplementary",FILES["supp"])

    # Centralised cleaning: strips strings, zero-pads ID columns, converts numerics.
    # Must run before any Level function so all downstream code sees clean data.
    clean_data(pt, hh)

    # Build pt_key lookup BEFORE level5 converts hh_member_person_no to numeric.
    # Maps (HH4 + person_no) join_key → employment_student_status.
    # Used by get_school_escort_trips to identify the escorted child's student status.
    _HH4 = ["id_municipality_code", "id_b_zone_code",
            "id_c_zone_code", "id_household_number"]
    pt_key = (
        pt[_HH4 + ["hh_member_person_no", "employment_student_status"]]
        .drop_duplicates(subset=_HH4 + ["hh_member_person_no"])
        .copy()
    )
    pt_key["join_key"] = (
        pt_key["id_municipality_code"].astype(str) + "_" +
        pt_key["id_b_zone_code"].astype(str) + "_" +
        pt_key["id_c_zone_code"].astype(str) + "_" +
        pt_key["id_household_number"].astype(str) + "_" +
        pt_key["hh_member_person_no"].astype(str).str.strip()
    )

    # Remaining numeric conversions (age/employment_student_status handled above).
    for df in [hh, pt, supp]:
        for col in ["sex", "annual_income", "home_ownership", "expansion_factor",
                    "num_trips", "hh_size_incl_under5", "hh_size_excl_under5"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    level1_data_quality(hh, pt, supp)
    level2_household_profile(hh)
    level3_age_by_status(hh)
    level4_trip_behavior(pt, hh)
    level5_student_independence(pt)
    level6_escorter_profile(pt, hh, pt_key)
    level7_escort_when_where(pt, hh, pt_key)
    level8_supplement_why_escort(supp, hh)
    level9_abm_cross_tabs(pt, hh, supp, pt_key)
    level10_distance_income_zone(pt, hh)
    level11_additional_analyses(pt, hh, pt_key)
    level11_pmt_gender(pt, supp)
    level12_school_charts(pt, pt_key)

    print(f"\n{'─'*70}")
    print(f"Done.  Outputs:")
    print(f"  Level1-3 → {OUT_L13}")
    print(f"  Level4   → {OUT_L4}")
    print(f"  Level5   → {OUT_L5}")
    print(f"  Level6   → {OUT_L6}")
    print(f"  Level7   → {OUT_L7}")
    print(f"  Level8   → {OUT_L8}")
    print(f"  Level9   → {SCRIPT_DIR / 'output' / 'Level9'}  +  L9_ABM_parameter_tables.xlsx")
    print(f"  Level10  → {SCRIPT_DIR / 'output' / 'Level10'}")
    print(f"  Level10+ → {SCRIPT_DIR / 'output' / 'Level10_additions'}")
    print(f"  Level11  → {SCRIPT_DIR / 'output' / 'Level11'}")
    print(f"  Level12  → (school PNGs in Level7 / Level9 / Level10_additions)")


# ═══════════════════════════════════════════════════════════════════════════
# ── LEVEL 12: School-classification PNG charts ────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def level12_school_charts(pt: pd.DataFrame,
                           pt_key: pd.DataFrame) -> None:
    """Plot all _school PNGs from previously written CSVs, plus derive T4_school."""
    section("LEVEL 12: School-classification PNG charts")

    HH_KEY_L  = ["id_municipality_code", "id_b_zone_code",
                  "id_c_zone_code", "id_household_number"]
    PERSON_KEY = HH_KEY_L + ["hh_member_person_no"]

    OUT_L7_  = SCRIPT_DIR / "output" / "Level7"
    OUT_L9_  = SCRIPT_DIR / "output" / "Level9"
    OUT_L10_ = SCRIPT_DIR / "output" / "Level10_additions"

    SCH_CLR = {
        "Preschool":    "#5b3fa8",
        "Kindergarten": "#3a7bbf",
        "Elem/middle":  "#2e9e5b",
        "High school":  "#e07b39",
        "University":   "#c03a3a",
    }
    CHILD_ORDER_STR = ["Preschool", "Kindergarten", "Elem/middle", "High school", "University"]
    MODE_COLS   = ["Car (%)", "Walk (%)", "Bicycle (%)", "Bus (%)", "Rail (%)", "Other (%)"]
    MODE_LABELS = ["Car", "Walk", "Bicycle", "Bus", "Rail", "Other"]
    MODE_COLORS = ["#3a7bbf", "#2e9e5b", "#f0c040", "#e07b39", "#c03a3a", "#aaaaaa"]

    # ── L7: departure hour distribution ───────────────────────────────────────
    df_l7       = pd.read_csv(OUT_L7_ / "L7_departure_hour_school.csv")
    show_hours  = list(range(5, 23))
    df_l7_plot  = (df_l7[df_l7["hour"].between(5, 22)]
                   .set_index("hour").reindex(show_hours, fill_value=0))
    n_total     = int(df_l7["n_trips"].sum())

    fig, ax = plt.subplots(figsize=(10, 4))
    xs = np.arange(len(show_hours))
    ax.bar(xs, df_l7_plot["pct"].values, color="#3a7bbf", alpha=0.85, width=0.7)
    ax.axvspan(-0.5,  5.5, alpha=0.08, color="#3a7bbf", zorder=0)
    ax.axvspan( 9.5, 14.5, alpha=0.08, color="#888888", zorder=0)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{h}:00" for h in show_hours], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Share of school escort trips (%)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ylim_l7 = float(df_l7_plot["pct"].max()) * 1.25
    ax.set_ylim(0, ylim_l7)
    ax.text(2.5,  ylim_l7 * 0.93, "Morning",   ha="center", va="top",
            fontsize=7, color="#2244aa", style="italic")
    ax.text(12.0, ylim_l7 * 0.93, "Afternoon", ha="center", va="top",
            fontsize=7, color="#555555", style="italic")
    ax.set_title(
        f"L7 — Departure Hour  (PT-based school escort trips,  n={n_total:,})\n"
        "Shading: morning 5–10 h  |  afternoon 14–19 h",
        fontweight="bold",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, "L7_departure_hour_school", OUT_L7_)

    # ── L9 T2: escorter profile (% female, mean age) ──────────────────────────
    t2 = pd.read_csv(OUT_L9_ / "L9_T2_escorter_profile_school.csv")
    t2 = t2[t2["Child level"].isin(CHILD_ORDER_STR)].copy()
    t2["Child level"] = pd.Categorical(t2["Child level"],
                                        categories=CHILD_ORDER_STR, ordered=True)
    t2 = t2.sort_values("Child level").reset_index(drop=True)

    clrs_b = [SCH_CLR.get(lv, "#888888") for lv in t2["Child level"]]
    y_b    = list(range(len(t2)))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].barh(y_b, t2["% female escorter"].values, color=clrs_b, height=0.6)
    axes[0].set_yticks(y_b)
    axes[0].set_yticklabels(t2["Child level"].values)
    axes[0].set_xlabel("% Female escorter")
    axes[0].set_xlim(0, 100)
    axes[0].set_title("(a) % Female escorter")
    axes[0].invert_yaxis()
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    for i, v in enumerate(t2["% female escorter"].values):
        axes[0].text(v + 1, i, f"{v:.1f}%", va="center", fontsize=8)
    axes[1].barh(y_b, t2["Mean escorter age"].values, color=clrs_b, height=0.6)
    axes[1].set_yticks(y_b)
    axes[1].set_yticklabels([])
    axes[1].set_xlabel("Mean escorter age (years)")
    axes[1].set_xlim(0, 65)
    axes[1].set_title("(b) Mean escorter age")
    axes[1].invert_yaxis()
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    for i, (v, n) in enumerate(zip(t2["Mean escorter age"].values,
                                   t2["n escort trips"].values)):
        axes[1].text(v + 0.5, i, f"{v:.1f} yr  (n={int(n)})", va="center", fontsize=8)
    fig.suptitle("Table 2 (school-only) — Escorter Profile by Child's School Level", y=1.02)
    fig.tight_layout()
    save_fig(fig, "L9_T2_escorter_profile_school", OUT_L9_)

    # ── L9 T3: mode share (stacked 100% horizontal bars) ─────────────────────
    t3 = pd.read_csv(OUT_L9_ / "L9_T3_mode_school.csv")
    t3 = t3[t3["Child level"].isin(CHILD_ORDER_STR)].copy()
    t3["Child level"] = pd.Categorical(t3["Child level"],
                                        categories=CHILD_ORDER_STR, ordered=True)
    t3 = t3.sort_values("Child level").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    y_c    = list(range(len(t3)))
    left_c = np.zeros(len(t3))
    for col, lbl, clr in zip(MODE_COLS, MODE_LABELS, MODE_COLORS):
        vals = t3[col].values.astype(float)
        ax.barh(y_c, vals, left=left_c, label=lbl, color=clr, height=0.6)
        for j, (v, l) in enumerate(zip(vals, left_c)):
            if v >= 5:
                ax.text(l + v / 2, j, f"{v:.0f}%",
                        va="center", ha="center", fontsize=8,
                        color="white" if v > 20 else "#222222")
        left_c += vals
    ax.set_yticks(y_c)
    ax.set_yticklabels(t3["Child level"].values)
    ax.set_xlabel("Share (%)")
    ax.set_xlim(0, 101)
    ax.set_title("Table 3 (school-only) — Escort Mode by Child's School Level")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, "L9_T3_mode_school", OUT_L9_)

    # ── L9 T6: departure time by escorter sex ─────────────────────────────────
    t6 = pd.read_csv(OUT_L9_ / "L9_T6_departure_school.csv")
    t6_plot = t6[t6["Escorter sex"].isin(["Female", "Male"])].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    x6 = np.arange(len(t6_plot))
    w6 = 0.35
    ax.bar(x6 - w6 / 2, t6_plot["% morning (5–10h)"].values,   width=w6,
           color="#3a7bbf", label="Morning (5–10h)")
    ax.bar(x6 + w6 / 2, t6_plot["% afternoon (14–19h)"].values, width=w6,
           color="#e07b39", label="Afternoon (14–19h)")
    ax.set_xticks(x6)
    ax.set_xticklabels(t6_plot["Escorter sex"].values)
    ax.set_ylabel("% of school escort trips")
    ax.set_ylim(0, 80)
    ax.set_title("Table 6 (school-only) — Departure Time by Escorter Sex")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", ls=":", lw=0.8, alpha=0.6)
    for i, (_, row) in enumerate(t6_plot.iterrows()):
        ph     = row["Peak morning hour"]
        ph_str = f"Peak:{int(ph)}h" if pd.notna(ph) else "Peak:—"
        ax.text(i - w6 / 2, row["% morning (5–10h)"] + 0.8,
                f"{ph_str}\nn={int(row['n trips'])}", ha="center", fontsize=7.5)
        ax.text(i + w6 / 2, row["% afternoon (14–19h)"] + 0.8,
                f"μ={row['Mean dep hour']:.1f}h",   ha="center", fontsize=7.5)
    fig.tight_layout()
    save_fig(fig, "L9_T6_departure_school", OUT_L9_)

    # ── L10B: morning peak contribution (school escort) ───────────────────────
    df_Bs        = pd.read_csv(OUT_L10_ / "B_morning_peak_contribution_school.csv")
    periods_plot = ["07:00–08:00", "08:00–09:00", "Peak  07–09", "All-day"]
    df_Bp        = (df_Bs[df_Bs["Period"].isin(periods_plot)]
                    .set_index("Period").reindex(periods_plot))
    non_sch      = df_Bp["Total vehicle trips"] - df_Bp["School escorting veh trips"]
    xs           = np.arange(len(periods_plot))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    ax.bar(xs, non_sch.values,
           color="#aaaaaa", label="Non-school-escort", width=0.55)
    ax.bar(xs, df_Bp["School escorting veh trips"].values,
           bottom=non_sch.values, color="#e07b39", label="School escorting", width=0.55)
    ax.set_xticks(xs)
    ax.set_xticklabels(periods_plot, rotation=15, ha="right")
    ax.set_ylabel("Vehicle trips")
    ax.set_title("(a) Total vehicle trips by period")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    pcts   = df_Bp["Contribution (%)"].values
    clrs_B = ["#3a7bbf" if p < 10 else "#e07b39" for p in pcts]
    ax.bar(xs, pcts, color=clrs_B, width=0.55)
    ax.set_xticks(xs)
    ax.set_xticklabels(periods_plot, rotation=15, ha="right")
    ax.set_ylabel("School escorting share of vehicle trips (%)")
    ax.set_title("(b) School escorting contribution (%)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", ls=":", lw=0.8, alpha=0.6)
    for i, (p, n_e) in enumerate(zip(pcts, df_Bp["School escorting veh trips"].values)):
        ax.text(i, p + 0.2, f"{p:.1f}%\n(n={int(n_e)})", ha="center", fontsize=8)
    fig.suptitle(
        "10B (school-only) — School Escorting Vehicle Trips as Share of All Vehicle Trips\n"
        "(vehicle modes 03–11; escorted person confirmed student via PT join)"
    )
    fig.tight_layout()
    save_fig(fig, "B_morning_peak_contribution_school", OUT_L10_)

    # ── L10C: HH escort rate by size & school-age count ──────────────────────
    csv_Cs = pd.read_csv(OUT_L10_ / "C_escort_rate_by_household_composition_school.csv")
    sz_grp = csv_Cs[csv_Cs["Dimension"] == "HH size"].copy().reset_index(drop=True)
    sc_grp = csv_Cs[csv_Cs["Dimension"] == "N school-age"].copy().reset_index(drop=True)
    for df_g in [sz_grp, sc_grp]:
        p = df_g["escort_rate"]
        df_g["ci95"] = 1.96 * (p * (1 - p) / df_g["n_total"]).pow(0.5) * 100

    CLR_BAR = "#3a7bbf"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    ax = axes[0]
    x0 = list(range(len(sz_grp)))
    ax.bar(x0, sz_grp["escort_rate_pct"].values,
           yerr=sz_grp["ci95"].values, color=CLR_BAR, width=0.55,
           error_kw=dict(lw=1.5, capsize=5, ecolor="#333333"))
    ax.set_xticks(x0)
    ax.set_xticklabels(sz_grp["Group"].astype(str).tolist())
    ax.set_xlabel("Household size")
    ax.set_ylabel("% households with school escort trip")
    ax.set_ylim(0, 60)
    ax.set_title("(a) School escort rate by household size")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", ls=":", lw=0.8, alpha=0.6)
    for i, row in sz_grp.iterrows():
        ax.text(i, row["escort_rate_pct"] + row["ci95"] + 0.8,
                f"{row['escort_rate_pct']:.1f}%\n(n={int(row['n_total'])})",
                ha="center", fontsize=8)
    ax = axes[1]
    x1 = list(range(len(sc_grp)))
    ax.bar(x1, sc_grp["escort_rate_pct"].values,
           yerr=sc_grp["ci95"].values, color=CLR_BAR, width=0.55,
           error_kw=dict(lw=1.5, capsize=5, ecolor="#333333"))
    ax.set_xticks(x1)
    ax.set_xticklabels(sc_grp["Group"].astype(str).tolist())
    ax.set_xlabel("Number of school-age household members")
    ax.set_ylabel("% households with school escort trip")
    ax.set_ylim(0, 65)
    ax.set_title("(b) School escort rate by n school-age children")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", ls=":", lw=0.8, alpha=0.6)
    for i, row in sc_grp.iterrows():
        ax.text(i, row["escort_rate_pct"] + row["ci95"] + 0.8,
                f"{row['escort_rate_pct']:.1f}%\n(n={int(row['n_total'])})",
                ha="center", fontsize=8)
    fig.suptitle(
        "10C (school-only) — Household School Escort Rate by Size & School-age Composition\n"
        "(school escort = household has ≥1 PT-classified school escort trip)"
    )
    fig.tight_layout()
    save_fig(fig, "C_escort_rate_by_household_composition_school", OUT_L10_)

    # ── L9 T4 (school): trip frequency — school escorters vs non-escorters ────
    # Derive from pt using get_school_escort_trips (pt_key already built in main).
    STUDENT_MAP_NUM = {12: "Preschool", 11: "Kindergarten", 10: "Elem/middle",
                        9: "High school",  8: "University"}

    pt_l = pt.copy()
    pt_l["went_out_num"]      = pd.to_numeric(pt_l["went_out"],      errors="coerce")
    pt_l["trip_purpose_str"]  = pt_l["trip_purpose"].astype(str).str.strip().str.zfill(2)
    pt_l["num_trips_num"]     = pd.to_numeric(pt_l["num_trips"],     errors="coerce")
    # hh_member_person_no is numeric after level5; re-pad for PERSON_KEY joins
    pt_l["pno_pad"] = (
        pd.to_numeric(pt_l["hh_member_person_no"], errors="coerce")
        .astype("Int64").astype(str).str.zfill(2)
    )

    valid_l  = pt_l[pt_l["went_out_num"] == 1].copy()
    esc_all  = valid_l[valid_l["trip_purpose_str"] == "12"].copy()
    sch_esc_t4 = get_school_escort_trips(esc_all, pt_key)

    sch_esc_t4["child_code"] = (
        pd.to_numeric(sch_esc_t4["employment_student_status_escorted"], errors="coerce")
        .round().astype("Int64")
    )
    # Normalise person key to zero-padded string for consistent joining
    sch_esc_t4["pno_pad"] = (
        pd.to_numeric(sch_esc_t4["hh_member_person_no"], errors="coerce")
        .astype("Int64").astype(str).str.zfill(2)
    )
    PKEY_PAD = HH_KEY_L + ["pno_pad"]

    sch_escorters = sch_esc_t4[PKEY_PAD].drop_duplicates()

    persons_pt = (
        pt_l[HH_KEY_L + ["pno_pad", "num_trips_num"]]
        .drop_duplicates(subset=PKEY_PAD)
    )

    sch_esc_with_trips = (
        sch_esc_t4[PKEY_PAD + ["child_code"]]
        .drop_duplicates(subset=PKEY_PAD)
        .merge(persons_pt, on=PKEY_PAD, how="left")
    )

    esc_keys = set(map(tuple, sch_escorters[PKEY_PAD].values.tolist()))
    all_persons = persons_pt.copy()
    all_persons["is_sch_esc"] = [
        tuple(r) in esc_keys
        for r in all_persons[PKEY_PAD].values.tolist()
    ]
    non_escorters = all_persons[~all_persons["is_sch_esc"]]

    rows_t4s = []
    for code in [12, 11, 10, 9, 8]:
        lbl  = STUDENT_MAP_NUM.get(code, str(code))
        vals = (sch_esc_with_trips[sch_esc_with_trips["child_code"] == code]
                ["num_trips_num"].dropna())
        if len(vals) == 0:
            continue
        rows_t4s.append({
            "Group":           f"School escorters ({lbl} child)",
            "Type":            "School escorter",
            "Child level":     lbl,
            "n persons":       int(len(vals)),
            "Mean num_trips":  round(float(vals.mean()), 2),
            "SD":              round(float(vals.std()),  2),
            "% ≥5 trips/day":  round((vals >= 5).sum() / len(vals) * 100, 1),
            "Median":          int(vals.median()),
        })
    ne_vals = non_escorters["num_trips_num"].dropna()
    rows_t4s.append({
        "Group":           "Non-escorters",
        "Type":            "Non-escorter",
        "Child level":     "—",
        "n persons":       int(len(ne_vals)),
        "Mean num_trips":  round(float(ne_vals.mean()), 2),
        "SD":              round(float(ne_vals.std()),  2),
        "% ≥5 trips/day":  round((ne_vals >= 5).sum() / len(ne_vals) * 100, 1),
        "Median":          int(ne_vals.median()),
    })
    t4s = pd.DataFrame(rows_t4s)
    t4s.to_csv(OUT_L9_ / "L9_T4_frequency_school.csv", index=False)
    print(f"  → L9_T4_frequency_school.csv")

    esc_rows = t4s[t4s["Type"] == "School escorter"].reset_index(drop=True)
    ne_row   = t4s[t4s["Type"] == "Non-escorter"].iloc[0]
    bar_colors = [SCH_CLR.get(lv, "#888888") for lv in esc_rows["Child level"]]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x_esc = np.arange(len(esc_rows))
    ax.bar(x_esc, esc_rows["Mean num_trips"].values,
           yerr=esc_rows["SD"].values, color=bar_colors, width=0.55, zorder=3,
           error_kw=dict(lw=1.5, capsize=5, ecolor="#333333"),
           label="School escorters (by child level)")
    ne_mean = float(ne_row["Mean num_trips"])
    ax.axhline(ne_mean, ls="--", lw=1.5, color="#888888",
               label=f"Non-escorters mean  (n={int(ne_row['n persons']):,})")
    ax.text(len(esc_rows) - 0.45, ne_mean + 0.08,
            f"Non-esc. μ={ne_mean:.2f}\nSD={ne_row['SD']:.2f}",
            fontsize=7.5, va="bottom", color="#555555")
    ax.set_xticks(x_esc)
    ax.set_xticklabels(
        [f"{r['Child level']}\n(n={int(r['n persons']):,})" for _, r in esc_rows.iterrows()],
        rotation=15, ha="right",
    )
    ax.set_ylabel("Mean daily trip count (num_trips)")
    ylim_t4 = max(float((esc_rows["Mean num_trips"] + esc_rows["SD"]).max()) + 0.8,
                  ne_mean + 1.5)
    ax.set_ylim(0, ylim_t4)
    ax.set_title(
        "Table 4 (school-only) — Daily Trip Frequency: School Escorters vs Non-escorters\n"
        "(school escorters identified via PT-based escort classification)"
    )
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", ls=":", lw=0.8, alpha=0.6)
    for i, (_, row) in enumerate(esc_rows.iterrows()):
        ax.text(i, row["Mean num_trips"] + row["SD"] + 0.12,
                f"{row['% ≥5 trips/day']:.0f}%\n≥5 trips",
                ha="center", fontsize=7.5, color="#444444")
    fig.tight_layout()
    save_fig(fig, "L9_T4_frequency_school", OUT_L9_)

    print(f"\n  Level12 school charts → {OUT_L9_}  |  {OUT_L7_}  |  {OUT_L10_}")


if __name__ == "__main__":
    main()
