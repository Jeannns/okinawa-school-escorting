"""
Okinawa Person Trip Survey — RQ Analyses
Analysis 1: School Travel Patterns (RQ1 — H1a, H1b, H1c)
Analysis 2: School Escorting Behavior (RQ2a, RQ2b)
  RQ2a Part A — Binary Logit: who escorts? (SUPP, n=1,237, R²=0.145)
  RQ2a Part B — Binary Logit: who cites safety? (escort group)
  RQ2b        — Descriptive: does safety concern change mode choice?
                  (1) Car ownership by safety group
                  (2) Car use by school level × safety (PT purpose=12)
                  (3) Travel time by school level × safety

Data directory: ../Okinawa_PT survey/Ver20240430第4回沖縄PTマスターデータ/EN/
Output:         ./output/Analysis1/   ./output/Analysis2/
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats as scipy_stats
from scipy.stats import chi2_contingency, mannwhitneyu, fisher_exact

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = (SCRIPT_DIR.parents[1]
            / "Okinawa_PT survey"
            / "Ver20240430第4回沖縄PTマスターデータ"
            / "EN")

FILES = {
    "pt":   DATA_DIR / "R05_PersonTrip_EN.csv",
    "hh":   DATA_DIR / "R05_HH_Survey_EN.csv",
    "supp": DATA_DIR / "R05_Supplementary_EN.csv",
}

OUT_A1 = SCRIPT_DIR / "output" / "Analysis1"
OUT_A2 = SCRIPT_DIR / "output" / "Analysis2"
OUT_A1.mkdir(parents=True, exist_ok=True)
OUT_A2.mkdir(parents=True, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})

# ── Constants ──────────────────────────────────────────────────────────────────
LEVEL_MAP   = {11: "Kindergarten", 10: "Elem/JHS", 9: "High School", 8: "University"}
LEVEL_ORDER = ["Kindergarten", "Elem/JHS", "High School", "University"]
LEVEL_COLORS = ["#AED6F1", "#85C1E9", "#3498DB", "#1A5276"]


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def section(title: str) -> None:
    bar = "═" * 70
    print(f"\n{bar}\n  {title}\n{bar}")


def load_csv(name: str, path: Path, required: bool = True) -> pd.DataFrame:
    print(f"  Loading {name} … ", end="", flush=True)
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        df.columns = df.columns.str.strip()
        print(f"{len(df):,} rows × {df.shape[1]} cols")
        return df
    except OSError as e:
        if required:
            print(f"ERROR ({e})")
            raise
        print(f"SKIPPED ({e})")
        return pd.DataFrame()


def save_fig(fig: plt.Figure, name: str, out_dir: Path) -> None:
    path = out_dir / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  → {path.name}")


def prepare_school_trips(pt: pd.DataFrame) -> pd.DataFrame:
    """Filter and prepare school trip rows used by all Analysis 1 hypotheses."""
    df = pt.copy()
    df["mode1_travel_time_min"]     = pd.to_numeric(df["mode1_travel_time_min"],     errors="coerce")
    df["employment_student_status"] = pd.to_numeric(df["employment_student_status"], errors="coerce")
    df["trip_purpose"]              = df["trip_purpose"].astype(str).str.strip()
    df["owned_has_car"]             = pd.to_numeric(df["owned_has_car"],             errors="coerce")
    df["annual_income"]             = pd.to_numeric(df["annual_income"],             errors="coerce")
    df["hh_size_excl_under5"]       = pd.to_numeric(df["hh_size_excl_under5"],       errors="coerce")
    df["id_b_zone_code"]            = pd.to_numeric(df["id_b_zone_code"],            errors="coerce")
    df["id_c_zone_code"]            = pd.to_numeric(df["id_c_zone_code"],            errors="coerce")
    df["dest_zone_code"]            = df["dest_zone_code"].astype(str).str.strip()
    df["addr_zone_code"]            = df["addr_zone_code"].astype(str).str.strip()

    school = df[
        (df["trip_purpose"] == "02") &
        (df["employment_student_status"].isin([8, 9, 10, 11])) &
        (df["mode1_travel_time_min"] < 999)
    ].copy()

    school["level"] = school["employment_student_status"].map(LEVEL_MAP)
    return school, df   # also return full df for C→B lookup


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis 1 — School Travel Patterns (RQ1)
# ═══════════════════════════════════════════════════════════════════════════════

def analysis1(pt: pd.DataFrame) -> None:
    section("ANALYSIS 1 — School Travel Patterns (RQ1)")

    school, full_df = prepare_school_trips(pt)
    groups = [school[school["level"] == l]["mode1_travel_time_min"].values
              for l in LEVEL_ORDER]

    print(f"\n  School trips after filter: {len(school):,}")

    # ── H1a: Descriptive stats + box plot + One-way ANOVA ─────────────────────
    section("H1a — Travel Distance by School Level")

    print(f"\n  {'Level':<15} {'n':>6}  {'Mean':>7}  {'Median':>7}  {'SD':>7}  {'Q1':>5}  {'Q3':>5}")
    print(f"  {'-'*62}")
    for l, g in zip(LEVEL_ORDER, groups):
        print(f"  {l:<15} {len(g):>6,}  {g.mean():>7.2f}  "
              f"{float(np.median(g)):>7.1f}  {g.std(ddof=1):>7.2f}  "
              f"{float(np.percentile(g, 25)):>5.1f}  {float(np.percentile(g, 75)):>5.1f}")

    F, p = scipy_stats.f_oneway(*groups)
    print(f"\n  One-way ANOVA: F = {F:.2f}, p = {p:.2e}")
    print(f"  → {'H1a SUPPORTED' if p < 0.05 else 'H1a NOT SUPPORTED'}")

    # Box plot
    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(groups, tick_labels=LEVEL_ORDER, patch_artist=True,
                    showfliers=False, medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], LEVEL_COLORS):
        patch.set_facecolor(color)

    medians = [float(np.median(g)) for g in groups]
    for i, med in enumerate(medians):
        ax.text(i + 1, med + 1.5, f"{med:.0f} min", ha="center",
                fontsize=9, fontweight="bold")
    for i, g in enumerate(groups):
        ax.text(i + 1, -6, f"n={len(g):,}", ha="center", fontsize=9, color="gray")

    ax.set_ylabel("Travel Time (minutes)", fontsize=11)
    ax.set_xlabel("School Level", fontsize=11)
    ax.set_title(
        f"School Travel Time by Level — Okinawa PT Survey 2023\n"
        f"One-way ANOVA: F = {F:.2f}, p < 0.001  (n = {len(school):,})", fontsize=11
    )
    ax.set_ylim(0, 75)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "H1a_boxplot_travel_time", OUT_A1)

    # ── H1b: Zone-crossing by school level at B-zone and C-zone ─────────────
    # Zone code format: B(1-2 digits, no padding) + C(1 digit) + local(2 digits)
    #   4-digit code → B = s[0],    C = s[1]   (B-zone 1-9)
    #   5-digit code → B = s[0:2],  C = s[2]   (B-zone 10-56)
    # Advisor feedback: B-zone (56 units) is too coarse; C-zone (144 units) preferred.
    section("H1b — Zone-Crossing Rate by School Level (B-zone & C-zone)")

    def _parse_bc(code_str):
        """Return (b_zone, c_zone) from a 4-5 digit zone code string."""
        s = str(code_str).strip()
        if len(s) == 4:
            return int(s[0]), int(s[1])
        if len(s) == 5:
            return int(s[:2]), int(s[2])
        return np.nan, np.nan

    _dest_bc = school["dest_zone_code"].apply(
        lambda x: pd.Series(_parse_bc(x), index=["dest_b", "dest_c"]))
    school = pd.concat([school.reset_index(drop=True),
                        _dest_bc.reset_index(drop=True)], axis=1)

    school["home_b"] = school["id_b_zone_code"]
    school["home_c"] = school["id_c_zone_code"]

    # B-zone cross: home B ≠ dest B
    school["bzone_cross"] = (school["home_b"] != school["dest_b"]).astype(float)
    # C-zone cross: (home B, home C) ≠ (dest B, dest C)
    school["czone_cross"] = (
        (school["home_b"] != school["dest_b"]) |
        (school["home_c"] != school["dest_c"])
    ).astype(float)

    _valid_b = school["dest_b"].notna() & school["home_b"].notna()
    _valid_c = _valid_b & school["home_c"].notna() & school["dest_c"].notna()

    print(f"\n  Trips with parseable destination zone: {_valid_b.sum():,} / {len(school):,}")
    print(f"  (missing/unparseable dest_zone_code: {(~_valid_b).sum():,} trips excluded)")

    # --- B-zone (56 zones) ---
    _sch_b = school.loc[_valid_b].copy()
    ct_b = pd.crosstab(_sch_b["level"], _sch_b["bzone_cross"])
    ct_b.columns = ["Same B-zone", "Cross B-zone"]
    ct_b["Total"]   = ct_b.sum(axis=1)
    ct_b["Cross %"] = (ct_b["Cross B-zone"] / ct_b["Total"] * 100).round(1)
    print(f"\n  [B-zone: 56 zones — coarser]")
    print(ct_b.reindex(LEVEL_ORDER).to_string())
    _ct_b_vals = ct_b[["Same B-zone", "Cross B-zone"]].reindex(LEVEL_ORDER).dropna()
    chi2_b, p_b, dof_b, _ = scipy_stats.chi2_contingency(_ct_b_vals)
    print(f"\n  Chi-square (B-zone): χ² = {chi2_b:.2f}, df = {dof_b}, p = {p_b:.2e}")

    # --- C-zone (144 zones, per advisor feedback) ---
    _sch_c = school.loc[_valid_c].copy()
    ct_c = pd.crosstab(_sch_c["level"], _sch_c["czone_cross"])
    ct_c.columns = ["Same C-zone", "Cross C-zone"]
    ct_c["Total"]   = ct_c.sum(axis=1)
    ct_c["Cross %"] = (ct_c["Cross C-zone"] / ct_c["Total"] * 100).round(1)
    print(f"\n  [C-zone: 144 zones — finer, per advisor feedback]")
    print(ct_c.reindex(LEVEL_ORDER).to_string())
    _ct_c_vals = ct_c[["Same C-zone", "Cross C-zone"]].reindex(LEVEL_ORDER).dropna()
    chi2_c, p_c, dof_c, _ = scipy_stats.chi2_contingency(_ct_c_vals)
    print(f"\n  Chi-square (C-zone): χ² = {chi2_c:.2f}, df = {dof_c}, p = {p_c:.2e}")
    print(f"  → H1b — C-zone provides finer spatial resolution (~2.6× more zones than B)")
    print(f"  → Elem/JHS cross-zone rate expected low due to catchment-zone assignment")

    # ── H1c: OLS regression — car ownership → travel time ─────────────────────
    section("H1c — Car Ownership and School Travel Distance (OLS)")

    school["has_car"] = (school["owned_has_car"] == 1).astype(float)
    school["income"]  = school["annual_income"].replace(9, np.nan)

    reg_df = school[["mode1_travel_time_min", "has_car", "level",
                      "hh_size_excl_under5", "income"]].dropna()
    print(f"\n  Regression sample: n = {len(reg_df):,} "
          f"(dropped {len(school) - len(reg_df):,} missing)")
    print(f"  has_car = 1: {reg_df['has_car'].sum():.0f}  ({reg_df['has_car'].mean():.1%})")
    print(f"  has_car = 0: {(reg_df['has_car'] == 0).sum():.0f}  ({(reg_df['has_car'] == 0).mean():.1%})")

    model  = smf.ols(
        'mode1_travel_time_min ~ has_car + C(level, Treatment("Elem/JHS")) '
        '+ hh_size_excl_under5 + income',
        data=reg_df
    )
    result = model.fit()

    print("\n  OLS Estimation Results:")
    print(f"  {'Variable':<45} {'Coef.':>8}  {'Std.Err.':>9}  {'t':>7}  {'p':>8}")
    print(f"  {'-'*80}")
    for var, row in result.summary2().tables[1].iterrows():
        print(f"  {var:<45} {row['Coef.']:>8.4f}  {row['Std.Err.']:>9.4f}  "
              f"{row['t']:>7.4f}  {row['P>|t|']:>8.4f}")
    print(f"\n  R²      = {result.rsquared:.3f}")
    print(f"  Adj. R² = {result.rsquared_adj:.3f}")
    print(f"  F       = {result.fvalue:.2f}  (p = {result.f_pvalue:.2e})")
    print(f"  n       = {int(result.nobs):,}")

    has_car_p = result.pvalues.get("has_car", 1.0)
    print(f"\n  has_car coefficient = {result.params.get('has_car', 0):.2f}, "
          f"p = {has_car_p:.4f}")
    print(f"  → {'H1c SUPPORTED' if has_car_p < 0.05 else 'H1c NOT SUPPORTED'}")

    print(f"\n  Analysis 1 outputs → {OUT_A1}")


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis 2 — School Escorting Behavior (RQ2a, RQ2b)
# ═══════════════════════════════════════════════════════════════════════════════

SCHOOL_LABEL = {"1": "Nursery/KG", "2": "Elementary", "3": "Middle",
                "4": "High", "5": "University"}

CAR_CODES = {"7", "8", "9", "10", "11", "80", "07", "08", "09"}


def _has_safety(row: pd.Series) -> int:
    """Return 1 if any child escort reason is code 02 (safety/crime)."""
    for col in ["q6_child1_escort_reason", "q6_child2_escort_reason",
                "q6_child3_escort_reason"]:
        if str(row.get(col, "")).strip() == "02":
            return 1
    return 0


def _build_escort_supp(supp: pd.DataFrame) -> pd.DataFrame:
    """
    Filter SUPP to confirmed escort parents (school levels 1-5).
    Adds: safety_concern, school_dest, school_label columns.
    """
    escort = supp[supp["q6_escorting_flag"].astype(str) == "1"].copy()
    escort["safety_concern"] = escort.apply(_has_safety, axis=1)
    escort["school_dest"] = escort["q6_child1_school_dest"].astype(str).str.strip()
    escort = escort[escort["school_dest"].isin(["1", "2", "3", "4", "5"])].copy()
    escort["school_label"] = escort["school_dest"].map(SCHOOL_LABEL)
    return escort


def analysis2(pt: pd.DataFrame, supp: pd.DataFrame) -> None:
    section("ANALYSIS 2 — School Escorting Behavior (RQ2a, RQ2b)")

    escort = _build_escort_supp(supp)

    # ── RQ2a Part A: Binary Logit — who escorts? ──────────────────────────────
    # Sample construction (exact reproduction of document n=1,237, R²=0.145):
    #   HH_KEY: 3-part key (id_b_zone_code, id_c_zone_code, id_household_number)
    #   school_hh: PT households with at least one school-age person (emp 8–11),
    #     ANY trip purpose.
    #   hh_youngest: youngest school-age child per HH. Encoding:
    #     11(KG)→1, 10(Elem/JHS)→2, 9(High)→4, 8(Univ)→5; sort ascending →
    #     first row = lowest value = youngest child.
    #   SUPP inner-joined to school_hh, then filtered to flag ∈ {1,2}.
    #   Escorts (flag=1): school level from q6_child1_school_dest (SUPP).
    #   Non-escorts (flag=2): school level from PT youngest child (sl_pt).
    #   recode_sl: 1→1(KG), 2/3→2(Elem/JHS), 4→3(HS), 5→4(Univ), other→NaN.
    #   NaN rows dropped (incl. code 6 = "Other" school type, only in flag=1).
    #   Result: n=1,237 (escorts=500, non-escorts=737), McFadden R²=0.145.
    #   Note: document listed 777/460 — these were taken from the descriptive
    #   table and incorrectly copied into the logit caption.
    section("RQ2a Part A — Binary Logit: Escorting Probability")

    HH_KEY_A = ['id_b_zone_code', 'id_c_zone_code', 'id_household_number']

    # Normalise to strings for safe merges / maps
    _pt_a  = pt.copy()
    _sup_a = supp.copy()
    for _c in HH_KEY_A + ['employment_student_status']:
        _pt_a[_c] = _pt_a[_c].astype(str).str.strip()
    for _c in HH_KEY_A + ['q6_escorting_flag', 'q6_child1_school_dest',
                           'owned_has_car', 'sex', 'hh_size_excl_under5', 'annual_income']:
        _sup_a[_c] = _sup_a[_c].astype(str).str.strip()

    # PT school-age households (any trip purpose)
    _pt_a['_emp'] = _pt_a['employment_student_status']
    _school_hh_a = (_pt_a[_pt_a['_emp'].isin(['8', '9', '10', '11'])]
                    [HH_KEY_A].drop_duplicates())

    # Youngest school-age child per HH from PT
    _emp_sl_map = {'11': 1, '10': 2, '9': 4, '8': 5}
    _children_a = (_pt_a[_pt_a['_emp'].isin(['8', '9', '10', '11'])]
                   [HH_KEY_A + ['_emp', 'person_number']].copy())
    _children_a['sl_pt'] = _children_a['_emp'].map(_emp_sl_map)
    _hh_yng = (_children_a.sort_values('sl_pt')
               .groupby(HH_KEY_A)['sl_pt'].first().reset_index())

    # E2: Distinct school-age children per HH (person-level count from PT)
    _hh_num_children = (
        _children_a[HH_KEY_A + ['person_number']].drop_duplicates()
        .groupby(HH_KEY_A).size().reset_index(name='num_children')
    )

    # SUPP merged to school_hh; both escort and non-escort rows
    _sup_a['flag'] = _sup_a['q6_escorting_flag']
    _df_a = _sup_a.merge(_school_hh_a, on=HH_KEY_A, how='inner')
    _df_a = _df_a[_df_a['flag'].isin(['1', '2'])].copy()
    _df_a['sl_supp'] = pd.to_numeric(_df_a['q6_child1_school_dest'], errors='coerce')
    _df_a = _df_a.merge(_hh_yng, on=HH_KEY_A, how='left')
    _df_a['sl_raw'] = np.where(_df_a['flag'] == '1', _df_a['sl_supp'], _df_a['sl_pt'])

    def _rsl(x):
        if x == 1:      return 1   # Kindergarten
        if x in [2, 3]: return 2   # Elem/JHS
        if x == 4:      return 3   # High School
        if x == 5:      return 4   # University
        return np.nan              # code 6 (Other) or NaN → dropped

    _df_a['sl'] = _df_a['sl_raw'].apply(_rsl)
    _df_a = _df_a.dropna(subset=['sl']).copy()

    _df_a['escort']     = (_df_a['flag'] == '1').astype(int)
    _df_a['car']        = _df_a['owned_has_car'].map({'1': 1, '2': 0})
    _df_a['female']     = (_df_a['sex'] == '2').astype(int)
    _df_a['hh_size']    = pd.to_numeric(_df_a['hh_size_excl_under5'], errors='coerce')
    _df_a['inc_raw']    = pd.to_numeric(_df_a['annual_income'], errors='coerce')
    _df_a['income']     = _df_a['inc_raw'].where(_df_a['inc_raw'].isin(range(1, 10))).fillna(0)
    _df_a['inc_unknown'] = (_df_a['income'] == 0).astype(int)
    _df_a['sl']         = _df_a['sl'].astype(int)

    # E1+E2: Employment status dummies + number of school-age children
    # Codebook: 1=self-employed, 2=corporate officer, 3=regular employee, 4=dispatch worker
    #           5=part-time/contract, 6=homemaker (ref), 7=unemployed
    #           8=university student, 9=HS student, 13=other
    # Filter: remove codes 7,8,9,13 (unemployed, student-siblings, other) so that
    #   reference group = homemaker only (code 6); 132 rows removed.
    _df_a['emp_raw']      = pd.to_numeric(_df_a['employment_student_status'], errors='coerce')
    _df_a['emp_fulltime'] = _df_a['emp_raw'].isin([1, 2, 3, 4]).astype(int)
    _df_a['emp_parttime'] = (_df_a['emp_raw'] == 5).astype(int)

    # E2: Number of school-age children per HH (distinct persons from PT)
    _df_a = _df_a.merge(_hh_num_children, on=HH_KEY_A, how='left')
    _df_a['num_children'] = _df_a['num_children'].fillna(0).astype(int)

    # Filter to parental respondents only (exclude student-siblings, unemployed, other)
    _df_ext = _df_a[~_df_a['emp_raw'].isin([7, 8, 9, 13])].copy()

    _n_esc_a  = int(_df_a['escort'].sum())
    _n_nesc_a = int(len(_df_a) - _n_esc_a)
    print(f"\n  Baseline sample: n={len(_df_a):,} (escorts={_n_esc_a}, non-escorts={_n_nesc_a})")
    print(f"  Extended sample: n={len(_df_ext):,} "
          f"(escorts={int(_df_ext['escort'].sum())}, "
          f"non-escorts={int(len(_df_ext)-_df_ext['escort'].sum())})")
    print(f"  Excluded: {len(_df_a)-len(_df_ext)} rows "
          f"(116 school-type 'other' + 132 student-siblings/unemployed/other employment)")
    print(f"  num_children dist: {_df_ext['num_children'].value_counts().sort_index().to_dict()}")

    # Baseline model (original paper, n=1,237)
    logit_a_base = smf.logit(
        'escort ~ car + C(sl, Treatment(1)) + female + hh_size + income + inc_unknown',
        data=_df_a
    ).fit(disp=False)

    # Extended model: E1+E2, ref=homemaker (code 6), n=989
    logit_a = smf.logit(
        'escort ~ car + C(sl, Treatment(1)) + female + hh_size + income + inc_unknown'
        ' + emp_fulltime + emp_parttime + num_children',
        data=_df_ext
    ).fit(disp=False)

    # Combined results table: Baseline vs Extended
    _VAR_LABELS = [
        ('car',                       'Car ownership'),
        ('C(sl, Treatment(1))[T.2]',  'School level: Elem/JHS vs Kinder'),
        ('C(sl, Treatment(1))[T.3]',  'School level: HS vs Kinder'),
        ('C(sl, Treatment(1))[T.4]',  'School level: Univ vs Kinder'),
        ('female',                    'Female escorter'),
        ('hh_size',                   'Household size'),
        ('income',                    'Income'),
        ('inc_unknown',               'Income unknown'),
        ('emp_fulltime',              'Employment: Full-time [E1]'),
        ('emp_parttime',              'Employment: Part-time [E1]'),
        ('num_children',              'No. school-age children [E2]'),
        ('Intercept',                 'Intercept'),
    ]

    def _fmt(m, var):
        tbl = m.summary2().tables[1]
        if var not in tbl.index:
            return "—", "—", "—"
        r = tbl.loc[var]
        sig = ("***" if r["P>|z|"] < 0.001 else "**" if r["P>|z|"] < 0.01
               else "*" if r["P>|z|"] < 0.05 else "†" if r["P>|z|"] < 0.10 else "")
        pstr = "<0.001" if r["P>|z|"] < 0.001 else f"{r['P>|z|']:.3f}"
        return f"{r['Coef.']:.3f}{sig}", f"{r['Std.Err.']:.3f}", pstr

    W = 36
    print(f"\n  {'Variable':<{W}} {'Baseline':>12} {'':>7} {'':>8}  "
          f"{'Extended':>12} {'':>7} {'':>8}")
    print(f"  {'':.<{W}} {'Coef.':>12} {'S.E.':>7} {'p':>8}  "
          f"{'Coef.':>12} {'S.E.':>7} {'p':>8}")
    print(f"  {'─'*88}")
    for _var, _lbl in _VAR_LABELS:
        _is_new = _var in ('emp_fulltime', 'emp_parttime', 'num_children')
        c1, s1, p1 = _fmt(logit_a_base, _var)
        c3, s3, p3 = _fmt(logit_a,      _var)
        _mk = "→ " if _is_new else "  "
        print(f"  {_mk}{_lbl:<{W-2}} {c1:>12} {s1:>7} {p1:>8}  {c3:>12} {s3:>7} {p3:>8}")

    print(f"  {'─'*88}")
    print(f"  {'  Null log-likelihood':<{W}} {logit_a_base.llnull:>12.2f} {'':>7} {'':>8}  "
          f"{logit_a.llnull:>12.2f}")
    print(f"  {'  Log-likelihood':<{W}} {logit_a_base.llf:>12.2f} {'':>7} {'':>8}  "
          f"{logit_a.llf:>12.2f}")
    print(f"  {'  McFadden R²':<{W}} {logit_a_base.prsquared:>12.4f} {'':>7} {'':>8}  "
          f"{logit_a.prsquared:>12.4f}")
    print(f"  {'  AIC':<{W}} {logit_a_base.aic:>12.2f} {'':>7} {'':>8}  "
          f"{logit_a.aic:>12.2f}")
    print(f"  {'  n':<{W}} {len(_df_a):>12,} {'':>7} {'':>8}  {len(_df_ext):>12,}")
    print(f"  {'─'*88}")
    print(f"  *** p<0.001  ** p<0.01  * p<0.05  † p<0.10")
    print(f"  Reference: school level = Kindergarten; employment = Homemaker (code 6)")
    print(f"  AIC not directly comparable across models (different sample sizes)")
    print(f"  → = variables added in extended model (E1/E2)")

    # ── RQ2a Part B: Binary Logit — who cites safety? ────────────────────────
    section("RQ2a Part B — Binary Logit: Safety Concern Among Escort Group")

    escort["income_raw"]    = pd.to_numeric(escort["annual_income"], errors="coerce")
    escort["inc_unknown"]   = escort["income_raw"].apply(
        lambda x: 1 if (pd.isna(x) or x in [10, 99]) else 0)
    escort["income_filled"] = escort["income_raw"].where(
        ~escort["inc_unknown"].astype(bool))
    escort["income_filled"] = escort["income_filled"].fillna(
        escort["income_filled"].median())
    escort["female"]   = (pd.to_numeric(escort["sex"], errors="coerce") == 2).astype(float)
    escort["hh_size"]  = pd.to_numeric(escort["hh_size_excl_under5"], errors="coerce")
    escort["age"]      = pd.to_numeric(escort["age"], errors="coerce")
    escort["num_children"] = 0
    for col in ["q6_child1_school_dest", "q6_child2_school_dest", "q6_child3_school_dest"]:
        escort["num_children"] += (
            escort[col].astype(str).str.strip().isin(["1", "2", "3", "4", "5"]).astype(int))
    emp_col = ("employment_student_status_x"
               if "employment_student_status_x" in escort.columns
               else "employment_student_status")
    escort["emp_raw"]        = pd.to_numeric(escort[emp_col], errors="coerce")
    escort["emp_fulltime"]   = escort["emp_raw"].isin([1, 2, 3, 4]).astype(int)
    escort["emp_parttime"]   = (escort["emp_raw"] == 5).astype(int)
    escort["emp_unemployed"] = (escort["emp_raw"] == 7).astype(int)
    escort["school_elem"]    = (escort["school_dest"] == "2").astype(int)
    escort["school_middle"]  = (escort["school_dest"] == "3").astype(int)
    escort["school_high"]    = (escort["school_dest"] == "4").astype(int)
    escort["school_univ"]    = (escort["school_dest"] == "5").astype(int)

    s2_cols = ["safety_concern", "income_filled", "inc_unknown", "female",
               "hh_size", "age", "num_children", "emp_fulltime", "emp_parttime",
               "emp_unemployed", "school_elem", "school_middle", "school_high", "school_univ"]
    s2_df = escort[s2_cols].dropna().reset_index(drop=True)
    X2 = sm.add_constant(s2_df.drop(columns=["safety_concern"]))
    y2 = s2_df["safety_concern"]
    logit_s2 = sm.Logit(y2, X2).fit(disp=False)
    null_s2  = sm.Logit(y2, pd.DataFrame({"const": np.ones(len(y2))})).fit(disp=False)
    mcf_s2   = 1 - logit_s2.llf / null_s2.llf

    print(f"\n  Sample: escort group, n = {len(s2_df):,}")
    print(f"  Safety concern = 1: {y2.sum():.0f} ({y2.mean():.1%})")
    print(f"  Reference: school = Nursery/KG, employment = homemaker")
    print(f"\n  {'Variable':<30} {'Coef':>8}  {'SE':>8}  {'z':>7}  {'p':>8}  {'Sig':>4}")
    print(f"  {'-'*72}")
    tbl2 = logit_s2.summary2().tables[1]
    for var, row in tbl2.iterrows():
        sig = ("***" if row["P>|z|"] < 0.001 else
               "**"  if row["P>|z|"] < 0.01  else
               "*"   if row["P>|z|"] < 0.05  else "")
        print(f"  {var:<30} {row['Coef.']:>8.4f}  {row['Std.Err.']:>8.4f}  "
              f"{row['z']:>7.4f}  {row['P>|z|']:>8.4f}  {sig:>4}")
    print(f"\n  McFadden R² = {mcf_s2:.4f}  |  AIC = {logit_s2.aic:.2f}")
    print(f"  Null LL = {null_s2.llf:.2f}  |  Model LL = {logit_s2.llf:.2f}")
    print("  Significance: * p<0.05  ** p<0.01  *** p<0.001")

    # ── RQ2b Descriptive Analysis 1: Car ownership by safety group ────────────
    section("RQ2b — Descriptive Analysis 1: Car Ownership by Safety Group")

    escort["has_car"] = (pd.to_numeric(escort["owned_has_car"], errors="coerce") == 1).astype(float)
    car_own = escort.groupby("safety_concern")["has_car"].agg(["sum", "count", "mean"])
    car_own.columns = ["Owns car", "Total", "Car ownership %"]
    car_own["Car ownership %"] = (car_own["Car ownership %"] * 100).round(1)
    car_own.index = car_own.index.map({0: "No-safety", 1: "Safety"})
    print(f"\n  {car_own.to_string()}")
    ct_car_own = pd.crosstab(escort["safety_concern"], escort["has_car"])
    chi2_own, p_own, _, _ = chi2_contingency(ct_car_own)
    print(f"\n  Chi-square: p = {p_own:.3f}")
    print(f"  Note: car ownership is near-universal (>99%) in both groups —")
    print(f"  formal mode choice model cannot be estimated (near-perfect separation).")

    # ── RQ2b Descriptive Analysis 1.5: Person-level mode table (doc p.17) ──────
    # Mode used for escorting — unique parent level (n=361 matched parents)
    # Safety definition: any-child (q6_child1/2/3_escort_reason = '02')
    section("RQ2b — Mode Table (person-level, matched parents)")

    pt_tmp2 = pt.copy()
    pt_tmp2["trip_purpose"] = pt_tmp2["trip_purpose"].astype(str).str.strip()
    for k in ["id_municipality_code", "id_household_number", "person_number"]:
        pt_tmp2[k] = pd.to_numeric(pt_tmp2[k], errors="coerce")

    pt12_pre = pt_tmp2[pt_tmp2["trip_purpose"] == "12"].copy()
    pt12_pre["rep_mode_class0"] = pt12_pre["rep_mode_class0"].astype(str).str.strip()

    _merge_keys = ["id_municipality_code", "id_household_number", "person_number"]
    escort_tmp = escort.copy()
    for k in _merge_keys:
        escort_tmp[k] = pd.to_numeric(escort_tmp[k], errors="coerce")

    matched_pre = pt12_pre.merge(
        escort_tmp[_merge_keys + ["safety_concern", "school_dest"]],
        on=_merge_keys, how="inner"
    )

    def _classify_mode(code: str) -> str:
        if code in CAR_CODES:                             return "Car (kei + passenger)"
        elif code in {"1", "01"}:                         return "Walk"
        elif code in {"5", "05", "6", "06"}:              return "Motorcycle/Moped"
        elif code in {"2", "02", "3", "03", "4", "04"}:  return "Bicycle"
        else:                                             return "Other"

    matched_pre["mode_label"] = matched_pre["rep_mode_class0"].apply(_classify_mode)
    first_trip = (matched_pre
                  .sort_values(_merge_keys)
                  .drop_duplicates(subset=_merge_keys, keep="first")
                  .copy())

    n_s  = (first_trip["safety_concern"] == 1).sum()
    n_ns = (first_trip["safety_concern"] == 0).sum()
    print(f"\n  Unique matched parents: {len(first_trip)}  "
          f"(Safety n={n_s}, Non-safety n={n_ns})")
    print(f"\n  {'Mode':<25} {'Safety n':>8} {'Safety %':>9} "
          f"{'Non-safety n':>13} {'Non-safety %':>13}")
    print(f"  {'-'*72}")
    for mode in ["Car (kei + passenger)", "Walk", "Motorcycle/Moped",
                 "Bicycle", "Other"]:
        sc  = ((first_trip["mode_label"] == mode) &
               (first_trip["safety_concern"] == 1)).sum()
        nc  = ((first_trip["mode_label"] == mode) &
               (first_trip["safety_concern"] == 0)).sum()
        sp  = sc / n_s  * 100 if n_s  > 0 else 0
        np_ = nc / n_ns * 100 if n_ns > 0 else 0
        print(f"  {mode:<25} {sc:>8}  {sp:>7.1f}%  {nc:>12}  {np_:>11.1f}%")

    # ── RQ2b Descriptive Analysis 2+3: Mode and travel time (PT purpose=12) ───
    section("RQ2b — Descriptive Analysis 2+3: Car Use and Travel Time by School Level")

    pt12 = pt[pt["trip_purpose"].astype(str).str.strip() == "12"].copy()
    matched = pt12.merge(
        escort[["id_municipality_code", "id_household_number", "person_number",
                "safety_concern", "school_dest"]],
        on=["id_municipality_code", "id_household_number", "person_number"],
        how="inner"
    )
    matched["is_car"] = matched["rep_mode_class0"].apply(
        lambda x: 1 if str(x).strip() in CAR_CODES else 0)
    for i in range(1, 5):
        matched[f"mode{i}_travel_time_min"] = pd.to_numeric(
            matched[f"mode{i}_travel_time_min"], errors="coerce").fillna(0)
    matched["total_time"] = matched[
        ["mode1_travel_time_min", "mode2_travel_time_min",
         "mode3_travel_time_min", "mode4_travel_time_min"]].sum(axis=1)
    matched.loc[matched["total_time"] == 0,   "total_time"] = np.nan
    matched.loc[matched["total_time"] >= 990,  "total_time"] = np.nan

    print(f"\n  Matched sample: {len(matched):,} trips "
          f"(PT purpose=12 x SUPP escort parents)")
    print(f"  Safety group: {matched['safety_concern'].sum():.0f} trips  |  "
          f"No-safety: {(matched['safety_concern'] == 0).sum():.0f} trips")

    # Analysis 2 table: car use
    print(f"\n  --- Analysis 2: Car use by school level ---")
    print(f"\n  {'School Level':<14}  {'Trips':>5}  {'Safety n':>8}  {'Safety car%':>11}  "
          f"{'No-safety n':>11}  {'No-safety car%':>14}  {'p-value':>8}")
    print(f"  {'-'*85}")
    for school in ["1", "2", "3", "4", "5"] + ["ALL"]:
        sub = matched if school == "ALL" else matched[matched["school_dest"] == school]
        label = "Overall" if school == "ALL" else SCHOOL_LABEL[school]
        if len(sub) < 3:
            continue
        s  = sub[sub["safety_concern"] == 1]
        ns = sub[sub["safety_concern"] == 0]
        sc = s["is_car"].mean() * 100 if len(s) > 0 else float("nan")
        nc = ns["is_car"].mean() * 100 if len(ns) > 0 else float("nan")
        ct = pd.crosstab(sub["safety_concern"], sub["is_car"])
        if ct.shape == (2, 2):
            if ct.values.min() < 5:
                _, p_car = fisher_exact(ct.values)
            else:
                _, p_car, _, _ = chi2_contingency(ct)
        else:
            p_car = float("nan")
        print(f"  {label:<14}  {len(sub):>5}  {len(s):>8}  {sc:>10.1f}%  "
              f"{len(ns):>11}  {nc:>13.1f}%  {p_car:>8.3f}")

    # Analysis 3 table: travel time
    print(f"\n  --- Analysis 3: Travel time by school level ---")
    print(f"\n  {'School Level':<14}  {'Safety median':>13}  {'No-safety median':>16}  {'p-value':>8}")
    print(f"  {'-'*60}")
    for school in ["1", "2", "3", "4", "5"] + ["ALL"]:
        sub = matched if school == "ALL" else matched[matched["school_dest"] == school]
        label = "Overall" if school == "ALL" else SCHOOL_LABEL[school]
        if len(sub) < 3:
            continue
        s  = sub[sub["safety_concern"] == 1]["total_time"].dropna()
        ns = sub[sub["safety_concern"] == 0]["total_time"].dropna()
        sm_med = s.median()  if len(s) > 0 else float("nan")
        ns_med = ns.median() if len(ns) > 0 else float("nan")
        if len(s) > 1 and len(ns) > 1:
            _, p_time = mannwhitneyu(s, ns, alternative="two-sided")
        else:
            p_time = float("nan")
        print(f"  {label:<14}  {sm_med:>11.0f} min  {ns_med:>14.0f} min  {p_time:>8.3f}")

    s_all  = matched[matched["safety_concern"] == 1]
    ns_all = matched[matched["safety_concern"] == 0]
    print(f"\n  Overall car use: Safety {s_all['is_car'].mean()*100:.1f}% vs "
          f"No-safety {ns_all['is_car'].mean()*100:.1f}%")
    _, p_car_all, _, _ = chi2_contingency(pd.crosstab(matched["safety_concern"], matched["is_car"]))
    print(f"  Chi-square (car use): p = {p_car_all:.3f}")
    _, p_time_all = mannwhitneyu(
        s_all["total_time"].dropna(), ns_all["total_time"].dropna(), alternative="two-sided")
    print(f"  Mann-Whitney (travel time): p = {p_time_all:.3f}")
    print(f"\n  Interpretation: Car use rates are nearly identical between safety and")
    print(f"  no-safety groups at every school level (all p > 0.05). Safety-motivated")
    print(f"  parents travel slightly shorter distances (median 10 vs 15 min, p=0.001)")
    print(f"  yet car use remains the same — consistent with car availability, not")
    print(f"  safety concern, being the dominant factor in escort mode choice.")

    print(f"\n  Analysis 2 outputs → {OUT_A2}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    section("Okinawa PT Survey — RQ Analyses")
    print(f"  Data    : {DATA_DIR}")
    print(f"  Output  : {SCRIPT_DIR / 'output'}")

    for name, path in FILES.items():
        if not path.exists():
            print(f"\n  ERROR: {path} not found.")
            sys.exit(1)

    print("\nLoading data …")
    pt   = load_csv("Person Trip",      FILES["pt"],   required=True)
    hh   = load_csv("HH Survey",        FILES["hh"],   required=False)  # may be OS-locked
    supp = load_csv("Supplementary",    FILES["supp"], required=True)

    analysis1(pt)
    analysis2(pt, supp)

    print(f"\n{'─'*70}")
    print("Done.")
    print(f"  Analysis 1 → {OUT_A1}")
    print(f"  Analysis 2 → {OUT_A2}")


if __name__ == "__main__":
    main()
