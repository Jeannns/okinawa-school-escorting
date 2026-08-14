"""
E2: Objective Safety — Crime Density vs. Escort Rate by Municipality
RQ2: Does objective crime exposure predict school escorting behaviour?

Data:
  - Okinawa crime record 2023 (7 CSV files, Shift-JIS encoded)
  - data/school_trips_los.csv (origin municipality, escorting_flag)
  - R05_Supplementary_EN.csv (q6_child*_escort_reason == '02' = safety concern)

Method:
  - Municipality code mapping: crime 472018 → // 10 % 1000 = 201 (PT survey code)
  - Crime density = crime_count / n_school_trips (municipality-level proxy)
  - Correlation (Pearson + Spearman) between crime density and escort rate
  - Scatter plot with labelled municipalities
  - Perceived safety breakdown: safety-motivated escorters vs. crime density

Outputs:
  figures/E2_crime_vs_escort_scatter.png
  figures/E2_crime_density_by_muni.png
  figures/E2_perceived_vs_objective_safety.png
  data/e2_crime_escort_muni.csv
"""

from pathlib import Path
import glob, subprocess, tempfile
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE  = Path(__file__).parent
DATA  = BASE / "data"
FIG   = BASE / "figures"
FIG.mkdir(exist_ok=True)

CRIME_DIR = BASE / "Okinawa crime record 2023"
SUPP_CSV  = Path("/sessions/awesome-beautiful-pascal/mnt/Okinawa/Okinawa_PT survey/"
                 "Ver20240430第4回沖縄PTマスターデータ/EN/R05_Supplementary_EN.csv")

SHORT_NAME = {
    201: "Naha", 205: "Itoman", 208: "Urasoe", 209: "Nanjo",
    210: "Tomigusuku", 211: "Ginowan", 212: "Okinawa C.",
    213: "Uruma", 214: "Haebaru", 215: "Yomitan", 324: "Chatan",
    325: "Kadena", 326: "Nishihara", 327: "Yonabaru", 328: "Kitanakagusuku",
    329: "Nakagusuku", 348: "Yaese", 350: "Okinawa C.", 362: "Yonabaru",
}

# ── 1. Load crime data (iconv workaround for macOS deadlock) ──────────────────
print("Loading crime data …")
crime_frames = []
for fpath in sorted(CRIME_DIR.glob("*.csv")):
    try:
        result = subprocess.run(
            ["iconv", "-f", "SHIFT-JIS", "-t", "UTF-8", str(fpath)],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0 and result.stdout.strip():
            from io import StringIO
            df = pd.read_csv(StringIO(result.stdout), dtype=str)
            crime_frames.append(df)
            print(f"  {fpath.name}: {len(df)} records")
    except Exception as e:
        print(f"  SKIP {fpath.name}: {e}")

crime = pd.concat(crime_frames, ignore_index=True)
print(f"  Total crime records: {len(crime):,}")

muni_col = [c for c in crime.columns if "市区町村コード" in c][0]
crime["muni_int"]     = pd.to_numeric(crime[muni_col], errors="coerce")
crime["pt_muni_code"] = (crime["muni_int"] // 10 % 1000).astype("Int64")

crime_by_muni = (crime.groupby("pt_muni_code").size()
                      .reset_index(name="crime_count"))
print(f"  Municipalities in crime data: {len(crime_by_muni)}")

# ── 2. Load school trips ───────────────────────────────────────────────────────
print("\nLoading school_trips_los.csv …")
trips = pd.read_csv(DATA / "school_trips_los.csv")
print(f"  {len(trips):,} rows")

escort_muni = (trips.groupby("origin_municipality_code")
               .agg(n_trips=("escorting_flag", "count"),
                    n_escorted=("escorting_flag", lambda x: (x == 1).sum()))
               .reset_index())
escort_muni["escort_rate"] = escort_muni["n_escorted"] / escort_muni["n_trips"]

# By school level
for level in ["Elementary", "Middle", "High School"]:
    sub = trips[trips["school_level"] == level]
    lv = (sub.groupby("origin_municipality_code")
            .agg(escort_rate_lv=("escorting_flag", lambda x: (x==1).mean()))
            .reset_index()
            .rename(columns={"escort_rate_lv": f"escort_{level[:3].lower()}"}))
    escort_muni = escort_muni.merge(lv, on="origin_municipality_code", how="left")

# ── 3. Merge crime + escort ────────────────────────────────────────────────────
df = escort_muni.merge(crime_by_muni, left_on="origin_municipality_code",
                       right_on="pt_muni_code", how="left")
df["crime_per100trips"] = df["crime_count"] / df["n_trips"] * 100
df["label"] = df["origin_municipality_code"].map(SHORT_NAME).fillna(
    df["origin_municipality_code"].astype(str))

# Restrict to municipalities with meaningful sample (n >= 10)
df_main = df[df["n_trips"] >= 10].dropna(subset=["crime_count"]).copy()
print(f"\nMunicipalities with n≥10 trips and crime data: {len(df_main)}")

# ── 4. Correlations ────────────────────────────────────────────────────────────
r_p, p_p = stats.pearsonr(df_main["crime_per100trips"], df_main["escort_rate"])
r_s, p_s = stats.spearmanr(df_main["crime_per100trips"], df_main["escort_rate"])
print(f"\nCorrelation (crime/100 trips vs escort rate):")
print(f"  Pearson r = {r_p:.3f}  p = {p_p:.3f}")
print(f"  Spearman ρ = {r_s:.3f}  p = {p_s:.3f}")

# ── 5. Figure 1: Scatter ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))

# Size ~ n_trips, color ~ escort rate
sc = ax.scatter(df_main["crime_per100trips"], df_main["escort_rate"] * 100,
                s=df_main["n_trips"] / 20, alpha=0.7,
                c=df_main["escort_rate"], cmap="RdYlGn_r",
                edgecolors="0.3", linewidths=0.5)

# Labels
for _, row in df_main.iterrows():
    ax.annotate(row["label"],
                (row["crime_per100trips"], row["escort_rate"] * 100),
                fontsize=8, ha="left", va="bottom",
                xytext=(4, 2), textcoords="offset points")

# Regression line
x_fit = np.linspace(df_main["crime_per100trips"].min(),
                    df_main["crime_per100trips"].max(), 100)
slope, intercept, *_ = stats.linregress(df_main["crime_per100trips"],
                                         df_main["escort_rate"] * 100)
ax.plot(x_fit, slope * x_fit + intercept, "k--", lw=1.2, alpha=0.6)

# Annotations
ax.text(0.97, 0.05,
        f"Pearson r = {r_p:.3f} (p = {p_p:.3f})\n"
        f"Spearman ρ = {r_s:.3f} (p = {p_s:.3f})",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

ax.set_xlabel("Crime incidents per 100 school trips (municipality-level)", fontsize=11)
ax.set_ylabel("School escort rate (%)", fontsize=11)
ax.set_title("Objective Crime Exposure vs. School Escort Rate\n"
             "(bubble size = number of school trips; colour = escort rate)",
             fontsize=12, fontweight="bold")
plt.colorbar(sc, ax=ax, label="Escort rate")

plt.tight_layout()
fig.savefig(FIG / "E2_crime_vs_escort_scatter.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved E2_crime_vs_escort_scatter.png")

# ── 6. Figure 2: Crime density bar chart ──────────────────────────────────────
df_sorted = df_main.sort_values("crime_per100trips", ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: crime per 100 trips
ax = axes[0]
colors = plt.cm.Reds(df_sorted["crime_per100trips"] /
                     df_sorted["crime_per100trips"].max() * 0.8 + 0.2)
bars = ax.barh(range(len(df_sorted)), df_sorted["crime_per100trips"],
               color=colors, edgecolor="0.4", linewidth=0.5)
ax.set_yticks(range(len(df_sorted)))
ax.set_yticklabels(df_sorted["label"], fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("Crime incidents per 100 school trips", fontsize=10)
ax.set_title("Objective Crime Density\nby Municipality", fontsize=11, fontweight="bold")
for i, val in enumerate(df_sorted["crime_per100trips"]):
    ax.text(val + 0.2, i, f"{val:.1f}", va="center", fontsize=8)

# Right: escort rate (sorted same order)
ax = axes[1]
escort_colors = plt.cm.RdYlGn_r(df_sorted["escort_rate"] * 1.2)
ax.barh(range(len(df_sorted)), df_sorted["escort_rate"] * 100,
        color=escort_colors, edgecolor="0.4", linewidth=0.5)
ax.set_yticks(range(len(df_sorted)))
ax.set_yticklabels(df_sorted["label"], fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("School escort rate (%)", fontsize=10)
ax.set_title("School Escort Rate\n(same municipality order)", fontsize=11, fontweight="bold")
ax.axvline(df_sorted["escort_rate"].mean() * 100, color="navy",
           ls="--", lw=1.2, label=f"Mean {df_sorted['escort_rate'].mean()*100:.1f}%")
ax.legend(fontsize=8)
for i, val in enumerate(df_sorted["escort_rate"] * 100):
    ax.text(val + 0.3, i, f"{val:.1f}%", va="center", fontsize=8)

plt.suptitle("Crime Density and Escort Rate by Municipality\n"
             "(ordered by crime density, highest first)",
             fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
fig.savefig(FIG / "E2_crime_density_by_muni.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved E2_crime_density_by_muni.png")

# ── 7. Perceived safety from supplementary survey ─────────────────────────────
print("\nLoading supplementary survey …")
try:
    supp = pd.read_csv(SUPP_CSV, low_memory=False)

    def has_safety_concern(row):
        for col in ["q6_child1_escort_reason", "q6_child2_escort_reason",
                    "q6_child3_escort_reason"]:
            if col in row.index and str(row[col]).strip() == "02":
                return 1
        return 0

    supp_esc = supp[supp["q6_escorting_flag"].astype(str).str.strip() == "1"].copy()
    supp_esc["safety_concern"] = supp_esc.apply(has_safety_concern, axis=1)
    supp_esc["addr_muni"] = pd.to_numeric(supp_esc["addr_municipality_code"], errors="coerce").astype("Int64")

    safety_by_muni = (supp_esc.groupby("addr_muni")
                      .agg(n_esc=("safety_concern", "count"),
                           n_safety=("safety_concern", "sum"))
                      .reset_index())
    safety_by_muni["safety_rate"] = safety_by_muni["n_safety"] / safety_by_muni["n_esc"]

    df_perc = df_main.merge(safety_by_muni, left_on="origin_municipality_code",
                            right_on="addr_muni", how="inner")
    df_perc = df_perc[df_perc["n_esc"] >= 5]
    print(f"  Municipalities with both crime + perceived safety data: {len(df_perc)}")

    if len(df_perc) >= 5:
        r_p2, p_p2 = stats.pearsonr(df_perc["crime_per100trips"], df_perc["safety_rate"])
        r_s2, p_s2 = stats.spearmanr(df_perc["crime_per100trips"], df_perc["safety_rate"])
        print(f"  Corr (crime density vs safety-motivated escort rate):")
        print(f"    Pearson r = {r_p2:.3f}  p = {p_p2:.3f}")
        print(f"    Spearman ρ = {r_s2:.3f}  p = {p_s2:.3f}")

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Left: crime vs perceived safety rate
        ax = axes[0]
        ax.scatter(df_perc["crime_per100trips"], df_perc["safety_rate"] * 100,
                   s=df_perc["n_esc"] * 20, alpha=0.75, color="#C44E52",
                   edgecolors="0.3", linewidths=0.5)
        for _, row in df_perc.iterrows():
            ax.annotate(row["label"],
                        (row["crime_per100trips"], row["safety_rate"] * 100),
                        fontsize=8, xytext=(3, 2), textcoords="offset points")
        if len(df_perc) >= 3:
            x2 = np.linspace(df_perc["crime_per100trips"].min(),
                             df_perc["crime_per100trips"].max(), 100)
            s2, i2, *_ = stats.linregress(df_perc["crime_per100trips"],
                                          df_perc["safety_rate"] * 100)
            ax.plot(x2, s2 * x2 + i2, "k--", lw=1.2, alpha=0.6)
        ax.set_xlabel("Crime incidents per 100 school trips", fontsize=10)
        ax.set_ylabel("% of escorters citing safety/crime as reason", fontsize=10)
        ax.set_title("Objective Crime vs. Safety-Motivated Escort\n"
                     f"Pearson r={r_p2:.3f} (p={p_p2:.3f}), "
                     f"Spearman ρ={r_s2:.3f} (p={p_s2:.3f})",
                     fontsize=10, fontweight="bold")

        # Right: objective crime vs overall escort rate comparison
        ax = axes[1]
        ax.scatter(df_perc["safety_rate"] * 100, df_perc["escort_rate"] * 100,
                   s=df_perc["crime_per100trips"] * 5, alpha=0.75, color="#4C72B0",
                   edgecolors="0.3", linewidths=0.5)
        for _, row in df_perc.iterrows():
            ax.annotate(row["label"],
                        (row["safety_rate"] * 100, row["escort_rate"] * 100),
                        fontsize=8, xytext=(3, 2), textcoords="offset points")
        ax.set_xlabel("% of escorters citing safety as reason", fontsize=10)
        ax.set_ylabel("Overall school escort rate (%)", fontsize=10)
        ax.set_title("Perceived Safety Concern vs. Overall Escort Rate\n"
                     "(bubble size = crime density)",
                     fontsize=10, fontweight="bold")

        plt.suptitle("Objective Crime, Perceived Safety, and School Escorting",
                     fontsize=12, fontweight="bold", y=1.02)
        plt.tight_layout()
        fig.savefig(FIG / "E2_perceived_vs_objective_safety.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  Saved E2_perceived_vs_objective_safety.png")
    else:
        print("  Too few municipalities for perceived safety plot — skipping.")

except Exception as e:
    print(f"  Supplementary survey load failed: {e}")
    df_perc = pd.DataFrame()

# ── 8. Save output table ───────────────────────────────────────────────────────
out_cols = ["origin_municipality_code", "label", "n_trips", "n_escorted",
            "escort_rate", "crime_count", "crime_per100trips",
            "escort_ele", "escort_mid", "escort_hig"]
out_cols_exist = [c for c in out_cols if c in df_main.columns]
df_main[out_cols_exist].to_csv(DATA / "e2_crime_escort_muni.csv", index=False)
print(f"\nSaved e2_crime_escort_muni.csv ({len(df_main)} municipalities)")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n══ E2 Summary ══")
print(f"  Crime records analysed:   {len(crime):,}")
print(f"  Municipalities matched:    {len(df_main)} / {len(escort_muni)}")
print(f"  Crime density range:       "
      f"{df_main['crime_per100trips'].min():.1f} – "
      f"{df_main['crime_per100trips'].max():.1f} per 100 trips")
print(f"  Escort rate range:         "
      f"{df_main['escort_rate'].min():.1%} – "
      f"{df_main['escort_rate'].max():.1%}")
print(f"  Pearson r (crime vs escort): {r_p:.3f} (p={p_p:.3f})")
print(f"  Spearman ρ:                  {r_s:.3f} (p={p_s:.3f})")
if not df_perc.empty:
    print(f"  Pearson r (crime vs perceived safety rate): {r_p2:.3f} (p={p_p2:.3f})")
highcrime = df_main.nlargest(3, "crime_per100trips")[["label","crime_per100trips","escort_rate"]]
print(f"\n  Highest crime density:")
for _, r in highcrime.iterrows():
    print(f"    {r['label']:18s} crime/100={r['crime_per100trips']:.1f}  escort={r['escort_rate']:.1%}")
print("\nDone.")
