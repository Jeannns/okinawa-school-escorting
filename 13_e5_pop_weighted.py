"""
E5: Population-Weighted Escort Rate by C-zone
RQ1: Are raw zone-level escort rates representative of the true population?

Data:
  - R05_HH_Survey_EN.csv  → school-age population proxy by C-zone
    (addr_zone_code first 3 digits = C-zone; students = employment_student_status 8–11)
  - data/school_trips_los.csv → zone-level escort rate (origin_czone)

Method:
  1. HH Survey: extract school-age respondents (employment_student_status ∈ {8,9,10,11})
     derive C-zone from addr_zone_code (D-zone 5-digit → first 3 digits)
     → school-age HH member count per C-zone = population weight
  2. School trips: escort rate per C-zone (overall + by school level)
  3. Population-weighted mean = Σ(rate_i × pop_i) / Σ(pop_i)
  4. Compare raw (unweighted) vs weighted rates
  5. Identify over- and under-represented zones

Outputs:
  figures/E5_pop_weight_comparison.png
  figures/E5_zone_deviation.png
  data/e5_zone_escort_popweight.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
DATA = BASE / "data"
FIG  = BASE / "figures"
FIG.mkdir(exist_ok=True)

HH_CSV = Path("/sessions/awesome-beautiful-pascal/mnt/Okinawa/Okinawa_PT survey/"
              "Ver20240430第4回沖縄PTマスターデータ/EN/R05_HH_Survey_EN.csv")

STUDENT_STATUS = [8, 9, 10, 11]   # Elementary, Middle, HS, University student
SCHOOL_STATUS  = [8, 9, 10]       # Elementary, Middle, HS only

# ── 1. HH Survey: school-age population by C-zone ────────────────────────────
print("Loading HH Survey …")
hh = pd.read_csv(HH_CSV, low_memory=False)
print(f"  {len(hh):,} rows")

# Filter to school-age (elementary through high school)
students = hh[hh["employment_student_status"].isin(SCHOOL_STATUS)].copy()
print(f"  School-age (elem–HS): {len(students):,}")

# Derive C-zone from addr_zone_code (D-zone 5-digit)
students["addr_zone_str"] = students["addr_zone_code"].astype(str).str.strip()
students["origin_czone"]  = pd.to_numeric(
    students["addr_zone_str"].str[:3], errors="coerce"
).astype("Int64")

# School level from employment_student_status
level_map = {8: "Elementary", 9: "Middle", 10: "High School"}
students["school_level"] = students["employment_student_status"].map(level_map)

# Population weight = count of school-age HH members per C-zone
pop_czone = (students.groupby("origin_czone").size()
             .reset_index(name="pop_count"))
print(f"  C-zones with school-age pop: {len(pop_czone)}")

pop_level = (students.groupby(["origin_czone","school_level"]).size()
             .reset_index(name="pop_count_lv"))

# ── 2. School trips: escort rate by C-zone ────────────────────────────────────
print("\nLoading school_trips_los.csv …")
trips = pd.read_csv(DATA / "school_trips_los.csv")
print(f"  {len(trips):,} rows")

trips["origin_czone_int"] = pd.to_numeric(trips["origin_czone"], errors="coerce").astype("Int64")

# Overall escort rate per C-zone
escort_czone = (trips.groupby("origin_czone_int")
                .agg(n_trips=("escorting_flag", "count"),
                     n_escorted=("escorting_flag", lambda x: (x==1).sum()))
                .reset_index()
                .rename(columns={"origin_czone_int": "origin_czone"}))
escort_czone["escort_rate"] = escort_czone["n_escorted"] / escort_czone["n_trips"]

# By school level
for level in ["Elementary", "Middle", "High School"]:
    sub = trips[trips["school_level"] == level]
    lv = (sub.groupby(trips.loc[sub.index, "origin_czone"].apply(
              lambda x: int(x) if pd.notna(x) else np.nan))
          .agg(n_lv=("escorting_flag","count"),
               n_esc_lv=("escorting_flag", lambda x: (x==1).sum()))
          .reset_index()
          .rename(columns={"origin_czone": "origin_czone",
                            "n_lv":f"n_{level[:3].lower()}",
                            "n_esc_lv":f"n_esc_{level[:3].lower()}"}))
    lv[f"rate_{level[:3].lower()}"] = lv[f"n_esc_{level[:3].lower()}"] / lv[f"n_{level[:3].lower()}"]
    escort_czone = escort_czone.merge(lv, on="origin_czone", how="left")

# ── 3. Merge trips + population ────────────────────────────────────────────────
df = escort_czone.merge(pop_czone, on="origin_czone", how="left")
df["pop_count"] = df["pop_count"].fillna(0)

# Restrict to zones with trips only (n_trips ≥ 5 for stability)
df_main = df[df["n_trips"] >= 5].copy()
df_w    = df_main.dropna(subset=["escort_rate"])

print(f"\nC-zones with ≥5 school trips: {len(df_main)}")
print(f"Of those with HH pop data:    {(df_w['pop_count'] > 0).sum()}")

# ── 4. Weighted vs unweighted escort rates ────────────────────────────────────
# Unweighted (simple mean of zone rates, equal-weight per zone)
raw_mean_zone = df_w["escort_rate"].mean()

# Weighted by HH school-age population per zone
w_df = df_w[df_w["pop_count"] > 0].copy()
pop_weighted_mean = np.average(w_df["escort_rate"], weights=w_df["pop_count"])

# Trip-count weighted (what we'd normally report)
trip_weighted_mean = np.average(df_w["escort_rate"], weights=df_w["n_trips"])

# Overall unweighted from raw data (benchmark)
overall_raw = (trips["escorting_flag"] == 1).mean()

print(f"\n── Escort Rate Estimates ──")
print(f"  Raw (per-trip, all trips):           {overall_raw:.3f}  ({overall_raw*100:.1f}%)")
print(f"  Zone mean (unweighted, per zone):    {raw_mean_zone:.3f}  ({raw_mean_zone*100:.1f}%)")
print(f"  Zone mean (trip-count weighted):     {trip_weighted_mean:.3f}  ({trip_weighted_mean*100:.1f}%)")
print(f"  Zone mean (HH school-age weighted):  {pop_weighted_mean:.3f}  ({pop_weighted_mean*100:.1f}%)")

# By school level
print("\n── By School Level (HH pop-weighted vs raw trip rate) ──")
for level, col in [("Elementary","ele"), ("Middle","mid"), ("High School","hig")]:
    rate_col = f"rate_{col}"
    n_col    = f"n_{col}"
    pop_col  = "pop_count"
    if rate_col not in w_df.columns:
        continue
    sub = w_df[[rate_col, pop_col]].dropna()
    sub = sub[sub[pop_col] > 0]
    if len(sub) == 0:
        continue
    wm = np.average(sub[rate_col], weights=sub[pop_col])
    rm = sub[rate_col].mean()
    print(f"  {level:12s}: raw zone mean={rm:.1%}  pop-weighted={wm:.1%}  diff={wm-rm:+.3f}")

# ── 5. Figure 1: Comparison bar chart ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: overall escort rate comparison
ax = axes[0]
estimates = {
    "Raw per-trip\n(all trips)": overall_raw * 100,
    "Zone mean\n(unweighted)": raw_mean_zone * 100,
    "Zone mean\n(trip-weighted)": trip_weighted_mean * 100,
    "Zone mean\n(HH pop-weighted)": pop_weighted_mean * 100,
}
colors = ["#4C72B0", "#AAAAAA", "#55A868", "#C44E52"]
bars = ax.bar(range(len(estimates)), list(estimates.values()),
              color=colors, alpha=0.85, edgecolor="0.3", width=0.6)
ax.set_xticks(range(len(estimates)))
ax.set_xticklabels(list(estimates.keys()), fontsize=9)
ax.set_ylabel("Escort rate (%)", fontsize=10)
ax.set_title("Escort Rate Estimates:\nRaw vs Zone-Weighted",
             fontsize=11, fontweight="bold")
ax.set_ylim(0, 80)
for i, (k, v) in enumerate(estimates.items()):
    ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

# Reference line
ax.axhline(overall_raw * 100, color="#4C72B0", ls="--", lw=1.0, alpha=0.5)

# Right: distribution of zone-level escort rates (weighted by HH pop)
ax = axes[1]
valid = w_df.dropna(subset=["escort_rate"])
pop_nonzero = valid[valid["pop_count"] > 0]
pop_zero    = valid[valid["pop_count"] == 0]

ax.hist(pop_nonzero["escort_rate"] * 100, bins=20, weights=pop_nonzero["pop_count"],
        alpha=0.7, color="#C44E52", edgecolor="0.4", label="Pop-weighted")
ax.hist(valid["escort_rate"] * 100, bins=20,
        alpha=0.4, color="#4C72B0", edgecolor="0.3", label="Unweighted (equal zone)")
ax.axvline(pop_weighted_mean * 100, color="#C44E52", ls="--", lw=1.5,
           label=f"Pop-wtd mean: {pop_weighted_mean*100:.1f}%")
ax.axvline(overall_raw * 100, color="#4C72B0", ls="--", lw=1.5,
           label=f"Raw mean: {overall_raw*100:.1f}%")
ax.set_xlabel("Zone-level escort rate (%)", fontsize=10)
ax.set_ylabel("Count / weighted count", fontsize=10)
ax.set_title("Distribution of Zone Escort Rates\n(pop-weighted vs unweighted)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=8)

plt.tight_layout()
fig.savefig(FIG / "E5_pop_weight_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("\n  Saved E5_pop_weight_comparison.png")

# ── 6. Figure 2: Zone deviation from weighted mean ────────────────────────────
df_dev = w_df[w_df["pop_count"] > 0].copy()
df_dev["deviation"] = df_dev["escort_rate"] - pop_weighted_mean
df_dev = df_dev.sort_values("deviation", ascending=False)

# Identify over/under-represented zones (by deviation)
top_over  = df_dev.head(10)
top_under = df_dev.tail(10)
highlight = pd.concat([top_over, top_under])

fig, ax = plt.subplots(figsize=(10, 8))

colors_dev = ["#C44E52" if d > 0 else "#4C72B0"
              for d in highlight["deviation"]]
ax.barh(range(len(highlight)), highlight["deviation"] * 100,
        color=colors_dev, alpha=0.8, edgecolor="0.3")
ax.set_yticks(range(len(highlight)))
ax.set_yticklabels([f"C-zone {int(z)}" for z in highlight["origin_czone"]], fontsize=8)
ax.invert_yaxis()
ax.axvline(0, color="black", lw=1.0)
ax.set_xlabel("Deviation from population-weighted mean (%pt)", fontsize=10)
ax.set_title("Top 10 Over- / Under-Represented Zones\n"
             f"(vs pop-weighted mean = {pop_weighted_mean*100:.1f}%)",
             fontsize=11, fontweight="bold")

red_patch  = mpatches.Patch(color="#C44E52", alpha=0.8, label="Above mean (over-represented)")
blue_patch = mpatches.Patch(color="#4C72B0", alpha=0.8, label="Below mean (under-represented)")
ax.legend(handles=[red_patch, blue_patch], fontsize=8)

# Annotate with n_trips and pop_count
for i, (_, row) in enumerate(highlight.iterrows()):
    note = f"n={int(row['n_trips'])}, pop={int(row['pop_count'])}"
    xpos = row["deviation"] * 100
    ha = "left" if xpos >= 0 else "right"
    offset = 0.3 if xpos >= 0 else -0.3
    ax.text(xpos + offset, i, note, va="center", ha=ha, fontsize=7, color="0.3")

plt.tight_layout()
fig.savefig(FIG / "E5_zone_deviation.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved E5_zone_deviation.png")

# ── 7. Save output table ───────────────────────────────────────────────────────
out = df_main[["origin_czone","n_trips","n_escorted","escort_rate","pop_count"]].copy()
out["pop_weight"] = out["pop_count"] / out["pop_count"].sum()
out["escort_rate_pct"] = (out["escort_rate"] * 100).round(1)
out.to_csv(DATA / "e5_zone_escort_popweight.csv", index=False)
print(f"\nSaved e5_zone_escort_popweight.csv ({len(out)} zones)")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n══ E5 Summary ══")
print(f"  C-zones analysed:              {len(df_main)} (n_trips ≥ 5)")
print(f"  C-zones with HH pop data:      {(w_df['pop_count'] > 0).sum()}")
print(f"  Total HH school-age proxy:     {int(pop_czone['pop_count'].sum()):,}")
print(f"  Raw per-trip escort rate:       {overall_raw:.1%}")
print(f"  Pop-weighted zone escort rate:  {pop_weighted_mean:.1%}")
print(f"  Difference:                    {(pop_weighted_mean - overall_raw)*100:+.2f} %pt")
print(f"  Interpretation: sample {'over' if pop_weighted_mean < overall_raw else 'under'}-represents "
      f"high-escort zones relative to the school-age population distribution.")
print("\nDone.")
