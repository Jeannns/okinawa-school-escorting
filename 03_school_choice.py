"""
03_school_choice.py — School District GIS Assignment + School Choice Analysis
==============================================================================
Requires: data/school_trips_los.csv     (from 01_data_prep.py)
          SSD mounted at /Volumes/Samsung_T5/For Jehan 260721

Analyses:
  P2a (#68) — GIS: join each C-zone centroid → school district polygon
               → build czone_to_school_district lookup (Elem + Middle)
  P2b (#69) — School choice flag (in-catchment vs out-of-catchment)
               → escort rate comparison: in-catchment vs school choice trips

Methodology:
  1. CZone.shp provides C-zone centroids (lat/lon already in attributes)
  2. School district polygons (16 cities × Elem/Middle) define catchment areas
  3. Spatial join: C-zone centroid → school district → home_district
  4. For each trip: if dest C-zone falls in DIFFERENT school district than origin
     → school_choice = 1 (out-of-catchment)
  5. High School trips are EXCLUDED (no catchment-based districts in Japan)

Note on escort rate by school type:
  Elementary/Middle trips are escort-relevant for the school choice analysis.
  High School trips dominate the LOS-joined sample (55% of trips) but are
  excluded here. The LOS-selection artifact (97.6% HS escort rate in LOS subset)
  does NOT affect this analysis.
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
    import geopandas as gpd
    HAS_GPD = True
except ImportError:
    HAS_GPD = False
    print("⚠️  geopandas not installed. Run: pip install geopandas --break-system-packages")

# ══════════════════════════════════════════════════════════════════════════════
# Paths
# ══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).resolve().parent
SSD_DIR    = Path("/Volumes/Samsung_T5/For Jehan 260721")  # Mac path
# Sandbox fallback (HFS+ FUSE mount)
if not SSD_DIR.exists():
    SSD_DIR = Path("/sessions/awesome-beautiful-pascal/mnt/For Jehan 260721")
PREP_DIR   = SCRIPT_DIR / "data"
TABLE_DIR  = PREP_DIR / "tables"
FIG_DIR    = PREP_DIR / "figures"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


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
# P2a — Build C-zone → School District Lookup
# ══════════════════════════════════════════════════════════════════════════════

def load_school_districts(ssd_dir: Path, school_type: str) -> "gpd.GeoDataFrame":
    """
    Load all school district polygons for one school type across 16 cities.
    school_type: 'ElementarySchoolDistrict' | 'MiddleSchoolDistrict'
    Returns a unified GeoDataFrame (EPSG:6668 / JGD2011 geographic).
    """
    urban_dir = ssd_dir / "Okinawa_UrbanPlanning_MasterData" / "R4"
    gdfs = []
    for city_dir in sorted(urban_dir.iterdir()):
        if not city_dir.is_dir():
            continue
        shps = list(city_dir.rglob(f"*{school_type}/*.shp"))
        if not shps:
            continue
        gdf = gpd.read_file(shps[0])
        gdfs.append(gdf)

    if not gdfs:
        return gpd.GeoDataFrame()

    combined = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
    combined = combined.set_crs(gdfs[0].crs, allow_override=True)
    # Reproject to geographic CRS for spatial join with C-zone centroids
    combined = combined.to_crs("EPSG:6668")

    label = "Elementary" if "Elementary" in school_type else "Middle"
    print(f"  {label} school districts loaded : {len(combined)} polygons, {combined['SHI'].nunique()} cities")
    return combined


def build_czone_school_lookup(ssd_dir: Path) -> pd.DataFrame:
    """
    P2a: Spatial join — C-zone centroid → school district polygon.
    Returns DataFrame: czone_int | elem_district | mid_district
    """
    if not HAS_GPD:
        return pd.DataFrame()

    # ── Load C-zone centroids ──────────────────────────────────────────────────
    czone_shp = (ssd_dir / "Okinawa_PT_MasterData"
                 / "01_PopulationFrame" / "shp" / "CZone.shp")
    czone = gpd.read_file(czone_shp)
    czone = czone.rename(columns={
        "Cゾーン": "czone_str",
        "緯度":   "czone_lat",
        "経度":   "czone_lon",
    })
    czone["czone_int"] = czone["czone_str"].apply(lambda x: int(str(x)))

    # Create point GeoDataFrame from the pre-computed centroid coordinates
    czone_pts = gpd.GeoDataFrame(
        czone[["czone_int", "czone_lat", "czone_lon"]],
        geometry=gpd.points_from_xy(czone["czone_lon"], czone["czone_lat"]),
        crs="EPSG:6668"
    )
    print(f"  C-zone centroids : {len(czone_pts)} points (EPSG:6668)")

    # ── Load school districts ──────────────────────────────────────────────────
    elem_gdf = load_school_districts(ssd_dir, "ElementarySchoolDistrict")
    mid_gdf  = load_school_districts(ssd_dir, "MiddleSchoolDistrict")

    # ── Spatial join ───────────────────────────────────────────────────────────
    results = czone_pts[["czone_int"]].copy()

    for label, gdf in [("elem_district", elem_gdf), ("mid_district", mid_gdf)]:
        if gdf.empty:
            results[label] = np.nan
            results[label + "_city"] = np.nan
            continue

        joined = gpd.sjoin(
            czone_pts, gdf[["SHICD", "SHI", "SC_NAME", "geometry"]],
            how="left", predicate="within"
        )
        # Some centroids may fall slightly outside polygons (boundary effect)
        # Use nearest for unmatched
        n_unmatched = joined["SC_NAME"].isna().sum()
        if n_unmatched > 0:
            unmatched_pts = czone_pts[czone_pts["czone_int"].isin(
                joined[joined["SC_NAME"].isna()]["czone_int"]
            )]
            nearest = gpd.sjoin_nearest(
                unmatched_pts, gdf[["SHICD", "SHI", "SC_NAME", "geometry"]],
                how="left"
            )
            # Merge nearest results into joined
            fix_idx = joined[joined["SC_NAME"].isna()].index
            nearest = nearest.set_index("czone_int")
            for col in ["SC_NAME", "SHI", "SHICD"]:
                joined.loc[fix_idx, col] = joined.loc[fix_idx, "czone_int"].map(
                    nearest[col]
                ).values
            n_fixed = n_unmatched - joined["SC_NAME"].isna().sum()
            print(f"    {label}: {n_unmatched} unmatched → fixed {n_fixed} via nearest-neighbor")

        results[label]           = joined["SC_NAME"].values
        results[label + "_city"] = joined["SHI"].values
        results[label + "_shicd"]= joined["SHICD"].values

    n_elem_matched = results["elem_district"].notna().sum()
    n_mid_matched  = results["mid_district"].notna().sum()
    print(f"\n  C-zones matched to Elementary district : {n_elem_matched}/{len(results)}")
    print(f"  C-zones matched to Middle district     : {n_mid_matched}/{len(results)}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# P2b — School Choice Flag + Escort Analysis
# ══════════════════════════════════════════════════════════════════════════════

def build_school_choice_flag(df_trips: pd.DataFrame, czone_lookup: pd.DataFrame) -> pd.DataFrame:
    """
    For each Elementary/Middle school trip:
      - Map origin_czone → home school district
      - Map dest_czone   → dest school district
      - school_choice = 1 if home ≠ dest school district
    """
    if czone_lookup.empty:
        df_trips["school_choice"] = np.nan
        return df_trips

    # Focus on Elem + Middle only
    elem_mid = df_trips[df_trips["school_level"].isin(["Elementary", "Middle"])].copy()
    print(f"  Elementary + Middle trips : {len(elem_mid):,}")

    # Merge lookup for ORIGIN (home district)
    lookup_o = czone_lookup.rename(columns={
        "elem_district": "home_elem_dist",
        "mid_district":  "home_mid_dist",
        "elem_district_city": "home_elem_city",
        "mid_district_city":  "home_mid_city",
    })
    elem_mid = elem_mid.merge(
        lookup_o[["czone_int","home_elem_dist","home_mid_dist","home_elem_city","home_mid_city"]],
        left_on="origin_czone", right_on="czone_int", how="left"
    ).drop(columns="czone_int")

    # Merge lookup for DESTINATION
    lookup_d = czone_lookup.rename(columns={
        "elem_district": "dest_elem_dist",
        "mid_district":  "dest_mid_dist",
    })
    elem_mid = elem_mid.merge(
        lookup_d[["czone_int","dest_elem_dist","dest_mid_dist"]],
        left_on="dest_czone", right_on="czone_int", how="left"
    ).drop(columns="czone_int")

    # School choice flag: home district ≠ dest district (for each school type)
    def choice_flag(row):
        lv = row.get("school_level")
        if lv == "Elementary":
            h = row.get("home_elem_dist")
            d = row.get("dest_elem_dist")
        elif lv == "Middle":
            h = row.get("home_mid_dist")
            d = row.get("dest_mid_dist")
        else:
            return np.nan
        if pd.isna(h) or pd.isna(d):
            return np.nan
        return 0 if h == d else 1

    elem_mid["school_choice"] = elem_mid.apply(choice_flag, axis=1)

    n_valid    = elem_mid["school_choice"].notna().sum()
    n_choice   = (elem_mid["school_choice"] == 1).sum()
    n_catchment= (elem_mid["school_choice"] == 0).sum()
    print(f"  Valid school choice flag  : {n_valid:,} ({n_valid/len(elem_mid):.1%} of Elem+Mid trips)")
    print(f"    In-catchment (flag=0)   : {n_catchment:,} ({n_catchment/n_valid:.1%})")
    print(f"    School choice (flag=1)  : {n_choice:,}   ({n_choice/n_valid:.1%})")

    return elem_mid


def analysis_p2b(elem_mid: pd.DataFrame) -> None:
    """
    P2b: Escort analysis — in-catchment vs school choice trips.
    """
    sub = elem_mid.dropna(subset=["school_choice"]).copy()
    sub["escort"]     = (sub["escorting_flag"] == 1).astype(int)
    sub["car_escort"] = ((sub["escorting_flag"] == 1) & (sub["rep_mode_class1"] == 6)).astype(int)
    sub["choice_lbl"] = sub["school_choice"].map({0: "In-catchment", 1: "School choice"})

    # ── Overall cross-tab ──────────────────────────────────────────────────────
    tbl_overall = (
        sub.groupby("choice_lbl")
        .agg(
            n              = ("escort", "count"),
            escort_rate    = ("escort", "mean"),
            car_escort_rate= ("car_escort", "mean"),
        )
        .round(4)
    )
    tbl_overall["escort_pct"]     = (tbl_overall["escort_rate"]     * 100).round(1)
    tbl_overall["car_escort_pct"] = (tbl_overall["car_escort_rate"] * 100).round(1)
    print("\n  Overall escort rates by school choice:")
    print(tbl_overall.to_string())
    save_table(tbl_overall, "P2b_escort_by_school_choice")

    # ── By school level ────────────────────────────────────────────────────────
    tbl_level = (
        sub.groupby(["school_level", "choice_lbl"])
        .agg(
            n              = ("escort", "count"),
            escort_rate    = ("escort", "mean"),
            car_escort_rate= ("car_escort", "mean"),
        )
        .round(4)
    )
    tbl_level["escort_pct"]     = (tbl_level["escort_rate"]     * 100).round(1)
    tbl_level["car_escort_pct"] = (tbl_level["car_escort_rate"] * 100).round(1)
    print("\n  Escort rates by school level × school choice:")
    print(tbl_level.to_string())
    save_table(tbl_level, "P2b_escort_by_level_x_choice")

    # ── With LOS: escort rate × bus_freq × school choice ──────────────────────
    sub_los = sub.dropna(subset=["bus_frequency_per_day"]).copy()
    if len(sub_los) > 50:
        bins   = [-1, 0, 10, 30, sub_los["bus_frequency_per_day"].max() + 1]
        labels = ["0 (no bus)", "1–10", "11–30", "31+"]
        sub_los["bus_grp"] = pd.cut(sub_los["bus_frequency_per_day"], bins=bins, labels=labels)
        tbl_bus = (
            sub_los.groupby(["choice_lbl", "bus_grp"], observed=True)
            .agg(
                n           = ("escort", "count"),
                escort_rate = ("escort", "mean"),
            )
            .round(4)
        )
        tbl_bus["escort_pct"] = (tbl_bus["escort_rate"] * 100).round(1)
        print("\n  Escort rate by school choice × bus frequency:")
        print(tbl_bus.to_string())
        save_table(tbl_bus, "P2b_escort_by_choice_x_bus")

    # ── Figure: grouped bar chart ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    choice_categories = tbl_overall.index.tolist()
    colors = ["#4C72B0", "#C44E52"]

    for ax, metric, ylabel, title_suffix in zip(
        axes,
        ["escort_pct", "car_escort_pct"],
        ["Escort rate (%)", "Car escort rate (%)"],
        ["Escort Rate", "Car Escort Rate"]
    ):
        vals = [tbl_overall.loc[c, metric] if c in tbl_overall.index else 0
                for c in choice_categories]
        bars = ax.bar(choice_categories, vals, color=colors, width=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                    f"{val:.1f}%", ha="center", fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_title(f"P2b: {title_suffix}\nIn-Catchment vs School Choice\n(Elementary + Middle school trips)")
        ax.set_ylim(0, max(vals) * 1.2 if vals else 100)

    fig.tight_layout()
    save_fig(fig, "P2b_escort_by_school_choice")

    # ── Figure: school level breakdown ─────────────────────────────────────────
    levels = sub["school_level"].unique()
    x      = np.arange(len(levels))
    width  = 0.35

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    for i, (choice, color) in enumerate(zip(["In-catchment", "School choice"], colors)):
        esc_rates = []
        for lv in levels:
            try:
                esc_rates.append(tbl_level.loc[(lv, choice), "escort_pct"])
            except KeyError:
                esc_rates.append(0)
        bars = ax2.bar(x + i * width, esc_rates, width, label=choice, color=color)
        for bar, v in zip(bars, esc_rates):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f"{v:.0f}%", ha="center", fontsize=8)

    ax2.set_xticks(x + width/2)
    ax2.set_xticklabels(levels)
    ax2.set_ylabel("Escort rate (%)")
    ax2.set_title("P2b: Escort Rate by School Level × School Choice\n(Okinawa PT Survey R5)")
    ax2.legend()
    ax2.set_ylim(0, 100)
    fig2.tight_layout()
    save_fig(fig2, "P2b_escort_by_level_x_choice")

    # ── Chi-square test ────────────────────────────────────────────────────────
    from scipy import stats
    ct = pd.crosstab(sub["choice_lbl"], sub["escort"])
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    print(f"\n  Chi-square test (escort ~ school_choice):")
    print(f"    χ²({dof}) = {chi2:.4f},  p = {p:.4f}")
    if p < 0.05:
        print("    → Statistically significant difference in escort rates")
    else:
        print("    → No significant difference in escort rates")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "█" * 70)
    print("  03_school_choice.py — School District GIS + Choice Analysis")
    print("█" * 70)
    print(f"  SSD: {SSD_DIR}")

    if not HAS_GPD:
        print("  ⚠️  geopandas required — exiting.")
        return
    if not SSD_DIR.exists():
        print("  ⚠️  SSD not found — plug in Samsung T5.")
        return

    # ── P2a: build C-zone → school district lookup ─────────────────────────────
    section("P2a — C-zone → School District Lookup (GIS)")
    czone_lookup = build_czone_school_lookup(SSD_DIR)

    if not czone_lookup.empty:
        save_table(czone_lookup.set_index("czone_int"),
                   "P2a_czone_school_district_lookup")

    # ── Load school trips ───────────────────────────────────────────────────────
    section("Load School Trips")
    trips_path = PREP_DIR / "school_trips_los.csv"
    df = pd.read_csv(trips_path, low_memory=False)
    df["escort"]     = (df["escorting_flag"] == 1).astype(int)
    df["car_escort"] = ((df["escorting_flag"] == 1) & (df["rep_mode_class1"] == 6)).astype(int)
    df["car_dist_km"] = df["car_distance_m"] / 1000
    print(f"  Loaded : {len(df):,} school trips (all levels)")

    # ── P2b: school choice flag + analysis ─────────────────────────────────────
    section("P2b — School Choice Flag + Escort Analysis")
    elem_mid = build_school_choice_flag(df, czone_lookup)
    analysis_p2b(elem_mid)

    # ── Save enriched Elem/Middle trips ───────────────────────────────────────
    out_path = PREP_DIR / "school_choice_trips.csv"
    cols_to_save = [c for c in elem_mid.columns
                    if not c.startswith("_")]
    elem_mid[cols_to_save].to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n  ✅  Saved: school_choice_trips.csv ({len(elem_mid):,} rows)")

    section("Summary")
    tables  = sorted(t.name for t in TABLE_DIR.glob("P2*.csv"))
    figures = sorted(f.name for f in FIG_DIR.glob("P2*.png"))
    print(f"  Tables  : {tables}")
    print(f"  Figures : {figures}")
    print(f"\n  Next: P1d (#67) — Spatial map of marginal effects by C-zone\n")


if __name__ == "__main__":
    main()
