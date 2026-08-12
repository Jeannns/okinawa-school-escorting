"""
01_data_prep.py — Okinawa Extended Analysis: Data Preparation
=============================================================
Run this ONCE before any of the analysis files (02, 03, 04).
Outputs two CSV files to ./data/:

  school_trips_los.csv   — PT school trips joined with Current LOS
                           (F1: zone code mapping  +  F2: LOS join)

  school_locations.csv   — School centroids derived from district
                           boundary shapefiles, 16 cities
                           (F3: school geocoding proxy)

Dependencies:
  - PT survey data      : ../Okinawa_PT survey/Ver20240430第4回沖縄PTマスターデータ/EN/
  - SSD data            : /Volumes/Samsung_T5/For Jehan 260721/
  - geopandas (F3 only) : pip install geopandas

Notes on LOS join rate:
  ~70% of school trips join to LOS. Remaining ~30% are long-distance
  OD pairs absent from the LOS sparse matrix. These rows are kept with
  NaN for LOS columns — downstream analyses can filter with dropna().
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# Paths
# ══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).resolve().parent

# PT survey — same location as analysis_rq.py
DATA_DIR = (SCRIPT_DIR.parents[1]
            / "Okinawa_PT survey"
            / "Ver20240430第4回沖縄PTマスターデータ"
            / "EN")

# Samsung T5 SSD
SSD_DIR = Path("/Volumes/Samsung_T5/For Jehan 260721")
if not SSD_DIR.exists():
    SSD_DIR = Path("/sessions/awesome-beautiful-pascal/mnt/For Jehan 260721")

# Output folder
OUT_DIR = SCRIPT_DIR / "data"
OUT_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def section(title: str) -> None:
    bar = "═" * 70
    print(f"\n{bar}\n  {title}\n{bar}")


def load_csv(label: str, path: Path) -> pd.DataFrame:
    print(f"  Loading {label} … ", end="", flush=True)
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    df.columns = df.columns.str.strip()
    print(f"{len(df):,} rows × {df.shape[1]} cols")
    return df


def check_ssd() -> bool:
    ok = SSD_DIR.exists()
    if not ok:
        print(f"\n  ⚠️  SSD not found at {SSD_DIR}")
        print("     Plug in Samsung T5 and re-run.")
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# F1 — Zone Code Mapping: PT D-zone → LOS C-zone integer
# ══════════════════════════════════════════════════════════════════════════════

def build_zone_map(ssd_dir: Path) -> dict:
    """
    Returns dict: {d_zone_int → c_zone_int}

    ZoneCodeTable structure (after skipping headers):
      b_zone  c_zone  d_zone  city_code  city_name  town_name  …
      '01'    '011'   '01101'  47201     那覇市      曙一丁目   …

    PT  origin_zone_code = D-zone as int (e.g. 17103)
    LOS origin_czone     = C-zone as int (e.g. 171)
    """
    section("F1 — Zone Code Mapping (D-zone → C-zone)")

    zt_path = ssd_dir / "沖縄PTマスターデータ" / "ゾーンコード表.xlsx"
    print(f"  Source: {zt_path.name}")

    raw = pd.read_excel(zt_path, header=2)
    raw.columns = [
        "b_zone", "c_zone", "d_zone", "city_code", "city_name",
        "town_name", "b2", "b3", "b2b", "ku"
    ]

    # Drop header rows and rows without zone codes
    zt = raw[raw["d_zone"] != "Ｄゾーン"].dropna(subset=["d_zone"]).copy()
    zt["d_zone_int"] = pd.to_numeric(zt["d_zone"], errors="coerce")
    zt["c_zone_int"] = pd.to_numeric(zt["c_zone"], errors="coerce")
    zt = zt.dropna(subset=["d_zone_int", "c_zone_int"])

    zone_map = dict(
        zip(zt["d_zone_int"].astype(int), zt["c_zone_int"].astype(int))
    )
    print(f"  Zone map: {len(zone_map):,} D-zone → C-zone entries")
    print(f"  C-zone range: {min(zone_map.values())} – {max(zone_map.values())}")
    return zone_map


# ══════════════════════════════════════════════════════════════════════════════
# F2 — Build School Trips + Join LOS
# ══════════════════════════════════════════════════════════════════════════════

# Trip purpose codes for school trips (string in PT data)
SCHOOL_PURPOSES = ["10", "11", "12"]   # 10=elementary, 11=middle, 12=high school

SCHOOL_LEVEL_MAP = {
    "10": "Elementary",
    "11": "Middle",
    "12": "High School",
}

def build_school_trips(data_dir: Path, zone_map: dict) -> pd.DataFrame:
    """
    Load PT survey → filter school trips → apply zone mapping.
    Returns DataFrame with origin_czone, dest_czone columns added.
    """
    section("F2 — School Trips Sample + Zone Mapping")

    pt   = load_csv("Person Trip",   data_dir / "R05_PersonTrip_EN.csv")
    hh   = load_csv("HH Survey",     data_dir / "R05_HH_Survey_EN.csv")
    supp = load_csv("Supplementary", data_dir / "R05_Supplementary_EN.csv")

    # ── Filter to school trips ─────────────────────────────────────────────
    school = pt[pt["trip_purpose"].isin(SCHOOL_PURPOSES)].copy()
    print(f"\n  School trips (purpose 10/11/12): {len(school):,}")

    # School level label
    school["school_level"] = school["trip_purpose"].map(SCHOOL_LEVEL_MAP)

    # ── Apply zone mapping ─────────────────────────────────────────────────
    school["origin_zone_int"] = pd.to_numeric(
        school["origin_zone_code"], errors="coerce"
    ).astype("Int64")
    school["dest_zone_int"] = pd.to_numeric(
        school["dest_zone_code"], errors="coerce"
    ).astype("Int64")

    school["origin_czone"] = school["origin_zone_int"].map(zone_map)
    school["dest_czone"]   = school["dest_zone_int"].map(zone_map)

    orig_cov = school["origin_czone"].notna().mean()
    dest_cov = school["dest_czone"].notna().mean()
    print(f"  origin_czone mapping coverage : {orig_cov:.1%}")
    print(f"  dest_czone   mapping coverage : {dest_cov:.1%}")

    # ── Merge escort flag from SUPP ────────────────────────────────────────
    # SUPP has escort / safety concern info (same logic as analysis_rq.py)
    supp_cols = [
        c for c in supp.columns
        if any(k in c.lower() for k in [
            "person_id", "household_id", "trip_id",
            "escort", "safety", "flag", "q6", "q5", "q4"
        ])
    ]
    on_cols = [c for c in ["person_id", "trip_id"] if c in supp.columns and c in school.columns]
    if supp_cols and on_cols:
        school = school.merge(
            supp[supp_cols].drop_duplicates(),
            on=on_cols,
            how="left"
        )
    else:
        print(f"  ⚠️  Skipping SUPP merge — no join key found (supp_cols={len(supp_cols)}, on_cols={on_cols})")

    # ── Merge HH attributes ────────────────────────────────────────────────
    hh_cols = [
        c for c in hh.columns
        if any(k in c.lower() for k in [
            "household_id", "car_ownership", "num_car", "income",
            "hh_size", "num_children"
        ])
    ]
    if hh_cols and "household_id" in school.columns:
        school = school.merge(
            hh[hh_cols].drop_duplicates(subset=["household_id"]),
            on="household_id",
            how="left"
        )

    print(f"  Final sample : {len(school):,} rows × {school.shape[1]} cols")
    return school


def join_los(school: pd.DataFrame, ssd_dir: Path) -> pd.DataFrame:
    """
    Join Current LOS attributes to school trip dataframe.
    Trips without a matching OD pair get NaN for LOS columns.
    """
    section("F2 (cont.) — Join Current LOS")

    los_path = (ssd_dir / "Okinawa_PT_MasterData"
                / "04_LOS_ModalSplitParameters" / "01_CurrentLOS.csv")
    los = load_csv("Current LOS", los_path)

    before = len(school)
    merged = school.merge(
        los,
        left_on=["origin_czone", "dest_czone"],
        right_on=["origin_czone", "destination_czone"],
        how="left"
    )
    assert len(merged) == before, "Row count changed after LOS join!"

    join_rate = merged["walk_time_min"].notna().mean()
    miss_n    = merged["walk_time_min"].isna().sum()
    print(f"  LOS join rate : {join_rate:.1%}  ({miss_n:,} trips with NaN LOS)")
    print(f"  Note: NaN = OD pair absent from LOS sparse matrix (long-distance/cross-area trips)")
    print(f"  Downstream analyses that need LOS: filter with .dropna(subset=['walk_time_min'])")

    # ── Report LOS variable coverage ───────────────────────────────────────
    los_vars = [
        "walk_time_min", "bike_time_min", "bus_frequency_per_day",
        "bus_travel_time_min", "bus_fare_yen",
        "car_distance_m", "car_travel_time_min",
        "monorail_travel_time_min", "monorail_fare_yen",
    ]
    print("\n  LOS variable summary (joined trips only):")
    joined = merged.dropna(subset=["walk_time_min"])
    for v in los_vars:
        if v in merged.columns:
            print(f"    {v:<35} mean={joined[v].mean():8.2f}  "
                  f"median={joined[v].median():8.2f}  "
                  f"max={joined[v].max():8.1f}")

    return merged


# ══════════════════════════════════════════════════════════════════════════════
# F3 — School Centroids from District Polygons
# ══════════════════════════════════════════════════════════════════════════════

def extract_school_centroids(ssd_dir: Path) -> pd.DataFrame:
    """
    Extract school location proxy (polygon centroid) from Urban Planning
    school district shapefiles. Covers 16 cities × 2 levels (Elem + Middle).

    Returns DataFrame: city, school_name, school_type, lat, lon
    Note: centroid ≈ school location (±500m) — adequate for C-zone level analysis.
    """
    section("F3 — School Centroids from District Shapefiles (16 cities)")

    try:
        import geopandas as gpd
    except ImportError:
        print("  ⚠️  geopandas not installed. Run: pip install geopandas")
        print("  Skipping F3 — school_locations.csv will not be created.")
        return pd.DataFrame()

    urban_dir = ssd_dir / "Okinawa_UrbanPlanning_MasterData" / "R4"
    if not urban_dir.exists():
        print(f"  ⚠️  Not found: {urban_dir}")
        return pd.DataFrame()

    school_types = {
        "ElementarySchoolDistrict": "Elementary",
        "MiddleSchoolDistrict":     "Middle",
    }

    records = []
    cities_found = 0

    for city_dir in sorted(urban_dir.iterdir()):
        if not city_dir.is_dir():
            continue

        # Navigate to school district shapefiles
        shp_root = (city_dir / "00_DataFolder(*)"
                    if False else None)  # dynamic search below

        for st_folder, st_label in school_types.items():
            # Find shapefiles recursively for this school type
            shps = list(city_dir.rglob(f"*{st_folder}/*.shp"))
            if not shps:
                continue

            shp = shps[0]
            try:
                gdf = gpd.read_file(shp)
                # Project to EPSG:4326 (WGS84) for lat/lon
                gdf_wgs = gdf.to_crs("EPSG:4326")
                # Compute centroid in projected CRS first (more accurate)
                gdf_proj = gdf.to_crs(gdf.crs)
                gdf_wgs["centroid_lat"] = gdf_proj.geometry.centroid.to_crs("EPSG:4326").y
                gdf_wgs["centroid_lon"] = gdf_proj.geometry.centroid.to_crs("EPSG:4326").x

                for _, row in gdf_wgs.iterrows():
                    records.append({
                        "city_code":   row.get("SHICD", None),
                        "city_name":   row.get("SHI",   None),
                        "school_name": row.get("SC_NAME", None),
                        "school_type": st_label,
                        "lat":         round(row["centroid_lat"], 6),
                        "lon":         round(row["centroid_lon"], 6),
                        "source_shp":  shp.name,
                    })
                cities_found += 1

            except Exception as e:
                print(f"    ⚠️  Could not read {shp.name}: {e}")

    if not records:
        print("  ⚠️  No school shapefiles found.")
        return pd.DataFrame()

    df = pd.DataFrame(records).drop_duplicates(
        subset=["city_name", "school_name", "school_type"]
    )
    n_elem = (df["school_type"] == "Elementary").sum()
    n_mid  = (df["school_type"] == "Middle").sum()
    print(f"  Shapefiles read : {cities_found} city-level files")
    print(f"  Elementary schools : {n_elem}")
    print(f"  Middle schools     : {n_mid}")
    print(f"  Total records      : {len(df)}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "█" * 70)
    print("  01_data_prep.py — Okinawa Extended Analysis: Data Preparation")
    print("█" * 70)
    print(f"  PT data  : {DATA_DIR}")
    print(f"  SSD data : {SSD_DIR}")
    print(f"  Output   : {OUT_DIR}")

    # ── Check SSD ─────────────────────────────────────────────────────────
    ssd_ok = check_ssd()

    # ── F1: Zone mapping ──────────────────────────────────────────────────
    if ssd_ok:
        zone_map = build_zone_map(SSD_DIR)
    else:
        print("  ⚠️  Skipping F1/F2/F3 — SSD required.")
        return

    # ── F2: School trips + LOS join ───────────────────────────────────────
    school = build_school_trips(DATA_DIR, zone_map)
    school_los = join_los(school, SSD_DIR)

    out_trips = OUT_DIR / "school_trips_los.csv"
    school_los.to_csv(out_trips, index=False, encoding="utf-8-sig")
    print(f"\n  ✅  Saved: {out_trips.name}  ({len(school_los):,} rows)")

    # ── F3: School centroids ───────────────────────────────────────────────
    school_locs = extract_school_centroids(SSD_DIR)
    if not school_locs.empty:
        out_locs = OUT_DIR / "school_locations.csv"
        school_locs.to_csv(out_locs, index=False, encoding="utf-8-sig")
        print(f"  ✅  Saved: {out_locs.name}  ({len(school_locs):,} schools)")

    # ── Summary ────────────────────────────────────────────────────────────
    section("Summary")
    los_vars = ["walk_time_min", "bus_frequency_per_day", "car_distance_m"]
    n_total  = len(school_los)
    n_los    = school_los["walk_time_min"].notna().sum()
    print(f"  school_trips_los.csv : {n_total:,} school trips")
    print(f"    └─ with LOS data   : {n_los:,} ({n_los/n_total:.1%})")
    print(f"    └─ NaN LOS         : {n_total-n_los:,} ({(n_total-n_los)/n_total:.1%})  ← filter out when needed")
    if not school_locs.empty:
        print(f"  school_locations.csv : {len(school_locs):,} schools (16 cities)")
    print(f"\n  Next: run 02_los_and_safety.py\n")


if __name__ == "__main__":
    main()
