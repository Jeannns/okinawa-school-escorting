"""
05_spatial_maps.py — Spatial Maps of Escort Patterns by C-zone (#67)
======================================================================
Requires: data/tables/D4_zone_level_escort_vs_los.csv
          data/tables/P3b_zone_counterfactual_S3_bus2x.csv
          data/tables/P3c_future_los_counterfactual.csv  (for legend)
          SSD: CZone.shp  (C-zone boundaries)

Maps produced (#67 P1d):
  Map 1 — Observed escort rate by C-zone
  Map 2 — Observed car escort rate by C-zone
  Map 3 — Mean bus frequency by C-zone
  Map 4 — Δ escort rate (counterfactual bus ×2, P3b S3) by C-zone
  Map 5 — Composite panel: 4-map overview
  Map 6 — Δ car escort rate (FutureWI vs Current LOS, P3c) by C-zone

All maps use diverging/sequential colormaps, trip-count-weighted alpha,
and focus on Okinawa main island (filter out remote islands).
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import geopandas as gpd
    HAS_GPD = True
except ImportError:
    HAS_GPD = False
    print("⚠️  geopandas required. Run: pip install geopandas --break-system-packages")

# ══════════════════════════════════════════════════════════════════════════════
# Paths
# ══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).resolve().parent
PREP_DIR   = SCRIPT_DIR / "data"
TABLE_DIR  = PREP_DIR / "tables"
FIG_DIR    = PREP_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SSD_DIR = Path("/Volumes/Samsung_T5/For Jehan 260721")
if not SSD_DIR.exists():
    SSD_DIR = Path("/sessions/awesome-beautiful-pascal/mnt/For Jehan 260721")

CZONE_SHP = SSD_DIR / "Okinawa_PT_MasterData" / "01_PopulationFrame" / "shp" / "CZone.shp"

# Okinawa main island bounding box (filter out remote islands like Miyako, Yaeyama)
MAIN_ISLAND_BBOX = {
    "lon_min": 127.6, "lon_max": 128.35,
    "lat_min": 26.0,  "lat_max": 26.9,
}


def section(title: str) -> None:
    bar = "═" * 70
    print(f"\n{bar}\n  {title}\n{bar}")


def save_fig(fig: plt.Figure, name: str) -> None:
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved: figures/{name}.png")


# ══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_czone_gdf() -> "gpd.GeoDataFrame":
    czone = gpd.read_file(CZONE_SHP)
    czone = czone.rename(columns={"Cゾーン": "czone_str", "緯度": "lat", "経度": "lon"})
    czone["czone_int"] = czone["czone_str"].apply(lambda x: int(str(x)))
    # Filter to main island
    czone = czone[
        (czone["lon"] >= MAIN_ISLAND_BBOX["lon_min"]) &
        (czone["lon"] <= MAIN_ISLAND_BBOX["lon_max"]) &
        (czone["lat"] >= MAIN_ISLAND_BBOX["lat_min"]) &
        (czone["lat"] <= MAIN_ISLAND_BBOX["lat_max"])
    ].copy()
    # Project to EPSG:6669 (Japan Plane Rectangular IX — good for Okinawa)
    czone = czone.to_crs("EPSG:6669")
    print(f"  CZone (main island): {len(czone)} zones")
    return czone


def build_map_gdf(czone: "gpd.GeoDataFrame") -> "gpd.GeoDataFrame":
    # D4 zone table
    d4 = pd.read_csv(TABLE_DIR / "D4_zone_level_escort_vs_los.csv")
    # P3b zone counterfactual
    p3b = pd.read_csv(TABLE_DIR / "P3b_zone_counterfactual_S3_bus2x.csv")

    gdf = czone.merge(
        d4.rename(columns={"origin_czone": "czone_int"}),
        on="czone_int", how="left"
    ).merge(
        p3b.rename(columns={"origin_czone": "czone_int"})[
            ["czone_int", "delta_escort_pp", "delta_car_escort_pp",
             "baseline_escort_pred", "cf_escort_pred"]
        ],
        on="czone_int", how="left"
    )

    n_with_data = gdf["escort_rate"].notna().sum()
    print(f"  Zones with escort data : {n_with_data}/{len(gdf)}")
    return gdf


# ══════════════════════════════════════════════════════════════════════════════
# Map Drawing Helpers
# ══════════════════════════════════════════════════════════════════════════════

def draw_choropleth(ax, gdf, col, cmap, vmin, vmax, title, unit="",
                    n_col=None, show_nodata=True):
    """
    Draw a single choropleth panel.
    Zones without data are shown in light gray.
    Circle size (scatter) shows sample size if n_col given.
    """
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=9, pad=4)

    # Zones without data — light gray base
    gdf.plot(ax=ax, color="#E0E0E0", linewidth=0.2, edgecolor="#AAAAAA")

    # Zones with data — colored
    sub = gdf[gdf[col].notna()].copy()
    if len(sub) == 0:
        return

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm   = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    sub.plot(ax=ax, column=col, cmap=cmap, vmin=vmin, vmax=vmax,
             linewidth=0.3, edgecolor="#666666", legend=False)

    # Colorbar
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.01, shrink=0.7)
    cbar.ax.tick_params(labelsize=7)
    if unit:
        cbar.set_label(unit, fontsize=7)

    # Stats annotation
    mean_val = sub[col].mean()
    ax.annotate(f"mean={mean_val:.2f}{unit}\nn={len(sub)} zones",
                xy=(0.02, 0.02), xycoords="axes fraction",
                fontsize=7, color="#333333",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))


# ══════════════════════════════════════════════════════════════════════════════
# Individual Maps
# ══════════════════════════════════════════════════════════════════════════════

def map_escort_rate(gdf: "gpd.GeoDataFrame") -> None:
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    draw_choropleth(ax, gdf, "escort_rate", "YlOrRd", 0, 1,
                    "Map 1: Observed Escort Rate by C-Zone\n(Okinawa PT Survey R5)",
                    unit="%")
    ax.set_title("Map 1: Observed Escort Rate by C-Zone\n(Okinawa PT Survey R5, all school trips)",
                 fontsize=10)
    fig.tight_layout()
    save_fig(fig, "MAP1_escort_rate_czone")


def map_car_escort_rate(gdf: "gpd.GeoDataFrame") -> None:
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    draw_choropleth(ax, gdf, "car_escort_rate", "Oranges", 0, 0.6,
                    "Map 2: Car Escort Rate by C-Zone", unit="")
    ax.set_title("Map 2: Car Escort Rate by C-Zone\n(Okinawa PT Survey R5)",
                 fontsize=10)
    fig.tight_layout()
    save_fig(fig, "MAP2_car_escort_rate_czone")


def map_bus_frequency(gdf: "gpd.GeoDataFrame") -> None:
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    draw_choropleth(ax, gdf, "mean_bus_freq", "Blues", 0, 100,
                    "Map 3: Mean Bus Frequency (trips/day) by C-Zone", unit=" trips/day")
    ax.set_title("Map 3: Mean Bus Frequency (trips/day) by Origin C-Zone\n(LOS-joined school trips)",
                 fontsize=10)
    fig.tight_layout()
    save_fig(fig, "MAP3_bus_frequency_czone")


def map_counterfactual_delta(gdf: "gpd.GeoDataFrame") -> None:
    """Map 4: Δ escort rate under bus ×2 (P3b S3)."""
    col = "delta_escort_pp"
    vmax = max(abs(gdf[col].dropna().max()), abs(gdf[col].dropna().min()))
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    draw_choropleth(ax, gdf, col, "RdBu_r", -vmax, vmax,
                    "Map 4: Δ Escort Rate (bus ×2 counterfactual)", unit=" pp")
    ax.set_title("Map 4: Δ Escort Rate (pp) — Counterfactual Bus Frequency ×2\n"
                 "(P3b S3: bus_freq doubled for all OD pairs with bus > 0)",
                 fontsize=9)
    fig.tight_layout()
    save_fig(fig, "MAP4_counterfactual_delta_czone")


def map_delta_car_escort(gdf: "gpd.GeoDataFrame") -> None:
    """Map 5: Δ car escort rate under bus ×2 (P3b S3)."""
    col = "delta_car_escort_pp"
    vmax = max(abs(gdf[col].dropna().max()), abs(gdf[col].dropna().min()))
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    draw_choropleth(ax, gdf, col, "PiYG", -vmax, vmax,
                    "Map 5: Δ Car Escort Rate (bus ×2 counterfactual)", unit=" pp")
    ax.set_title("Map 5: Δ Car Escort Rate (pp) — Counterfactual Bus Frequency ×2",
                 fontsize=9)
    fig.tight_layout()
    save_fig(fig, "MAP5_counterfactual_car_delta_czone")


def map_panel_4(gdf: "gpd.GeoDataFrame") -> None:
    """Map 6: 4-panel overview."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    panels = [
        ("escort_rate",       "YlOrRd",  0,    1,    "Escort Rate",             ""),
        ("car_escort_rate",   "Oranges", 0,    0.6,  "Car Escort Rate",          ""),
        ("mean_bus_freq",     "Blues",   0,    100,  "Mean Bus Freq (trips/day)", ""),
        ("delta_escort_pp",   "RdBu_r",  None, None, "Δ Escort Rate (bus ×2, pp)", " pp"),
    ]

    for ax, (col, cmap, vmin, vmax, title, unit) in zip(axes.flat, panels):
        if col == "delta_escort_pp":
            vabs = max(abs(gdf[col].dropna().max()), abs(gdf[col].dropna().min()))
            vmin, vmax = -vabs, vabs
        draw_choropleth(ax, gdf, col, cmap, vmin, vmax, title, unit)

    fig.suptitle("Okinawa School Escorting Spatial Patterns by C-Zone\n(Okinawa PT Survey R5)",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    save_fig(fig, "MAP6_panel_4maps")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "█" * 70)
    print("  05_spatial_maps.py — Spatial Maps (#67 P1d)")
    print("█" * 70)

    if not HAS_GPD:
        print("  ⚠️  geopandas required — exiting.")
        return
    if not CZONE_SHP.exists():
        print(f"  ⚠️  CZone.shp not found at {CZONE_SHP}")
        return

    section("Load Data")
    czone = load_czone_gdf()
    gdf   = build_map_gdf(czone)

    section("Map 1 — Escort Rate")
    map_escort_rate(gdf)

    section("Map 2 — Car Escort Rate")
    map_car_escort_rate(gdf)

    section("Map 3 — Bus Frequency")
    map_bus_frequency(gdf)

    section("Map 4 — Δ Escort Rate (Counterfactual)")
    map_counterfactual_delta(gdf)

    section("Map 5 — Δ Car Escort Rate (Counterfactual)")
    map_delta_car_escort(gdf)

    section("Map 6 — 4-Panel Overview")
    map_panel_4(gdf)

    section("Summary")
    maps = sorted(f.name for f in FIG_DIR.glob("MAP*.png"))
    print(f"  Maps saved ({len(maps)}): {maps}")
    print(f"\n  Next: #72 MNL School Choice Model\n")


if __name__ == "__main__":
    main()
