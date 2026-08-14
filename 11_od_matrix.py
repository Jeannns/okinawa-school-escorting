"""
A4: OD Matrix — Home Municipality/B-zone → School Municipality
RQ1: School travel spatial clustering patterns
Input:  data/school_trips_los.csv
Output: data/od_municipality.csv
        data/od_bzone.csv
        figures/A4_od_heatmap_muni.png
        figures/A4_od_crossmuni_by_level.png
        figures/A4_od_top_flows.png
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
DATA = BASE / "data"
FIG  = BASE / "figures"
FIG.mkdir(exist_ok=True)

SSD_DIR = Path("/Volumes/Samsung_T5/For Jehan 260721")
if not SSD_DIR.exists():
    SSD_DIR = Path("/sessions/awesome-beautiful-pascal/mnt/For Jehan 260721")

ZONE_TABLE = SSD_DIR / "Okinawa_PT_MasterData" / "ZoneCodeTable.xlsx"

# ── Municipality name lookup ──────────────────────────────────────────────────
def load_muni_map():
    """Return dict {muni_code_3digit: 'City name'}"""
    xf = pd.read_excel(ZONE_TABLE, header=None)
    df = xf.iloc[3:].copy()
    df.columns = ["b_zone","c_zone","d_zone","muni_code_full",
                  "muni_name","block_name","bz2","bz3","bz2b","category"]
    df = df[pd.to_numeric(df["muni_code_full"], errors="coerce").notna()].copy()
    df["muni_code_full"] = df["muni_code_full"].astype(int)
    df["muni_code"] = df["muni_code_full"] % 1000
    muni_map = (df[["muni_code","muni_name"]]
                .drop_duplicates("muni_code")
                .set_index("muni_code")["muni_name"]
                .to_dict())
    return muni_map

# ── Short labels for plots ────────────────────────────────────────────────────
SHORT_NAME = {
    "那覇市":   "Naha",
    "宜野湾市": "Ginowan",
    "浦添市":   "Urasoe",
    "糸満市":   "Itoman",
    "沖縄市":   "Okinawa C.",
    "豊見城市": "Tomigusuku",
    "うるま市": "Uruma",
    "南城市":   "Nanjo",
    "読谷村":   "Yomitan",
    "嘉手納町": "Kadena",
    "北谷町":   "Chatan",
    "中頭郡北中城村": "Kitanakagusuku",
    "中城村":   "Nakagusuku",
    "西原町":   "Nishihara",
    "与那原町": "Yonabaru",
    "南風原町": "Haebaru",
    "八重瀬町": "Yaese",
    "Unknown":  "Unknown",
}

LEVEL_ORDER = ["Elementary", "Middle", "High School"]
LEVEL_COLOR = {
    "Elementary":  "#4C72B0",
    "Middle":      "#55A868",
    "High School": "#C44E52",
}

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading school_trips_los.csv …")
df = pd.read_csv(DATA / "school_trips_los.csv")
print(f"  {len(df):,} rows, {df['school_level'].value_counts().to_dict()}")

muni_map = load_muni_map()
print(f"  Loaded municipality names for {len(muni_map)} codes")

df["origin_muni_name"] = df["origin_municipality_code"].map(muni_map).fillna("Unknown")
df["dest_muni_name"]   = df["dest_municipality_code"].map(muni_map).fillna("Unknown")
df["origin_short"] = df["origin_muni_name"].map(SHORT_NAME).fillna(df["origin_muni_name"])
df["dest_short"]   = df["dest_muni_name"].map(SHORT_NAME).fillna(df["dest_muni_name"])
df["is_cross_muni"] = (df["origin_municipality_code"] != df["dest_municipality_code"]).astype(int)

# ── A4-1: Municipality-level OD matrix ───────────────────────────────────────
print("\n[A4-1] Municipality OD matrix …")
od_muni = (df.groupby(["origin_muni_name","dest_muni_name"])
             .size().reset_index(name="trips"))
od_pivot = od_muni.pivot(index="origin_muni_name", columns="dest_muni_name", values="trips").fillna(0)

# Save raw OD
od_muni.to_csv(DATA / "od_municipality.csv", index=False)
print(f"  Saved od_municipality.csv ({len(od_muni)} OD pairs)")

# Reorder: sort by total trips from each origin
origin_order = (od_pivot.sum(axis=1).sort_values(ascending=False).index.tolist())
dest_order   = (od_pivot.sum(axis=0).sort_values(ascending=False).index.tolist())
od_pivot = od_pivot.loc[origin_order, dest_order]

# Map to short names
od_pivot.index   = [SHORT_NAME.get(x, x) for x in od_pivot.index]
od_pivot.columns = [SHORT_NAME.get(x, x) for x in od_pivot.columns]

# ── Figure 1: Heatmap ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 9))
data_arr = od_pivot.values.astype(float)
# Normalize by row (origin total → % to each destination)
row_sum = data_arr.sum(axis=1, keepdims=True)
row_sum[row_sum == 0] = 1
pct_arr = data_arr / row_sum * 100

im = ax.imshow(pct_arr, cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)
plt.colorbar(im, ax=ax, label="% of trips from origin municipality")

ax.set_xticks(range(len(od_pivot.columns)))
ax.set_yticks(range(len(od_pivot.index)))
ax.set_xticklabels(od_pivot.columns, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(od_pivot.index, fontsize=8)
ax.set_xlabel("Destination (School Municipality)", fontsize=10)
ax.set_ylabel("Origin (Home Municipality)", fontsize=10)
ax.set_title("OD Matrix: Home Municipality → School Municipality\n"
             "(Cell = % of trips from that origin going to that destination)",
             fontsize=11, fontweight="bold")

# Annotate cells with absolute trip counts (only cells > 30 trips for clarity)
for i in range(len(od_pivot.index)):
    for j in range(len(od_pivot.columns)):
        val = int(data_arr[i, j])
        if val >= 30:
            color = "white" if pct_arr[i, j] > 60 else "black"
            ax.text(j, i, str(val), ha="center", va="center",
                    fontsize=6.5, color=color)

# Diagonal line to mark same-municipality
diag_cities = [c for c in od_pivot.index if c in od_pivot.columns]
for city in diag_cities:
    xi = list(od_pivot.columns).index(city)
    yi = list(od_pivot.index).index(city)
    ax.add_patch(plt.Rectangle((xi - 0.5, yi - 0.5), 1, 1,
                                fill=False, edgecolor="steelblue",
                                linewidth=1.5, linestyle="--"))

plt.tight_layout()
fig.savefig(FIG / "A4_od_heatmap_muni.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved A4_od_heatmap_muni.png")

# ── A4-2: Cross-municipality rate by school level ────────────────────────────
print("\n[A4-2] Cross-municipality rate by school level …")
cross = (df.groupby("school_level")["is_cross_muni"]
           .agg(["mean","sum","count"])
           .reset_index())
cross.columns = ["school_level","cross_rate","cross_trips","total_trips"]
cross["cross_rate_pct"] = cross["cross_rate"] * 100
cross["same_rate_pct"]  = 100 - cross["cross_rate_pct"]
cross = cross[cross["school_level"].isin(LEVEL_ORDER)]
cross["school_level"] = pd.Categorical(cross["school_level"], categories=LEVEL_ORDER, ordered=True)
cross = cross.sort_values("school_level")

print("  Cross-municipality rates:")
for _, r in cross.iterrows():
    print(f"    {r['school_level']}: {r['cross_rate_pct']:.1f}% cross-muni "
          f"({r['cross_trips']:.0f}/{r['total_trips']:.0f})")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Stacked bar — same vs cross
ax = axes[0]
bar_w = 0.5
x = np.arange(len(cross))
bars_same  = ax.bar(x, cross["same_rate_pct"],  bar_w, label="Same municipality",  color="#4C72B0", alpha=0.85)
bars_cross = ax.bar(x, cross["cross_rate_pct"], bar_w, bottom=cross["same_rate_pct"],
                    label="Cross-municipality", color="#C44E52", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(cross["school_level"], fontsize=10)
ax.set_ylabel("% of school trips", fontsize=10)
ax.set_title("Within vs. Cross-Municipality School Travel\nby School Level", fontsize=11, fontweight="bold")
ax.set_ylim(0, 105)
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.legend(fontsize=9)
for i, (_, r) in enumerate(cross.iterrows()):
    ax.text(i, r["cross_rate_pct"] / 2 + r["same_rate_pct"],
            f"{r['cross_rate_pct']:.1f}%", ha="center", va="center",
            fontsize=9, color="white", fontweight="bold")

# Right: Top cross-municipality flows
ax = axes[1]
cross_flows = df[df["is_cross_muni"] == 1].copy()
top_flows = (cross_flows.groupby(["origin_short","dest_short"])
             .size().reset_index(name="trips")
             .sort_values("trips", ascending=False).head(12))
top_flows["label"] = top_flows["origin_short"] + " → " + top_flows["dest_short"]
colors = [LEVEL_COLOR.get("High School")] * len(top_flows)
ax.barh(range(len(top_flows)), top_flows["trips"], color="#6A8EBF", alpha=0.85)
ax.set_yticks(range(len(top_flows)))
ax.set_yticklabels(top_flows["label"], fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("Number of school trips", fontsize=10)
ax.set_title("Top Cross-Municipality\nSchool Travel Flows", fontsize=11, fontweight="bold")
for i, val in enumerate(top_flows["trips"]):
    ax.text(val + 2, i, str(val), va="center", fontsize=8)

plt.tight_layout()
fig.savefig(FIG / "A4_od_crossmuni_by_level.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved A4_od_crossmuni_by_level.png")

# ── A4-3: Top flows per school level ─────────────────────────────────────────
print("\n[A4-3] Top flows by school level …")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, level in zip(axes, LEVEL_ORDER):
    sub = df[df["school_level"] == level].copy()
    top = (sub.groupby(["origin_short","dest_short"])
              .size().reset_index(name="trips")
              .sort_values("trips", ascending=False).head(10))
    top["label"] = top["origin_short"] + " → " + top["dest_short"]
    # mark same-muni
    top["same"] = top["origin_short"] == top["dest_short"]
    bar_colors = [LEVEL_COLOR[level] if s else "#AAAAAA" for s in top["same"]]
    ax.barh(range(len(top)), top["trips"], color=bar_colors, alpha=0.85)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["label"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Trips", fontsize=9)
    ax.set_title(f"{level}\n(blue=same, gray=cross muni)", fontsize=10, fontweight="bold",
                 color=LEVEL_COLOR[level])
    for i, val in enumerate(top["trips"]):
        ax.text(val + 1, i, str(val), va="center", fontsize=7.5)
    total_cross = (sub["is_cross_muni"].sum() / len(sub) * 100) if len(sub) > 0 else 0
    ax.text(0.98, 0.02, f"Cross-muni: {total_cross:.1f}%",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            color="dimgray")

plt.suptitle("Top 10 School Trip Flows by School Level\n(Home Municipality → School Municipality)",
             fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
fig.savefig(FIG / "A4_od_top_flows.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved A4_od_top_flows.png")

# ── A4-4: B-zone level OD ───────────────────────────────────────────────────
print("\n[A4-4] B-zone OD (home id_b_zone_code → school dest_czone) …")
# dest_czone is a 2–3 digit C-zone code; first 2 chars = B-zone
df["dest_bzone"] = (df["dest_czone"].dropna()
                    .astype(str).str.replace(r"\.0$","",regex=True)
                    .str.zfill(3).str[:2].astype(int, errors="ignore"))
od_bzone = (df.dropna(subset=["id_b_zone_code","dest_bzone"])
              .groupby(["id_b_zone_code","dest_bzone","school_level"])
              .size().reset_index(name="trips"))
od_bzone.to_csv(DATA / "od_bzone.csv", index=False)
print(f"  Saved od_bzone.csv ({len(od_bzone)} origin–dest–level triplets)")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n══ A4 OD Matrix Summary ══")
total = len(df)
cross_total = df["is_cross_muni"].sum()
print(f"  Total school trips analysed : {total:,}")
print(f"  Cross-municipality trips     : {cross_total:,} ({cross_total/total*100:.1f}%)")
print(f"  Same-municipality trips      : {total-cross_total:,} ({(total-cross_total)/total*100:.1f}%)")
print()
print("  By school level:")
for _, r in cross.iterrows():
    print(f"    {r['school_level']:12s}  cross={r['cross_rate_pct']:.1f}%  "
          f"n={r['total_trips']:.0f}")
print()
# Biggest school attractors
attractors = df.groupby("dest_muni_name")["is_cross_muni"].agg(["sum","count"]).reset_index()
attractors.columns = ["muni","cross_in","total_in"]
attractors["cross_in_pct"] = attractors["cross_in"] / attractors["total_in"] * 100
attractors = attractors.sort_values("total_in", ascending=False)
print("  Top 5 school destination municipalities:")
for _, r in attractors.head(5).iterrows():
    print(f"    {r['muni']:12s}  total_in={r['total_in']:.0f}  "
          f"from_cross={r['cross_in']:.0f} ({r['cross_in_pct']:.1f}%)")

print("\nDone.")
