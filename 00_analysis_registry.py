"""
00_analysis_registry.py
=======================
Registry สำหรับ track ทุก analysis ใน Okinawa PT Survey Study
- รันเพื่อดู status ปัจจุบัน
- แก้ REGISTRY เพื่อเพิ่ม/ลด analysis
- แก้ STATUS ใน entry ที่ต้องการ update ผล
"""

from pathlib import Path
from datetime import date

SCRIPT_DIR = Path(__file__).resolve().parent

# ══════════════════════════════════════════════════════════════════════
# REGISTRY — แก้ไขได้เลย
# ══════════════════════════════════════════════════════════════════════
# status: "done" | "pending" | "not_needed" | "in_progress"

REGISTRY = [

    # ── Data Preparation ─────────────────────────────────────────────
    {
        "id":      "F0",
        "script":  "01_data_prep.py",
        "name":    "Data Preparation",
        "rq":      "prep",
        "desc":    "Zone code mapping (D→C zone), school trips sample, join CurrentLOS, school centroids",
        "inputs":  ["R05_PersonTrip_EN.csv", "R05_HH_Survey_EN.csv", "R05_Supplementary_EN.csv",
                    "ゾーンコード表.xlsx (SSD)", "01_CurrentLOS.csv (SSD)", "school shapefiles (SSD)"],
        "outputs": ["school_trips_los.csv", "school_locations.csv"],
        "status":  "done",
        "key_result": "10,676 school trips; 7,550 (70.7%) LOS-joined; 217 schools (16 cities)",
        "notes":   "SSD required for F1/F3; preprocessed outputs in data/ allow offline use",
    },

    # ── RQ1: School Travel Patterns ──────────────────────────────────
    {
        "id":      "A1",
        "script":  "analysis_rq.py",
        "name":    "H1a: Distance by School Level",
        "rq":      "RQ1",
        "desc":    "One-way ANOVA: travel distance × school level (kindergarten / elem / HS / university)",
        "inputs":  ["R05_PersonTrip_EN.csv", "R05_HH_Survey_EN.csv"],
        "outputs": ["output/Analysis1/"],
        "status":  "done",
        "key_result": "Distance increases with school level; HS mean ~3.8 km",
        "notes":   "",
    },
    {
        "id":      "A2",
        "script":  "analysis_rq.py",
        "name":    "H1b: Zone-crossing by School Level",
        "rq":      "RQ1",
        "desc":    "Chi-square test: zone-crossing rate × school level; B-zone origin ≠ B-zone dest",
        "inputs":  ["R05_PersonTrip_EN.csv"],
        "outputs": ["output/Analysis2/"],
        "status":  "done",
        "key_result": "70.9% out-of-catchment; HS zone-crossing >> Elementary",
        "notes":   "",
    },
    {
        "id":      "A3",
        "script":  "analysis_rq.py",
        "name":    "H1c: OLS Travel Time ~ Car Ownership",
        "rq":      "RQ1",
        "desc":    "OLS: travel_time ~ car_ownership + school_level + hh_size + income",
        "inputs":  ["R05_PersonTrip_EN.csv", "R05_HH_Survey_EN.csv"],
        "outputs": ["output/Analysis1/"],
        "status":  "done",
        "key_result": "Car ownership → longer travel time (upstream effect on school choice)",
        "notes":   "",
    },
    {
        "id":      "A4",
        "script":  "11_od_matrix.py",
        "name":    "OD Matrix: Home Municipality → School Municipality",
        "rq":      "RQ1",
        "desc":    "Spatial clustering: from which home municipalities do students travel to which school municipalities; cross-municipality rates by school level",
        "inputs":  ["school_trips_los.csv", "ZoneCodeTable.xlsx (SSD)"],
        "outputs": ["data/od_municipality.csv", "data/od_bzone.csv",
                    "figures/A4_od_heatmap_muni.png",
                    "figures/A4_od_crossmuni_by_level.png",
                    "figures/A4_od_top_flows.png"],
        "status":  "done",
        "key_result": "66.2% same-muni; Middle school highest cross-muni rate (38.8%); "
                      "Naha is top attractor (3,180 trips); Urasoe most 'imported' (39.4% from outside)",
        "notes":   "Municipality-level OD heatmap + cross-muni rate + B-zone OD CSV",
    },
    {
        "id":      "A5",
        "script":  "06_mnl_school_choice.py",
        "name":    "MNL School Choice Model",
        "rq":      "RQ1",
        "desc":    "Discrete choice: household chooses school among alternatives; IVs: distance, LOS, car ownership",
        "inputs":  ["school_choice_trips.csv", "school_locations.csv", "CZone.shp (SSD)"],
        "outputs": ["P2c_mnl_base.csv", "P2c_mnl_carinteract.csv", "P2c_counterfactual.png"],
        "status":  "done",
        "key_result": "Car HH choose more distant schools; distance elasticity significant",
        "notes":   "Requires CZone.shp from SSD",
    },

    # ── RQ2a: Escort Decision ────────────────────────────────────────
    {
        "id":      "B1",
        "script":  "analysis_rq.py",
        "name":    "Stage 1 Binary Logit: Escort Probability",
        "rq":      "RQ2a",
        "desc":    "escort(0/1) ~ car_ownership + school_level + escorter_sex + hh_size + child_sex + income",
        "inputs":  ["R05_PersonTrip_EN.csv", "R05_HH_Survey_EN.csv", "R05_Supplementary_EN.csv"],
        "outputs": ["output/Analysis1/", "output/Analysis2/"],
        "status":  "done",
        "key_result": "n=1,353; car ownership dominant (OR≈5×); McF-R²=0.145",
        "notes":   "Uses SUPP Q6 escort flag; n limited by SUPP coverage",
    },
    {
        "id":      "B2",
        "script":  "analysis_rq.py",
        "name":    "Stage 2 Binary Logit: Safety vs Resource",
        "rq":      "RQ2a",
        "desc":    "safety_concern(0/1) ~ car_ownership + school_level + ... (escorting parents only)",
        "inputs":  ["R05_Supplementary_EN.csv"],
        "outputs": ["output/Analysis2/"],
        "status":  "done",
        "key_result": "No-car HH more likely to cite safety concern; threat appraisal when coping limited",
        "notes":   "n=922 (escorting parents only)",
    },
    {
        "id":      "B3",
        "script":  "02_los_and_safety.py",
        "name":    "P1a–P1c: LOS + Safety Logit",
        "rq":      "RQ2a",
        "desc":    "Binary logit: escort ~ walk_time + bus_freq + car_dist + car_ownership; safety × bus interaction",
        "inputs":  ["school_trips_los.csv", "R05_Supplementary_EN.csv"],
        "outputs": ["P1a_safety_x_bus_heatmap.png", "P1b_logit_escort_LOS.csv", "P1c_logit_safety_x_bus.csv"],
        "status":  "done",
        "key_result": "LOS flat (p>0.1 all); car_ownership dominates; safety × bus n.s.",
        "notes":   "McF-R²=0.019; null LOS finding",
    },
    {
        "id":      "B4",
        "script":  "08_stratified_logit.py",
        "name":    "P4a: Stratified Logit (Car / No-car HH)",
        "rq":      "RQ2a",
        "desc":    "Separate logit for car-HH vs no-car HH; test if LOS effect differs",
        "inputs":  ["school_trips_los.csv"],
        "outputs": ["P4a_stratified_logit_summary.csv", "P4a_coef_comparison.png"],
        "status":  "done",
        "key_result": "No-car escort rate 8.5% vs car 56.7%; LOS absent in both groups → structural break",
        "notes":   "",
    },

    # ── RQ2b: Escort Mode Choice ─────────────────────────────────────
    {
        "id":      "C1",
        "script":  "09_mnl_escort_mode.py",
        "name":    "P4b: MNL Escort Mode (0/1/2)",
        "rq":      "RQ2b",
        "desc":    "MNL: no escort / non-car escort / car escort ~ LOS variables + car ownership",
        "inputs":  ["school_trips_los.csv"],
        "outputs": ["P4b_mnl_base_coef.csv", "P4b_mnl_base_ame.csv", "P4b_predicted_probs.png"],
        "status":  "done",
        "key_result": "McF-R²=0.0038; walk_time→car escort β=+0.60** (p=0.009); overall LOS weak",
        "notes":   "n=7,506 (LOS-joined sample)",
    },
    {
        "id":      "C2",
        "script":  "10_school_choice_escort.py",
        "name":    "MNL-I: Distance × Car Ownership Interaction",
        "rq":      "RQ2b",
        "desc":    "MNL escort mode + school trip distance + distance×car_ownership interaction",
        "inputs":  ["school_trips_los.csv", "CZone.shp (SSD)"],
        "outputs": ["D5_mnl_interaction.csv", "D3_interaction_dist_car.png"],
        "status":  "done",
        "key_result": "McF-R²: 0.004→0.031 (Δ=0.027); Dist×Car→car escort β=−0.72*** → car HH always-on, no-car HH distance-triggered",
        "notes":   "LR test χ²=433.69, p<0.001***; largest McF-R² gain of all analyses",
    },

    # ── LOS Counterfactual ───────────────────────────────────────────
    {
        "id":      "D1",
        "script":  "04_counterfactual.py",
        "name":    "P3b: Bus Frequency Counterfactual",
        "rq":      "RQ2 policy",
        "desc":    "Dose-response: doubling bus frequency → escort rate change; 4 scenarios",
        "inputs":  ["school_trips_los.csv", "01_CurrentLOS.csv (SSD)"],
        "outputs": ["P3b_scenario_summary.csv", "P3b_dose_response_escort_vs_bus.png"],
        "status":  "done",
        "key_result": "S1 (+5 trips/day): escort Δ=+0.06pp; S3 (×2 bus): Δ=+0.85pp — negligible policy effect",
        "notes":   "Null result: confirms LOS improvement cannot change escort behavior in car-dominant city",
    },
    {
        "id":      "D2",
        "script":  "04_counterfactual.py",
        "name":    "P3c: Future LOS Counterfactual",
        "rq":      "RQ2 policy",
        "desc":    "Compare Current vs Future LOS (with/without network improvement) → escort rate",
        "inputs":  ["school_trips_los.csv", "02_FutureLOS_No...csv", "03_FutureLOS_With...csv (SSD)"],
        "outputs": ["P3c_future_los_escort_comparison.png", "P3c_los_delta_summary.csv"],
        "status":  "done",
        "key_result": "Future LOS improvement → escort Δ≈0 — same null finding across time horizon",
        "notes":   "Requires both Future LOS files from SSD",
    },

    # ── Spatial ──────────────────────────────────────────────────────
    {
        "id":      "E1",
        "script":  "05_spatial_maps.py",
        "name":    "Spatial Maps (MAP1–6)",
        "rq":      "RQ2 spatial",
        "desc":    "C-zone level maps: escort rate, car escort rate, bus frequency, counterfactual delta",
        "inputs":  ["school_trips_los.csv outputs", "CZone.shp (SSD)"],
        "outputs": ["MAP1–MAP6 .png"],
        "status":  "done",
        "key_result": "High escort rate concentrated in suburban zones; bus frequency not aligned with escort patterns",
        "notes":   "",
    },

    # ── Trip Chaining ────────────────────────────────────────────────
    {
        "id":      "TC",
        "script":  "07_trip_chaining.py",
        "name":    "Trip Chaining Among Escorts",
        "rq":      "RQ2 mechanism",
        "desc":    "Do escorting parents chain the escort trip with a work/errand trip?",
        "inputs":  ["school_trips_los.csv", "R05_OkinawaPT_Person_Master.csv (SSD)"],
        "outputs": ["TC_overview.png", "TC_logit.csv", "TC_desc_overall.csv"],
        "status":  "done",
        "key_result": "35.8% of escort trips chained with work trip; has_work_trip significant predictor",
        "notes":   "Suggests escort is embedded in parental activity chain, not standalone decision",
    },

    # ── School Choice × Escort ───────────────────────────────────────
    {
        "id":      "SC",
        "script":  "03_school_choice.py",
        "name":    "P2a/P2b: School Choice × Escort",
        "rq":      "RQ1+RQ2",
        "desc":    "GIS catchment assignment → in/out-of-catchment → escort rate difference + counterfactual",
        "inputs":  ["school_trips_los.csv", "CZone.shp", "school shapefiles (SSD)"],
        "outputs": ["P2a_czone_school_district_lookup.csv", "P2b_escort_by_school_choice.csv"],
        "status":  "done",
        "key_result": "70.9% out-of-catchment; escort rate: in 2.5% vs out 2.0% → not significant",
        "notes":   "",
    },

]

# ══════════════════════════════════════════════════════════════════════
# PRINT SUMMARY
# ══════════════════════════════════════════════════════════════════════

STATUS_ICON = {
    "done":        "✅",
    "pending":     "⏳",
    "in_progress": "🔄",
    "not_needed":  "—",
}

def print_registry():
    print(f"\n{'═'*90}")
    print(f"  OKINAWA ANALYSIS REGISTRY   (as of {date.today()})")
    print(f"{'═'*90}\n")

    rq_groups = {}
    for entry in REGISTRY:
        rq = entry["rq"]
        rq_groups.setdefault(rq, []).append(entry)

    for rq, entries in rq_groups.items():
        print(f"  ── {rq} {'─'*(50-len(rq))}")
        for e in entries:
            icon    = STATUS_ICON.get(e["status"], "?")
            script  = Path(e["script"]).name if e["script"] != "— (not yet built)" else "— (not yet built)"
            print(f"  {icon}  [{e['id']:4s}] {e['name']}")
            print(f"         Script : {script}")
            print(f"         Desc   : {e['desc']}")
            if e["key_result"]:
                print(f"         Result : {e['key_result']}")
            if e["notes"]:
                print(f"         Notes  : {e['notes']}")
            print()

    # Status summary
    total   = len(REGISTRY)
    done    = sum(1 for e in REGISTRY if e["status"] == "done")
    pending = sum(1 for e in REGISTRY if e["status"] == "pending")
    wip     = sum(1 for e in REGISTRY if e["status"] == "in_progress")
    print(f"{'═'*90}")
    print(f"  Total: {total}  |  ✅ Done: {done}  |  🔄 In progress: {wip}  |  ⏳ Pending: {pending}")
    print(f"{'═'*90}\n")

    # Check which scripts exist on disk
    print("  Script existence check:")
    for e in REGISTRY:
        s = e["script"]
        if s == "— (not yet built)":
            print(f"    ⏳  [{e['id']}] {s}")
        else:
            p = SCRIPT_DIR / s
            exists = "✅" if p.exists() else "❌ MISSING"
            print(f"    {exists}  [{e['id']}] {s}")
    print()


if __name__ == "__main__":
    print_registry()
