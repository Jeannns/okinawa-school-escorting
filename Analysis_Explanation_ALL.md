# Okinawa PT Survey — Analysis Explanation (ทุก Analysis อย่างละเอียด)

> รวม 23 analyses | อัปเดต 2026-08-14  
> เป้าหมาย: อธิบาย *ทำไม / ข้อมูลอะไร / ลิงก์ยังไง / ผลคืออะไร / วิจารณ์อะไรได้บ้าง*  
> ภาษาไทยสำหรับผู้อ่านที่รู้จักบริบท school escorting และ safety perception

---

## แผนที่ว่า Analysis ไหนมีอยู่เดิม — ไหนเพิ่มใหม่

| สถานะ | หมายความว่า |
|---|---|
| 🟢 **มีอยู่เดิม** | สร้างและรันอยู่ก่อน session นี้ ผลถูก track ใน registry แล้ว |
| 🟡 **มีอยู่แต่ไม่ได้ track** | code อยู่ใน script เดิมแล้ว รันได้ผลแล้ว แต่ยังไม่ได้ใส่ใน registry — เพิ่ง add entry เข้าไปใน session นี้ |
| 🔴 **สร้างใหม่ session นี้** | script ใหม่ทั้งหมด เขียนขึ้นในช่วง session นี้หลังได้ข้อมูลเพิ่ม |

### ตาราง Origin ของแต่ละ Analysis

| ID | ชื่อ | สถานะ | เหตุผล |
|---|---|---|---|
| F0 | Data Preparation | 🟢 มีอยู่เดิม | `01_data_prep.py` เป็น pipeline หลัก มีก่อน session นี้ |
| A1 | Distance by Level | 🟢 มีอยู่เดิม | อยู่ใน `analysis_rq.py` มาแต่แรก |
| A2 | Zone-Crossing | 🟢 มีอยู่เดิม | อยู่ใน `analysis_rq.py` มาแต่แรก |
| A3 | OLS Travel Time ~ Car | 🟢 มีอยู่เดิม | อยู่ใน `analysis_rq.py` มาแต่แรก |
| **A4** | **OD Matrix** | 🔴 **สร้างใหม่ session นี้** | `11_od_matrix.py` เขียนใหม่ทั้งหมดใน session นี้ เพราะก่อนหน้ายังเป็น pending |
| A5 | MNL School Choice | 🟢 มีอยู่เดิม | `06_mnl_school_choice.py` มีก่อน |
| **D1** | **Escort × Walk Quartile** | 🟡 **มีแต่ไม่ได้ track** | code อยู่ใน `02_los_and_safety.py` แล้ว ผลมีอยู่ แต่เพิ่ง add เข้า registry session นี้ |
| **D2** | **Escort × Bus Frequency** | 🟡 **มีแต่ไม่ได้ track** | เหมือน D1 |
| **D3** | **Car Distance × Escort** | 🟡 **มีแต่ไม่ได้ track** | เหมือน D1 |
| **D4** | **Zone LOS vs Escort** | 🟡 **มีแต่ไม่ได้ track** | เหมือน D1 |
| B1 | Logit: Escort Prob | 🟢 มีอยู่เดิม | `analysis_rq.py` มาแต่แรก |
| B2 | Logit: Safety vs Resource | 🟢 มีอยู่เดิม | `analysis_rq.py` มาแต่แรก |
| B3 | LOS + Safety Logit | 🟢 มีอยู่เดิม | `02_los_and_safety.py` มาแต่แรก |
| B4 | Stratified Logit | 🟢 มีอยู่เดิม | `08_stratified_logit.py` มาแต่แรก |
| C1 | MNL Escort Mode | 🟢 มีอยู่เดิม | `09_mnl_escort_mode.py` มาแต่แรก |
| C2 | MNL + Dist×Car Interact | 🟢 มีอยู่เดิม | `10_school_choice_escort.py` มาแต่แรก |
| P3b | Bus Counterfactual | 🟢 มีอยู่เดิม | `04_counterfactual.py` มาแต่แรก (เดิม ID ว่า D1 ก่อน rename) |
| P3c | Future LOS Counterfactual | 🟢 มีอยู่เดิม | `04_counterfactual.py` มาแต่แรก (เดิม ID ว่า D2 ก่อน rename) |
| E1 | Spatial Maps | 🟢 มีอยู่เดิม | `05_spatial_maps.py` มาแต่แรก |
| TC | Trip Chaining | 🟢 มีอยู่เดิม | `07_trip_chaining.py` มาแต่แรก |
| SC | School Choice × Escort | 🟢 มีอยู่เดิม | `03_school_choice.py` มาแต่แรก |
| **E2** | **Crime Density vs Escort** | 🔴 **สร้างใหม่ session นี้** | `12_e2_crime_safety.py` เขียนใหม่ หลังจากได้ crime data CSV 7 ไฟล์ |
| **E5** | **Pop-Weighted Escort Rate** | 🔴 **สร้างใหม่ session นี้** | `13_e5_pop_weighted.py` เขียนใหม่ หลังจาก HH Survey population proxy พร้อม |

### สรุปจำนวน

| สถานะ | จำนวน | IDs |
|---|---|---|
| 🟢 มีอยู่เดิม | 16 | F0, A1, A2, A3, A5, B1, B2, B3, B4, C1, C2, P3b, P3c, E1, TC, SC |
| 🟡 มีแต่ไม่ได้ track | 4 | D1, D2, D3, D4 |
| 🔴 สร้างใหม่ session นี้ | 3 | A4, E2, E5 |
| **รวม** | **23** | |

### Note เรื่อง P3b / P3c (rename)

ก่อน session นี้ registry มี `D1` = Bus Counterfactual และ `D2` = Future LOS Counterfactual ซึ่งซ้ำกับ ID ของ descriptive analyses ที่เพิ่งจะ add เข้ามา จึง **rename** เป็น P3b และ P3c ให้ตรงกับ Analysis Roadmap v2 ตัว code (`04_counterfactual.py`) ไม่ได้เปลี่ยน เปลี่ยนแค่ชื่อ ID ใน registry

---

## ภาพรวม Dataset

| ไฟล์หลัก | ที่มา | บทบาท |
|---|---|---|
| `R05_PersonTrip_EN.csv` | PT Survey มาสเตอร์ | ทุก trip ของทุกคนในวันสำรวจ |
| `R05_HH_Survey_EN.csv` | PT Survey มาสเตอร์ | ข้อมูลครัวเรือน รายได้ รถยนต์ จำนวนคน |
| `R05_Supplementary_EN.csv` | PT Survey มาสเตอร์ | Q6 escort reason / เหตุผลที่พ่อแม่ escort |
| `ゾーンコード表.xlsx` | SSD external | ตาราง D-zone → C-zone → B-zone → municipality |
| `01_CurrentLOS.csv` | SSD external | LOS ปัจจุบัน: walk_time, bus_freq, car_dist ต่อ OD pair |
| `school_trips_los.csv` | **preprocessed** | ผล join ของทั้งหมดข้างต้น → ใช้ใน analyses ส่วนใหญ่ |

Zone hierarchy: D-zone (5 หลัก) → C-zone (3 หลัก) → B-zone (2 หลัก) → municipality

---

## F0 — Data Preparation (Preprocessing)

### ทำไมต้องทำ
ข้อมูล PT Survey มาในรูป Person Trip แบบ "ทุกคน ทุก trip" ซึ่งรวมทั้ง trips ไปทำงาน ซื้อของ นันทนาการ ฯลฯ ก่อนจะทำ analysis ใดก็ตามต้องกรอง trip ที่เกี่ยวกับ **การเดินทางไปโรงเรียน** ออกมาก่อน และต้อง join กับ LOS (Level of Service) เพื่อให้แต่ละ trip มีข้อมูลว่า ถ้าเดินจะใช้เวลาเท่าไหร่ รถเมล์มีกี่เที่ยว ขับรถไกลแค่ไหน

### ข้อมูลที่ใช้และการลิงก์
```
R05_PersonTrip_EN.csv
   ↓ กรอง trip_purpose ∈ {10,11,12}  (elementary / middle / high school)
   ↓ แต่ละ trip มี origin_zone_code (D-zone) + dest_zone_code
   ↓
ゾーนコード表.xlsx (SSD)
   ↓ map D-zone → C-zone (3 หลัก)
   ↓
01_CurrentLOS.csv (SSD)
   ↓ join บน (origin_czone, dest_czone) คู่ → ได้ walk_time, bus_freq, car_dist, fare
   ↓
school_trips_los.csv  ← intermediate file หลัก (10,676 rows)
```

- Zone mapping coverage: origin 71%, destination 71% (บางโซนชายขอบไม่อยู่ใน LOS table)
- 70.7% ของ trips ได้ LOS สำเร็จ → 7,550 trips สำหรับ LOS analyses

### ผล
- 10,676 school trips ทั้งหมด (Elementary 2,224 / Middle 2,539 / HS 5,913)
- 7,550 trips มี LOS ครบ
- 217 โรงเรียน ใน 16 เมือง

### วิจารณ์ในมุม escort & safety
School trips ใน dataset นี้คือ **trip ที่ destination คือโรงเรียน** — ซึ่งรวมทั้ง trip ของ **นักเรียนเอง** และ **trip ของพ่อแม่ที่ไปส่ง** (ถ้า purpose ของพ่อแม่ถูก code ว่าไปโรงเรียน) ในทางปฏิบัติ `escorting_flag=1` หมายถึง "เดินทางนี้มีการ escort" ซึ่งเมื่อดูจากข้อมูลพบว่า HS escort rate ~97.9% vs Elementary ~1.8% — สะท้อนว่าใน Okinawa การ escort นักเรียน HS ด้วยรถยนต์คือ **ค่าปกติ** ไม่ใช่ safety concern แต่เป็น car-dependent lifestyle

---

## A1 — H1a: Travel Distance by School Level

### ทำไมต้องทำ
hypothesis พื้นฐานที่สุดของการศึกษาการเดินทางไปโรงเรียน: **ระดับชั้นสูงขึ้น = เดินไกลขึ้น** เพราะ catchment area กว้างขึ้น (โรงเรียนอาชีวะ/HS มีน้อยกว่า เฉพาะทางกว่า) การพิสูจน์ก่อนเป็นพื้นฐานสำหรับทุก analysis ถัดไป

### ข้อมูลที่ใช้และการลิงก์
```
R05_PersonTrip_EN.csv
   ↓ กรอง school trips + คำนวณ travel distance จาก zone coordinates
R05_HH_Survey_EN.csv
   ↓ join บน household key → ได้ car_ownership, income, hh_size
```

วิธีการ: One-way ANOVA เปรียบเทียบ mean distance ข้าม school level (kindergarten, elementary, middle, HS, university)

### ผล
ระยะทางเพิ่มขึ้นตามระดับชั้น — HS mean ~3.8 km (ยืนยัน hypothesis)

### วิจารณ์
ระยะทางที่เพิ่มขึ้นตาม school level นี้สำคัญมากในมุม escort: ยิ่งไกลยิ่งมีแรงจูงใจ escort ด้วยรถ แต่ผลที่น่าสนใจคือ HS ไกล + escort rate สูง ไม่ใช่เพราะ **safety concern** แต่เพราะ **ระยะทางเกิน walking threshold** และพ่อแม่มีรถ การตีความนี้จะชัดขึ้นเมื่อดูร่วมกับ B3/B4

---

## A2 — H1b: Zone-Crossing Rate by School Level

### ทำไมต้องทำ
ใน Okinawa มี school choice (เลือกโรงเรียนนอก catchment ได้) ต้องการรู้ว่า school level ไหน **ข้าม B-zone มากที่สุด** เพราะถ้าเด็กข้าม zone เยอะ = เดินทางไม่ได้ = ต้องพึ่งรถ = โอกาส escort สูง

### ข้อมูลที่ใช้และการลิงก์
```
R05_PersonTrip_EN.csv
   ↓ origin_b_zone ≠ dest_b_zone → binary cross-zone flag
   ↓ Chi-square test: zone-crossing × school_level
```

### ผล
70.9% ของ school trips ข้าม zone (out-of-catchment) — HS ข้ามมากที่สุด

### วิจารณ์
Zone-crossing ≠ escorted อัตโนมัติ แต่บ่งชี้ว่าระบบโรงเรียน Okinawa มี **school choice สูงมาก** ซึ่งทำให้ระยะทางยาวขึ้น และทำให้ **walking / cycling ทำได้ยาก** จึงนำไปสู่ escort ด้วยรถ ในมุม safety: zone-crossing เอง ไม่ได้วัด safety concern โดยตรง แต่เป็น structural factor ที่ทำให้ car dependence สูง

---

## A3 — H1c: OLS Travel Time ~ Car Ownership

### ทำไมต้องทำ
ต้องการทดสอบว่า **การมีรถทำให้เดินทางนานขึ้นหรือไม่** (paradox: ครอบครัวมีรถ อาจเลือกโรงเรียนไกลกว่า → travel time สูงกว่า แม้รถเร็วกว่า)

### ข้อมูลที่ใช้และการลิงก์
```
R05_PersonTrip_EN.csv + R05_HH_Survey_EN.csv
   ↓ join household key
   ↓ OLS: travel_time ~ car_ownership + school_level + hh_size + income
```

### ผล
Car ownership → **travel time นาน** (upstream school choice effect): ครอบครัวมีรถเลือกโรงเรียนไกล

### วิจารณ์
ผลนี้ชี้ให้เห็น **endogeneity** ที่สำคัญ: รถไม่ใช่แค่ mode of transport แต่ยังเปลี่ยน school choice ด้วย ทำให้ยากที่จะ identify ว่า escort เกิดจาก "รถอยู่แล้วก็ส่ง" vs "เส้นทางยาวจนต้องขับส่ง" นี่เป็น identification challenge ที่ต้องระวังเมื่อตีความ logit models

---

## A4 — OD Matrix: Home Municipality → School Municipality

### ทำไมต้องทำ
ต้องการเห็น **spatial flow patterns**: นักเรียนจาก municipality ไหนไปเรียนที่ municipality ไหน มากแค่ไหน? และ cross-municipality rate แตกต่างกันตาม school level ไหม? ข้อมูลนี้ช่วยอธิบายว่าทำไม HS escort rate สูง (HS = เดินไกล, ข้าม municipality มาก)

### ข้อมูลที่ใช้และการลิงก์
```
school_trips_los.csv
   ↓ origin_municipality_code + dest_municipality_code (อยู่ใน PT Survey แล้ว)
ZoneCodeTable.xlsx (SSD)
   ↓ map municipality_code_full (6 หลัก) → muni_code_3digit
   ↓ ได้ชื่อ municipality แต่ละโซน
   ↓
OD matrix: count trips per (origin_muni, dest_muni) pair
```

### ผล
- 66.2% เดินทางในเมืองเดิม (same municipality)
- Middle school cross-muni สูงสุด 38.8% (school choice สูง)
- Naha เป็น top attractor รับนักเรียน 3,180 trips
- Urasoe รับ import 39.4% จาก municipality อื่น

### วิจารณ์ (escort & safety)
Cross-municipality rate สูงมาก (33.8% โดยเฉลี่ย) หมายความว่านักเรียน Okinawa จำนวนมาก **เดินทางข้ามเมือง** ซึ่งทำให้ไม่สามารถ walk ได้ → ต้องใช้รถ → escort rate สูง ในมุม safety: ถ้าเด็กต้องข้ามหลาย zone ผ่านถนนสายหลัก พ่อแม่ก็มีเหตุผลด้าน safety ที่จะขับส่ง อย่างไรก็ตาม เราไม่พบหลักฐานว่า safety perception เป็น predictor หลักจาก B3/E2

---

## A5 — MNL School Choice Model

### ทำไมต้องทำ
ต้องการ model ว่า **ครอบครัวเลือกโรงเรียนอย่างไร** เมื่อมีทางเลือกหลายโรงเรียน (discrete choice) โดยพิจารณา distance, LOS, car ownership เป็น predictors สำหรับ school choice → เชื่อมกับ escort ในภายหลัง

### ข้อมูลที่ใช้และการลิงก์
```
school_choice_trips.csv (derived จาก school_trips_los.csv)
school_locations.csv (จาก shapefiles SSD)
CZone.shp (SSD) → calculate GIS distance ระหว่าง home zone กับทุก school alternatives
   ↓ MNL: P(school j | household i) ~ distance_ij + LOS_ij + car_i
```

### ผล
Car-owning households เลือกโรงเรียนไกลกว่า — distance elasticity significant

### วิจารณ์
School choice model บอกว่า **รถ unlocks distant school options** ซึ่ง reinforces escort ให้สูงขึ้น นี่คือ feedback loop: มีรถ → เลือกโรงเรียนไกล → ต้องขับส่ง → escort สูง ในแง่ policy: การปรับปรุง transit อย่างเดียวไม่พอ เพราะ school choice behavior เองก็ต้องเปลี่ยน

---

## D1 — Escort Rate by Walk Time Quartile

### ทำไมต้องทำ
ตั้ง hypothesis ง่ายๆ: ถ้า walking ยากขึ้น (walk_time นาน) ควรมี escort มากขึ้น ถ้า D1 เห็น monotone relationship (Q1 < Q2 < Q3 < Q4) จะเป็นหลักฐาน descriptive ว่า LOS matters ก่อนใส่ logit

### ข้อมูลที่ใช้และการลิงก์
```
school_trips_los.csv
   ↓ walk_time_min (มาจาก CurrentLOS join)
   ↓ bin เป็น Q1-Q4 quartile
   ↓ escort_rate = mean(escorting_flag==1) ต่อ quartile
   ↓ Chi-square test ว่า distribution แตกต่างหรือไม่
```

### ผล
Q1=53.6%, Q2=51.7%, Q3=53.1%, Q4=57.2% — **flat ไม่มี trend ชัดเจน**

### วิจารณ์
ผลนี้ counter-intuitive: แม้แต่ zone ที่ walk_time ใกล้มาก (Q1 median=11.5 นาที) ก็ยังมี escort rate 53.6% ใกล้เคียงกับ Q4 (119 นาที) นี่บอกว่าการตัดสินใจ escort **ไม่ขึ้นกับ accessibility** เลย หรือบอกว่า HS trips (escort rate ~97.9%) กระจายอยู่ทั้ง 4 quartile จึง mask ความสัมพันธ์ที่อาจมีใน Elementary/Middle ซึ่งเป็น limitation สำคัญของ D1

---

## D2 — Escort Rate by Bus Frequency Group

### ทำไมต้องทำ
ทดสอบว่า **transit availability** (วัดด้วยความถี่รถเมล์) มีผลต่อ escort หรือไม่ ถ้า bus บ่อย ควร escort น้อยลงเพราะนักเรียนนั่งรถเมล์เองได้

### ข้อมูลที่ใช้และการลิงก์
```
school_trips_los.csv
   ↓ bus_frequency_per_day (จาก CurrentLOS)
   ↓ แบ่งกลุ่ม: 0 / 1-5 / 6-15 / 16+ trips per day
   ↓ escort_rate ต่อกลุ่ม + Chi-square test
```

### ผล
no_bus=55.3%, frequent(16+)=54.2% — **แทบไม่ต่างกัน**

### วิจารณ์
ถ้า transit จริงๆ แก้ escort ได้ ควรเห็น escort_rate ลดลงชัดในกลุ่ม bus บ่อย แต่ผลบอกว่าไม่ใช่ ทำไม? เหตุผลที่น่าสนใจคือ: (1) HS escort rate ~97.9% ไม่ว่าจะมี bus หรือไม่ — มัน structural ไม่ใช่ LOS-driven, (2) พ่อแม่ Okinawa อาจ prefer car ด้วยเหตุผลอื่น (ฝน, ความสะดวก, กลัวเปล่า), (3) bus ใน Okinawa ไม่น่าเชื่อถือพอที่จะแทนที่รถได้จริง

---

## D3 — Car Distance Distribution: School Level × Escort

### ทำไมต้องทำ
ถ้า escort เกิดเมื่อ "ไกลเกินไป" ควรเห็น **car distance ของ escorted trips มากกว่า non-escorted** อย่างมีนัยสำคัญ D3 ทดสอบ hypothesis นี้แยกตาม school level

### ข้อมูลที่ใช้และการลิงก์
```
school_trips_los.csv
   ↓ car_distance_m (จาก CurrentLOS)
   ↓ กลุ่ม: escorted (flag=1) vs non-escorted (flag=2)
   ↓ Box plot + Kruskal-Wallis test แยกตาม school_level
```

### ผล
- Elementary: escorted mean 5.6 km vs non-escort 4.2 km (+1.4 km)
- HS: escorted 5.1 km vs non-escort 4.7 km (+0.4 km เล็กน้อย)

หมายเหตุ: Elementary escort n=29 เท่านั้น (sample เล็กมาก)

### วิจารณ์
Distance effect มีแต่ **เล็กมาก** — escorted trips ไกลกว่าแค่ 0.4–1.4 km ซึ่งต่ำกว่าที่คาด ชี้ว่าระยะทางคือ partial explanation ไม่ใช่ dominant driver ของ escort decision ที่น่าสนใจคือ HS non-escorted trips (2.1% = 124 trips) ก็มี distance ใกล้เคียงกัน แสดงว่า HS escort เกือบ "universal" โดยไม่สนระยะทาง — บ่งชี้ว่าเป็น **cultural norm** มากกว่า rational distance threshold

---

## D4 — Zone-Level: Escort Rate vs Mean LOS

### ทำไมต้องทำ
ดู LOS effect ในระดับ **zone** (aggregate) แทน individual trip เผื่อว่า noise ระดับ trip ทำให้ไม่เห็น pattern ถ้า zone ที่มี LOS แย่มี escort rate สูง จะเป็น spatial evidence ว่า accessibility matters

### ข้อมูลที่ใช้และการลิงก์
```
school_trips_los.csv
   ↓ aggregate ต่อ C-zone:
     - escort_rate = proportion escorted trips
     - mean_walk_time = mean walk_time_min
     - mean_bus_freq = mean bus_frequency_per_day
   ↓ scatter plot + bivariate Pearson/Spearman correlation
```

### ผล
139 C-zones — correlation:
- walk_time vs escort_rate: r=+0.181 (อ่อนมาก)
- bus_freq vs escort_rate: r=−0.175 (อ่อนมาก ทิศทางถูก)

### วิจารณ์
Zone-level correlation ยืนยัน individual-level null: LOS ไม่ predict escort behavior ทั้งในระดับ trip และระดับ zone ทิศทางของ correlation ถูกต้อง (walk นาน = escort มากขึ้น นิดหน่อย; bus น้อย = escort มากขึ้น นิดหน่อย) แต่ effect size เล็กมากจน **ไม่มี practical significance** ในมุม policy: การ improve LOS ของ zone หนึ่งไม่น่าจะเปลี่ยน escort rate อย่างมีนัยสำคัญ

---

## B1 — Stage 1 Binary Logit: Escort Probability

### ทำไมต้องทำ
หลังจาก D1-D4 ยืนยัน LOS null ด้วย descriptive แล้ว B1 ทำ **multivariate logit** เพื่อ control หลายตัวแปรพร้อมกัน และหา dominant predictor ของ escort decision อย่างเป็น formal model

### ข้อมูลที่ใช้และการลิงก์
```
R05_PersonTrip_EN.csv → school trips + escorting_flag
R05_HH_Survey_EN.csv → join household key → car_ownership, income, hh_size
R05_Supplementary_EN.csv → join Q6 → escort reason, child sex, escorter sex
   ↓ Binary logit: escort(0/1) ~ car_ownership + school_level + 
     sex_child + sex_escorter + hh_size + annual_income
```

หมายเหตุ: ใช้ SUPP Q6 flag → n=1,353 (จำกัดโดย SUPP coverage)

### ผล
- car_ownership dominant predictor (OR≈5×)
- McFadden R²=0.145 (ดีพอสมควรสำหรับ cross-sectional logit)
- n=1,353

### วิจารณ์
OR≈5× สำหรับ car ownership แปลว่า **ครอบครัวมีรถมีโอกาส escort 5 เท่าของครอบครัวไม่มีรถ** — นี่คือ finding หลักของ study ที่บอกว่า escort เป็นเรื่องของ resource availability ไม่ใช่ safety concern ข้อจำกัดสำคัญ: n=1,353 เล็กมากเพราะขึ้นอยู่กับ SUPP coverage และ SUPP Q6 ถาม escorting parents เท่านั้น ดังนั้น sample อาจ biased

---

## B2 — Stage 2 Binary Logit: Safety vs Resource

### ทำไมต้องทำ
B1 บอกว่า **ใครเป็นคน escort** แต่ B2 ถาม **ทำไมถึง escort** โดยเฉพาะ: ในบรรดาพ่อแม่ที่ escort แล้ว กลุ่มไหนให้เหตุผลว่าเป็นเรื่อง safety? ทฤษฎีที่ทดสอบ: "no-car households → coping resources จำกัด → appraise threat สูง"

### ข้อมูลที่ใช้และการลิงก์
```
R05_Supplementary_EN.csv
   ↓ filter: q6_escorting_flag == '1' (escorting parents เท่านั้น)
   ↓ safety_concern = 1 if any q6_child*_escort_reason == '02'
   ↓ Binary logit: safety_concern(0/1) ~ car_ownership + school_level + ...
   n = 922 (escorting parents only)
```

### ผล
**No-car households มีโอกาสให้เหตุผล safety สูงกว่า car households** — สอดคล้องกับ "threat appraisal when coping limited"

### วิจารณ์
ผลนี้น่าสนใจมาก: แม้ค car ownership จะ predict escort (B1) แต่คนที่ escort **โดยไม่มีรถ** (เช่น ขี่จักรยาน, เดินไปด้วยกัน) มักให้เหตุผลว่าเป็น safety มากกว่าคนที่ขับรถส่ง ชี้ให้เห็น **two-track escorting**: (1) car-based escort = convenience/habit, (2) non-car escort = genuine safety concern ข้อจำกัด: escort_reason = subjective self-report, อาจมี social desirability bias

---

## B3 — P1a–P1c: LOS + Safety Logit

### ทำไมต้องทำ
รวม LOS variables (walk_time, bus_freq, car_dist) กับ safety_concern ใน logit เดียว เพื่อทดสอบว่า LOS และ safety perception interact กันหรือไม่ เช่น "zone ที่ bus น้อย + พ่อแม่กังวลเรื่อง safety → escort มากขึ้นพิเศษ"

### ข้อมูลที่ใช้และการลิงก์
```
school_trips_los.csv  ← LOS variables
R05_Supplementary_EN.csv  ← safety_concern (escort reason == '02')
   ↓ join บน household composite key
   ↓ subset: trips ที่ match กับ SUPP (~7-8% ของ school trips เท่านั้น)
   ↓ logit: escort ~ walk_time + bus_freq + car_dist + car_ownership
   ↓ interaction model: escort ~ ... + safety_concern × bus_freq
```

### ผล
- LOS variables: p>0.1 ทั้งหมด → **ไม่มีนัยสำคัญ**
- safety × bus interaction: not significant
- McFadden R²=0.019 (ต่ำมาก)
- car_ownership ยังคง dominant

### วิจารณ์
แม้แต่เมื่อ control LOS + safety พร้อมกัน car ownership ก็ยังชนะ และ interaction ที่คาดว่าจะเห็น (safety concern ทำให้ LOS matter มากขึ้น) ก็ไม่เกิด นี่อาจเป็นเพราะ **sample ที่ match SUPP เล็กมาก** (7-8%) และ safety_concern ใน dataset นี้เป็น rare event (ส่วนใหญ่ escort เพราะ convenience ไม่ใช่ safety)

---

## B4 — P4a: Stratified Logit (Car / No-car HH)

### ทำไมต้องทำ
ตั้งสมมติฐานว่า mechanism ของ escort ใน car-HH vs no-car HH **ต่างกัน** ถ้า logit coefficients ต่างกันมากระหว่างสองกลุ่ม แปลว่ามี structural break → ควร model แยก

### ข้อมูลที่ใช้และการลิงก์
```
school_trips_los.csv
   ↓ split ตาม owned_has_car (car-HH vs no-car HH)
   ↓ logit แยก 2 กลุ่ม: escort ~ LOS + school_level + hh_size
   ↓ เปรียบเทียบ coefficients ระหว่าง 2 กลุ่ม
```

### ผล
- No-car escort rate: 8.5% vs car escort rate: 56.7% (ต่างกันมาก)
- LOS ไม่ significant ใน **ทั้งสองกลุ่ม** → structural break confirmed
- รูปแบบ null LOS ไม่ขึ้นกับการมีรถ

### วิจารณ์
Stratification นี้เผยให้เห็นว่า **null LOS result ไม่ใช่ artifact** ของการ pool สองกลุ่มที่ต่างกัน แม้แยกดูแต่ละกลุ่ม LOS ก็ยังไม่ significant ข้อสังเกตที่น่าสนใจ: no-car HH escort น้อยมาก (8.5%) แต่เมื่อ escort จะให้เหตุผล safety มากกว่า (จาก B2) — นี่แปลว่า **quality ของ escort ต่างกัน** ระหว่างสองกลุ่ม ไม่ใช่แค่ quantity

---

## C1 — P4b: MNL Escort Mode (0/1/2)

### ทำไมต้องทำ
จนถึง B1-B4 เราถามแค่ "escort หรือเปล่า?" C1 ถามต่อ: **ถ้า escort แล้วใช้ mode อะไร?** (no escort / non-car escort / car escort) เพราะ policy implication ต่างกัน: ถ้าเป็น car escort = ผลต่อ traffic congestion

### ข้อมูลที่ใช้และการลิงก์
```
school_trips_los.csv
   ↓ สร้าง 3-category outcome:
     0 = no escort (flag=2)
     1 = non-car escort (flag=1 AND rep_mode_class1 ≠ 6)
     2 = car escort (flag=1 AND rep_mode_class1 == 6)
   ↓ MNL: mode_choice(0/1/2) ~ walk_time + bus_freq + car_dist + car_owned + school_level
   n = 7,506 (LOS-joined)
```

### ผล
- McFadden R²=0.0038 (ต่ำมาก)
- walk_time → car escort β=+0.60** (p=0.009): zone ที่เดินนานขึ้น = car escort มากขึ้น
- bus_freq → non-car escort: n.s.

### วิจารณ์
เป็น analysis เดียวใน study ที่ **walk_time มี significant effect** (p=0.009) แต่เฉพาะในการเลือกระหว่าง no_escort กับ car_escort ไม่ใช่ว่า walk_time บอกว่า "escort ไหม" แต่บอกว่า "ถ้าจะ escort จะใช้รถไหม" ซึ่ง nuance นี้สำคัญมาก: LOS มีผลต่อ **mode** ของ escort มากกว่า **decision** ที่จะ escort

---

## C2 — MNL-I: Distance × Car Ownership Interaction

### ทำไมต้องทำ
ต่อจาก C1 โดย add interaction term: **distance × car_ownership** เพื่อทดสอบว่า "car HH escort ทุกกรณี" ในขณะที่ "no-car HH escort เฉพาะเมื่อไกล" (distance-triggered threshold)

### ข้อมูลที่ใช้และการลิงก์
```
school_trips_los.csv
   ↓ เหมือน C1 แต่เพิ่ม interaction: car_dist × car_owned
   ↓ MNL: mode_choice ~ ... + car_dist × car_owned
   ↓ LR test เปรียบ C1 vs C2
```

### ผล
- McFadden R² เพิ่มจาก 0.004 → 0.031 (Δ=0.027) — **ใหญ่ที่สุดใน study**
- Dist×Car → car escort β=−0.72*** (p<0.001): car HH ไม่ sensitive ต่อ distance เลย
- LR test χ²=433.69, p<0.001

### วิจารณ์
นี่คือ **best model** ของ study ในแง่ fit เนื้อหาที่ได้: car HH ขับส่งลูกโดยไม่สนใจระยะทาง (always-on behavior) แต่ no-car HH จะ escort ด้วย non-car mode เฉพาะเมื่อระยะทางสั้นพอที่จะเดิน/ขี่จักรยาน ข้อจำกัด: แม้ McF-R²=0.031 จะดีกว่าก่อน แต่ยังต่ำ บ่งชี้ว่ายังมีตัวแปรอื่น (เช่น หลักสูตรพิเศษ, เพื่อน, ความปลอดภัยเฉพาะจุด) ที่โมเดลไม่ได้จับ

---

## P3b — Bus Frequency Counterfactual

### ทำไมต้องทำ
ถ้า LOS null (B3, D1-D4) — แปลว่า policy ของ **เพิ่มความถี่รถเมล์** ไม่มีผลต่อ escort rate? P3b จำลองว่าถ้าเพิ่ม bus frequency หลายระดับ escort rate จะเปลี่ยนแค่ไหน

### ข้อมูลที่ใช้และการลิงก์
```
school_trips_los.csv + logit coefficients จาก B3
   ↓ สร้าง 4 scenarios: +5 trips/day, +10, ×1.5, ×2
   ↓ predict escort_prob ด้วย coefficient × new_bus_freq
   ↓ คำนวณ Δescort_rate ต่อ scenario
```

### ผล
- S1 (+5 trips/day): escort Δ=+0.06 pp
- S3 (×2 bus): Δ=+0.85 pp
- **ทุก scenario เปลี่ยนน้อยมาก**

### วิจารณ์
Counterfactual analysis นี้ confirm null result ด้วยตัวเลขที่ชัดเจน: แม้เพิ่ม bus ถึง 2 เท่า escort rate เปลี่ยนแค่ <1 pp ใน Okinawa context นี้หมายความว่า **transit policy ไม่ใช่ lever ที่ถูกต้อง** สำหรับ escort behavior ซึ่งสำคัญมากในแง่ policy recommendation อย่างไรก็ตาม ต้องระวัง: logit coefficients ที่ใช้ทำ counterfactual มาจาก model ที่ McF-R²=0.019 — ถ้า model fit ต่ำ counterfactual reliability ก็ต่ำด้วย

---

## P3c — Future LOS Counterfactual

### ทำไมต้องทำ
เหมือน P3b แต่ใช้ **LOS จริงในอนาคต** (ที่รัฐบาลวางแผนไว้) แทนที่จะ simulate แบบ P3b เพื่อทดสอบว่า network improvement ที่วางแผนจริงจะเปลี่ยน escort behavior ได้แค่ไหน

### ข้อมูลที่ใช้และการลิงก์
```
school_trips_los.csv + B3 logit
02_FutureLOS_No...csv (SSD) → network without improvement
03_FutureLOS_With...csv (SSD) → network with planned improvement
   ↓ replace current LOS ด้วย future LOS ทั้งสองเวอร์ชัน
   ↓ predict escort rate → compare ΔWith vs ΔWithout
```

### ผล
Future LOS improvement → escort Δ≈0 — **null ทั้ง time horizon**

### วิจารณ์
แม้แต่ planned real-world network improvement ก็ไม่เปลี่ยน escort behavior นี่คือ finding ที่แข็งแกร่งมาก: **ไม่ใช่แค่ simulation null แต่ future-real-data null ด้วย** สำหรับ thesis: นี่เป็นหลักฐานสำคัญที่ justify ว่าโมเดลต้องมอง mechanism อื่น (car ownership, school choice) ไม่ใช่แค่ transit improvement

---

## E1 — Spatial Maps (MAP1–6)

### ทำไมต้องทำ
วิธีการที่ดีที่สุดในการสื่อสาร spatial pattern คือ **แผนที่** E1 สร้าง choropleth maps ระดับ C-zone สำหรับตัวแปรสำคัญ เพื่อดูว่า high-escort zones อยู่ที่ไหน และ bus frequency pattern ตรงกับ escort pattern หรือเปล่า

### ข้อมูลที่ใช้และการลิงก์
```
school_trips_los.csv → aggregate escort_rate, car_escort_rate ต่อ C-zone
CZone.shp (SSD) → geometry polygon ของแต่ละ C-zone
   ↓ merge บน czone_code → choropleth map
   ↓ MAP1: escort rate, MAP2: car escort rate
   ↓ MAP3: bus frequency, MAP4: counterfactual Δ escort
   ↓ MAP5: counterfactual Δ car escort, MAP6: panel 4 maps
```

### ผล
High escort rate concentrated ใน suburban zones; bus frequency **ไม่ align** กับ escort patterns

### วิจารณ์
MAP3 vs MAP1 เป็น visual proof ของ null finding: ถ้า bus frequency ช่วยลด escort rate ควรเห็น inverse spatial pattern แต่ไม่เป็นเช่นนั้น ในมุม safety: suburban zones ที่ escort rate สูงอาจมีถนนสายหลักที่อันตราย แต่เราไม่มีข้อมูล road safety ระดับ zone มายืนยัน

---

## TC — Trip Chaining Among Escorts

### ทำไมต้องทำ
ถ้าพ่อแม่ escort ลูกไปโรงเรียนแล้ว**ไปทำงานต่อเลย** (trip chaining) แสดงว่า escort ถูก embed เข้าใน activity pattern ของพ่อแม่ ไม่ใช่ standalone decision นี่มีนัยต่อ policy: การเปลี่ยน escort behavior ต้องเปลี่ยน commute patterns ด้วย

### ข้อมูลที่ใช้และการลิงก์
```
school_trips_los.csv → escort trips (escorting_flag=1)
R05_OkinawaPT_Person_Master.csv (SSD) → all trips ของพ่อแม่ในวันเดียวกัน
   ↓ identify if escorter's next trip = work/errand
   ↓ binary: is_chained_with_work (0/1)
   ↓ logit: is_chained ~ car_owned + employment + school_level + hh_size
```

### ผล
35.8% ของ escort trips ถูก chain กับ work trip — `has_work_trip` เป็น significant predictor

### วิจารณ์
Trip chaining finding นี้สำคัญมากในมุม mechanism: escort ไม่ได้เกิดเพราะพ่อแม่กังวลเรื่อง safety แต่เพราะ **"ไปทางเดียวกัน"** กับที่ทำงาน ซึ่ง reinforces ว่า escort เป็น rational activity-chain optimization ไม่ใช่ emotion-driven safety behavior อย่างไรก็ตาม: 64.2% ที่ไม่ chain กับงาน อาจมีเหตุผลอื่น — รวมถึง safety concern ที่แท้จริง

---

## SC — P2a/P2b: School Choice × Escort

### ทำไมต้องทำ
ถ้านักเรียนอยู่ **นอก catchment** (in-district = โรงเรียนในเขต / school choice = เลือกเอง) ระยะทางไกลกว่า → น่าจะ escort มากกว่า SC ทดสอบว่า school choice decision เชื่อมกับ escort หรือไม่

### ข้อมูลที่ใช้และการลิงก์
```
school_trips_los.csv
CZone.shp + school shapefiles (SSD)
   ↓ GIS: assign catchment school ให้แต่ละ home zone
   ↓ flag: in-catchment vs school-choice
   ↓ compare escort_rate: in vs out of catchment
```

### ผล
- 70.9% out-of-catchment (school choice สูงมาก)
- escort rate: in-catchment 2.5% vs out-of-catchment 2.0% → **ไม่มีนัยสำคัญ**

### วิจารณ์
ผล counter-intuitive: คาดว่า out-of-catchment (ไกลกว่า) จะ escort มากกว่า แต่ไม่เป็น ทำไม? อาจเป็นเพราะ escort rates ต่ำมาก (~2%) สำหรับ Elementary/Middle ทั้งสองกลุ่ม จึงไม่มี power พอ หรือ HS trips ที่ escort ~97.9% dominate ทั้ง in/out จน mask ความแตกต่าง ข้อจำกัด: GIS catchment assignment ขึ้นกับ shapefile quality ที่มาจาก SSD

---

## E2 — Objective Safety: Crime Density vs Escort Rate

### ทำไมต้องทำ
B2 เจอว่า no-car HH cite safety concern มากกว่า แต่นั่นคือ **subjective safety** (self-reported) E2 ถามว่า **objective safety** (crime ที่เกิดขึ้นจริง) correlate กับ escort rate หรือไม่? ถ้าใช่: พ่อแม่ escort เพราะ assess crime risk จริง ถ้าไม่ใช่: safety perception เป็น psychological/cultural ไม่ใช่ evidence-based

### ข้อมูลที่ใช้และการลิงก์
```
Okinawa crime record 2023 (7 CSV ไฟล์, Shift-JIS encoded)
   ↓ iconv -f SHIFT-JIS -t UTF-8  (macOS deadlock workaround)
   ↓ รวม 2,588 records: ひったくり, 車上ねらい, 部品ねらい, 自動販売機ねらい,
     自動車盗, オートバイ盗, 自転車盗
   ↓ municipality code: 472018 // 10 % 1000 = 201 (Naha) → match PT survey code
   ↓ crime_count per municipality

school_trips_los.csv → escort_rate per municipality
   ↓ crime_per100trips = crime_count / n_trips * 100  (normalize by exposure)

R05_Supplementary_EN.csv → safety_concern rate per municipality
   ↓ q6_child*_escort_reason == '02' → safety_concern flag
```

### ผล
- Pearson r=0.236 (p=0.345): ไม่มีนัยสำคัญ
- Spearman ρ=−0.373 (p=0.128): ไม่มีนัยสำคัญ
- Crime vs perceived safety rate: r=−0.410 (p=0.102): ไม่มีนัยสำคัญ
- outlier: Nanjo (muni=209) crime/100trips=688.9 แต่ n=18 trips เท่านั้น

### วิจารณ์
Null finding นี้สำคัญมาก: พ่อแม่ Okinawa **ไม่ได้ escort มากขึ้นในพื้นที่ที่มี crime สูงจริง** ชี้ให้เห็นว่า safety perception เป็น subjective/cultural construct ไม่ใช่ rational response ต่อ objective risk ข้อจำกัดสำคัญ: (1) crime data ระดับ municipality coarse เกิน — ควรได้ระดับ C-zone แต่ไม่มี town-block→zone crosswalk; (2) crime types เป็น property crime ไม่ใช่ violent crime ต่อเด็ก; (3) Nanjo outlier ที่ sample เล็กมาก ทำให้ correlation ไม่เสถียร; (4) n=18 municipalities เท่านั้น = low power

---

## E5 — Population-Weighted Escort Rate

### ทำไมต้องทำ
ตัวเลข escort rate 55.5% ที่คำนวณจาก sample อาจ **biased** ถ้า zone บางโซน over- หรือ under-represented ในข้อมูล E5 ทดสอบว่าถ้า weight ด้วย school-age population จริงต่อ zone จะได้ตัวเลขต่างออกไปแค่ไหน — เพื่อประเมิน representativeness ของ sample

### ข้อมูลที่ใช้และการลิงก์
```
R05_HH_Survey_EN.csv
   ↓ filter: employment_student_status ∈ {8,9,10} (Elementary/Middle/HS)
   ↓ addr_zone_code (D-zone 5 หลัก) → first 3 digits = C-zone
   ↓ count per C-zone = school-age HH member proxy
   (proxy ไม่ใช่ population จริง — เป็น enrolled students ใน HH Survey sample)

school_trips_los.csv
   ↓ escort_rate per C-zone (origin_czone)
   ↓ merge บน origin_czone

→ population-weighted mean = Σ(rate_i × pop_i) / Σ(pop_i)
```

### ผล
| ประเภท | Escort Rate |
|---|---|
| Raw per-trip | 55.5% |
| Zone mean (unweighted) | 54.6% |
| Zone mean (trip-weighted) | 55.4% |
| Zone mean (HH pop-weighted) | 56.9% |

- ต่างกัน +1.44 pp — sample slightly under-represents high-escort zones
- Elementary: 1.9%, Middle: 3.8%, HS: 97.8% (uniform ทั้งนั้น ต่างกันน้อย)

### วิจารณ์
1.44 pp ถือว่าเล็กมาก แปลว่า sample ค่อนข้าง representative ไม่มี major zone bias ข้อที่น่าสนใจ: HS escort ~97.9% เกือบ universal ไม่มีความแตกต่างระหว่าง zone มากนัก ทำให้ pop-weighting ไม่ค่อยเปลี่ยนผลเท่าไหร่ ข้อจำกัด: HH Survey school-age count เป็น proxy ไม่ใช่ actual school-age population ต่อ zone — ใช้ residential address ของ household members ไม่ใช่ enrollment data จริง

---

## สรุปสุดท้าย: ตาราง All Analyses

| ID | ชื่อ | RQ | Method | ผลหลัก | Escort/Safety Critique |
|---|---|---|---|---|---|
| **F0** | Data Prep | prep | Zone mapping + LOS join | 10,676 trips; 7,550 LOS-joined | HS 97.9% escort = car culture ไม่ใช่ safety |
| **A1** | Distance by Level | RQ1 | ANOVA | HS mean ~3.8 km; ไกลขึ้นตาม level | ระยะไกล → ต้องรถ → escort สูง |
| **A2** | Zone-Crossing | RQ1 | Chi-square | 70.9% out-of-catchment; HS ข้ามสุด | Zone-cross = structural cause ของ car dependence |
| **A3** | OLS Travel Time ~ Car | RQ1 | OLS | Car ownership → travel time ยาว | Endogeneity: รถ → เลือกโรงเรียนไกล → escort |
| **A4** | OD Matrix | RQ1 | OD matrix + heatmap | 66.2% same-muni; Middle cross 38.8% | Naha top attractor; cross-muni flow explain car escort |
| **A5** | MNL School Choice | RQ1 | MNL | Car HH เลือกโรงเรียนไกล | School choice reinforces car escort feedback loop |
| **D1** | Escort × Walk Quartile | RQ1 desc | Chi-square | Q1=53.6% Q4=57.2% — flat | LOS ไม่ predict escort descriptively |
| **D2** | Escort × Bus Freq | RQ1 desc | Chi-square | no_bus≈frequent=55% — flat | Transit null: bus frequency ไม่ช่วยลด escort |
| **D3** | Car Distance × Escort | RQ1 desc | Kruskal-Wallis | Escort trips ไกลกว่านิดหน่อย; effect เล็ก | Distance partial — ไม่ใช่ dominant factor |
| **D4** | Zone LOS vs Escort | RQ1 desc | Scatter + corr | r(walk,escort)=+0.181; r(bus,escort)=−0.175 | Zone-level ยืนยัน LOS null; weak spatial signal |
| **B1** | Logit: Escort Prob | RQ2a | Binary logit | car_own OR≈5×; McF-R²=0.145; n=1,353 | Car = dominant predictor; not safety |
| **B2** | Logit: Safety vs Resource | RQ2a | Binary logit | No-car HH cite safety more | Two-track: car=convenience; no-car=safety concern |
| **B3** | LOS + Safety Logit | RQ2a | Binary logit | LOS n.s.; safety×bus n.s.; McF-R²=0.019 | Even with safety control, LOS still null |
| **B4** | Stratified Logit | RQ2a | Binary logit (split) | No-car 8.5% vs car 56.7%; LOS null ทั้งสอง | Structural break confirmed; LOS absent in both groups |
| **C1** | MNL Escort Mode | RQ2b | MNL (3 cat) | walk_time→car escort β=+0.60** | LOS มีผลต่อ mode ของ escort ไม่ใช่ decision |
| **C2** | MNL + Dist×Car Interact | RQ2b | MNL + interaction | Δ McF-R²=+0.027; Dist×Car β=−0.72*** | Car HH always-on; no-car HH distance-triggered |
| **P3b** | Bus Counterfactual | RQ2 policy | Scenario simulation | ×2 bus → Δ=+0.85 pp เท่านั้น | Transit policy ไม่ใช่ lever ที่ถูก |
| **P3c** | Future LOS Counterfactual | RQ2 policy | Scenario simulation | Future network → Δ≈0 | Real-world improvement ก็ยัง null |
| **E1** | Spatial Maps (MAP1–6) | RQ2 spatial | GIS choropleth | Suburban high-escort; bus ไม่ align | Visual proof ของ LOS null spatially |
| **TC** | Trip Chaining | RQ2 mechanism | Logit + activity chain | 35.8% chain กับ work trip | Escort = rational commute chain ไม่ใช่ standalone safety |
| **SC** | School Choice × Escort | RQ1+RQ2 | GIS + logit | 70.9% out-catchment; in vs out n.s. | School choice ไม่ predict escort directly |
| **E2** | Crime Density vs Escort | RQ2 safety | Scatter + corr | r=0.236 (n.s.); crime ≠ predict escort | Objective safety ไม่ correlate กับ escort → subjective/cultural |
| **E5** | Pop-Weighted Escort Rate | RQ1 repr. | Weighted mean | 55.5% → 56.9% (+1.44 pp); representative | Sample ค่อนข้างดี; HS escort uniform ทุก zone |

---

## ข้อสรุปใหญ่ (Narrative)

Okinawa school escorting เป็น **car-dependent structural behavior** ไม่ใช่ safety-driven behavior:

1. **HS escort ~97.9%** เพราะนักเรียนถูกขับส่งด้วยรถ — ระยะทางไกล, cross-municipality, car-HH มีรถอยู่แล้ว
2. **LOS ไม่มีผล** ทั้งใน descriptive (D1-D4), logit (B3), counterfactual (P3b, P3c) และ spatial (E1)
3. **Car ownership** คือ dominant predictor (B1: OR≈5×) และ interaction term ที่ทรงพลังที่สุด (C2: Δ McF-R²=+0.027)
4. **Safety** มีบทบาทใน **no-car households** เท่านั้น (B2) และไม่ correlate กับ objective crime (E2) → safety perception เป็น subjective construction
5. **Trip chaining** (TC) ยืนยันว่า escort ถูก embed ใน commute schedule — เปลี่ยนได้ต้องเปลี่ยน work travel behavior ด้วย
6. **Sample representative** (E5 diff=+1.44 pp) — ตัวเลขจากการศึกษานี้สะท้อน population ได้ดีพอสมควร

Policy implication: การแก้ปัญหา school escort (traffic congestion, emission) ใน Okinawa ต้องโฟกัส **school district reform** (ลด out-of-catchment) และ **land use** (โรงเรียนใกล้บ้านมากขึ้น) มากกว่า transit improvement

---

## เปรียบเทียบ: ก่อนได้ข้อมูล SSD vs หลังได้ข้อมูล SSD

> อ้างอิงจาก **Okinawa basic analysis_r4.pdf** (July 14, 2026) — เวอร์ชันก่อนที่จะได้ SSD มา

### สถานะก่อนได้ SSD (r4 = July 14, 2026)

ตอนนั้นมี 3 Analysis หลัก + Extensions บางส่วน:

| ID เดิม (r4) | ชื่อ | สถานะ | ตรงกับ ID ปัจจุบัน |
|---|---|---|---|
| Analysis 1 | ANOVA + Chi-square + OLS travel time | ✅ Done | A1, A2, A3 |
| Analysis 2A | Binary logit escort decision | ✅ Done | B1 |
| Analysis 2B | Binary logit safety motivation | ✅ Done | B2 |
| Analysis 2 Trip Chaining | Trip chaining analysis | ✅ Done | TC |
| Analysis 3 | Mode choice chi-square + Mann-Whitney | ✅ Done (preliminary) | C1, C2 (ขยายภายหลัง) |
| E1_old | Employment status as IV ใน escort logit | ✅ Done | merged เข้า B1 extended |
| E2_old | Number of school-age children per HH | ✅ Done | merged เข้า B1 extended |
| E3_old | Trip chaining | ✅ Done | TC |
| E4_old | Triangular: home-school-workplace spatial | ⏳ Waiting | **ยังไม่ครบ** (A4 OD matrix ครอบคลุมบางส่วน) |
| P1 | School location GIS geocodes | ⏳ Waiting | → E1, SC, A5 (หลังได้ SSD) |
| P2 | Objective crime / accident records | ⏳ Waiting | → E2 (หลังได้ SSD) |
| P3 | Level of Service (LOS) transit + pedestrian | ⏳ Waiting | → F0, D1-D4, B3, C1, C2, P3b, P3c (หลังได้ SSD) |

**Note สำคัญ:** r4 ระบุว่า crime data มีอยู่แล้ว แต่ "ไม่สามารถ merge ได้" เพราะ crime data เป็น municipality level ขณะที่ survey ใช้ PT zone code (B/C-zone) — geographic incompatibility ตอนนั้นยังไม่มีวิธีแก้

---

### สิ่งที่เพิ่มขึ้นหลังจากได้ SSD มา

SSD มีข้อมูล 3 ชุดหลัก และแต่ละชุดปลดล็อก analyses ที่ต่างกัน:

#### 📁 จาก LOS Data (CurrentLOS.csv) — แก้ P3
| Analysis ที่เพิ่มได้ | สาเหตุ |
|---|---|
| F0 (Data Prep pipeline) | ต้องมี LOS + ZoneCodeTable เพื่อสร้าง `school_trips_los.csv` |
| D1, D2, D3, D4 | Descriptive: LOS × escort rate — ต้องมี LOS data |
| B3 | Full logit with LOS + safety — ต้องมี LOS |
| C1, C2 | MNL escort mode ที่ครบกว่า (3 categories) — ต้องมี LOS ใน trips |
| P3b, P3c | Counterfactual simulation — ต้องรู้ baseline LOS ก่อน |

#### 📁 จาก School GIS + CZone.shp — แก้ P1
| Analysis ที่เพิ่มได้ | สาเหตุ |
|---|---|
| E1 (Spatial Maps MAP1–6) | Choropleth map ต้องมี GIS boundaries |
| SC (School Choice × Escort) | ต้องรู้ school location เพื่อคำนวณ catchment |
| A5 (MNL School Choice) | ต้องมี school geocodes เป็น choice alternatives |

#### 📁 จาก ZoneCodeTable.xlsx — ใช้ร่วมกับ LOS
| Analysis ที่เพิ่มได้ | สาเหตุ |
|---|---|
| A4 (OD Matrix) | ต้องมี municipality name mapping จาก ZoneCodeTable |

#### 📁 จาก Crime Records CSV (7 ไฟล์ Shift-JIS) — แก้ P2
| Analysis ที่เพิ่มได้ | สาเหตุ |
|---|---|
| E2 (Crime Density vs Escort) | Crime data พร้อม; แก้ geographic incompatibility โดยใช้ municipality-level join แทน zone-level |

> **การแก้ปัญหา geographic incompatibility (จาก r4):**  
> r4 บอกว่า crime data ไม่สามารถ merge ได้เพราะ geographic mismatch — ตอนนั้นพยายาม join กับ C-zone/B-zone ซึ่งไม่มีใน crime data  
> หลังได้ SSD จึงค้นพบว่า `crime_municipality_code // 10 % 1000` → ตรงกับ `origin_municipality_code` ใน PT survey  
> ทำให้ join ได้ที่ **municipality level** (แทนที่จะเป็น zone level) — ปัญหาหายไป แต่ resolution ต่ำกว่าที่ตั้งใจ

#### 📁 ใหม่ทั้งหมด — ไม่ได้อยู่ใน r4 WBS เลย
| Analysis | เหตุผลที่เพิ่ม |
|---|---|
| E5 (Pop-Weighted Escort Rate) | ตรวจสอบ sample representativeness โดยใช้ HH Survey เป็น population proxy — ไม่ได้วางแผนไว้ใน r4 |

---

### ตาราง Before vs After สรุป

| หมวด | ก่อน SSD (r4) | หลัง SSD (ปัจจุบัน) |
|---|---|---|
| จำนวน analyses | ~8 (3 หลัก + 5 extensions) | 23 |
| ฐาน dataset | SUPP survey (n=922–1,353 parents) | PT trip-level (n=10,676 school trips) |
| LOS data | ❌ ไม่มี | ✅ walk_time, bus_freq, car_distance per zone |
| School GIS | ❌ ไม่มี | ✅ geocoded schools + C-zone boundaries |
| Crime data | ⚠️ มีแต่ merge ไม่ได้ | ✅ municipality-level join ทำได้ (null result) |
| Spatial analysis | ❌ ไม่มี | ✅ 6 choropleth maps (E1) |
| Counterfactual | ❌ ไม่มี | ✅ bus ×2 และ future LOS scenarios (P3b, P3c) |
| School choice model | ⏳ Planned | ✅ MNL (A5) + catchment analysis (SC) |
| Population weighting | ❌ ไม่มี | ✅ HH survey proxy (E5) |
| **ข้อสรุปหลัก** | car ownership dominant; safety secondary | ยืนยันเหมือนเดิม + LOS null robust ทุก specification |

### สิ่งที่ยังค้างอยู่จาก r4

- **E4_old (Triangular: home-school-workplace spatial)** — A4 OD Matrix ครอบคลุม spatial flow บางส่วน แต่ full triangular network analysis (3-node: home → school → workplace) ยังไม่ได้ทำ ต้องมี employer geocode ซึ่งยังไม่มีใน SSD
