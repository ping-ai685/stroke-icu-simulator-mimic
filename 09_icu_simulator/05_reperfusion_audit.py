"""
ICU 路径模拟器 · 审计：MIMIC-IV 中能否识别溶栓和取栓，以及时间（只输出汇总数字）

为 readiness v0.3 的"再灌注后 24 小时监护期"做准备。只看缺血性卒中（AIS）病人，另报全队列。

来源：
  溶栓  a. 用药给药记录 emar：alteplase / tenecteplase，事件为"已给药"；剂量来自 emar_detail，
           ≤ 4 mg 视为导管通管（Cathflo），不算溶栓
        b. 处方 prescriptions：同上药名，记录剂量
        c. ICD 操作编码：ICD-10-PCS 3E03317 / 3E05317 / 3E06317（溶栓药注入），ICD-9 99.10
        d. ICD 诊断编码：Z92.82 / V45.88（入院前 24 小时内在其他机构用过 tPA，"drip and ship"）
  取栓  e. ICD-10-PCS 03CG3ZZ / 03CK3ZZ / 03CL3ZZ / 03CM3ZZ / 03CN3ZZ / 03CP3ZZ / 03CQ3ZZ
           （经皮颅内及颈部动脉取栓），ICD-9 39.74
        f. 医嘱表：放射 · 神经介入（Interventional Neuro）的医嘱时间，作为介入时间的近似
时间：相对于入 ICU 的小时数（溶栓用给药时间；取栓的 ICD 只有日期，用医嘱时间近似）

输出：outputs/s6_reperfusion_audit.md
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
import project_paths  # noqa: E402

B = project_paths.MIMIC_IV
OUT = Path(__file__).resolve().parent / "outputs"
CH = 5_000_000
DRUG = "alteplase|tenecteplase|activase|tnkase|\\btpa\\b|\\brt-pa\\b"

c = pd.read_csv(ROOT / "04_outputs/tables/patient_level_cohort.csv", parse_dates=["intime", "admittime"])
hadm = set(c.hadm_id)
ais = set(c.loc[c.stroke_subtype == "AIS", "hadm_id"])
it = c.set_index("hadm_id")["intime"]
src = {}

# ---- a. emar --------------------------------------------------------------
print("emar ...", flush=True)
em = []
for ch in pd.read_csv(f"{B}/hosp/emar.csv.gz", usecols=["hadm_id", "emar_id", "charttime", "medication", "event_txt"],
                      chunksize=CH):
    ch = ch[ch.hadm_id.isin(hadm) & ch.medication.str.contains(DRUG, case=False, na=False, regex=True)]
    em.append(ch)
em = pd.concat(em)
med_names = em.medication.value_counts()
em = em[em.event_txt.isin(["Administered", "Started", "Bolus Given", "Partial Administered"])]
print("emar_detail ...", flush=True)
ids = set(em.emar_id)
dt = []
for ch in pd.read_csv(f"{B}/hosp/emar_detail.csv.gz", usecols=["emar_id", "dose_given", "dose_given_unit"],
                      chunksize=CH, dtype=str):
    dt.append(ch[ch.emar_id.isin(ids)])
dt = pd.concat(dt)
dt["dose"] = pd.to_numeric(dt.dose_given, errors="coerce")
dose = dt.groupby("emar_id").dose.max()
em["dose"] = em.emar_id.map(dose)
em["charttime"] = pd.to_datetime(em.charttime)
em_lysis = em[(em.dose > 4) | em.dose.isna()]
em_cath = em[em.dose <= 4]
src["a 给药记录(剂量>4mg或缺)"] = set(em_lysis.hadm_id)
first_em = em_lysis.groupby("hadm_id").charttime.min()
lag_em = (first_em - first_em.index.map(it)).dt.total_seconds() / 3600

# ---- b. prescriptions ------------------------------------------------------
print("prescriptions ...", flush=True)
pr = []
for ch in pd.read_csv(f"{B}/hosp/prescriptions.csv.gz", usecols=["hadm_id", "drug", "dose_val_rx", "starttime"],
                      chunksize=CH, dtype={"dose_val_rx": str}):
    pr.append(ch[ch.hadm_id.isin(hadm) & ch.drug.str.contains(DRUG, case=False, na=False, regex=True)])
pr = pd.concat(pr)
pr["dose"] = pd.to_numeric(pr.dose_val_rx, errors="coerce")
src["b 处方(剂量>4mg)"] = set(pr.loc[pr.dose > 4, "hadm_id"])

# ---- c/d/e. ICD -------------------------------------------------------------
px = pd.read_csv(f"{B}/hosp/procedures_icd.csv.gz", dtype={"icd_code": str}, parse_dates=["chartdate"])
px = px[px.hadm_id.isin(hadm)]
LYS_PCS = ["3E03317", "3E05317", "3E06317"]
EVT_PCS = ["03CG3ZZ", "03CK3ZZ", "03CL3ZZ", "03CM3ZZ", "03CN3ZZ", "03CP3ZZ", "03CQ3ZZ"]
src["c ICD操作：溶栓"] = set(px.loc[((px.icd_version == 10) & px.icd_code.isin(LYS_PCS)) |
                               ((px.icd_version == 9) & (px.icd_code == "9910")), "hadm_id"])
evt = px[((px.icd_version == 10) & px.icd_code.isin(EVT_PCS)) | ((px.icd_version == 9) & (px.icd_code == "3974"))]
src["e ICD操作：取栓"] = set(evt.hadm_id)
evt_day = (evt.groupby("hadm_id").chartdate.min() - evt.groupby("hadm_id").chartdate.min().index.map(it).normalize()
           ).dt.days
dx = pd.read_csv(f"{B}/hosp/diagnoses_icd.csv.gz", dtype={"icd_code": str})
src["d 诊断：外院tPA后转入"] = set(dx.loc[dx.hadm_id.isin(hadm) & dx.icd_code.isin(["Z9282", "V4588"]), "hadm_id"])

# ---- f. 神经介入医嘱 ---------------------------------------------------------
print("poe ...", flush=True)
po = []
for ch in pd.read_csv(f"{B}/hosp/poe.csv.gz", usecols=["hadm_id", "ordertime", "order_type", "order_subtype"],
                      chunksize=CH):
    po.append(ch[ch.hadm_id.isin(hadm) & (ch.order_subtype == "Interventional Neuro")])
po = pd.concat(po)
po["ordertime"] = pd.to_datetime(po.ordertime)
first_po = po.groupby("hadm_id").ordertime.min()
lag_po = (first_po - first_po.index.map(it)).dt.total_seconds() / 3600
src["f 神经介入医嘱"] = set(po.hadm_id)

# ---------------------------------------------------------------------------
rows = []
for k, s in src.items():
    rows.append({"来源": k, "全队列住院数": len(s), "其中AIS": len(s & ais),
                 "占AIS%": round(len(s & ais) / len(ais) * 100, 1)})
t1 = pd.DataFrame(rows)

lysis_any = src["a 给药记录(剂量>4mg或缺)"] | src["b 处方(剂量>4mg)"] | src["c ICD操作：溶栓"]
lysis_here = lysis_any
dripship = src["d 诊断：外院tPA后转入"]
evt_any = src["e ICD操作：取栓"]
t2 = pd.DataFrame([
    {"类别": "本院溶栓（a/b/c 任一）", "AIS住院数": len(lysis_here & ais), "占AIS%": round(len(lysis_here & ais) / len(ais) * 100, 1)},
    {"类别": "外院溶栓后转入（d）", "AIS住院数": len(dripship & ais), "占AIS%": round(len(dripship & ais) / len(ais) * 100, 1)},
    {"类别": "任一溶栓", "AIS住院数": len((lysis_here | dripship) & ais), "占AIS%": round(len((lysis_here | dripship) & ais) / len(ais) * 100, 1)},
    {"类别": "取栓（e）", "AIS住院数": len(evt_any & ais), "占AIS%": round(len(evt_any & ais) / len(ais) * 100, 1)},
    {"类别": "溶栓或取栓", "AIS住院数": len((lysis_here | dripship | evt_any) & ais),
     "占AIS%": round(len((lysis_here | dripship | evt_any) & ais) / len(ais) * 100, 1)},
])
ov = pd.DataFrame([{"两两重叠": f"{a} ∩ {b}", "住院数": len(src[a] & src[b])}
                   for i, a in enumerate(list(src)[:4]) for b in list(src)[i + 1:4]])


def q(s):
    s = s.dropna()
    return (f"n={len(s)}; 中位数 {s.median():.1f} h; 四分位 {s.quantile(.25):.1f} 至 {s.quantile(.75):.1f} h; "
            f"入ICU前 {(s < 0).mean():.0%}，入ICU后0–24h {s.between(0, 24).mean():.0%}，24h以后 {(s > 24).mean():.0%}")


lines = ["# 溶栓与取栓的可识别性审计", "",
         f"全队列 {len(hadm):,} 次住院，其中缺血性卒中（AIS）{len(ais):,}。", "",
         "## 1. 各来源", "", t1.to_markdown(index=False) if hasattr(t1, "to_markdown") and False else "", ""]


def md(df):
    head = list(map(str, df.columns))
    o = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    o += ["| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |" for r in df.values]
    return "\n".join(o)


mn = med_names[med_names >= 5]
lines = ["# 溶栓与取栓的可识别性审计", "",
         f"全队列 {len(hadm):,} 次住院，其中缺血性卒中（AIS）{len(ais):,}。", "",
         "## 1. 各来源识别到的住院数", "", md(t1), "",
         "## 2. 合并后（AIS）", "", md(t2), "",
         "## 3. 溶栓来源之间的重叠（全队列）", "", md(ov), "",
         "## 4. 时间（相对入 ICU）", "",
         f"- 溶栓给药时间（emar 首次，剂量 > 4 mg 或缺）：{q(lag_em)}",
         f"- 神经介入医嘱时间（首次）：{q(lag_po)}",
         f"- 取栓 ICD 日期 − 入 ICU 日期（天）：" +
         ", ".join(f"{int(k)}天 {v}" for k, v in evt_day.value_counts().sort_index().head(6).items()), "",
         f"- 给药记录中剂量 ≤ 4 mg（通管用）的住院：{em_cath.hadm_id.nunique()}",
         "", "## 5. 给药记录中的药名（≥ 5 条）", "", md(mn.rename("记录数").reset_index()), ""]
(OUT / "s6_reperfusion_audit.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
