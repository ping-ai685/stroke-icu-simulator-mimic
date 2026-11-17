"""
ICU 路径模拟器 · readiness 定义的数据审计（只输出汇总数字）

问题：操作性定义中第二层（高级监测或干预需求）及若干例外规则，在 MIMIC-IV 中能否识别？
对每类记录，统计在 ICU 停留期间（入 ICU 至 min(出 ICU, 死亡)）至少有一条记录的病人数和比例，
全队列和"入 ICU 单元为神经中间监护 / 神经过渡病房"两组分别统计。

另外：
  - 医嘱表（poe / poe_detail）：列出本队列住院期间的医嘱类型和子类型，寻找神经检查医嘱
  - 下拉选项类条目（氧疗方式、呼吸机模式、卒中护理计划）：列出出现 ≥ 20 次的取值
    （少于 20 次的取值不列出，避免出现个别自由文本）

输出：09_icu_simulator/outputs/s4_*.csv 与 s4_summary.md
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
import project_paths  # noqa: E402

B = project_paths.MIMIC_IV
HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
CH = 5_000_000
MIN_VALUE_COUNT = 20

c = pd.read_csv(ROOT / "04_outputs/tables/patient_level_cohort.csv",
                parse_dates=["intime", "outtime", "deathtime"])
c["end"] = c["outtime"]
d = c.deathtime.notna() & (c.deathtime < c.outtime)
c.loc[d, "end"] = c.loc[d, "deathtime"]
c = c[c.end > c.intime].copy()
c["inter"] = c.first_careunit.isin(["Neuro Intermediate", "Neuro Stepdown"])
N, NI = len(c), int(c.inter.sum())
ids = set(c.stay_id)
win = c.set_index("stay_id")[["intime", "end"]]

GROUPS = {
    # chartevents
    "颅内压数值": ("chartevents", [220765, 227989]),
    "神经引流(EVD)护理记录": ("chartevents", [223926, 223930, 223931, 223932, 224764, 224765,
                                     227981, 227982, 227983, 227984]),
    "氧疗方式": ("chartevents", [226732]),
    "呼吸机模式": ("chartevents", [223849, 229314]),
    "BiPAP/CPAP 参数": ("chartevents", [227577, 227578, 227579, 227580, 227581, 227582, 227583]),
    "RASS 镇静评分": ("chartevents", [228096]),
    "约束": ("chartevents", [227945, 227946, 227962, 227963, 224063, 227669, 227670, 227671]),
    "癫痫发作记录": ("chartevents", [223923, 223925]),
    "瞳孔": ("chartevents", [223907, 224733, 227121, 227288]),
    "意识水平": ("chartevents", [226104]),
    "卒中护理计划": ("chartevents", [229641, 229644, 229646, 229647, 229648, 229649, 229650,
                               229652, 229656]),
    "吞咽评估(护理记录)": ("chartevents", [229379, 228345]),
    # procedureevents
    "无创通气(过程)": ("procedureevents", [225794]),
    "有创通气(过程)": ("procedureevents", [225792]),
    "EVD / 脑室引流(过程)": ("procedureevents", [229519, 229520, 225456, 226475]),
    "颅内压导管(过程)": ("procedureevents", [229518, 229521, 226124, 226474]),
    "拔管": ("procedureevents", [227194]),
    "非计划拔管": ("procedureevents", [225468, 225477]),
    "吞咽评估(过程)": ("procedureevents", [229298, 229380]),
    # inputevents
    "高渗盐水 3%": ("inputevents", [225161]),
    "高渗盐水 23.4%": ("inputevents", [228341]),
    "甘露醇": ("inputevents", [227531]),
    "胰岛素(常规)": ("inputevents", [223258]),
    "红细胞输注": ("inputevents", [225168]),
    "血小板输注": ("inputevents", [225170]),
    "血浆输注": ("inputevents", [220970]),
    # outputevents
    "脑室引流量": ("outputevents", [226606, 226607]),
}
VALUE_ITEMS = {226732: "氧疗方式", 223849: "呼吸机模式", 229649: "卒中护理计划-类型",
               229652: "卒中护理计划-干预", 229647: "卒中护理计划-问题"}
TIME_COL = {"chartevents": "charttime", "procedureevents": "starttime",
            "inputevents": "starttime", "outputevents": "charttime"}


def in_window(df, tcol):
    df = df.join(win, on="stay_id")
    t = pd.to_datetime(df[tcol])
    return df[(t >= df.intime) & (t < df.end)]


hits = {}
values = {k: pd.Series(dtype=int) for k in VALUE_ITEMS}
insulin_cont = set()
for table in ["chartevents", "procedureevents", "inputevents", "outputevents"]:
    items = {i for g, (t, its) in GROUPS.items() if t == table for i in its}
    cols = ["stay_id", "itemid", TIME_COL[table]]
    if table == "chartevents":
        cols.append("value")
    if table == "inputevents":
        cols.append("ordercategorydescription")
    print(f"扫描 {table} ...", flush=True)
    sub = "hosp" if table == "labevents" else "icu"
    for ch in pd.read_csv(f"{B}/{sub}/{table}.csv.gz", usecols=cols, chunksize=CH):
        ch = ch[ch.itemid.isin(items) & ch.stay_id.isin(ids)]
        if ch.empty:
            continue
        ch = in_window(ch, TIME_COL[table])
        for g, (t, its) in GROUPS.items():
            if t == table:
                hits.setdefault(g, set()).update(ch.loc[ch.itemid.isin(its), "stay_id"])
        if table == "chartevents":
            for it in VALUE_ITEMS:
                vc = ch.loc[ch.itemid == it, "value"].value_counts()
                values[it] = values[it].add(vc, fill_value=0)
        if table == "inputevents":
            m = (ch.itemid == 223258) & ch.ordercategorydescription.str.contains("Continuous", na=False)
            insulin_cont.update(ch.loc[m, "stay_id"])
hits["胰岛素(持续静脉)"] = insulin_cont

inter_ids = set(c.loc[c.inter, "stay_id"])
rows = [{"记录类别": g, "病人数": len(s), "占全队列%": round(len(s) / N * 100, 1),
         "中间监护病人数": len(s & inter_ids), "占中间监护%": round(len(s & inter_ids) / NI * 100, 1)}
        for g, s in hits.items()]
t1 = pd.DataFrame(rows)
t1.to_csv(OUT / "s4_item_availability.csv", index=False)

vrows = []
for it, name in VALUE_ITEMS.items():
    v = values[it]
    v = v[v >= MIN_VALUE_COUNT].sort_values(ascending=False).head(25)
    for val, n in v.items():
        vrows.append({"条目": name, "取值": val, "记录数": int(n)})
t2 = pd.DataFrame(vrows)
t2.to_csv(OUT / "s4_value_distributions.csv", index=False)

# ---- 医嘱表 ----------------------------------------------------------------
print("扫描 poe ...", flush=True)
hadm = set(c.hadm_id)
poe = []
for ch in pd.read_csv(f"{B}/hosp/poe.csv.gz", usecols=["poe_id", "hadm_id", "order_type", "order_subtype"],
                      chunksize=CH):
    poe.append(ch[ch.hadm_id.isin(hadm)])
poe = pd.concat(poe)
t3 = (poe.groupby(["order_type", "order_subtype"], dropna=False)
      .agg(医嘱数=("poe_id", "size"), 住院数=("hadm_id", "nunique")).reset_index()
      .sort_values("医嘱数", ascending=False))
t3 = t3[t3["医嘱数"] >= MIN_VALUE_COUNT]
t3.to_csv(OUT / "s4_poe_order_types.csv", index=False)
neuro = t3[t3.order_subtype.astype(str).str.contains("neuro|neur|check|vital|monitor", case=False)]

det = pd.read_csv(f"{B}/hosp/poe_detail.csv.gz")
det = det[det.poe_id.isin(set(poe.poe_id))]
t4 = (det.groupby("field_name").agg(记录数=("poe_id", "size")).reset_index()
      .sort_values("记录数", ascending=False))
t4.to_csv(OUT / "s4_poe_detail_fields.csv", index=False)


def md(df):
    head = list(map(str, df.columns))
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |" for r in df.values]
    return "\n".join(out)


lines = ["# readiness 定义的数据审计", "",
         f"全队列 {N:,} 人；入 ICU 单元为神经中间监护 / 神经过渡病房 {NI:,} 人。", "",
         "## 1. 各类记录：ICU 停留期间至少有一条记录的病人", "", md(t1), "",
         "## 2. 下拉选项类条目的取值（出现 ≥ 20 次）", "", md(t2), "",
         "## 3. 医嘱表：本队列住院期间的医嘱类型（≥ 20 条）", "", md(t3.head(60)), "",
         "与神经检查 / 生命体征 / 监测相关的子类型：", "", md(neuro) if len(neuro) else "（没有）", "",
         "## 4. 医嘱细节表的字段", "", md(t4), ""]
(OUT / "s4_summary.md").write_text("\n".join(lines), encoding="utf-8")
print("完成")
