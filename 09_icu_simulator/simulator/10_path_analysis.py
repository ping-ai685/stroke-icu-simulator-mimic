"""
单中心 ICU 路径模拟器 · 第 1 步：路径分析补充（技术设计 v0.2 第 3 节）

补充：入科来源、ICU 内单元之间的移动、离开 ICU 后的去向、同次住院再入 ICU、年代划分的核实，
以及《ICU时间论文_分析方案_v1.2》第 12 节预先设定的两项分析：
  12.1 T3 中的监护强度
  12.2 readiness 定义的临床意义（离开时状态 × 再入 ICU / 离开后院内死亡）

输入：paper4_formal/work/hourly.pkl、patients.pkl；MIMIC-IV 原始表
输出：simulator/outputs/path_analysis.md 与 csv（只有汇总数字）
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
sys.path.insert(0, str(ROOT / "03_code"))
import project_paths  # noqa: E402
import variable_itemid_map as V  # noqa: E402

B = project_paths.MIMIC_IV
HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"; OUT.mkdir(exist_ok=True)
FW = HERE.parent / "paper4_formal" / "work"
CH = 5_000_000
INTER = ["Neuro Intermediate", "Neuro Stepdown"]

H = pd.read_pickle(FW / "hourly.pkl")
P = pd.read_pickle(FW / "patients.pkl").set_index("stay_id")
hadm = set(P.hadm_id)
T = {}

# ---------------------------------------------------------------------------
# 0. 年代划分的核实
# ---------------------------------------------------------------------------
pat = pd.read_csv(f"{B}/hosp/patients.csv.gz", usecols=["subject_id", "anchor_year", "anchor_year_group"])
P = P.drop(columns=["anchor_year_group"]).join(pat.set_index("subject_id"), on="subject_id")
mid = P.anchor_year_group.str[:4].astype(int) + 1
P["approx_year"] = mid + (P.intime.dt.year - P.anchor_year)
P["split_group"] = np.where(P.anchor_year_group.isin(["2008 - 2010", "2011 - 2013", "2014 - 2016"]), "建模(按组)",
                            np.where(P.anchor_year_group == "2017 - 2019", "验证(按组)", "2020–2022(按组)"))
P["split_approx"] = np.select([P.approx_year <= 2016, P.approx_year <= 2019], ["建模(近似年)", "验证(近似年)"],
                              default="2020–2022(近似年)")
T["0_year_split"] = pd.crosstab(P.split_group, P.split_approx, margins=True, margins_name="合计")
T["0_years_after_anchor"] = (P.intime.dt.year - P.anchor_year).clip(upper=6).value_counts().sort_index() \
    .rename("入院年 − 锚定年（≥6 合并）").to_frame()

# ---------------------------------------------------------------------------
# 1. 入科来源
# ---------------------------------------------------------------------------
adm = pd.read_csv(f"{B}/hosp/admissions.csv.gz",
                  usecols=["hadm_id", "admittime", "deathtime", "admission_location", "discharge_location"],
                  parse_dates=["admittime", "deathtime"])
adm = adm[adm.hadm_id.isin(hadm)].set_index("hadm_id")
tr = pd.read_csv(f"{B}/hosp/transfers.csv.gz", usecols=["hadm_id", "eventtype", "careunit", "intime", "outtime"],
                 parse_dates=["intime", "outtime"])
tr = tr[tr.hadm_id.isin(hadm)].sort_values(["hadm_id", "intime"])
# ICU 类单元（明确列出）。icustays 的 first_careunit 中偶有 Neurology、Medicine、PACU 等普通单元，不能直接当作 ICU 名单
icu_units = {"Cardiac Vascular Intensive Care Unit (CVICU)", "Coronary Care Unit (CCU)", "Intensive Care Unit (ICU)",
             "Medical Intensive Care Unit (MICU)", "Medical/Surgical Intensive Care Unit (MICU/SICU)",
             "Neuro Surgical Intensive Care Unit (Neuro SICU)", "Surgical Intensive Care Unit (SICU)",
             "Trauma SICU (TSICU)", "Neuro Intermediate", "Neuro Stepdown"}


trg = {h: g for h, g in tr.groupby("hadm_id")}


def prev_unit(h, t):
    g = trg.get(h)
    if g is None:
        return None
    x = g[g.intime < t - pd.Timedelta(minutes=1)]
    return x.careunit.iloc[-1] if len(x) else None


def next_unit(h, t):
    g = trg.get(h)
    if g is None:
        return None, None
    x = g[g.intime >= t - pd.Timedelta(minutes=1)]
    x = x[~x.careunit.isin(icu_units) | x.careunit.isna()]
    if not len(x):
        return None, None
    r = x.iloc[0]
    return r.careunit, r.eventtype


P["prev_unit"] = [prev_unit(h, t) for h, t in zip(P.hadm_id, P.intime)]


def src(u, loc):
    if u is None or (isinstance(u, float) and np.isnan(u)):
        return "直接入 ICU（无此前记录）"
    u = str(u)
    if "Emergency" in u:
        return "急诊"
    if any(k in u for k in ["PACU", "Operating", "Procedure", "Cath", "Interventional"]):
        return "手术室 / 操作室 / 麻醉恢复室"
    if u in icu_units:
        return "其他 ICU（同次住院此前的 ICU）"
    return "普通病房 / 其他院内单元"


P["entry"] = [src(u, None) for u in P.prev_unit]
P["outside"] = P.hadm_id.map(adm.admission_location).str.contains("TRANSFER FROM HOSPITAL", na=False)
t1 = pd.crosstab(P.entry, P.subtype, margins=True, margins_name="合计")
t1["占全部%"] = (t1["合计"] / len(P) * 100).round(1)
T["1_entry_source"] = t1
T["1_outside_transfer"] = pd.DataFrame({"外院转入%": (P.groupby("subtype").outside.mean() * 100).round(1)})

# ---------------------------------------------------------------------------
# 2. ICU 内单元之间的移动（逐小时单元类型的变化）
# ---------------------------------------------------------------------------
h2 = H[["stay_id", "k", "unit_type"]].sort_values(["stay_id", "k"])
prev = h2.groupby("stay_id").unit_type.shift()
mv = h2[prev.notna() & (prev != h2.unit_type)].assign(frm=prev)
mv = mv[mv.frm.isin(["正规ICU", "神经中间监护/过渡病房"]) & mv.unit_type.isin(["正规ICU", "神经中间监护/过渡病房"])]
first_type = h2.groupby("stay_id").unit_type.first()
rows = []
for ft in ["正规ICU", "神经中间监护/过渡病房"]:
    ids = first_type[first_type == ft].index
    m = mv[mv.stay_id.isin(ids)]
    down = m[m.frm == "正规ICU"]
    up = m[m.frm == "神经中间监护/过渡病房"]
    rows.append({"首个单元类型": ft, "人数": len(ids),
                 "曾 正规ICU→中间监护 %": round(down.stay_id.nunique() / len(ids) * 100, 1),
                 "曾 中间监护→正规ICU %": round(up.stay_id.nunique() / len(ids) * 100, 1),
                 "首次移动时间 中位(小时)": float(m.groupby("stay_id").k.min().median()) if len(m) else np.nan})
T["2_within_icu_moves"] = pd.DataFrame(rows).set_index("首个单元类型")

# ---------------------------------------------------------------------------
# 3. 离开 ICU 后的去向、再入 ICU、离开后院内死亡
# ---------------------------------------------------------------------------
alive = P[~P.died & ~P.truncated].copy()
nx = [next_unit(h, t) for h, t in zip(alive.hadm_id, alive.outtime)]
alive["next_unit"], alive["next_event"] = [a for a, _ in nx], [b for _, b in nx]


def dest(u, e):
    if e == "discharge" or u is None or (isinstance(u, float) and np.isnan(u)):
        return "直接出院"
    u = str(u)
    if any(k in u for k in ["PACU", "Operating", "Procedure", "Cath", "Interventional"]):
        return "手术室 / 操作室"
    if "Emergency" in u:
        return "急诊"
    return "普通病房"


alive["dest"] = [dest(u, e) for u, e in zip(alive.next_unit, alive.next_event)]
icu = pd.read_csv(f"{B}/icu/icustays.csv.gz", usecols=["hadm_id", "stay_id", "intime"], parse_dates=["intime"])
icu = icu[icu.hadm_id.isin(hadm)]
later = icu.merge(alive[["hadm_id", "outtime"]].reset_index().rename(columns={"stay_id": "idx"}), on="hadm_id")
later = later[later.intime > later.outtime]
first_re = later.groupby("idx").intime.min()
alive["readmit"] = alive.index.isin(first_re.index)
alive["readmit_h"] = (first_re.reindex(alive.index) - alive.outtime).dt.total_seconds() / 3600
dt = alive.hadm_id.map(adm.deathtime)
alive["post_death"] = dt.notna() & (dt > alive.outtime)
alive["composite"] = alive.readmit | alive.post_death
T["3_destination"] = pd.crosstab(alive.dest, alive.subtype, margins=True, margins_name="合计")
T["3_readmit_death"] = pd.DataFrame({
    "人数": alive.groupby("subtype").size(),
    "同次住院再入ICU%": (alive.groupby("subtype").readmit.mean() * 100).round(1),
    "再入时间 中位(小时)": alive.groupby("subtype").readmit_h.median().round(1),
    "离开ICU后院内死亡%": (alive.groupby("subtype").post_death.mean() * 100).round(1)})
T["3_readmit_death"].loc["合计"] = [len(alive), round(alive.readmit.mean() * 100, 1),
                                   round(alive.readmit_h.median(), 1), round(alive.post_death.mean() * 100, 1)]

# ---------------------------------------------------------------------------
# 12.1 T3 中的监护强度
# ---------------------------------------------------------------------------
print("扫描 chartevents（GCS 时间）...", flush=True)
ids = set(P.index)
fr = []
for ch in pd.read_csv(f"{B}/icu/chartevents.csv.gz", usecols=["stay_id", "itemid", "charttime", "value"], chunksize=CH):
    fr.append(ch[ch.itemid.isin([V.GCS_EYE, V.GCS_VERBAL, V.GCS_MOTOR, 223758]) & ch.stay_id.isin(ids)])
ce = pd.concat(fr)
code_ce = ce[ce.itemid == 223758]
g = ce[ce.itemid != 223758][["stay_id", "charttime"]].drop_duplicates()
g["h"] = (pd.to_datetime(g.charttime) - g.stay_id.map(P.intime)).dt.total_seconds() / 3600
g = g[(g.h > -6) & (g.h <= 720)]
cnt6 = np.zeros(len(H)); cnt4 = np.zeros(len(H))
Hk = H[["stay_id", "k"]].reset_index(drop=True)
gh = {s: np.sort(x.values) for s, x in g.groupby("stay_id").h}
for sid, idx in Hk.groupby("stay_id").groups.items():
    hs = gh.get(sid, np.array([]))
    ks = Hk.k.values[idx].astype(float)
    cnt6[idx] = np.searchsorted(hs, ks, side="right") - np.searchsorted(hs, ks - 6, side="right")
    cnt4[idx] = np.searchsorted(hs, ks, side="right") - np.searchsorted(hs, ks - 4, side="right")
N = H.N_unk0 | H.N_gap6 | H.N_drop2 | H.N_sed | H.N_rass | H.N_seiz
v02 = (H.R | H.C | N | H.M).to_numpy()
extra = (H.CBP | H.P).to_numpy()
seg = np.select([v02, extra], ["T1", "T2"], default="T3")
H3 = H.assign(seg=seg, hi6=cnt6 >= 3, hi4=cnt4 >= 2)
H3["phase"] = pd.cut(H3.k, [-1, 23, 71, 167, 335, 720], labels=["第1天", "第2–3天", "第4–7天", "第8–14天", "第15–30天"])
t3h = H3[H3.seg == "T3"]
r121 = []
for ut in ["正规ICU", "神经中间监护/过渡病房", "全部"]:
    x = t3h if ut == "全部" else t3h[t3h.unit_type == ut]
    pm = x.groupby("stay_id").hi6.mean()
    r121.append({"单元类型": ut, "T3小时数": len(x), "高强度监护%（6h≥3次，小时口径）": round(x.hi6.mean() * 100, 1),
                 "高强度监护%（病人口径均值）": round(pm.mean() * 100, 1),
                 "敏感性：4h≥2次（小时口径）%": round(x.hi4.mean() * 100, 1)})
T["12_1_T3_intensity"] = pd.DataFrame(r121).set_index("单元类型")
T["12_1_T3_intensity_by_phase"] = (t3h.groupby(["unit_type", "phase"], observed=True).hi6.mean() * 100).round(1) \
    .unstack("phase")

# ---------------------------------------------------------------------------
# 12.2 readiness 定义的临床意义
# ---------------------------------------------------------------------------
u03 = pd.Series(v02 | extra, index=H.index)
x = pd.DataFrame({"stay_id": H.stay_id.values, "k": H.k.values, "m": ~u03.values})
grp = (x.m != x.groupby("stay_id").m.shift()).cumsum()
run = x.groupby([x.stay_id, grp]).cumcount()
r03 = x[x.m & (run >= 6)].groupby("stay_id").k.first()
lastm = x.groupby("stay_id").tail(1).set_index("stay_id").m
# 仅舒适治疗：方案 v1.2 设定的来源（医嘱表的代码状态字段）只有"全力抢救 / DNR"等选项，没有"仅舒适治疗"，
# 因此改用 ICU 护理记录中的 Code Status 条目（itemid 223758）。见方案修订记录。
code_vals = code_ce.value.value_counts()
cmo = code_ce[code_ce.value.str.contains("Comfort|CMO", case=False, na=False)]
cmo_first = pd.to_datetime(cmo.groupby("stay_id").charttime.min())
alive["cmo_before_exit"] = alive.index.map(cmo_first).to_series(index=alive.index).lt(alive.outtime).fillna(False)
T["12_2_code_status_values"] = code_vals[code_vals >= 20].rename("记录数").to_frame()
alive["grp"] = np.select([alive.index.isin(r03.index) & alive.index.map(lastm).astype(bool),
                          alive.index.isin(r03.index)], ["1 离开时已达到", "2 曾达到、离开时又不满足"],
                         default="3 从未达到")
lu = H.groupby("stay_id").unit_type.last()
alive["exit_inter"] = (alive.index.map(lu) == "神经中间监护/过渡病房").astype(float)
alive["after_hours"] = alive.outtime.dt.hour.map(lambda h: h >= 19 or h < 7).astype(float)
T["12_2_groups"] = pd.DataFrame({
    "人数": alive.groupby("grp").size(),
    "再入ICU%": (alive.groupby("grp").readmit.mean() * 100).round(1),
    "离开后院内死亡%": (alive.groupby("grp").post_death.mean() * 100).round(1),
    "复合%": (alive.groupby("grp").composite.mean() * 100).round(1)})


def logit(df, y):
    X = pd.DataFrame({
        "曾达到、离开时又不满足(对已达到)": (df.grp == "2 曾达到、离开时又不满足").astype(float),
        "从未达到(对已达到)": (df.grp == "3 从未达到").astype(float),
        "年龄(每10岁)": df.age_at_adm_capped / 10, "女性": (df.gender == "F").astype(float),
        "脑出血": (df.subtype == "ICH").astype(float), "蛛网膜下腔出血": (df.subtype == "SAH").astype(float),
        "ICU停留(log小时)": np.log(df.end_h.clip(lower=1)), "离开时在中间监护": df.exit_inter,
        "再灌注": df.reperfusion.fillna(False).astype(float), "晚7点至早7点离开": df.after_hours})
    for e in ["2011 - 2013", "2014 - 2016", "2017 - 2019", "2020 - 2022"]:
        X[f"年代 {e}"] = (df.anchor_year_group == e).astype(float)
    X = sm.add_constant(X)
    f = sm.GLM(df[y].astype(float), X, family=sm.families.Binomial()).fit()
    ci = f.conf_int()
    return pd.DataFrame({"比值比": np.exp(f.params).round(2),
                         "95%CI": [f"{np.exp(a):.2f}–{np.exp(b):.2f}" for a, b in ci.values],
                         "P": f.pvalues.map(lambda v: "<0.001" if v < 0.001 else f"{v:.3f}")}).drop(index="const")


T["12_2_logit_readmit"] = logit(alive, "readmit")
T["12_2_logit_post_death"] = logit(alive, "post_death")
T["12_2_logit_composite"] = logit(alive, "composite")
T["12_2_sens_excl_cmo_readmit"] = logit(alive[~alive.cmo_before_exit], "readmit")
T["12_2_sens_excl_cmo_post_death"] = logit(alive[~alive.cmo_before_exit], "post_death")
T["12_2_sens_excl_cmo_composite"] = logit(alive[~alive.cmo_before_exit], "composite")
T["12_2_cmo_count"] = pd.DataFrame({"离开前有仅舒适治疗医嘱的人数": [int(alive.cmo_before_exit.sum())]})

# v0.2 作为次要暴露
x2 = pd.DataFrame({"stay_id": H.stay_id.values, "m": ~v02})
grp2 = (x2.m != x2.groupby("stay_id").m.shift()).cumsum()
run2 = x2.groupby([x2.stay_id, grp2]).cumcount()
r02 = x2[x2.m & (run2 >= 6)].groupby("stay_id").size()
last2 = x2.groupby("stay_id").tail(1).set_index("stay_id").m
alive2 = alive.copy()
alive2["grp"] = np.select([alive2.index.isin(r02.index) & alive2.index.map(last2).astype(bool),
                           alive2.index.isin(r02.index)], ["1 离开时已达到", "2 曾达到、离开时又不满足"],
                          default="3 从未达到")
T["12_2_v02_groups"] = pd.DataFrame({"人数": alive2.groupby("grp").size(),
                                     "再入ICU%": (alive2.groupby("grp").readmit.mean() * 100).round(1)})
T["12_2_v02_logit_readmit"] = logit(alive2, "readmit")


# ---------------------------------------------------------------------------
def md(df):
    df = df.to_frame() if isinstance(df, pd.Series) else df
    idx = df.index
    head = [" / ".join(map(str, idx.names)) if idx.nlevels > 1 else str(idx.name or "")] + list(map(str, df.columns))
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for i, r in zip(idx, df.values):
        i = " / ".join(map(str, i)) if isinstance(i, tuple) else str(i)
        out.append("| " + " | ".join([i] + ["" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r]) + " |")
    return "\n".join(out)


TITLE = {"0_year_split": "0 年代划分：按 anchor_year_group vs 按近似真实入院年", "0_years_after_anchor": "0 入院年与锚定年之差",
         "1_entry_source": "1 入科来源（ICU 前最后所在单元）", "1_outside_transfer": "1 外院转入（入院来源）",
         "2_within_icu_moves": "2 ICU 内单元之间的移动", "3_destination": "3 离开 ICU 后的去向（活着离开者）",
         "3_readmit_death": "3 同次住院再入 ICU 与离开后院内死亡（活着离开者）",
         "12_1_T3_intensity": "12.1 T3 中的监护强度", "12_1_T3_intensity_by_phase": "12.1 T3 中高强度监护%（按单元类型 × 阶段）",
         "12_2_groups": "12.2 离开时状态与结局（v0.3）", "12_2_logit_readmit": "12.2 再入 ICU（logistic，调整后）",
         "12_2_logit_post_death": "12.2 离开后院内死亡（logistic，调整后）", "12_2_logit_composite": "12.2 复合结局（logistic，调整后）",
         "12_2_sens_excl_cmo_readmit": "12.2 敏感性：排除离开前已有仅舒适治疗医嘱者（再入 ICU）",
         "12_2_cmo_count": "12.2 仅舒适治疗医嘱", "12_2_sens_excl_cmo_post_death": "12.2 敏感性：排除仅舒适治疗者（离开后院内死亡）", "12_2_sens_excl_cmo_composite": "12.2 敏感性：排除仅舒适治疗者（复合结局）", "12_2_code_status_values": "12.2 ICU 护理记录中 Code Status 的取值（≥20 条）", "12_2_v02_groups": "12.2 次要暴露：v0.2 离开时状态与再入 ICU",
         "12_2_v02_logit_readmit": "12.2 次要暴露：v0.2（再入 ICU，logistic）"}
for k, t in T.items():
    t.to_csv(OUT / f"path_{k}.csv")
lines = ["# 单中心 ICU 路径分析补充", "",
         "第 12 节两项分析按《ICU时间论文_分析方案_v1.2》事先设定；其余为描述性补充。只作关联分析，不作因果解释。", ""]
for k, t in T.items():
    lines += [f"## {TITLE.get(k, k)}", "", md(t), ""]
(OUT / "path_analysis.md").write_text("\n".join(lines), encoding="utf-8")
print("完成")
