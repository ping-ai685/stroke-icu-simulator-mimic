"""
ICU 路径模拟器 · 按操作性定义 v0.3 计算 ICU 转出准备状态（readiness），同时重算 v0.2 以便比较
v0.3 = v0.2 + 持续静脉降压药（并入循环）+ 再灌注后监护期（第五条）。定义见
the readiness definition document, version 0.3 (Supplementary S2 of the paper)。以下为 v0.2 部分的原说明。


定义见 the readiness definition document, version 0.2 (Supplementary S2 of the paper)。在每个整点（入 ICU 后第 k 小时）
判断四条生理条件；四条同时满足、连续 6 小时（第 k−6 … k 个整点全部满足）的第一个 k
为 readiness 确认时间。只用该时点及以前的信息。

实现中的具体选择（定义文件没有写死的地方，在此写明）：
  呼吸  未满足 = 有创通气过程进行中；或此前 6 小时内记录过 ETT / 高流量鼻导管 / BiPAP 面罩 /
        CPAP 面罩；或此前 6 小时内记录过呼吸机模式（Standby 除外）
  循环  未满足 = 升压药持续输注进行中
  意识  未满足 = 此前 6 小时内没有 GCS 记录（未知）；或此前 6 小时内有经确认的 GCS 下降；
        或镇静药持续输注进行中；或此前 6 小时内 RASS ≥ +3；或此前 24 小时内有癫痫发作记录
        GCS 下降：与此前 12 小时内（≥ 2 次记录）的中位数相比下降 ≥ 2 分；语言分可评估时用总分，
        否则用睁眼 + 运动分；下一次评估（6 小时内）仍然下降，即为确认，确认时间为复测时间
  代谢  未满足 = 最近一次结果为危急值（不因过期而恢复，直到出现正常结果）；或此前 6 小时内用过
        高渗盐水 / 甘露醇 / 输血；或持续静脉胰岛素进行中。同一时刻多个结果取最异常的一个
  第二层（完整 readiness）未满足 = 此前 12 小时内有神经引流记录（状态非"已拔除"）或脑室引流量，
        或颅内压数值
敏感性分析：稳定持续 4 / 12 / 24 小时；GCS 下降 ≥ 3 分；有"睡眠呼吸暂停用 CPAP"医嘱者不把
        CPAP / BiPAP 面罩算作 ICU 级支持

输出：outputs/s5_*.csv、s5_summary.md（汇总）；work/readiness_by_stay.csv（个体层面，仅本机）
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.insert(0, str(ROOT / "03_code"))
import project_paths  # noqa: E402
import variable_itemid_map as V  # noqa: E402

B = project_paths.MIMIC_IV
HERE = Path(__file__).resolve().parent
OUT, WORK = HERE / "outputs", HERE / "work"
CH = 5_000_000
MAXH = 720
H = pd.Timedelta(hours=1)

# ---------------------------------------------------------------------------
# 队列与整点网格
# ---------------------------------------------------------------------------
c = pd.read_csv(ROOT / "04_outputs/tables/patient_level_cohort.csv",
                parse_dates=["intime", "outtime", "deathtime"])
c["died"] = c.icu_mortality.astype(int) == 1
c["end"] = c.outtime
d = c.deathtime.notna() & (c.deathtime < c.outtime)
c.loc[d, "end"] = c.loc[d, "deathtime"]
c = c[c.end > c.intime].copy()
c["end_h"] = (c.end - c.intime).dt.total_seconds() / 3600
c["K"] = np.floor(c.end_h.clip(upper=MAXH)).astype(int)
ids, hadm = set(c.stay_id), set(c.hadm_id)
it = c.set_index("stay_id")["intime"]
Kmax = c.set_index("stay_id")["K"]

grid = pd.DataFrame({"stay_id": np.repeat(c.stay_id.values, c.K.values + 1),
                     "k": np.concatenate([np.arange(n + 1) for n in c.K.values])})
grid["t"] = grid.stay_id.map(it) + grid.k * H
print(f"整点网格：{len(grid):,} 个整点，{c.stay_id.nunique():,} 人", flush=True)


def rel(df, col):
    return (pd.to_datetime(df[col]) - df.stay_id.map(it)).dt.total_seconds() / 3600


def marks(stay, lo, hi):
    """lo ≤ k ≤ hi 的整点（两端为小时数，已取整），裁剪到每人的 [0, K]。"""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    fin = np.isfinite(lo) & np.isfinite(hi)
    stay, lo, hi = stay[fin], lo[fin], hi[fin]
    lo = np.ceil(lo).astype(int).clip(min=0)
    hi = np.minimum(np.floor(hi).astype(int), stay.map(Kmax).values)
    ok = hi >= lo
    s, lo, hi = stay.values[ok], lo[ok], hi[ok]
    if len(s) == 0:
        return pd.DataFrame(columns=["stay_id", "k"])
    n = hi - lo + 1
    return pd.DataFrame({"stay_id": np.repeat(s, n),
                         "k": np.concatenate([np.arange(a, b + 1) for a, b in zip(lo, hi)])})


def active_interval(df):            # starttime ≤ t < endtime
    s, e = rel(df, "starttime"), rel(df, "endtime")
    return marks(df.stay_id, s.values, np.ceil(e.values) - 1)


def recent_event(df, col, hours):   # 事件时间 ≤ t < 事件时间 + hours
    e = rel(df, col)
    return marks(df.stay_id, e.values, np.ceil(e.values + hours) - 1)


def flag(name, *frames):
    f = pd.concat([x for x in frames if len(x)]).drop_duplicates() if any(len(x) for x in frames) \
        else pd.DataFrame(columns=["stay_id", "k"])
    f[name] = True
    return f


# ---------------------------------------------------------------------------
# 读取原始记录
# ---------------------------------------------------------------------------
GCS_IT = [V.GCS_EYE, V.GCS_VERBAL, V.GCS_MOTOR]
CHART_IT = GCS_IT + [226732, 223849, 229314, 228096, 223923, 223926, 223930, 223931, 223932,
                     224764, 224765, 227981, 227982, 227983, 227984, 220765, 227989]
print("chartevents ...", flush=True)
fr = []
for ch in pd.read_csv(f"{B}/icu/chartevents.csv.gz",
                      usecols=["stay_id", "itemid", "charttime", "value", "valuenum"], chunksize=CH):
    fr.append(ch[ch.itemid.isin(CHART_IT) & ch.stay_id.isin(ids)])
ce = pd.concat(fr); del fr
ce["h"] = rel(ce, "charttime")
ce = ce[(ce.h >= -24) & (ce.h <= ce.stay_id.map(c.set_index("stay_id").end_h))]

print("labevents ...", flush=True)
LABS = {"sodium": (V.SODIUM, 125, 155), "potassium": (V.POTASSIUM, 3.0, 6.0),
        "glucose": (V.GLUCOSE, 54, 400), "hemoglobin": (V.HEMOGLOBIN, 7, None),
        "platelet": (V.PLATELET, 50, None), "ph": (V.PH, 7.25, 7.55), "lactate": (V.LACTATE, None, 4)}
lab_ids = {i: k for k, (ids_, _, _) in LABS.items() for i in ids_}
fr = []
for ch in pd.read_csv(f"{B}/hosp/labevents.csv.gz", usecols=["hadm_id", "itemid", "charttime", "valuenum"],
                      chunksize=CH):
    fr.append(ch[ch.itemid.isin(lab_ids) & ch.hadm_id.isin(hadm)])
lab = pd.concat(fr).dropna(subset=["valuenum"]); del fr
lab = lab.merge(c[["hadm_id", "stay_id"]], on="hadm_id")
lab["var"] = lab.itemid.map(lab_ids)
lab["h"] = rel(lab, "charttime")

print("procedure / input / output events ...", flush=True)
proc = pd.read_csv(f"{B}/icu/procedureevents.csv.gz", usecols=["stay_id", "itemid", "starttime", "endtime"])
proc = proc[proc.stay_id.isin(ids) & proc.itemid.isin(V.MECH_VENT)]
BP_IT = [222042, 229624, 229234, 225153, 221429, 222051]   # 尼卡地平×2、氯维地平、拉贝洛尔、艾司洛尔、硝普钠
INP_IT = V.VASOPRESSORS + V.SEDATIVES + BP_IT + [223258, 225161, 228341, 227531, 225168, 225170, 220970]
inp = pd.read_csv(f"{B}/icu/inputevents.csv.gz",
                  usecols=["stay_id", "itemid", "starttime", "endtime", "ordercategorydescription"])
inp = inp[inp.stay_id.isin(ids) & inp.itemid.isin(INP_IT)]
outp = pd.read_csv(f"{B}/icu/outputevents.csv.gz", usecols=["stay_id", "itemid", "charttime"])
outp = outp[outp.stay_id.isin(ids) & outp.itemid.isin([226606, 226607])]
osa = set()
for ch in pd.read_csv(f"{B}/hosp/poe.csv.gz", usecols=["hadm_id", "order_subtype"], chunksize=CH):
    osa.update(ch.loc[ch.hadm_id.isin(hadm) & (ch.order_subtype == "CPAP for OSA"), "hadm_id"])
osa_stays = set(c.loc[c.hadm_id.isin(osa), "stay_id"])

# ---------------------------------------------------------------------------
# 呼吸
# ---------------------------------------------------------------------------
dev = ce[ce.itemid == 226732]
ADV = ["Endotracheal tube", "High flow nasal cannula", "Bipap mask", "CPAP mask"]
dev_adv = dev[dev.value.str.strip().isin(ADV)]
dev_adv_noosa = dev_adv[~(dev_adv.stay_id.isin(osa_stays) &
                          dev_adv.value.str.strip().isin(["Bipap mask", "CPAP mask"]))]
mode = ce[ce.itemid.isin([223849, 229314]) & ce.value.notna() & (ce.value.str.strip() != "Standby")]
R_vent = active_interval(proc)
R_mode = recent_event(mode, "charttime", 6)
R = flag("R_unmet", R_vent, recent_event(dev_adv, "charttime", 6), R_mode)
R_osa = flag("R_unmet_osa", R_vent, recent_event(dev_adv_noosa, "charttime", 6), R_mode)

# ---------------------------------------------------------------------------
# 循环
# ---------------------------------------------------------------------------
C = flag("C_unmet", active_interval(inp[inp.itemid.isin(V.VASOPRESSORS)]))
CBP = flag("CBP_unmet", active_interval(inp[inp.itemid.isin(BP_IT) &
                                             inp.ordercategorydescription.str.contains("Continuous", na=False)]))

# ---------------------------------------------------------------------------
# 意识
# ---------------------------------------------------------------------------
g = ce[ce.itemid.isin(GCS_IT)].copy()
g.loc[(g.itemid == V.GCS_VERBAL) & g.value.isin(V.GCS_VERBAL_NOT_ASSESSABLE_VALUES), "valuenum"] = np.nan
g = g.pivot_table(index=["stay_id", "h"], columns="itemid", values="valuenum", aggfunc="last").reset_index()
g = g.rename(columns={V.GCS_EYE: "eye", V.GCS_VERBAL: "verbal", V.GCS_MOTOR: "motor"})
g = g.dropna(subset=["eye", "motor"]).sort_values(["stay_id", "h"])
g["em"] = g.eye + g.motor
g["tot"] = g.em + g.verbal


def confirmed_drops(df, thr):
    out = []
    for sid, x in df.groupby("stay_id", sort=False):
        h, em, tot = x.h.values, x.em.values, x.tot.values
        for i in range(1, len(h) - 1):
            lo = np.searchsorted(h, h[i] - 12, side="left")
            if i - lo < 2:
                continue
            base_tot, base_em = tot[lo:i], em[lo:i]
            use_tot = not np.isnan(tot[i]) and not np.isnan(tot[i + 1]) and not np.isnan(base_tot).any()
            cur, nxt, base = (tot[i], tot[i + 1], np.median(base_tot)) if use_tot else \
                             (em[i], em[i + 1], np.median(base_em))
            if cur <= base - thr and h[i + 1] - h[i] <= 6 and nxt <= base - thr:
                out.append((sid, h[i + 1]))
    return pd.DataFrame(out, columns=["stay_id", "h"])


print("GCS 下降 ...", flush=True)
drops2, drops3 = confirmed_drops(g, 2), confirmed_drops(g, 3)
# GCS 未知：此前 6 小时内没有记录
last = pd.merge_asof(grid.assign(hh=grid.k.astype(float)).sort_values("hh"),
                     g[["stay_id", "h"]].sort_values("h"), left_on="hh", right_on="h",
                     by="stay_id", direction="backward")
unk = last.loc[last.h.isna() | (last.hh - last.h > 6), ["stay_id", "k"]]
sed = active_interval(inp[inp.itemid.isin(V.SEDATIVES)])
rass = recent_event(ce[(ce.itemid == 228096) & (ce.valuenum >= 3)], "charttime", 6)
# 只计记录了发作类型的条目；该条目约 98% 的记录取值为空，不代表发作（2026-11-05 修正）
seiz = recent_event(ce[(ce.itemid == 223923) & ce.value.isin(["Focal", "General", "Status"])], "charttime", 24)
drop2_m = recent_event(drops2.rename(columns={"h": "hx"}).assign(
    hx_t=lambda x: x.stay_id.map(it) + pd.to_timedelta(x.hx, unit="h")), "hx_t", 6)
unk0 = last.loc[last.h.isna(), ["stay_id", "k"]]
gap = last.loc[last.h.notna() & (last.hh - last.h > 6), ["stay_id", "k"]]
NPARTS = {"N_unk0": unk0, "N_gap": gap, "N_drop": drop2_m, "N_sed": sed, "N_rass": rass, "N_seiz": seiz}
N = flag("N_unmet", unk, recent_event(drops2.rename(columns={"h": "hx"}).assign(
    hx_t=lambda x: x.stay_id.map(it) + pd.to_timedelta(x.hx, unit="h")), "hx_t", 6), sed, rass, seiz)
N3 = flag("N_unmet_g3", unk, recent_event(drops3.rename(columns={"h": "hx"}).assign(
    hx_t=lambda x: x.stay_id.map(it) + pd.to_timedelta(x.hx, unit="h")), "hx_t", 6), sed, rass, seiz)

# ---------------------------------------------------------------------------
# 代谢
# ---------------------------------------------------------------------------
lab_abn = []
for v, (_, lo, hi) in LABS.items():
    x = lab[lab["var"] == v].copy()
    x["abn"] = False
    if lo is not None:
        x["abn"] |= x.valuenum < lo
    if hi is not None:
        x["abn"] |= x.valuenum > hi
    x = x.sort_values(["h", "abn"])                     # 同一时刻：异常排在后面，取最异常
    x = x.groupby(["stay_id", "h"], as_index=False).abn.last().sort_values("h")
    m = pd.merge_asof(grid.assign(hh=grid.k.astype(float)).sort_values("hh"), x,
                      left_on="hh", right_on="h", by="stay_id", direction="backward")
    lab_abn.append(m.loc[m.abn.fillna(False).astype(bool), ["stay_id", "k"]])
ins_cont = inp[(inp.itemid == 223258) & inp.ordercategorydescription.str.contains("Continuous", na=False)]
M = flag("M_unmet", *lab_abn,
         recent_event(inp[inp.itemid.isin([225161, 228341, 227531, 225168, 225170, 220970])], "starttime", 6),
         active_interval(ins_cont))

# ---------------------------------------------------------------------------
# 第二层：神经干预
# ---------------------------------------------------------------------------
DRAIN = [223926, 223930, 223931, 223932, 224764, 224765, 227981, 227982, 227983, 227984]
drain = ce[ce.itemid.isin(DRAIN) & (ce.value.str.strip() != "Discontinued")]
icp = ce[ce.itemid.isin([220765, 227989]) & ce.valuenum.notna()]
L2 = flag("L2_unmet", recent_event(drain, "charttime", 12), recent_event(outp, "charttime", 12),
          recent_event(icp, "charttime", 12))

# ---------------------------------------------------------------------------
# 再灌注后监护期（v0.3 第五条）
# ---------------------------------------------------------------------------
print("再灌注 ...", flush=True)
c["admittime"] = pd.to_datetime(c.admittime)
ais_h = set(c.loc[c.stroke_subtype == "AIS", "hadm_id"])
h2s = c.set_index("hadm_id")["stay_id"]
DRUG = "alteplase|tenecteplase|activase|tnkase"
em = []
for ch in pd.read_csv(f"{B}/hosp/emar.csv.gz", usecols=["hadm_id", "emar_id", "charttime", "medication", "event_txt"],
                      chunksize=CH):
    em.append(ch[ch.hadm_id.isin(ais_h) & ch.medication.str.contains(DRUG, case=False, na=False)
                 & ~ch.medication.str.contains("Clearance|Ventricular|Dialysis|Flush", case=False, na=False)
                 & ch.event_txt.isin(["Administered", "Started", "Bolus Given", "Partial Administered"])])
em = pd.concat(em)
dose = []
for ch in pd.read_csv(f"{B}/hosp/emar_detail.csv.gz", usecols=["emar_id", "dose_given"], chunksize=CH, dtype=str):
    dose.append(ch[ch.emar_id.isin(set(em.emar_id))])
dose = pd.concat(dose)
dose["d"] = pd.to_numeric(dose.dose_given, errors="coerce")
em["d"] = em.emar_id.map(dose.groupby("emar_id").d.max())
em = em[(em.d > 4) | em.d.isna()]
em["stay_id"] = em.hadm_id.map(h2s)
exact = em.groupby("stay_id").charttime.min()
exact_h = (pd.to_datetime(exact) - exact.index.map(it)).dt.total_seconds() / 3600

px = pd.read_csv(f"{B}/hosp/procedures_icd.csv.gz", dtype={"icd_code": str}, parse_dates=["chartdate"])
px = px[px.hadm_id.isin(ais_h)]
lys_px = px[((px.icd_version == 10) & px.icd_code.isin(["3E03317", "3E05317", "3E06317"])) |
            ((px.icd_version == 9) & (px.icd_code == "9910"))]
evt_px = px[((px.icd_version == 10) & px.icd_code.isin(["03CG3ZZ", "03CK3ZZ", "03CL3ZZ", "03CM3ZZ", "03CN3ZZ",
                                                         "03CP3ZZ", "03CQ3ZZ"])) |
            ((px.icd_version == 9) & (px.icd_code == "3974"))]
dxr = pd.read_csv(f"{B}/hosp/diagnoses_icd.csv.gz", dtype={"icd_code": str})
drip = set(dxr.loc[dxr.hadm_id.isin(ais_h) & dxr.icd_code.isin(["Z9282", "V4588"]), "stay_id" if False else "hadm_id"])

S = c.set_index("stay_id")
rp = pd.DataFrame(index=S.index)
rp["lys_date"] = lys_px.groupby("hadm_id").chartdate.min().reindex(S.hadm_id).values
rp["evt_date"] = evt_px.groupby("hadm_id").chartdate.min().reindex(S.hadm_id).values
rp["drip"] = S.hadm_id.isin(drip).values
rp["exact_h"] = exact_h.reindex(S.index)
rp["any_lys"] = rp.lys_date.notna() | rp.drip | rp.exact_h.notna()
rp["any"] = rp.any_lys | rp.evt_date.notna()


def date_window(dcol):
    dd = (pd.to_datetime(rp[dcol]) - S.intime.dt.normalize()).dt.days
    okb = dd.between(-1, 3)
    lo = ((pd.to_datetime(rp[dcol]) - S.intime).dt.total_seconds() / 3600).clip(lower=0)
    return okb, lo


okb_l, lo_l = date_window("lys_date")
okb_e, lo_e = date_window("evt_date")
drip_b = rp.drip & ((S.intime - S.admittime).dt.total_seconds() / 3600 <= 24)
rp["grade"] = np.select([rp.exact_h.notna(), okb_l | okb_e | drip_b, rp["any"]], ["A", "B", "C"], default="")
# 代理窗口起点：ICD 日期（若在入 ICU 之后）否则 0
proxy_lo = pd.concat([lo_l.where(okb_l), lo_e.where(okb_e)], axis=1).min(axis=1).fillna(0)
proxy_lo_lys = lo_l.where(okb_l).fillna(0)


def p_flag(name, mask, lo):
    s = rp.index[np.asarray(mask, bool)]
    lo = lo.reindex(s).values
    return flag(name, marks(pd.Series(s), lo, np.ceil(lo + 24) - 1))


ab = rp.grade.isin(["A", "B"])
P = p_flag("P_unmet", ab, proxy_lo)
P_lys = p_flag("P_unmet_lys", ab & rp.any_lys, proxy_lo_lys)
lo_exact = proxy_lo.copy()
lo_exact[rp.grade == "A"] = rp.exact_h[rp.grade == "A"]
P_exact = p_flag("P_unmet_exact", ab, lo_exact)

# ---------------------------------------------------------------------------
# 合并
# ---------------------------------------------------------------------------
G = grid[["stay_id", "k"]].copy()
for f in [R, C, CBP, N, M, L2, P, P_lys, P_exact] + [flag(k, v) for k, v in NPARTS.items()]:
    f["k"] = f.k.astype(int)
    G = G.merge(f, on=["stay_id", "k"], how="left")
G = G.fillna(False).sort_values(["stay_id", "k"]).reset_index(drop=True)
FLAGS = ["R_unmet", "C_unmet", "CBP_unmet", "N_unmet", "M_unmet", "L2_unmet", "P_unmet", "P_unmet_lys",
         "P_unmet_exact"] + list(NPARTS)
for col in FLAGS:
    G[col] = G[col].astype(bool)
base2 = ~(G.R_unmet | G.C_unmet | G.N_unmet | G.M_unmet)
base3 = base2 & ~G.CBP_unmet
G["v02"] = base2
G["v03"] = base3 & ~G.P_unmet
G["v03_full"] = G.v03 & ~G.L2_unmet
G["v03_lysonly"] = base3 & ~G.P_unmet_lys
G["v03_exact"] = base3 & ~G.P_unmet_exact
G["v03_noreperf"] = base3


def first_ready(col, dur=6):
    x = G[["stay_id", "k", col]]
    grp = (x[col] != x.groupby("stay_id")[col].shift()).cumsum()
    run = x.groupby([x.stay_id, grp]).cumcount()
    return x[x[col] & (run >= dur)].groupby("stay_id").k.first()


VERS = {"v02": "v0.2 生理", "v03": "v0.3 主分析", "v03_full": "v0.3 + 无神经引流/颅压",
        "v03_lysonly": "敏感性A：只对溶栓加规则", "v03_exact": "敏感性B：精确给药时间优先",
        "v03_noreperf": "敏感性C：只加降压药"}
res = c.set_index("stay_id")[["died", "end_h", "first_careunit", "last_careunit", "outtime"]].copy()
for v in VERS:
    res[v] = first_ready(v)
res = res.join(rp[["grade", "any", "any_lys"]])
res.reset_index().to_csv(WORK / "readiness_v03_by_stay.csv", index=False)

res["truncated"] = res.end_h > MAXH
alive = res[~res.died & ~res.truncated]
dead = res[res.died & ~res.truncated]


def summarise(col, df):
    r = df[col].notna()
    delay = (df.end_h - df[col])[r]
    return {"人数": len(df), "达到%": round(r.mean() * 100, 1),
            "入ICU后8小时内达到%": round((df[col] <= 8).mean() * 100, 1),
            "readiness时间中位数(h)": round(df.loc[r, col].median(), 1),
            "延迟中位数(h)": round(delay.median(), 1),
            "延迟四分位(h)": f"{delay.quantile(.25):.1f}–{delay.quantile(.75):.1f}",
            "延迟>24h%": round((delay > 24).mean() * 100, 1)}


t1 = pd.DataFrame({VERS[v]: summarise(v, alive) for v in VERS}).T
t1["ICU内死亡者达到%"] = [round(dead[v].notna().mean() * 100, 1) for v in VERS]

alive = alive.assign(grp=np.where(alive.last_careunit.isin(["Neuro Intermediate", "Neuro Stepdown"]),
                                  "神经中间监护/过渡病房", "其他ICU"))
rows = []
for u in ["Neuro Intermediate", "Neuro Stepdown", "Surgical Intensive Care Unit (SICU)",
          "Neuro Surgical Intensive Care Unit (Neuro SICU)", "Trauma SICU (TSICU)",
          "Medical Intensive Care Unit (MICU)"]:
    x = alive[alive.last_careunit == u]
    rows.append({"单元": u, "人数": len(x),
                 "v0.2 8小时内达到%": round((x.v02 <= 8).mean() * 100, 1),
                 "v0.3 8小时内达到%": round((x.v03 <= 8).mean() * 100, 1),
                 "v0.2 延迟中位数(h)": round((x.end_h - x.v02).median(), 1),
                 "v0.3 延迟中位数(h)": round((x.end_h - x.v03).median(), 1)})
t2 = pd.DataFrame(rows).set_index("单元")

# ICU 时间分解：治疗（到 v0.2）+ 监护代理（v0.2 到 v0.3）+ 其余（v0.3 到离开）
def decomp(df):
    e = df.end_h
    a = df.v02.fillna(e).clip(upper=e)
    b = df.v03.fillna(e).clip(upper=e).clip(lower=a)
    tot = e.sum()
    return {"人数": len(df), "ICU总时间(人·天)": round(tot / 24),
            "到v0.2(器官支持)%": round(a.sum() / tot * 100, 1),
            "v0.2到v0.3(降压/再灌注监护)%": round((b - a).sum() / tot * 100, 1),
            "v0.3之后(其余监护+等待)%": round((e - b).sum() / tot * 100, 1)}


t3 = pd.DataFrame({g: decomp(alive[alive.grp == g]) for g in ["其他ICU", "神经中间监护/过渡病房"]} |
                  {"合计": decomp(alive)}).T

t4 = pd.crosstab(res.grade.replace("", "无再灌注"), res.any_lys.map({True: "有溶栓证据", False: "仅取栓/无"}),
                 margins=True, margins_name="合计")

LAB = {"R_unmet": "呼吸", "C_unmet": "升压药", "CBP_unmet": "降压药泵入", "N_unk0": "意识：尚无GCS记录",
       "N_gap": "意识：GCS间隔>6h", "N_drop": "意识：确认的GCS下降", "N_sed": "意识：持续镇静",
       "N_rass": "意识：躁动", "N_seiz": "意识：癫痫", "M_unmet": "代谢", "P_unmet": "再灌注监护期"}
Gi = G.set_index(["stay_id", "k"])
r_ok = alive[alive.v03.notna()]
prev = [(s, int(k) - 7) for s, k in r_ok.v03.items() if int(k) - 7 >= 0]
t5 = (Gi.loc[prev, list(LAB)].mean() * 100).round(1).rename(index=LAB).rename("v0.3最后才满足的条件%").to_frame()
t5.loc["（第一个整点就开始稳定、无前一整点）"] = round((r_ok.v03 - 7 < 0).mean() * 100, 1)

nr = alive[alive.v03.isna()]
lastk = G.groupby("stay_id").tail(1).set_index("stay_id")
t6 = (lastk.loc[nr.index, list(LAB)].mean() * 100).round(1).rename(index=LAB)
t6 = t6.rename(f"始终未达到v0.3的活着转出者（{len(nr)}人）离开时未满足%").to_frame()
datamiss = lastk.loc[nr.index]
only_missing = (datamiss.N_unk0 | datamiss.N_gap) & ~(datamiss.R_unmet | datamiss.C_unmet | datamiss.CBP_unmet |
                datamiss.N_drop | datamiss.N_sed | datamiss.N_rass | datamiss.N_seiz | datamiss.M_unmet | datamiss.P_unmet)
t6.loc["其中：只因GCS记录缺失（数据缺失）"] = round(only_missing.mean() * 100, 1)

st = pd.read_csv(WORK / "states_full_stay.csv", usecols=["stay_id", "window_idx", "state"])
rr = r_ok.v03.rename("k").reset_index().assign(window_idx=lambda x: (x.k // 6).astype(int))
t7 = (rr.merge(st, on=["stay_id", "window_idx"], how="left").state.value_counts(normalize=True) * 100
      ).round(1).rename("v0.3 readiness时点的状态%").to_frame()


def md(df):
    df = df.to_frame() if isinstance(df, pd.Series) else df
    head = [str(df.index.name or "")] + [str(x) for x in df.columns]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join([str(i)] + ["" if pd.isna(v) else str(v) for v in r]) + " |"
            for i, r in zip(df.index, df.values)]
    return "\n".join(out)


# readiness 后 24 小时内恢复持续降压药泵入（记录空档 vs 真实停泵的检查）
bpi = inp[inp.itemid.isin(BP_IT) & inp.ordercategorydescription.str.contains("Continuous", na=False)].copy()
bpi["sh"] = rel(bpi, "starttime")
rr3 = r_ok.v03.rename("rk").reset_index()
mm = rr3.merge(bpi[["stay_id", "sh"]], on="stay_id", how="left")
resumed = mm[(mm.sh > mm.rk) & (mm.sh <= mm.rk + 24)].stay_id.nunique()
had_bp = rr3.stay_id.isin(set(bpi.stay_id)).sum()
t8 = pd.DataFrame({"人数": [len(rr3), had_bp, resumed]},
                  index=["活着转出且达到v0.3", "其中曾用降压药泵入", "readiness后24小时内又开始降压药泵入"])

# ICU 时间的互斥分解（逐小时按当时的实际条件状态，不用首次 readiness 时间）
#   T1：v0.2 条件未满足（器官支持、意识、代谢；含 GCS 未知）
#   T2：v0.2 满足，但有持续静脉降压药或处于再灌注监护期
#   T3：两者都满足（其余：不可观测的监护需求与可能的等待）
Gh = G[G.stay_id.isin(set(alive.index))].merge(
    alive[["grp"]].reset_index(), on="stay_id")
Gh["seg"] = np.select([~Gh.v02, Gh.v02 & ~Gh.v03], ["T1", "T2"], default="T3")
only_gcs = ~Gh.v02 & (Gh.N_unk0 | Gh.N_gap) & ~(Gh.R_unmet | Gh.C_unmet | Gh.M_unmet | Gh.N_drop | Gh.N_sed |
                                                  Gh.N_rass | Gh.N_seiz)
t9 = (pd.crosstab(Gh.grp, Gh.seg, normalize="index") * 100).round(1)
t9.loc["合计"] = (Gh.seg.value_counts(normalize=True) * 100).round(1)
t9["小时数"] = list(Gh.groupby("grp").size()) + [len(Gh)]
t9["T1中仅因GCS记录缺失%"] = list((only_gcs.groupby(Gh.grp).sum() / (Gh.seg == "T1").groupby(Gh.grp).sum() * 100
                                 ).round(1)) + [round(only_gcs.sum() / (Gh.seg == "T1").sum() * 100, 1)]
# 首次 readiness 之后又失去 v0.3 条件的小时
after = Gh.merge(alive.v03.rename("rk").reset_index(), on="stay_id")
after = after[after.rk.notna() & (after.k > after.rk)]
t9_note = round((~after.v03).mean() * 100, 1)

for name, t in [("time_decomposition_hourly", t9), ("bp_resumption", t8), ("overview", t1), ("by_unit", t2), ("time_decomposition", t3), ("reperfusion_grades", t4),
                ("last_condition", t5), ("never_ready", t6), ("state", t7)]:
    t.to_csv(OUT / f"s7_{name}.csv")
lines = ["# readiness v0.3 计算结果（与 v0.2 比较）", "",
         f"活着转出 {len(alive):,}，ICU 内死亡 {len(dead):,}（超过 30 天截断者不计）。延迟 = 离开 ICU − readiness。", "",
         "## 1. 总览", "", md(t1), "", "## 2. 按转出前单元", "", md(t2), "",
         "## 3. ICU 时间分解（活着转出者）", "", md(t3), "",
         "## 4. 再灌注时间证据分级（全队列，人数）", "", md(t4), "",
         "## 5. v0.3 最后才满足的条件（稳定开始前一个整点未满足；可多选）", "", md(t5), "",
         "## 6. 始终未达到 v0.3 的活着转出者", "", md(t6), "",
         "## 7. v0.3 readiness 时点的状态", "", md(t7), "",
         "## 8. readiness 后又恢复降压药泵入", "", md(t8), "",
         "## 9. ICU 时间的互斥分解（逐小时，活着转出者，%）", "", md(t9), "",
         f"首次达到 v0.3 之后的时间里，有 {t9_note}% 的小时 v0.3 条件又不满足。", ""]
(OUT / "s7_summary.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
