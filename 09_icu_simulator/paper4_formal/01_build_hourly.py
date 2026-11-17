"""
第四篇（ICU 时间论文）· 正式分析 · 第 1 步：逐小时条件表

按《ICU时间论文_分析方案_v1.0.md》（已锁定）。readiness 的规则与探索阶段的
06_readiness_v03.py 完全相同（规则说明见该脚本），本脚本只在此基础上：
  - 加入敏感性分析需要的变体：GCS 有效期 4 / 8 小时（S6）、GCS 下降 ≥ 3 分（S4）、
    睡眠呼吸暂停 CPAP（S5）、再灌注规则三种变体（S3）
  - 加入分层变量：该小时所在的 ICU 单元（transfers 表）、年代（anchor_year_group）、性别、年龄
  - 主分析人群为全部病人（含 ICU 内死亡者）

输出（个体层面，仅本机）：paper4_formal/work/hourly.pkl、patients.pkl
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
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
# 逐小时条件表（保存各组成部分，由 02 组合成主分析和各敏感性分析的口径）
# ---------------------------------------------------------------------------
print("组装逐小时表 ...", flush=True)
gap4 = last.loc[last.h.notna() & (last.hh - last.h > 4), ["stay_id", "k"]]
gap8 = last.loc[last.h.notna() & (last.hh - last.h > 8), ["stay_id", "k"]]
drop3_m = recent_event(drops3.rename(columns={"h": "hx"}).assign(
    hx_t=lambda x: x.stay_id.map(it) + pd.to_timedelta(x.hx, unit="h")), "hx_t", 6)
COMP = {"R": R, "R_osa": R_osa, "C": C, "CBP": CBP, "M": M, "L2": L2,
        "P": P, "P_lys": P_lys, "P_exact": P_exact,
        "N_unk0": flag("x", unk0), "N_gap6": flag("x", gap), "N_gap4": flag("x", gap4), "N_gap8": flag("x", gap8),
        "N_drop2": flag("x", drop2_m), "N_drop3": flag("x", drop3_m), "N_sed": flag("x", sed),
        "N_rass": flag("x", rass), "N_seiz": flag("x", seiz)}
H_ = grid[["stay_id", "k"]].copy()
for name, f in COMP.items():
    f = f[["stay_id", "k"]].copy()
    f["k"] = f.k.astype(int)
    f[name] = True
    H_ = H_.merge(f.drop_duplicates(["stay_id", "k"]), on=["stay_id", "k"], how="left")
for name in COMP:
    H_[name] = H_[name].fillna(False).astype(bool)

# 该小时所在的单元（transfers 表）
tr = pd.read_csv(f"{B}/hosp/transfers.csv.gz", usecols=["hadm_id", "careunit", "intime", "outtime"],
                 parse_dates=["intime", "outtime"])
tr = tr[tr.hadm_id.isin(hadm) & tr.careunit.notna()].merge(c[["hadm_id", "stay_id"]], on="hadm_id")
tr["a"] = rel(tr.rename(columns={"intime": "_s"}), "_s")
tr["b"] = rel(tr.rename(columns={"outtime": "_e"}), "_e")
tr = tr.sort_values("a")
u = pd.merge_asof(H_.assign(kk=H_.k.astype(float)).sort_values("kk")[["stay_id", "k", "kk"]],
                  tr[["stay_id", "a", "b", "careunit"]], left_on="kk", right_on="a", by="stay_id",
                  direction="backward")
u.loc[~(u.kk < u.b), "careunit"] = np.nan
H_ = H_.merge(u[["stay_id", "k", "careunit"]], on=["stay_id", "k"], how="left")
INTER = ["Neuro Intermediate", "Neuro Stepdown"]
ICU_LIKE = set(pd.read_csv(f"{B}/icu/icustays.csv.gz", usecols=["first_careunit"]).first_careunit) - set(INTER)
H_["unit_type"] = np.select([H_.careunit.isin(INTER), H_.careunit.isin(ICU_LIKE)],
                            ["神经中间监护/过渡病房", "正规ICU"], default="未匹配")
H_.to_pickle(WORK / "hourly.pkl")

# 病人表
pat = pd.read_csv(f"{B}/hosp/patients.csv.gz", usecols=["subject_id", "anchor_year_group"])
P_ = c[["stay_id", "subject_id", "hadm_id", "stroke_subtype", "gender", "age_at_adm_capped", "intime", "outtime",
        "end_h", "K", "died", "first_careunit", "last_careunit"]].merge(pat, on="subject_id", how="left")
P_["subtype"] = P_.stroke_subtype.replace({"ICH+SAH": "ICH"})
P_ = P_.merge(rp[["grade", "any", "any_lys"]].rename(columns={"any": "reperfusion", "any_lys": "thrombolysis"}),
              left_on="stay_id", right_index=True, how="left")
P_["truncated"] = P_.end_h > MAXH
P_.to_pickle(WORK / "patients.pkl")
print(f"完成：{len(H_):,} 个小时，{len(P_):,} 人；未匹配单元的小时 {(H_.unit_type == '未匹配').mean():.2%}")
