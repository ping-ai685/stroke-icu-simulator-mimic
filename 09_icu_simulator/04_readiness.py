"""
ICU 路径模拟器 · 按操作性定义 v0.2 计算 ICU 转出准备状态（readiness）

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
INP_IT = V.VASOPRESSORS + V.SEDATIVES + [223258, 225161, 228341, 227531, 225168, 225170, 220970]
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
# 合并与 readiness
# ---------------------------------------------------------------------------
G = grid[["stay_id", "k"]].copy()
for f in [R, R_osa, C, N, N3, M, L2]:
    f["k"] = f.k.astype(int)
    G = G.merge(f, on=["stay_id", "k"], how="left")
G = G.fillna(False).sort_values(["stay_id", "k"]).reset_index(drop=True)
for col in ["R_unmet", "R_unmet_osa", "C_unmet", "N_unmet", "N_unmet_g3", "M_unmet", "L2_unmet"]:
    G[col] = G[col].astype(bool)
G["phys"] = ~(G.R_unmet | G.C_unmet | G.N_unmet | G.M_unmet)
G["full"] = G.phys & ~G.L2_unmet
G["phys_g3"] = ~(G.R_unmet | G.C_unmet | G.N_unmet_g3 | G.M_unmet)
G["phys_osa"] = ~(G.R_unmet_osa | G.C_unmet | G.N_unmet | G.M_unmet)


def first_ready(col, dur):
    """每人第一个 k，使 k−dur … k 全部满足；返回 Series(stay_id -> k)。"""
    x = G[["stay_id", "k", col]]
    grp = (x[col] != x.groupby("stay_id")[col].shift()).cumsum()
    run = x.groupby([x.stay_id, grp]).cumcount()
    ok = x[col] & (run >= dur)
    return x[ok].groupby("stay_id").k.first()


res = c[["stay_id", "died", "end_h", "K", "first_careunit", "last_careunit", "outtime"]].set_index("stay_id")
specs = {"phys_6h": ("phys", 6), "full_6h": ("full", 6), "phys_4h": ("phys", 4), "phys_12h": ("phys", 12),
         "phys_24h": ("phys", 24), "phys_gcs3": ("phys_g3", 6), "phys_osa": ("phys_osa", 6)}
for name, (col, dur) in specs.items():
    res[name] = first_ready(col, dur)
res.reset_index().to_csv(WORK / "readiness_by_stay.csv", index=False)

# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
res["truncated"] = res.end_h > MAXH
alive = res[~res.died & ~res.truncated]
dead = res[res.died & ~res.truncated]


def summarise(col, df):
    r = df[col].notna()
    delay = (df.end_h - df[col])[r]
    return {"人数": len(df), "达到readiness%": round(r.mean() * 100, 1),
            "readiness时间中位数(h)": round(df.loc[r, col].median(), 1),
            "延迟中位数(h)": round(delay.median(), 1),
            "延迟四分位(h)": f"{delay.quantile(.25):.1f}–{delay.quantile(.75):.1f}",
            "延迟≤6h%": round((delay <= 6).mean() * 100, 1),
            "延迟>24h%": round((delay > 24).mean() * 100, 1)}


t1 = pd.DataFrame({f"{k} · 活着转出": summarise(k, alive) for k in specs}).T
t1["ICU内死亡者达到readiness%"] = [round(dead[k].notna().mean() * 100, 1) for k in specs]
t1.to_csv(OUT / "s5_readiness_overview.csv")

top = alive.last_careunit.value_counts()
top = top[top >= 100].index
t2 = pd.DataFrame({u: summarise("phys_6h", alive[alive.last_careunit == u]) for u in top}).T
t2["完整readiness延迟中位数(h)"] = [round((alive.end_h - alive.full_6h)[alive.last_careunit == u].median(), 1)
                             for u in top]
t2.to_csv(OUT / "s5_by_unit.csv")

r_ok = alive[alive.phys_6h.notna()]
rt = r_ok.index.to_series().map(it) + pd.to_timedelta(r_ok.phys_6h, unit="h")
t3 = pd.DataFrame({"达到readiness的时刻": rt.dt.hour.value_counts().reindex(range(24), fill_value=0),
                   "实际转出的时刻": r_ok.outtime.dt.hour.value_counts().reindex(range(24), fill_value=0)})
t3.index.name = "时(点)"
t3.to_csv(OUT / "s5_hour_of_day.csv")

# 卡在哪一条：最后一段稳定开始前一个整点，哪几条未满足（活着转出且达到 readiness）
Gi = G.set_index(["stay_id", "k"])
rows = []
for sid, k in r_ok.phys_6h.items():
    k0 = int(k) - 6 - 1
    if k0 >= 0:
        r = Gi.loc[(sid, k0)]
        rows.append({"呼吸": r.R_unmet, "循环": r.C_unmet, "意识": r.N_unmet, "代谢": r.M_unmet})
bott = pd.DataFrame(rows)
t4 = (bott.mean() * 100).round(1).rename("最后才满足的条件%").to_frame()
t4.loc["入ICU即稳定(前6小时内达到)"] = round((r_ok.phys_6h <= 6).mean() * 100, 1)
t4.to_csv(OUT / "s5_last_condition.csv")

# 活着转出但始终未达到：转出时哪几条未满足
nr = alive[alive.phys_6h.isna()]
ex = G.groupby("stay_id").tail(1).set_index("stay_id").loc[nr.index]
t5 = (ex[["R_unmet", "C_unmet", "N_unmet", "M_unmet"]].mean() * 100).round(1)
t5.index = ["呼吸", "循环", "意识", "代谢"]
t5 = t5.rename(f"未达到readiness的活着转出者（{len(nr)}人）转出时未满足%").to_frame()
t5.to_csv(OUT / "s5_never_ready_reasons.csv")

# 与状态的对应：readiness 时点所在 6 小时窗口的状态
st = pd.read_csv(WORK / "states_full_stay.csv", usecols=["stay_id", "window_idx", "state"])
rr = r_ok.phys_6h.rename("k").reset_index()
rr["window_idx"] = (rr.k // 6).astype(int)
rr = rr.merge(st, on=["stay_id", "window_idx"], how="left")
t6 = (rr.state.value_counts(normalize=True) * 100).round(1).rename("readiness时点的状态%").to_frame()
# 各状态窗口末整点的生理条件满足比例
ge = G[(G.k % 6 == 5)].assign(window_idx=lambda x: x.k // 6).merge(st, on=["stay_id", "window_idx"])
t6["该状态窗口中生理条件满足%"] = (ge.groupby("state").phys.mean() * 100).round(1)
t6.to_csv(OUT / "s5_state_concordance.csv")


def md(df):
    df = df.to_frame() if isinstance(df, pd.Series) else df
    head = [str(df.index.name or "")] + [str(x) for x in df.columns]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join([str(i)] + ["" if pd.isna(v) else str(v) for v in r]) + " |"
            for i, r in zip(df.index, df.values)]
    return "\n".join(out)


lines = ["# ICU 转出准备状态（readiness）计算结果", "",
         f"- 病人 {len(res):,}（活着转出 {len(alive):,}，ICU 内死亡 {len(dead):,}，"
         f"超过 30 天截断 {int(res.truncated.sum())}）；有睡眠呼吸暂停 CPAP 医嘱者 {len(osa_stays)}",
         "- 延迟 = 实际离开 ICU 时间 − readiness 确认时间（只在达到 readiness 的病人中计算）", "",
         "## 1. 总览", "", md(t1), "",
         "## 2. 按转出前所在单元（生理 readiness）", "", md(t2), "",
         "## 3. 一天中的时刻：达到 readiness vs 实际转出（活着转出且达到者）", "", md(t3), "",
         "## 4. 最后才满足的条件（稳定开始前一个整点未满足的比例；可多选）", "", md(t4), "",
         "## 5. 活着转出但始终未达到 readiness 的病人，转出时未满足的条件", "", md(t5), "",
         "## 6. 与四状态的对应", "", md(t6), ""]
(OUT / "s5_summary.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
