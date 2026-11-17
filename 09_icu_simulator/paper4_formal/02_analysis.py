"""
第四篇（ICU 时间论文）· 正式分析 · 第 2 步：按分析方案 v1.0 逐节计算

输入：work/hourly.pkl、work/patients.pkl（01_build_hourly.py）
输出：outputs/*.csv 与 outputs/formal_results.md（只有汇总数字）

各节编号与《ICU时间论文_分析方案_v1.0.md》一致。
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from lifelines import AalenJohansenFitter

HERE = Path(__file__).resolve().parent
OUT, WORK = HERE / "outputs", HERE / "work"
RNG = np.random.default_rng(20261106)
NBOOT = 1000

H = pd.read_pickle(WORK / "hourly.pkl")
P = pd.read_pickle(WORK / "patients.pkl").set_index("stay_id")
H = H.merge(P[["subtype", "anchor_year_group", "reperfusion", "died", "truncated", "last_careunit"]],
            left_on="stay_id", right_index=True)
H["phase"] = pd.cut(H.k, [-1, 23, 71, 167, 335, 720],
                    labels=["第1天", "第2–3天", "第4–7天", "第8–14天", "第15–30天"])
SEGS = ["T1", "T2", "T3"]
SUBTYPE = {"AIS": "缺血性卒中", "ICH": "脑出血", "SAH": "蛛网膜下腔出血"}


# ---------------------------------------------------------------------------
# 口径组合
# ---------------------------------------------------------------------------
def unmet(h, gap="N_gap6", drop="N_drop2", R="R", P_="P", reperf=True):
    N = h.N_unk0 | h[gap] | h[drop] | h.N_sed | h.N_rass | h.N_seiz
    v02 = h[R] | h.C | N | h.M
    extra = h.CBP | (h[P_] if reperf else False)
    return v02, extra


def segment(h, insufficient=False, v02_only=False, **kw):
    v02, extra = unmet(h, **kw)
    if v02_only:
        seg = np.where(v02, "T1", "T3")
    else:
        seg = np.select([v02, extra], ["T1", "T2"], default="T3")
    if insufficient:
        gcs_only = (h.N_unk0 | h[kw.get("gap", "N_gap6")]) & ~(h[kw.get("R", "R")] | h.C | h.M | h[kw.get("drop", "N_drop2")]
                                                            | h.N_sed | h.N_rass | h.N_seiz)
        seg = np.where(v02 & gcs_only, "T0", seg)
    return pd.Series(seg, index=h.index)


def first_ready(h, met, dur=6):
    x = pd.DataFrame({"stay_id": h.stay_id.values, "k": h.k.values, "m": np.asarray(met)})
    grp = (x.m != x.groupby("stay_id").m.shift()).cumsum()
    run = x.groupby([x.stay_id, grp]).cumcount()
    return x[x.m & (run >= dur)].groupby("stay_id").k.first()


# ---------------------------------------------------------------------------
# 分解：小时口径 + 病人口径，以病人为单位的自助法
# ---------------------------------------------------------------------------
def decomp(h, seg, segs=SEGS, boot=True):
    cnt = pd.crosstab(h.stay_id, seg).reindex(columns=segs, fill_value=0)
    m = cnt.to_numpy(float)
    tot = m.sum(1)
    share = m / tot[:, None]
    out = {"人数": len(cnt), "小时数": int(tot.sum())}
    hour = m.sum(0) / tot.sum()
    if boot:
        idx = RNG.integers(0, len(m), size=(NBOOT, len(m)))
        bh = np.stack([m[i].sum(0) / m[i].sum() for i in idx])
        bp = np.stack([share[i].mean(0) for i in idx])
    for j, s in enumerate(segs):
        ci = f" ({np.percentile(bh[:, j], 2.5) * 100:.1f}–{np.percentile(bh[:, j], 97.5) * 100:.1f})" if boot else ""
        out[f"{s} 小时口径%"] = f"{hour[j] * 100:.1f}{ci}"
        q = np.percentile(share[:, j], [25, 50, 75]) * 100
        cim = f" ({np.percentile(bp[:, j], 2.5) * 100:.1f}–{np.percentile(bp[:, j], 97.5) * 100:.1f})" if boot else ""
        out[f"{s} 病人口径 均值%"] = f"{share[:, j].mean() * 100:.1f}{cim}"
        out[f"{s} 病人口径 中位(四分位)%"] = f"{q[1]:.1f} ({q[0]:.1f}–{q[2]:.1f})"
    return out, hour


def strat(h, seg, col, order=None, boot=True):
    rows = {}
    keys = order if order is not None else sorted(h[col].dropna().unique())
    for k in keys:
        m = (h[col] == k).to_numpy()
        if m.sum() == 0:
            continue
        rows[k] = decomp(h[m], seg[m], boot=boot)[0]
    return pd.DataFrame(rows).T


seg_main = segment(H)
tables = {}

# ---------------------------------------------------------------------------
# 研究流程与表 1
# ---------------------------------------------------------------------------
flow = pd.DataFrame({"人数": [6368, 6368 - len(P), len(P), int(P.truncated.sum()), int(P.died.sum())]},
                    index=["第一篇队列", "排除：死亡时间不晚于入 ICU", "分析人群", "其中超过 30 天截断",
                           "其中 ICU 内死亡"])
t1 = []
for st in ["AIS", "ICH", "SAH", "全部"]:
    x = P if st == "全部" else P[P.subtype == st]
    t1.append({"卒中类型": SUBTYPE.get(st, st), "人数": len(x),
               "年龄 中位(四分位)": f"{x.age_at_adm_capped.median():.0f} ({x.age_at_adm_capped.quantile(.25):.0f}–{x.age_at_adm_capped.quantile(.75):.0f})",
               "女性%": round((x.gender == "F").mean() * 100, 1),
               "ICU 停留 中位(四分位) 小时": f"{x.end_h.median():.0f} ({x.end_h.quantile(.25):.0f}–{x.end_h.quantile(.75):.0f})",
               "ICU 内死亡%": round(x.died.mean() * 100, 1),
               "再灌注治疗%": round(x.reperfusion.fillna(False).astype(bool).mean() * 100, 1),
               "首个单元为中间监护/过渡%": round(x.first_careunit.isin(["Neuro Intermediate", "Neuro Stepdown"]).mean() * 100, 1)})
tables["table1"] = pd.DataFrame(t1).set_index("卒中类型")

# ---------------------------------------------------------------------------
# 5.1 主分析
# ---------------------------------------------------------------------------
overall, overall_hour = decomp(H, seg_main)
tables["5_1_overall"] = pd.DataFrame({"全部病人": overall}).T
tables["5_1_by_outcome"] = strat(H.assign(o=np.where(H.died, "ICU内死亡", "活着转出")), seg_main, "o")
tables["5_1_by_subtype"] = strat(H, seg_main, "subtype", ["AIS", "ICH", "SAH"]).rename(index=SUBTYPE)
tables["5_1_by_unit_type"] = strat(H, seg_main, "unit_type", ["正规ICU", "神经中间监护/过渡病房", "未匹配"])
big_units = H.careunit.value_counts()
big_units = [u for u in big_units.index if H.loc[H.careunit == u, "stay_id"].nunique() >= 100]
tables["5_1_by_unit"] = strat(H, seg_main, "careunit", big_units)
PH = ["第1天", "第2–3天", "第4–7天", "第8–14天", "第15–30天"]
tables["5_1_by_phase"] = strat(H, seg_main, "phase", PH)
tables["5_1_by_era"] = strat(H, seg_main, "anchor_year_group")
ais = H.subtype == "AIS"
tables["5_1_by_reperfusion_AIS"] = strat(H[ais].assign(r=np.where(H[ais].reperfusion.fillna(False), "再灌注", "无再灌注")),
                                         seg_main[ais], "r")
cross = {}
for ut in ["正规ICU", "神经中间监护/过渡病房"]:
    for ph in PH:
        m = ((H.unit_type == ut) & (H.phase == ph)).to_numpy()
        if m.sum():
            r, _ = decomp(H[m], seg_main[m])
            cross[(ut, ph)] = {k: v for k, v in r.items() if "小时口径" in k or k in ("人数", "小时数")}
tables["5_1_unit_x_phase"] = pd.DataFrame(cross).T

# ---------------------------------------------------------------------------
# 5.2 重复性检验
# ---------------------------------------------------------------------------
def hour_share(mask, s="T3"):
    x = seg_main[mask]
    return (x == s).mean() * 100


eras = sorted(H.anchor_year_group.dropna().unique())
ov_t3 = (seg_main == "T3").mean() * 100
ov_unit = hour_share(H.unit_type == "神经中间监护/过渡病房") > hour_share(H.unit_type == "正规ICU")
ov_phase = np.sign(np.diff([hour_share(H.phase == p) for p in PH]))
rows = []
for e in eras:
    me = H.anchor_year_group == e
    t3 = hour_share(me)
    u = hour_share(me & (H.unit_type == "神经中间监护/过渡病房")) > hour_share(me & (H.unit_type == "正规ICU"))
    ph = np.sign(np.diff([hour_share(me & (H.phase == p)) for p in PH]))
    rows.append({"年代": e + ("（新冠时期）" if e.startswith("2020") else ""), "人数": H.loc[me, "stay_id"].nunique(),
                 "T3 小时口径%": round(t3, 1), "与全队列之差(百分点)": round(t3 - ov_t3, 1),
                 "条件1 差≤5": abs(t3 - ov_t3) <= 5, "条件2 单元顺序相同": u == ov_unit,
                 "条件3 阶段方向相同": bool((ph == ov_phase).all()),
                 "阶段方向": " ".join("↑" if v > 0 else "↓" if v < 0 else "→" for v in ph)})
rep = pd.DataFrame(rows).set_index("年代")
rep.loc["全队列", ["T3 小时口径%", "阶段方向"]] = [round(ov_t3, 1),
                                               " ".join("↑" if v > 0 else "↓" if v < 0 else "→" for v in ov_phase)]
tables["5_2_reproducibility"] = rep
stable = {c: bool(rep.loc[rep.index != "全队列", c].all()) for c in ["条件1 差≤5", "条件2 单元顺序相同", "条件3 阶段方向相同"]}

# ---------------------------------------------------------------------------
# 5.3 分数响应模型（描述性）
# ---------------------------------------------------------------------------
cnt = pd.crosstab(H.stay_id, seg_main).reindex(columns=SEGS, fill_value=0)
pp = P.join((cnt.T3 / cnt.sum(1)).rename("t3"), how="inner")
main_unit = H.groupby("stay_id").unit_type.agg(lambda s: s.value_counts().index[0])
pp = pp.join(main_unit.rename("main_unit"))
X = pd.DataFrame({
    "脑出血(对缺血性)": (pp.subtype == "ICH").astype(float),
    "蛛网膜下腔出血(对缺血性)": (pp.subtype == "SAH").astype(float),
    "年龄(每10岁)": pp.age_at_adm_capped / 10,
    "女性": (pp.gender == "F").astype(float),
    "主要在中间监护/过渡病房": (pp.main_unit == "神经中间监护/过渡病房").astype(float),
    "再灌注治疗": pp.reperfusion.fillna(False).astype(float),
    "ICU停留(log小时)": np.log(pp.end_h.clip(lower=1)),
})
for e in eras[1:]:
    X[f"年代 {e}(对{eras[0]})"] = (pp.anchor_year_group == e).astype(float)
X = sm.add_constant(X)
fit = sm.GLM(pp.t3, X, family=sm.families.Binomial()).fit(cov_type="HC0")
ci = fit.conf_int()
tables["5_3_fractional_logit"] = pd.DataFrame({
    "比值比": np.exp(fit.params).round(2), "95%CI": [f"{np.exp(a):.2f}–{np.exp(b):.2f}" for a, b in ci.values],
    "P": fit.pvalues.map(lambda v: "<0.001" if v < 0.001 else f"{v:.3f}")}).drop(index="const")

# ---------------------------------------------------------------------------
# 5.4 准备状态的时间
# ---------------------------------------------------------------------------
v02u, extra = unmet(H)
r02 = first_ready(H, ~v02u)
r03 = first_ready(H, ~(v02u | extra))
P["r02"], P["r03"] = r02, r03


def cif(rcol, groups):
    rows = {}
    for gname, gm in groups.items():
        x = P[gm]
        t = x[rcol].where(x[rcol].notna(), x.end_h.clip(upper=720))
        ev = np.where(x[rcol].notna(), 1, np.where(x.died & ~x.truncated, 2, 0))
        aj = AalenJohansenFitter(calculate_variance=False, seed=1)
        aj.fit(np.asarray(t, float), np.asarray(ev, int), event_of_interest=1)
        cf = aj.cumulative_density_
        rows[gname] = {f"{h}小时": round(float(cf[cf.index <= h].iloc[-1, 0]) * 100, 1) if (cf.index <= h).any() else 0.0
                       for h in [8, 24, 48, 72, 168]}
        rows[gname]["人数"] = len(x)
    return pd.DataFrame(rows).T


G = {"全部": P.index == P.index}
G.update({SUBTYPE[s]: (P.subtype == s).to_numpy() for s in ["AIS", "ICH", "SAH"]})
G["首个单元 正规ICU"] = ~P.first_careunit.isin(["Neuro Intermediate", "Neuro Stepdown"]).to_numpy()
G["首个单元 中间监护/过渡"] = P.first_careunit.isin(["Neuro Intermediate", "Neuro Stepdown"]).to_numpy()
tables["5_4_cif_v02"] = cif("r02", G)
tables["5_4_cif_v03"] = cif("r03", G)
al = P[~P.died & ~P.truncated & P.r03.notna()]
dl = al.end_h - al.r03
tables["5_4_delay_alive"] = pd.DataFrame({"活着转出且达到v0.3": {
    "人数": len(al), "中位(四分位)小时": f"{dl.median():.1f} ({dl.quantile(.25):.1f}–{dl.quantile(.75):.1f})",
    "≤6小时%": round((dl <= 6).mean() * 100, 1), "6–24小时%": round(dl.between(6, 24, inclusive="right").mean() * 100, 1),
    ">24小时%": round((dl > 24).mean() * 100, 1)}}).T
both = P[P.r02.notna() & P.r03.notna()]
dd = both.r03 - both.r02
tables["5_4_v02_v03_gap"] = pd.DataFrame({"两者都达到": {
    "人数": len(both), "差为0%": round((dd == 0).mean() * 100, 1),
    "差 中位(四分位)小时": f"{dd.median():.0f} ({dd.quantile(.25):.0f}–{dd.quantile(.75):.0f})"}}).T

# ---------------------------------------------------------------------------
# 5.5 达到后再次失去
# ---------------------------------------------------------------------------
H["u03"] = (v02u | extra).to_numpy()
aft = H.merge(P.r03.rename("rk").dropna(), left_on="stay_id", right_index=True)
aft = aft[aft.k > aft.rk]
hour_rel = aft.u03.mean() * 100
a2 = aft[["stay_id", "k", "u03"]].copy()
grp = (a2.u03 != a2.groupby("stay_id").u03.shift()).cumsum()
runlen = a2.groupby([a2.stay_id, grp]).u03.transform("size")
pat_rel = a2[a2.u03 & (runlen >= 6)].stay_id.nunique() / P.r03.notna().sum() * 100
rel = aft[aft.u03]
REASON = {"R": "呼吸", "C": "升压药", "CBP": "降压药泵入", "N_unk0": "意识：尚无GCS记录", "N_gap6": "意识：GCS间隔>6h",
          "N_drop2": "意识：确认的GCS下降", "N_sed": "意识：持续镇静", "N_rass": "意识：躁动", "N_seiz": "意识：癫痫",
          "M": "代谢", "P": "再灌注监护期"}
tables["5_5_relapse"] = pd.concat([
    pd.DataFrame({"值": [round(hour_rel, 1), round(pat_rel, 1)]},
                 index=["首次达到v0.3后，条件不满足的小时%（小时比例）",
                        "首次达到v0.3后，至少一次连续≥6小时不满足的病人%"]),
    (rel[list(REASON)].mean() * 100).round(1).rename(index=REASON).rename("值").to_frame()
    .rename(index=lambda s: f"不满足的小时中：{s}%")])

# ---------------------------------------------------------------------------
# 5.6 一天中的时刻
# ---------------------------------------------------------------------------
ra = P[~P.died & ~P.truncated & P.r03.notna()]
rt = ra.intime + pd.to_timedelta(ra.r03, unit="h")
tables["5_6_hour_of_day"] = pd.DataFrame({
    "首次达到v0.3": rt.dt.hour.value_counts().reindex(range(24), fill_value=0),
    "实际离开ICU": ra.outtime.dt.hour.value_counts().reindex(range(24), fill_value=0)}).rename_axis("时")
alive = P[~P.died & ~P.truncated]
after = alive.outtime.dt.hour.map(lambda h: h >= 19 or h < 7)
ah = alive.assign(a=after).groupby("last_careunit").agg(人数=("a", "size"), 下班后离开ICU百分比=("a", "mean"))
ah = ah[ah.人数 >= 100]
ah["下班后离开ICU百分比"] = (ah.下班后离开ICU百分比 * 100).round(1)
ah.loc["全部"] = [len(alive), round(after.mean() * 100, 1)]
tables["5_6_after_hours"] = ah

# ---------------------------------------------------------------------------
# 6 敏感性分析
# ---------------------------------------------------------------------------
def brief(h, s, segs):
    r, _ = decomp(h, s, segs=segs, boot=False)
    return {k: v for k, v in r.items() if "小时口径" in k or "均值" in k}


sens = {"主分析": brief(H, seg_main, SEGS)}
sens["S1 GCS缺失单列为T0"] = brief(H, segment(H, insufficient=True), ["T0", "T1", "T2", "T3"])
sens["S2 只用v0.2"] = brief(H, segment(H, v02_only=True), ["T1", "T3"])
sens["S3a 再灌注只对溶栓"] = brief(H, segment(H, P_="P_lys"), SEGS)
sens["S3b 精确给药时间优先"] = brief(H, segment(H, P_="P_exact"), SEGS)
sens["S3c 不加再灌注规则"] = brief(H, segment(H, reperf=False), SEGS)
sens["S4 GCS下降≥3"] = brief(H, segment(H, drop="N_drop3"), SEGS)
sens["S5 睡眠呼吸暂停CPAP"] = brief(H, segment(H, R="R_osa"), SEGS)
sens["S6a GCS有效期4h"] = brief(H, segment(H, gap="N_gap4"), SEGS)
sens["S6b GCS有效期8h"] = brief(H, segment(H, gap="N_gap8"), SEGS)
m7 = (~H.died).to_numpy()
sens["S7 只分析活着转出者"] = brief(H[m7], seg_main[m7], SEGS)
m8 = (~H.truncated).to_numpy()
sens["S8 排除30天截断者"] = brief(H[m8], seg_main[m8], SEGS)
tables["6_sensitivity"] = pd.DataFrame(sens).T

INTER = ["Neuro Intermediate", "Neuro Stepdown"]
H9 = H.assign(ut9=np.where(H.last_careunit.isin(INTER), "神经中间监护/过渡病房", "正规ICU"))
s9 = pd.concat({"主分析（逐小时所在单元）": strat(H, seg_main, "unit_type", ["正规ICU", "神经中间监护/过渡病房"], boot=False),
                "S9（最后所在单元）": strat(H9, seg_main, "ut9", ["正规ICU", "神经中间监护/过渡病房"], boot=False)})
tables["6_S9_unit_by_last"] = s9[[c for c in s9.columns if "小时口径" in c or c == "人数"]]

s10 = {}
for dur in [4, 6, 12, 24]:
    r = first_ready(H, ~(v02u | extra), dur)
    s10[f"稳定{dur}小时"] = {"达到v0.3%": round(len(r) / len(P) * 100, 1), "达到时间中位(小时)": float(r.median())}
tables["6_S10_stability"] = pd.DataFrame(s10).T

# ---------------------------------------------------------------------------
# 方法说明：逐小时分解与按首次 readiness 切分的差异归因
# ---------------------------------------------------------------------------
old = np.where(H.k < H.stay_id.map(r02).fillna(np.inf), "T1",
               np.where(H.k < H.stay_id.map(r03).fillna(np.inf), "T2", "T3"))
after03 = H.k >= H.stay_id.map(r03).fillna(np.inf)
before03 = ~after03
gcs_only = (H.N_unk0 | H.N_gap6) & ~(H.R | H.C | H.M | H.N_drop2 | H.N_sed | H.N_rass | H.N_seiz)
att = pd.DataFrame({
    "小时数": [int(((old == "T3") & (seg_main != "T3")).sum()),
              int((((old == "T3") & (seg_main != "T3")) & gcs_only & v02u).sum()),
              int(((old != "T3") & (seg_main == "T3") & before03).sum())]},
    index=["旧法记为T3、新法记为T1/T2（首次达到后再次失去）",
           "　其中仅因GCS记录缺失",
           "旧法记为T1/T2、新法记为T3（首次达到之前，条件已满足但尚未连续满6小时或其后又中断）"])
att["占全部小时%"] = (att.小时数 / len(H) * 100).round(2)
tables["7_method_attribution"] = att
tables["7_method_crosstab"] = pd.crosstab(pd.Series(old, name="旧法"), pd.Series(seg_main.values, name="新法"))


# ---------------------------------------------------------------------------
def md(df):
    df = df.to_frame() if isinstance(df, pd.Series) else df
    idx = df.index.names
    head = [" / ".join(str(i) for i in idx if i) if len(idx) > 1 else str(df.index.name or "")] + list(map(str, df.columns))
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for i, r in zip(df.index, df.values):
        i = " / ".join(map(str, i)) if isinstance(i, tuple) else str(i)
        out.append("| " + " | ".join([i] + ["" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r]) + " |")
    return "\n".join(out)


for k, t in tables.items():
    t.to_csv(OUT / f"{k}.csv")
TITLES = {"table1": "表 1 病人特征", "5_1_overall": "5.1 主分析：全部病人", "5_1_by_outcome": "5.1 按 ICU 结局",
          "5_1_by_subtype": "5.1 按卒中类型", "5_1_by_unit_type": "5.1 按该小时所在单元类型",
          "5_1_by_unit": "5.1 按单个单元（≥100人）", "5_1_by_phase": "5.1 按住院阶段", "5_1_by_era": "5.1 按年代",
          "5_1_by_reperfusion_AIS": "5.1 缺血性卒中：按是否再灌注", "5_1_unit_x_phase": "5.1 单元类型 × 住院阶段（小时口径）",
          "5_2_reproducibility": "5.2 重复性检验", "5_3_fractional_logit": "5.3 病人 T3 比例的关联因素（分数响应模型，描述性）",
          "5_4_cif_v02": "5.4 首次达到 v0.2 的累积发生率%（ICU 内死亡为竞争事件）",
          "5_4_cif_v03": "5.4 首次达到 v0.3 的累积发生率%（ICU 内死亡为竞争事件）",
          "5_4_delay_alive": "5.4 首次达到 v0.3 到实际离开 ICU", "5_4_v02_v03_gap": "5.4 v0.2 与 v0.3 的时间差",
          "5_5_relapse": "5.5 达到后再次失去", "5_6_hour_of_day": "5.6 一天中的时刻（活着转出者）",
          "5_6_after_hours": "5.6 晚上 7 点至早上 7 点离开 ICU 的比例（按最后所在单元）",
          "6_sensitivity": "6 敏感性分析 S1–S8（不含置信区间）", "6_S9_unit_by_last": "6 S9 单元归类方法",
          "6_S10_stability": "6 S10 稳定持续时间", "7_method_attribution": "方法说明：两种分解的差异归因",
          "7_method_crosstab": "方法说明：旧法 × 新法（小时数）"}
lines = ["# 第四篇（ICU 时间论文）正式分析结果", "",
         "按《ICU时间论文_分析方案_v1.0.md》（2026-11-06 锁定）。括号内为以病人为单位自助法 1,000 次的 95% 置信区间。",
         "T1 = 器官支持等需求（v0.2 未满足）；T2 = 额外可观测的卒中特异需求；T3 = 未被可观测需求解释的 ICU 时间。", "",
         "## 研究流程", "", md(flow), "",
         f"**5.2 是否在不同年代中稳定**：条件 1 {'满足' if stable['条件1 差≤5'] else '不满足'}；"
         f"条件 2 {'满足' if stable['条件2 单元顺序相同'] else '不满足'}；条件 3 {'满足' if stable['条件3 阶段方向相同'] else '不满足'}。", ""]
for k, t in tables.items():
    lines += [f"## {TITLES.get(k, k)}", "", md(t), ""]
(OUT / "formal_results.md").write_text("\n".join(lines), encoding="utf-8")
print("完成")
