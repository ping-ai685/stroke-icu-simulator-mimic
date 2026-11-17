"""
ICU 路径模拟器 · 第二步：把四状态延伸到整个 ICU 停留，并检查 72 小时以后的状态
是否还是原来的样子。

做法（不重新拟合任何东西）：
  - 变量提取规则与第一篇 02_extract_timewindow_variables.py、02b 第一层范围检查完全相同，
    只是窗口不再截止于 72 小时，而截止于 min(出 ICU, 死亡)，最长 30 天（120 个窗口）。
  - 预处理用第二篇冻结的参数（07_paper2_eicu/frozen_params/）：前向填补上限、训练集中位数、
    z 分数参数。
  - 状态用第一篇的模型 .model_k4.pt 解码，分类发射概率加 1e-4 下限（与第二篇相同）。

检查标准在运行前写定（借用第二篇的迁移标准，把“72 小时以后”当作一个新的人群）：
  C0 复现：前 12 个窗口的原始变量与第一篇的 timewindow_level_raw.csv 一致（最大差 < 1e-6）
  C1 状态画像：每个状态在 72 小时以后的 21 变量平均画像与 72 小时以内的相关系数 ≥ 0.80
  C2 每个状态在 72 小时以后至少占 5% 的窗口
  C3 6 小时转移矩阵（72 小时以后 vs 以内）16 个格子的秩相关 ≥ 0.80
  C4 每个状态的自我转移概率相差不超过 0.10
另外报告（不设门槛）：
  - 每个窗口在其所属状态下的发射对数似然，72 小时以内与以后的分布
  - 前 72 小时：整段解码与只用 72 小时解码的状态一致率
  - 各状态在各时段的占比；转出前最后一个窗口的状态分布（按活着转出 / ICU 内死亡）

输出：09_icu_simulator/outputs/（汇总表）与 09_icu_simulator/work/（个体层面的中间文件，
仅本机，不外传）
"""
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.insert(0, str(ROOT / "03_code"))
import project_paths  # noqa: E402
import variable_itemid_map as V  # noqa: E402
from pomegranate_patches import CategoricalMixed  # noqa: E402,F401

BASE = project_paths.MIMIC_IV
HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"; OUT.mkdir(exist_ok=True)
WORK = HERE / "work"; WORK.mkdir(exist_ok=True)
FP = ROOT / "07_paper2_eicu/frozen_params"
T1 = ROOT / "04_outputs/tables"

WH = 6                     # 窗口小时数
MAXW = 120                 # 最长 30 天
N72 = 12                   # 前 72 小时 = 12 个窗口
EPS = 1e-4
CHUNK = 5_000_000

# ---------------------------------------------------------------------------
# 0. 队列与观察窗口
# ---------------------------------------------------------------------------
c = pd.read_csv(T1 / "patient_level_cohort.csv", parse_dates=["intime", "outtime", "deathtime"])
c["died_icu"] = c["icu_mortality"].astype(int) == 1
c["obs_end"] = c["outtime"]
d = c["deathtime"].notna() & (c["deathtime"] < c["outtime"])
c.loc[d, "obs_end"] = c.loc[d, "deathtime"]
c["max_w"] = ((c["obs_end"] - c["intime"]).dt.total_seconds() // (WH * 3600))
bad = c["max_w"] < 0
print(f"死亡时间早于入 ICU 的病人：{int(bad.sum())}，不纳入（无观察窗口）")
c = c[~bad].copy()
c["truncated_30d"] = c["max_w"] > MAXW - 1
c["max_w"] = c["max_w"].clip(upper=MAXW - 1).astype(int)
stay_ids, hadm_ids = set(c.stay_id), set(c.hadm_id)
intime_by_stay = c.set_index("stay_id")["intime"]
intime_by_hadm = c.set_index("hadm_id")["intime"]
stay_by_hadm = c.set_index("hadm_id")["stay_id"]
maxw_by_stay = c.set_index("stay_id")["max_w"]


def widx(t, it):
    w = ((t - it).dt.total_seconds() // (WH * 3600))
    return w.where((w >= 0) & (w < MAXW))


# ---------------------------------------------------------------------------
# 1. 提取（规则同第一篇 02）
# ---------------------------------------------------------------------------
print("扫描 chartevents ...", flush=True)
frames = []
for ch in pd.read_csv(f"{BASE}/icu/chartevents.csv.gz",
                      usecols=["stay_id", "charttime", "itemid", "value", "valuenum"],
                      chunksize=CHUNK):
    ch = ch[ch.stay_id.isin(stay_ids) & ch.itemid.isin(V.ALL_CHARTEVENTS_ITEMIDS)]
    if len(ch):
        frames.append(ch)
chart = pd.concat(frames, ignore_index=True); del frames
chart["charttime"] = pd.to_datetime(chart["charttime"])
chart["window_idx"] = widx(chart.charttime, chart.stay_id.map(intime_by_stay))
chart = chart.dropna(subset=["window_idx"]); chart["window_idx"] = chart.window_idx.astype(int)
chart = chart.sort_values("charttime", kind="stable")
chart.loc[(chart.itemid == V.GCS_VERBAL) & chart.value.isin(V.GCS_VERBAL_NOT_ASSESSABLE_VALUES),
          "valuenum"] = np.nan


def last(df, ids, idcol="stay_id"):
    sub = df[df.itemid.isin(ids if isinstance(ids, list) else [ids])]
    return sub.groupby([idcol, "window_idx"])["valuenum"].last()


S = {
    "heart_rate": last(chart, V.HEART_RATE), "sbp_arterial": last(chart, V.SBP_ARTERIAL),
    "sbp_noninvasive": last(chart, V.SBP_NONINVASIVE), "map_arterial": last(chart, V.MAP_ARTERIAL),
    "map_noninvasive": last(chart, V.MAP_NONINVASIVE), "resp_rate": last(chart, V.RESP_RATE),
    "spo2": last(chart, V.SPO2), "temp_f": last(chart, V.TEMP_F), "temp_c_direct": last(chart, V.TEMP_C),
    "gcs_eye": last(chart, V.GCS_EYE), "gcs_verbal": last(chart, V.GCS_VERBAL),
    "gcs_motor": last(chart, V.GCS_MOTOR),
}
del chart

print("扫描 labevents ...", flush=True)
frames = []
for lb in pd.read_csv(f"{BASE}/hosp/labevents.csv.gz",
                      usecols=["hadm_id", "charttime", "itemid", "valuenum"], chunksize=CHUNK):
    lb = lb[lb.hadm_id.isin(hadm_ids) & lb.itemid.isin(V.ALL_LABEVENTS_ITEMIDS)]
    if len(lb):
        frames.append(lb)
lab = pd.concat(frames, ignore_index=True); del frames
lab["charttime"] = pd.to_datetime(lab["charttime"])
lab["window_idx"] = widx(lab.charttime, lab.hadm_id.map(intime_by_hadm))
lab = lab.dropna(subset=["window_idx"]); lab["window_idx"] = lab.window_idx.astype(int)
lab["stay_id"] = lab.hadm_id.map(stay_by_hadm)
lab = lab.sort_values("charttime", kind="stable")
for name, ids in [("wbc", V.WBC), ("hemoglobin", V.HEMOGLOBIN), ("platelet", V.PLATELET),
                  ("creatinine", V.CREATININE), ("bun", V.BUN), ("sodium", V.SODIUM),
                  ("potassium", V.POTASSIUM), ("glucose", V.GLUCOSE)]:
    S[name] = last(lab, ids)
del lab

print("区间事件与尿量 ...", flush=True)


def interval_flags(df, ids):
    sub = df[df.itemid.isin(ids) & df.stay_id.isin(stay_ids)].dropna(subset=["starttime", "endtime"]).copy()
    it = sub.stay_id.map(intime_by_stay)
    ws = ((sub.starttime - it).dt.total_seconds() // (WH * 3600)).clip(lower=0)
    we = ((sub.endtime - it).dt.total_seconds() // (WH * 3600)).clip(upper=MAXW - 1)
    sub = sub.assign(ws=ws, we=we)[lambda x: x.we >= x.ws]
    sub["window_idx"] = [list(range(int(a), int(b) + 1)) for a, b in zip(sub.ws, sub.we)]
    e = sub[["stay_id", "window_idx"]].explode("window_idx").drop_duplicates()
    e["window_idx"] = e.window_idx.astype(int)
    return e.assign(flag=1).set_index(["stay_id", "window_idx"])["flag"]


proc = pd.read_csv(f"{BASE}/icu/procedureevents.csv.gz",
                   usecols=["stay_id", "starttime", "endtime", "itemid"], parse_dates=["starttime", "endtime"])
inp = pd.read_csv(f"{BASE}/icu/inputevents.csv.gz",
                  usecols=["stay_id", "starttime", "endtime", "itemid"], parse_dates=["starttime", "endtime"])
S["mech_vent"] = interval_flags(proc, V.MECH_VENT)
S["crrt"] = interval_flags(proc, V.CRRT_MODES)
S["vasopressor"] = interval_flags(inp, V.VASOPRESSORS)
S["sedative"] = interval_flags(inp, V.SEDATIVES)
del proc, inp
outp = pd.read_csv(f"{BASE}/icu/outputevents.csv.gz",
                   usecols=["stay_id", "charttime", "itemid", "value"], parse_dates=["charttime"])
outp = outp[outp.stay_id.isin(stay_ids) & outp.itemid.isin(V.URINE_OUTPUT)]
outp["window_idx"] = widx(outp.charttime, outp.stay_id.map(intime_by_stay))
outp = outp.dropna(subset=["window_idx"]); outp["window_idx"] = outp.window_idx.astype(int)
S["urine_output_ml"] = outp.groupby(["stay_id", "window_idx"])["value"].sum()
del outp

rows = [(s, w) for s, m in maxw_by_stay.items() for w in range(m + 1)]
wide = pd.DataFrame(index=pd.MultiIndex.from_tuples(rows, names=["stay_id", "window_idx"]))
for k_, s in S.items():
    wide[k_] = s.reindex(wide.index)
wide = wide.reset_index()
wide["sbp"] = wide.sbp_arterial.combine_first(wide.sbp_noninvasive)
wide["map"] = wide.map_arterial.combine_first(wide.map_noninvasive)
wide["temp_c"] = wide.temp_c_direct.combine_first((wide.temp_f - 32) * 5 / 9)
for f in ["mech_vent", "crrt", "vasopressor", "sedative"]:
    wide[f] = wide[f].fillna(0).astype(int)
wide = wide.drop(columns=["sbp_arterial", "sbp_noninvasive", "map_arterial", "map_noninvasive",
                          "temp_f", "temp_c_direct"])

# 02b 第一层：不可能值设为缺失
RANGE = {"heart_rate": (20, 300), "sbp": (40, 300), "map": (20, 250), "resp_rate": (0, 80),
         "spo2": (0, 100), "temp_c": (25, 45), "gcs_eye": (1, 4), "gcs_verbal": (1, 5),
         "gcs_motor": (1, 6), "urine_output_ml": (0, 20000), "wbc": (0, 100), "hemoglobin": (2, 20),
         "platelet": (0, 2000), "creatinine": (0, 20), "bun": (0, 200), "sodium": (100, 200),
         "potassium": (1, 10), "glucose": (0, 1500)}
for v, (lo, hi) in RANGE.items():
    wide.loc[~wide[v].between(lo, hi) & wide[v].notna(), v] = np.nan
wide.to_csv(WORK / "timewindow_raw_full_stay.csv", index=False)

# ---- C0 复现检查 -----------------------------------------------------------
ref = pd.read_csv(T1 / "timewindow_level_raw.csv")
# 第一篇的窗口没有在死亡时截止，这里只比较两边都存在的窗口
m = ref.merge(wide, on=["stay_id", "window_idx"], how="inner", suffixes=("_ref", "_new"))
print(f"C0：第一篇窗口 {len(ref):,}，其中在本次窗口中也存在的 {len(m):,}")
c0 = []
for v in list(RANGE) + ["mech_vent", "crrt", "vasopressor", "sedative"]:
    a, b = m[f"{v}_ref"].astype(float), m[f"{v}_new"].astype(float)
    both_nan = a.isna() & b.isna()
    diff = (a - b).abs().where(~both_nan)
    mismatch_nan = int((a.isna() ^ b.isna()).sum())
    c0.append({"variable": v, "max_abs_diff": float(diff.max(skipna=True) if diff.notna().any() else 0),
               "nan_pattern_mismatch": mismatch_nan})
c0 = pd.DataFrame(c0)
c0.to_csv(OUT / "s2_C0_replay_check.csv", index=False)
print(c0.to_string(index=False))

# ---------------------------------------------------------------------------
# 2. 冻结预处理（同第二篇 10_eicu_apply_frozen_preprocessing.py 的 A1）
# ---------------------------------------------------------------------------
FEATS = pd.read_csv(FP / "feature_order.csv").feature.tolist()
imp = pd.read_csv(FP / "imputation_medians_train.csv").set_index("variable")
zp = pd.read_csv(FP / "zscore_params_train.csv", index_col=0)
CONT = [f[:-2] for f in FEATS if f.endswith("_z")]
wide = wide.sort_values(["stay_id", "window_idx"]).reset_index(drop=True)
for v in CONT:
    wide[f"{v}_missing_raw"] = wide[v].isna().astype(int)
    wide[v] = wide.groupby("stay_id")[v].ffill(limit=int(imp.loc[v, "ffill_limit_windows"]))
    wide[v] = wide[v].fillna(imp.loc[v, "train_median_post_ffill"])
    wide[f"{v}_z"] = (wide[v] - zp.loc[v, "mean"]) / zp.loc[v, "std"]

# ---------------------------------------------------------------------------
# 3. 解码（第一篇模型 + 1e-4 下限）
# ---------------------------------------------------------------------------
lbl = pd.read_csv(T1 / "state_label_mapping.csv").set_index("raw_state")["name"].to_dict()
model = torch.load(ROOT / ".model_k4.pt", weights_only=False)
for dist in model.distributions:
    for sub in dist.distributions[-4:]:
        p = torch.clamp(sub.probs.detach().clone(), EPS, 1 - EPS)
        p = p / p.sum(dim=-1, keepdim=True)
        sub.probs = torch.nn.Parameter(p, requires_grad=False)
        sub._log_probs = torch.log(p)


def emis(X):                                     # (n, K) log P(x | state)
    return torch.stack([dd.log_probability(X) for dd in model.distributions], dim=1)


def decode(X):
    if X.shape[0] == 1:
        return torch.argmax(emis(X) + model.starts, dim=1).numpy()
    return model.predict(X.unsqueeze(0))[0].numpy()


torch.set_grad_enabled(False)
states, ll, s72 = [], [], []
for sid, g in wide.groupby("stay_id", sort=False):
    X = torch.tensor(g[FEATS].to_numpy(dtype=np.float32))
    st = decode(X)
    states.append(st)
    e = emis(X).detach().numpy()
    ll.append(e[np.arange(len(st)), st])
    s72.append(decode(X[:N72]))
wide["raw_state"] = np.concatenate(states)
wide["state"] = wide.raw_state.map(lbl)
wide["emission_ll"] = np.concatenate(ll)
first = wide.window_idx < N72
wide.loc[first, "state_72h_only"] = pd.Series(np.concatenate(s72), index=wide.index[first]).map(lbl)
wide[["stay_id", "window_idx", "state", "emission_ll", "state_72h_only"] +
     [f"{v}_missing_raw" for v in CONT]].to_csv(WORK / "states_full_stay.csv", index=False)

# ---------------------------------------------------------------------------
# 4. 检查
# ---------------------------------------------------------------------------
STATES = [lbl[i] for i in [1, 3, 0, 2]]   # 保留-低支持、受损-低支持、受损-肾、受损-呼吸
early, late = wide[first], wide[~first]
pe, pl = early.groupby("state")[FEATS].mean(), late.groupby("state")[FEATS].mean()
c1 = {s: float(np.corrcoef(pe.loc[s], pl.loc[s])[0, 1]) for s in STATES}
prev_e = early.state.value_counts(normalize=True).reindex(STATES).fillna(0)
prev_l = late.state.value_counts(normalize=True).reindex(STATES).fillna(0)


def tmat(df):
    t = df[["stay_id", "window_idx", "state"]].copy()
    t["nxt"] = t.groupby("stay_id")["state"].shift(-1)
    t = t[t.groupby("stay_id")["window_idx"].shift(-1) == t.window_idx + 1]
    return pd.crosstab(t.state, t.nxt, normalize="index").reindex(index=STATES, columns=STATES).fillna(0)


# 72 小时以内：起点在前 11 个窗口的转移；以后：起点在第 12 个窗口及以后
te = tmat(wide[wide.window_idx < N72])
tl = tmat(wide[wide.window_idx >= N72])
c3 = float(pd.Series(te.to_numpy().ravel()).corr(pd.Series(tl.to_numpy().ravel()), method="spearman"))
c4 = max(abs(te.loc[s, s] - tl.loc[s, s]) for s in STATES)

crit = pd.DataFrame([
    {"标准": "C0 前72小时原始变量与第一篇一致", "结果": f"最大差 {c0.max_abs_diff.max():.2g}；缺失模式不一致 {int(c0.nan_pattern_mismatch.sum())} 处",
     "通过": bool(c0.max_abs_diff.max() < 1e-6 and c0.nan_pattern_mismatch.sum() == 0)},
    {"标准": "C1 各状态画像相关 ≥ 0.80", "结果": "；".join(f"{s}: {r:.3f}" for s, r in c1.items()),
     "通过": bool(min(c1.values()) >= 0.80)},
    {"标准": "C2 72小时以后各状态 ≥ 5%", "结果": "；".join(f"{s}: {p:.1%}" for s, p in prev_l.items()),
     "通过": bool(prev_l.min() >= 0.05)},
    {"标准": "C3 转移矩阵秩相关 ≥ 0.80", "结果": f"{c3:.3f}", "通过": bool(c3 >= 0.80)},
    {"标准": "C4 自我转移相差 ≤ 0.10", "结果": f"{c4:.3f}", "通过": bool(c4 <= 0.10)},
])
crit.to_csv(OUT / "s2_criteria.csv", index=False)

# 补充 1：发射对数似然
llt = pd.DataFrame({"72小时以内": early.groupby("state").emission_ll.median(),
                    "72小时以后": late.groupby("state").emission_ll.median()}).reindex(STATES).round(2)
llt.loc["全部"] = [early.emission_ll.median().round(2), late.emission_ll.median().round(2)]
llt.to_csv(OUT / "s2_emission_ll_median.csv")

# 补充 2：前 72 小时两种解码的一致率
agree = float((early.state == early.state_72h_only).mean())

# 补充 3：各时段状态占比
bands = pd.cut(wide.window_idx, [0, 12, 20, 28, 56, MAXW], right=False,
               labels=["0–3天", "3–5天", "5–7天", "7–14天", "14–30天"])
prev_t = (pd.crosstab(bands, wide.state, normalize="index") * 100).reindex(columns=STATES).round(1)
prev_t["窗口数"] = bands.value_counts().reindex(prev_t.index)
prev_t.to_csv(OUT / "s2_prevalence_by_period.csv")

# 补充 4：最后一个窗口的状态
lastw = wide.groupby("stay_id").tail(1).merge(c[["stay_id", "died_icu", "truncated_30d"]], on="stay_id")
lastw = lastw[~lastw.truncated_30d]
end_t = (pd.crosstab(lastw.died_icu.map({False: "活着转出", True: "ICU内死亡"}), lastw.state,
                     normalize="index") * 100).reindex(columns=STATES).round(1)
end_t["人数"] = lastw.died_icu.map({False: "活着转出", True: "ICU内死亡"}).value_counts()
end_t.to_csv(OUT / "s2_last_window_state.csv")

pd.concat({"72小时以内": te.round(3), "72小时以后": tl.round(3)}).to_csv(OUT / "s2_transition_matrices.csv")
pd.concat({"72小时以内": pe.round(3), "72小时以后": pl.round(3)}).to_csv(OUT / "s2_state_profiles.csv")


def md(df):
    df = df.to_frame() if isinstance(df, pd.Series) else df
    head = [str(df.index.name or "")] + [str(x) for x in df.columns]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join([str(i)] + ["" if pd.isna(v) else str(v) for v in r]) + " |"
            for i, r in zip(df.index, df.values)]
    return "\n".join(out)


lines = ["# 第二步：四状态延伸到整个 ICU 停留", "",
         f"- 病人 {wide.stay_id.nunique():,}，窗口 {len(wide):,}（72 小时以内 {len(early):,}，以后 {len(late):,}）",
         f"- 停留超过 30 天、在第 30 天截断的病人：{int(c.truncated_30d.sum())}",
         f"- 死亡时间早于入 ICU、未纳入的病人：{int(bad.sum())}", "",
         "## 预设标准", "", md(crit.set_index("标准")), "",
         "## 72 小时以内与以后的转移矩阵（行：当前状态，列：下一个窗口的状态）", "",
         "72 小时以内：", "", md(te.round(3)), "", "72 小时以后：", "", md(tl.round(3)), "",
         "## 发射对数似然中位数（越低表示模型对这些窗口描述得越差）", "", md(llt), "",
         f"## 前 72 小时：整段解码与只用 72 小时解码的一致率：{agree:.1%}", "",
         "## 各时段的状态占比（%）", "", md(prev_t), "",
         "## 出 ICU 前最后一个窗口的状态（%，不含 30 天截断者）", "", md(end_t), ""]
(OUT / "s2_summary.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
