"""
ICU 路径模拟器 · 为 B 准备的检查：神经功能检查（GCS 评估）的频次，在病人之间是否真的有差别

“一次评估” = GCS 睁眼 / 语言 / 运动中任意一项在同一时刻的记录（同一 charttime 只算一次）。
注意：这里量的是“记录下来的”评估频次，不一定等于实际执行的频次。

只用完整的观察单位：
  - 天：入 ICU 后完整的 24 小时（病人整天都在 ICU 且未死亡）
  - 窗口：完整的 6 小时窗口（去掉每位病人最后一个可能不完整的窗口）

问题：
  1. 每天评估多少次？按频次归类：约每小时（≥20 次/天）、约每 2 小时（10–19）、
     约每 4 小时（5–9）、更少（1–4）、无记录（0）
  2. 频次随 ICU 天数怎么变（逐步减少？）
  3. 同一状态、同一天数的病人之间，频次是否不同（B 需要的“变异”）
  4. 不同 ICU 单元之间是否不同（单元规程）
  5. 状态、天数、单元一共能解释多少差别；剩下的就是病人之间“同样情况、不同做法”的部分

输出：09_icu_simulator/outputs/s3_*.csv 与 s3_summary.md（只有汇总数字）
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

BASE = project_paths.MIMIC_IV
HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
WH, MAXW = 6, 120
GCS = [V.GCS_EYE, V.GCS_VERBAL, V.GCS_MOTOR]

c = pd.read_csv(ROOT / "04_outputs/tables/patient_level_cohort.csv",
                parse_dates=["intime", "outtime", "deathtime"])
c["obs_end"] = c["outtime"]
d = c["deathtime"].notna() & (c["deathtime"] < c["outtime"])
c.loc[d, "obs_end"] = c.loc[d, "deathtime"]
c = c[c.obs_end > c.intime].copy()
c["obs_h"] = ((c.obs_end - c.intime).dt.total_seconds() / 3600).clip(upper=MAXW * WH)
st = pd.read_csv(HERE / "work/states_full_stay.csv", usecols=["stay_id", "window_idx", "state"])

print("扫描 chartevents（GCS）...", flush=True)
frames = []
ids = set(c.stay_id)
for ch in pd.read_csv(f"{BASE}/icu/chartevents.csv.gz", usecols=["stay_id", "charttime", "itemid"],
                      chunksize=5_000_000):
    ch = ch[ch.itemid.isin(GCS) & ch.stay_id.isin(ids)]
    if len(ch):
        frames.append(ch[["stay_id", "charttime"]])
g = pd.concat(frames).drop_duplicates()
g["charttime"] = pd.to_datetime(g.charttime)
g = g.merge(c[["stay_id", "intime", "obs_end"]], on="stay_id")
g = g[(g.charttime >= g.intime) & (g.charttime < g.obs_end)]
g["h"] = (g.charttime - g.intime).dt.total_seconds() / 3600
g = g[g.h < MAXW * WH]

# ---- 天层面 ---------------------------------------------------------------
c["full_days"] = (c.obs_h // 24).astype(int)
days = pd.DataFrame([(s, k) for s, n in zip(c.stay_id, c.full_days) for k in range(n)],
                    columns=["stay_id", "day"])
g["day"] = (g.h // 24).astype(int)
cnt = g.groupby(["stay_id", "day"]).size().rename("n")
days = days.join(cnt, on=["stay_id", "day"]).fillna({"n": 0})
CAT = ["约每小时(≥20)", "约每2小时(10–19)", "约每4小时(5–9)", "更少(1–4)", "无记录(0)"]
days["cat"] = pd.cut(days.n, [-1, 0, 4, 9, 19, 1e9], labels=CAT[::-1]).astype(str)
days["day_band"] = pd.cut(days.day, [-1, 0, 1, 2, 6, 13, 30],
                          labels=["第1天", "第2天", "第3天", "第4–7天", "第8–14天", "第15–30天"])
t1 = (pd.crosstab(days.day_band, days.cat, normalize="index") * 100).reindex(columns=CAT).round(1)
t1["病人日数"] = days.day_band.value_counts().reindex(t1.index)
t1["中位次数"] = days.groupby("day_band", observed=True).n.median()
t1.to_csv(OUT / "s3_daily_frequency_by_day.csv")

# 单元（入 ICU 的单元，前 3 天）
days = days.merge(c[["stay_id", "first_careunit"]], on="stay_id")
top = c.first_careunit.value_counts()
top = top[top >= 100].index
e = days[(days.day <= 2) & days.first_careunit.isin(top)]
t2 = (pd.crosstab(e.first_careunit, e.cat, normalize="index") * 100).reindex(columns=CAT).round(1)
t2["病人日数"] = e.first_careunit.value_counts().reindex(t2.index)
t2["中位次数"] = e.groupby("first_careunit").n.median()
t2 = t2.sort_values("病人日数", ascending=False)
t2.to_csv(OUT / "s3_daily_frequency_by_unit_day1to3.csv")

# 同一病人随时间是否改变频次类别
pc = days.groupby("stay_id").cat.nunique()
long3 = c.loc[c.full_days >= 3, "stay_id"]
changed = float((pc.reindex(long3) > 1).mean())

# ---- 窗口层面：状态 × 天数 ---------------------------------------------------
g["window_idx"] = (g.h // WH).astype(int)
wc = g.groupby(["stay_id", "window_idx"]).size().rename("n")
st = st.join(wc, on=["stay_id", "window_idx"]).fillna({"n": 0})
st = st.merge(c[["stay_id", "first_careunit"]], on="stay_id")
lastw = st.groupby("stay_id").window_idx.transform("max")
st = st[st.window_idx < lastw]                         # 去掉最后一个可能不完整的窗口
WCAT = ["≥5次(约每小时)", "3–4次(约每2小时)", "1–2次(约每4小时)", "0次"]
st["cat"] = pd.cut(st.n, [-1, 0, 2, 4, 1e9], labels=WCAT[::-1]).astype(str)
st["day_band"] = pd.cut(st.window_idx, [0, 4, 12, 28, MAXW], right=False,
                        labels=["第1天", "第2–3天", "第4–7天", "第8–30天"])
STATES = ["neurologically preserved-low support", "neurological impairment-low support",
          "neurological impairment-renal dysfunction", "neurological impairment-respiratory support"]
t3 = (pd.crosstab([st.state, st.day_band], st.cat, normalize="index") * 100).reindex(columns=WCAT).round(1)
t3["窗口数"] = st.groupby(["state", "day_band"], observed=True).size()
t3 = t3.reindex(pd.MultiIndex.from_product([STATES, ["第1天", "第2–3天", "第4–7天", "第8–30天"]]))
t3.to_csv(OUT / "s3_window_frequency_by_state_day.csv")

# ---- 能解释多少：窗口评估次数 ~ 状态 + 天数 + 单元 ------------------------------
def r2(cols):
    X = pd.get_dummies(st[cols].astype(str), drop_first=True).astype(float)
    X.insert(0, "const", 1.0)
    y = st.n.to_numpy(float)
    beta, *_ = np.linalg.lstsq(X.to_numpy(), y, rcond=None)
    res = y - X.to_numpy() @ beta
    return 1 - res.var() / y.var()


# 病人之间（同样状态、天数、单元下）的差别：加入病人本身后还能多解释多少
st["patient_mean"] = st.groupby("stay_id").n.transform("mean")
r2_tab = pd.DataFrame({"模型": ["状态", "状态 + 天数", "状态 + 天数 + 单元"],
                       "R²": [r2(["state"]), r2(["state", "day_band"]),
                              r2(["state", "day_band", "first_careunit"])]})
X = pd.get_dummies(st[["state", "day_band", "first_careunit"]].astype(str), drop_first=True).astype(float)
X.insert(0, "const", 1.0); X["patient_mean"] = st.patient_mean
beta, *_ = np.linalg.lstsq(X.to_numpy(), st.n.to_numpy(float), rcond=None)
res = st.n.to_numpy(float) - X.to_numpy() @ beta
r2_tab.loc[len(r2_tab)] = ["状态 + 天数 + 单元 + 病人自身平均频次", 1 - res.var() / st.n.var()]
r2_tab["R²"] = r2_tab["R²"].round(3)
r2_tab.to_csv(OUT / "s3_variance_explained.csv", index=False)


def md(df):
    df = df.to_frame() if isinstance(df, pd.Series) else df
    head = [" / ".join(map(str, df.index.names)) if df.index.nlevels > 1 else str(df.index.name or "")] + \
           [str(x) for x in df.columns]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for i, r in zip(df.index, df.values):
        i = " / ".join(map(str, i)) if isinstance(i, tuple) else str(i)
        out.append("| " + " | ".join([i] + ["" if pd.isna(v) else str(v) for v in r]) + " |")
    return "\n".join(out)


lines = ["# 神经功能检查（GCS 记录）频次", "",
         f"- 病人 {c.stay_id.nunique():,}；完整病人日 {len(days):,}；完整 6 小时窗口 {len(st):,}",
         f"- 住满 3 天的病人中，频次类别在住院期间至少变过一次的比例：{changed:.1%}", "",
         "## 1–2. 每天的评估次数，按 ICU 天数（% 病人日）", "", md(t1), "",
         "## 4. 前 3 天，按入 ICU 的单元（% 病人日；单元 ≥ 100 人）", "", md(t2), "",
         "## 3. 每个 6 小时窗口的评估次数，按状态和天数（% 窗口）", "", md(t3), "",
         "## 5. 窗口评估次数的差别能被解释多少（R²）", "", md(r2_tab.set_index("模型")), ""]
(OUT / "s3_summary.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
