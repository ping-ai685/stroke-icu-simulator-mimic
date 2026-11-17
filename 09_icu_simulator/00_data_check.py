"""
ICU 路径模拟器 · 第一步：数据检查（只输出汇总数字，不输出任何个体记录）

回答四个问题，队列为第一篇的 ICU 卒中队列（patient_level_cohort.csv，6,368 人）：
  1. ICU 停留时长的分布：多少人超过 72 小时
  2. 终点：活着转出 ICU 与 ICU 内死亡各多少人
  3. 72 小时以后，第一篇的变量是否还有足够的记录
     （每个 6 小时窗口里是否至少有一次记录，与前 72 小时用同样的口径比较）
  4. 一天 24 小时中，GCS 记录的时刻分布与活着转出 ICU 的时刻分布

观察窗口截止于 min(出 ICU 时间, 死亡时间)：MIMIC-IV 中死亡病人的出科时间
可能晚于死亡时间（第一篇的经验）。MIMIC-IV 去标识化时保留了一天中的时刻，
因此第 4 项的时刻分布可以解读。

输出：09_icu_simulator/outputs/ 下的汇总 CSV 和 data_check_summary.md
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "03_code"))
import project_paths  # noqa: E402
import variable_itemid_map as V  # noqa: E402

BASE = project_paths.MIMIC_IV
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)
WINDOW_H = 6
MAX_DAYS = 30            # 可用性统计只看入 ICU 后 30 天以内
CHUNK = 20_000_000


def md(df):
    """DataFrame / Series -> Markdown 表格（不依赖 tabulate）。"""
    df = df.to_frame() if isinstance(df, pd.Series) else df
    head = [str(df.index.name or "")] + [str(x) for x in df.columns]
    rows = [[str(i)] + ["" if pd.isna(v) else str(v) for v in r] for i, r in zip(df.index, df.values)]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 队列与终点
# ---------------------------------------------------------------------------
c = pd.read_csv(ROOT / "04_outputs/tables/patient_level_cohort.csv",
                parse_dates=["intime", "outtime", "deathtime", "dod"])
n = len(c)
c["died_icu"] = c["icu_mortality"].astype(int) == 1
c["end"] = c["outtime"]
dead_before_out = c["deathtime"].notna() & (c["deathtime"] < c["outtime"])
c.loc[dead_before_out, "end"] = c.loc[dead_before_out, "deathtime"]
c["obs_h"] = (c["end"] - c["intime"]).dt.total_seconds() / 3600
c["n_windows"] = np.ceil(c["obs_h"] / WINDOW_H).clip(lower=1).astype(int)

lines = ["# ICU 路径模拟器 · 数据检查汇总", "",
         f"队列：第一篇 ICU 卒中队列，n = {n:,}", ""]

# ---- 问题 1：停留时长 ----
los_h = c["obs_h"]
q = los_h.quantile([0.25, 0.5, 0.75, 0.9]).round(1)
bands = pd.cut(los_h, [0, 24, 48, 72, 120, 168, 336, np.inf],
               labels=["≤1天", "1–2天", "2–3天", "3–5天", "5–7天", "7–14天", ">14天"],
               right=True)
t1 = (pd.crosstab(bands, c["died_icu"].map({True: "ICU内死亡", False: "活着转出"}),
                  margins=True, margins_name="合计"))
t1.to_csv(OUT / "q1_los_bands.csv")
over72 = (los_h > 72).sum()
lines += ["## 1. ICU 停留时长（截止于出 ICU 或死亡）", "",
          f"- 中位数 {q[0.5]} 小时（四分位 {q[0.25]}–{q[0.75]}），第 90 百分位 {q[0.9]} 小时",
          f"- 超过 72 小时：{over72:,} 人（{over72 / n:.1%}）", "",
          md(t1), ""]

# ---- 问题 2：终点 ----
t2 = (c.groupby("stroke_subtype")["died_icu"].agg(["size", "sum"])
      .rename(columns={"size": "人数", "sum": "ICU内死亡"}))
t2["活着转出"] = t2["人数"] - t2["ICU内死亡"]
t2.loc["合计"] = t2.sum()
t2["ICU内死亡比例"] = (t2["ICU内死亡"] / t2["人数"]).map("{:.1%}".format)
t2.to_csv(OUT / "q2_endpoints.csv")
n_shift = int(dead_before_out.sum())
lines += ["## 2. 终点", "", md(t2), "",
          f"- 死亡时间早于出 ICU 时间的病人：{n_shift:,} 人（观察窗口已在死亡时截止）", ""]

# ---------------------------------------------------------------------------
# 问题 3、4：扫描 chartevents 与 labevents
# ---------------------------------------------------------------------------
chart_vars = {
    "heart_rate": [V.HEART_RATE], "sbp": [V.SBP_ARTERIAL, V.SBP_NONINVASIVE],
    "map": [V.MAP_ARTERIAL, V.MAP_NONINVASIVE], "resp_rate": [V.RESP_RATE],
    "spo2": [V.SPO2], "temp": [V.TEMP_F, V.TEMP_C],
    "gcs_eye": [V.GCS_EYE], "gcs_verbal": [V.GCS_VERBAL], "gcs_motor": [V.GCS_MOTOR],
}
lab_vars = {"wbc": V.WBC, "hemoglobin": V.HEMOGLOBIN, "platelet": V.PLATELET,
            "creatinine": V.CREATININE, "bun": V.BUN, "sodium": V.SODIUM,
            "potassium": V.POTASSIUM, "glucose": V.GLUCOSE}
item2var = {i: v for v, ids in chart_vars.items() for i in ids}
lab2var = {i: v for v, ids in lab_vars.items() for i in ids}

stay_info = c.set_index("stay_id")[["intime", "end", "n_windows"]]
hadm_info = c.set_index("hadm_id")[["stay_id", "intime", "end"]]

hits = []          # (stay_id, window, var) 去重后的集合
gcs_hours = np.zeros(24, dtype=np.int64)


def keep_windows(df, intime, end):
    rel = (df["charttime"] - intime).dt.total_seconds() / 3600
    ok = (rel >= 0) & (df["charttime"] <= end) & (rel < MAX_DAYS * 24)
    df = df[ok].copy()
    df["w"] = (rel[ok] // WINDOW_H).astype(int)
    return df


print("扫描 chartevents ...", flush=True)
for i, ch in enumerate(pd.read_csv(f"{BASE}/icu/chartevents.csv.gz",
                                   usecols=["stay_id", "itemid", "charttime"],
                                   chunksize=CHUNK)):
    ch = ch[ch["itemid"].isin(item2var.keys()) & ch["stay_id"].isin(stay_info.index)]
    if ch.empty:
        continue
    ch["charttime"] = pd.to_datetime(ch["charttime"])
    ch = ch.join(stay_info, on="stay_id")
    ch = keep_windows(ch, ch["intime"], ch["end"])
    ch["var"] = ch["itemid"].map(item2var)
    hits.append(ch[["stay_id", "w", "var"]].drop_duplicates())
    g = ch[ch["var"] == "gcs_motor"]["charttime"].dt.hour.value_counts()
    gcs_hours[g.index.values] += g.values
    print(f"  chunk {i}", flush=True)

print("扫描 labevents ...", flush=True)
for i, lb in enumerate(pd.read_csv(f"{BASE}/hosp/labevents.csv.gz",
                                   usecols=["hadm_id", "itemid", "charttime"],
                                   chunksize=CHUNK)):
    lb = lb[lb["itemid"].isin(lab2var.keys()) & lb["hadm_id"].isin(hadm_info.index)]
    if lb.empty:
        continue
    lb["charttime"] = pd.to_datetime(lb["charttime"])
    lb = lb.join(hadm_info, on="hadm_id")
    lb = keep_windows(lb, lb["intime"], lb["end"])
    lb["var"] = lb["itemid"].map(lab2var)
    hits.append(lb[["stay_id", "w", "var"]].drop_duplicates())
    print(f"  chunk {i}", flush=True)

hits = pd.concat(hits).drop_duplicates()

# 分母：每个时段内实际存在的窗口数（病人仍在 ICU 且未死亡）
period_edges = [0, 12, 20, 28, 56, MAX_DAYS * 4]
period_names = ["0–3天(前72小时)", "3–5天", "5–7天", "7–14天", f"14–{MAX_DAYS}天"]
denom = {}
for lo, hi, name in zip(period_edges[:-1], period_edges[1:], period_names):
    denom[name] = int((c["n_windows"].clip(upper=hi) - lo).clip(lower=0).sum())
hits["period"] = pd.cut(hits["w"], period_edges, right=False, labels=period_names)
num = hits.groupby(["var", "period"], observed=False).size().unstack("period")
avail = (num / pd.Series(denom)).round(3)
order = list(chart_vars) + list(lab_vars)
avail = avail.reindex(order)
avail.loc["(窗口数)"] = pd.Series(denom)
avail.to_csv(OUT / "q3_window_availability.csv")
pct = avail.copy()
pct.iloc[:-1] = (avail.iloc[:-1] * 100).round(1)
lines += ["## 3. 各时段中至少有一次记录的 6 小时窗口比例（%）", "",
          "分母为该时段内病人仍在 ICU 且未死亡的窗口数。", "",
          md(pct), ""]

# ---- 问题 4：时刻分布 ----
alive = c[~c["died_icu"]]
out_hours = alive["outtime"].dt.hour.value_counts().reindex(range(24), fill_value=0)
death_hours = c.loc[c["died_icu"], "end"].dt.hour.value_counts().reindex(range(24), fill_value=0)
t4 = pd.DataFrame({"GCS运动评分记录次数": gcs_hours,
                   "活着转出ICU人数": out_hours.values,
                   "ICU内死亡人数": death_hours.values}, index=range(24))
t4.index.name = "时刻(时)"
t4.to_csv(OUT / "q4_hour_of_day.csv")
dow = alive["outtime"].dt.day_name().value_counts()
dow.to_csv(OUT / "q4_transfer_day_of_week.csv")
day_share = out_hours.loc[8:19].sum() / out_hours.sum()
lines += ["## 4. 一天 24 小时中的分布", "",
          f"- 活着转出 ICU 发生在 08:00–19:59 的比例：{day_share:.1%}", "",
          md(t4), "",
          "活着转出 ICU 的星期分布：", "", md(dow), ""]

(OUT / "data_check_summary.md").write_text("\n".join(lines), encoding="utf-8")
print("完成：", OUT / "data_check_summary.md")
