"""
第四篇 · 事后分析（分析方案 v1.1 第 11 节，看到正式结果之后增加；论文中须标明为事后分析）

1. 各年代中两类单元所占小时的比例
2. 每类单元内部重复 5.2 的条件 1（各年代 T3 与该单元类型全部年代合计相差 ≤ 5 个百分点）
   和条件 3（住院阶段之间 T3 的变化方向与该单元类型合计相同）
3. 按单元类型分别报告的 T1 / T2 / T3（带置信区间），作为论文主要呈现

输出：outputs/posthoc_by_unit.md 与 csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT, WORK = HERE / "outputs", HERE / "work"
RNG = np.random.default_rng(20261106)

H = pd.read_pickle(WORK / "hourly.pkl")
P = pd.read_pickle(WORK / "patients.pkl").set_index("stay_id")
H = H.join(P[["anchor_year_group"]], on="stay_id")
N = H.N_unk0 | H.N_gap6 | H.N_drop2 | H.N_sed | H.N_rass | H.N_seiz
v02 = H.R | H.C | N | H.M
extra = H.CBP | H.P
H["seg"] = np.select([v02, extra], ["T1", "T2"], default="T3")
H["phase"] = pd.cut(H.k, [-1, 23, 71, 167, 335, 720],
                    labels=["第1天", "第2–3天", "第4–7天", "第8–14天", "第15–30天"])
PH = ["第1天", "第2–3天", "第4–7天", "第8–14天", "第15–30天"]
UT = ["正规ICU", "神经中间监护/过渡病房"]
eras = sorted(H.anchor_year_group.dropna().unique())

t1 = (pd.crosstab(H.anchor_year_group, H.unit_type, normalize="index") * 100).round(1)[UT]
t1["小时数"] = H.groupby("anchor_year_group").size()

rows = []
for ut in UT:
    hu = H[H.unit_type == ut]
    ov = (hu.seg == "T3").mean() * 100
    ov_dir = np.sign(np.diff([(hu[hu.phase == p].seg == "T3").mean() for p in PH]))
    for e in eras:
        he = hu[hu.anchor_year_group == e]
        t3 = (he.seg == "T3").mean() * 100
        d = np.sign(np.diff([(he[he.phase == p].seg == "T3").mean() for p in PH]))
        rows.append({"单元类型": ut, "年代": e, "人数": he.stay_id.nunique(), "T3%": round(t3, 1),
                     "与该单元合计之差": round(t3 - ov, 1), "条件1 差≤5": abs(t3 - ov) <= 5,
                     "条件3 阶段方向相同": bool((d == ov_dir).all()),
                     "阶段方向": " ".join("↑" if v > 0 else "↓" if v < 0 else "→" for v in d)})
    rows.append({"单元类型": ut, "年代": "全部年代合计", "人数": hu.stay_id.nunique(), "T3%": round(ov, 1),
                 "阶段方向": " ".join("↑" if v > 0 else "↓" if v < 0 else "→" for v in ov_dir)})
t2 = pd.DataFrame(rows).set_index(["单元类型", "年代"])


def ci_table(h):
    cnt = pd.crosstab(h.stay_id, h.seg).reindex(columns=["T1", "T2", "T3"], fill_value=0).to_numpy(float)
    idx = RNG.integers(0, len(cnt), size=(1000, len(cnt)))
    b = np.stack([cnt[i].sum(0) / cnt[i].sum() for i in idx])
    est = cnt.sum(0) / cnt.sum()
    return {s: f"{est[j] * 100:.1f} ({np.percentile(b[:, j], 2.5) * 100:.1f}–{np.percentile(b[:, j], 97.5) * 100:.1f})"
            for j, s in enumerate(["T1", "T2", "T3"])} | {"人数": len(cnt), "小时数": int(cnt.sum())}


t3 = pd.DataFrame({ut: ci_table(H[H.unit_type == ut]) for ut in UT} | {"全部（合并）": ci_table(H)}).T


def md(df):
    idx = df.index
    head = [" / ".join(map(str, idx.names)) if idx.nlevels > 1 else str(idx.name or "")] + list(map(str, df.columns))
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for i, r in zip(idx, df.values):
        i = " / ".join(map(str, i)) if isinstance(i, tuple) else str(i)
        out.append("| " + " | ".join([i] + ["" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r]) + " |")
    return "\n".join(out)


for n, t in [("posthoc_unit_share_by_era", t1), ("posthoc_reproducibility_within_unit", t2),
             ("main_by_unit_type", t3)]:
    t.to_csv(OUT / f"{n}.csv")
ok = {ut: (bool(t2.loc[ut].drop("全部年代合计")["条件1 差≤5"].all()),
           bool(t2.loc[ut].drop("全部年代合计")["条件3 阶段方向相同"].all())) for ut in UT}
lines = ["# 第四篇 · 事后分析：按单元类型（分析方案 v1.1 第 11 节）", "",
         "**本文件中的分析是看到正式结果之后增加的事后分析。** 5.2 的预设结论（不同年代中不稳定）不变。", "",
         "## 主要呈现：按单元类型的 T1 / T2 / T3（小时口径，%，95% 置信区间）", "", md(t3), "",
         "## 各年代中两类单元所占小时的比例（%）", "", md(t1), "",
         "## 单元类型内部的重复性（事后）", "", md(t2), "",
         "单元类型内部：" + "；".join(f"{ut} 条件1 {'满足' if a else '不满足'}、条件3 {'满足' if b else '不满足'}"
                              for ut, (a, b) in ok.items()), ""]
(OUT / "posthoc_by_unit.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
