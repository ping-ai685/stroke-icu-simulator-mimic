"""
单中心 ICU 路径模拟器 · 正式数据审计（锁定方案 v1.0 下）

A. 年代划分
B. 各模块的风险集与事件数（按集合）
C. 训练标签一致性检查（每项给出违规数，必须为 0；不为 0 的写入实施记录）

只读 21_build_training_v1.py 的输出；只输出汇总数字。
"""
from pathlib import Path

import os
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WORK, OUT = Path(os.environ.get("SIM_WORK", HERE / "work")), Path(os.environ.get("SIM_OUT", HERE / "outputs"))
P = pd.read_pickle(WORK / "v1_patients.pkl")
H = pd.read_pickle(WORK / "v1_hourly.pkl")
D20 = pd.read_pickle(WORK / "v1_d20.pkl")
D2 = pd.read_pickle(WORK / "v1_d2.pkl")
A = pd.read_pickle(WORK / "v1_p1a.pkl")
PB = pd.read_pickle(WORK / "v1_p1b.pkl")
notes = pd.read_csv(WORK / "v1_build_notes.csv", index_col=0).iloc[:, 0]
SPL = ["internal_train", "internal_val", "test", "test2020"]

# ---------------------------------------------------------------------------
# A. 年代划分
# ---------------------------------------------------------------------------
tA = pd.DataFrame({
    "人数": P.int_split.value_counts().reindex(SPL),
    "近似入院年代范围": P.groupby("int_split").approx_year.agg(lambda x: f"{x.min()}–{x.max()}").reindex(SPL),
    "ICU内死亡": P.groupby("int_split").died.sum().reindex(SPL),
    "30天截断": P.groupby("int_split").trunc.sum().reindex(SPL)})
tA.loc["合计"] = [len(P), "", int(P.died.sum()), int(P.trunc.sum())]
tA2 = pd.crosstab(P.anchor_year_group, P.int_split).reindex(columns=SPL, fill_value=0)

# ---------------------------------------------------------------------------
# B. 风险集与事件
# ---------------------------------------------------------------------------
def counts(df, ycol, split_col="int_split", labels=None):
    out = {}
    for s in SPL:
        x = df[df[split_col] == s]
        r = {"行数": len(x), "病人": x.stay_id.nunique()}
        vc = x[ycol].value_counts()
        for k, name in (labels or {}).items():
            r[name] = int(vc.get(k, 0))
        out[s] = r
    return pd.DataFrame(out).T


tB = {
    "D1 死亡（每小时）": counts(H, "y_death", labels={True: "事件：死亡"}),
    "D2-0 初始状态": counts(D20, "fstate", labels={0: "状态0", 1: "状态1", 2: "状态2", 3: "状态3"}),
    "D2 状态转移（每 6 小时）": counts(D2, "fstate", labels={0: "到状态0", 1: "到状态1", 2: "到状态2", 3: "到状态3"}),
    "D3a 不满足→满足（每小时）": counts(H[H.in_D3 & ~H.met03].assign(y=lambda x: x.y_met_next.astype(bool)), "y",
                                 labels={True: "事件：变为满足"}),
    "D3b 满足→不满足（每小时）": counts(H[H.in_D3 & H.met03].assign(y=lambda x: ~x.y_met_next.astype(bool)), "y",
                                 labels={True: "事件：变为不满足"}),
    "C1 去向（每小时）": counts(H[H.in_C1], "y_dest", labels={1: "转单元", 2: "离开ICU"}),
    "P1a 离开后去向": counts(A.reset_index().assign(y=lambda x: x.direct_leave), "y",
                         labels={True: "直接离开医院", False: "转院内其他单元"}),
    "P1b 离开后 72 小时（每小时）": counts(PB, "y_post", labels={1: "再入ICU", 2: "院内死亡", 3: "出院"}),
}

# ---------------------------------------------------------------------------
# C. 标签一致性检查
# ---------------------------------------------------------------------------
chk = {}
g = H.groupby("stay_id")
nt = P[~P.trunc]
chk["每人行数 = K + 1"] = int((g.size() != P.K.reindex(g.size().index) + 1).sum())
chk["每人第 0 行存在"] = int((g.k.min() != 0).sum())
chk["行号连续"] = int((g.k.max() - g.k.min() + 1 != g.size()).sum())
ev = H.groupby("stay_id").apply(lambda x: int(x.y_death.sum() + (x.y_dest == 2).sum()))
chk["未截断者恰好一个终点事件"] = int((ev.reindex(nt.index) != 1).sum())
chk["截断者没有终点事件"] = int((ev.reindex(P[P.trunc].index) != 0).sum())
chk["截断者恰好 720 行"] = int((g.size().reindex(P[P.trunc].index) != 720).sum())
lastrow = H[H.k == H.K]
chk["终点事件只出现在最后一行"] = int((H.y_death & (H.k != H.K)).sum() + ((H.y_dest == 2) & (H.k != H.K)).sum())
chk["死亡者的终点为死亡、存活者为离开"] = int(((lastrow.died & ~lastrow.trunc) != lastrow.y_death).sum())
chk["死亡所在行不进入 C1"] = int((H.y_death & H.in_C1).sum())
chk["C1 离开数 = 存活且未截断人数"] = int(abs((H.y_dest == 2).sum() - (~P.died & ~P.trunc).sum()))
chk["D3 只在 k < K 且下一整点有值"] = int((H.in_D3 & H.y_met_next.isna()).sum() + (~H.in_D3 & H.y_met_next.notna()).sum())
chk["readiness 确认需连续满足 ≥ 6 小时"] = int((H.confirmed_now & ((H.run_h < 6) | ~H.met03)).sum())
chk["确认后标记不回退"] = int((H.groupby("stay_id").ever_conf.diff() < 0).sum()) if H.ever_conf.dtype != bool else \
    int((H.ever_conf.astype(int).groupby(H.stay_id).diff() < 0).sum())
chk["第 0–5 行状态为入科初期（−1）"] = int(((H.k < 6) & (H.fstate != -1)).sum())
chk["第 6 行起状态非缺失"] = int(((H.k >= 6) & (H.fstate.isna() | (H.fstate == -1))).sum())
chk["D2-0 人数 = T > 6 的人数"] = int(abs(len(D20) - (P["T"] > 6).sum()))
chk["D2 只用完整窗口"] = int((D2["T"] <= 6 * (D2.window_idx + 1)).sum())
chk["D2 的前一状态非缺失"] = int(D2.prev_state.isna().sum())
chk["D2 生效整点在风险集内"] = int((D2.k_eff > D2.stay_id.map(P.K)).sum())
chk["D2 单元类型非缺失"] = int(D2.inter.isna().sum())
pg = PB.groupby("stay_id")
chk["P1b 每人 ≤ 72 行"] = int((pg.size() > 72).sum())
chk["P1b 每人 ≤ 1 个事件"] = int(((PB.y_post > 0).groupby(PB.stay_id).sum() > 1).sum())
lastp = PB[PB.h == pg.h.transform("max")]
chk["P1b 事件只在最后一行"] = int(((PB.y_post > 0) & (PB.h != pg.h.transform("max"))).sum())
chk["P1b 无事件者恰好 72 行"] = int(((lastp.y_post == 0) & (lastp.h != 71)).sum())
chk["P1b 不含直接离开医院者"] = int(PB.stay_id.isin(A[A.direct_leave].index).sum())
chk["各集合病人不重叠"] = int(P.groupby(P.index).int_split.nunique().gt(1).sum())
tC = pd.Series(chk, name="违规数").to_frame()
tC["结果"] = np.where(tC["违规数"] == 0, "通过", "未通过")

info = pd.Series({
    "P1b 因事件时间早于离开 ICU 而排除的人数": int(notes.get("p1b_excluded_negative_delta", 0)),
    "P1b 同一时刻多个事件（按 死亡 > 再入 > 出院 取舍）的人数": int(notes.get("p1b_exact_ties", 0)),
    "ICU 停留结束时刻恰为整点的人数": int(((P["T"] % 1) == 0).sum()),
    "直接离开医院者的去向：" + "、".join(f"{k} {v}" for k, v in A[A.direct_leave].discharge_location.fillna("缺失")
                                     .value_counts().items()): ""}, name="说明").to_frame()


def md(df):
    df = df.to_frame() if isinstance(df, pd.Series) else df
    head = [str(df.index.name or "")] + list(map(str, df.columns))
    o = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    o += ["| " + " | ".join([str(i)] + ["" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r]) + " |"
          for i, r in zip(df.index, df.values)]
    return "\n".join(o)


lines = ["# 正式数据审计（分析方案 v1.0）", "",
         f"标签一致性检查：{(tC['违规数'] == 0).sum()} / {len(tC)} 项通过。", "",
         "## A. 年代划分", "", md(tA), "", "按 anchor_year_group 与本研究划分的对照：", "", md(tA2), "",
         "## B. 各模块的风险集与事件", ""]
for k, t in tB.items():
    lines += [f"### {k}", "", md(t), ""]
lines += ["## C. 训练标签一致性检查", "", md(tC), "", "## 说明", "", md(info), ""]
(OUT / "data_audit_v1.md").write_text("\n".join(lines), encoding="utf-8")
for k, t in tB.items():
    t.to_csv(OUT / f"audit_B_{k.split()[0]}.csv")
tC.to_csv(OUT / "audit_C_label_checks.csv")
print("\n".join(lines))
