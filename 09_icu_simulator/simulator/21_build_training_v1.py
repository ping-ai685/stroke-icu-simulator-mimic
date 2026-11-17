"""
单中心 ICU 路径模拟器 · 按已锁定的分析方案 v1.0（第 2.1 节时间约定）生成训练数据表

与 20_build_training_table.py 的区别（全部依据 v1.0）：
  - 行 k 的风险集：ICU 停留结束时刻 T > k；事件行 K = ceil(T) − 1；30 天截断者为第 0–719 行、无事件
  - D2-0：第 6 个整点仍在 ICU 且存活者的窗口 0 状态；D2：从第 12 个整点起，只用完整窗口的转移
  - C1：死亡所在的行不进入；离开 = 第 K 行（活着、未截断）；转单元 = 第 k 与 k+1 个整点单元类型不同
  - D3：只取第 k+1 个整点仍在风险集中的行（k < K）
  - P1a：出院时刻与离开 ICU 相差 ≤ 1 小时为直接离开医院
  - P1b：4 类（再入 ICU / 院内死亡 / 出院 / 都没有），h = max(0, ceil(Δ) − 1)，Δ ≤ 72；
         同一行多事件按精确时间，相同时 死亡 > 再入 ICU > 出院；Δ < 0 的病人排除并计数
过滤态沿用 20 步已计算的窗口过滤态（windows_train.pkl 中的 fstate）。

输出（个体层面，仅本机）：simulator/work/v1_*.pkl
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
import project_paths  # noqa: E402

B = project_paths.MIMIC_IV
HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get("SIM_WORK", HERE / "work"))
FW = HERE.parent / "paper4_formal" / "work"
ICU_UNITS = {"Cardiac Vascular Intensive Care Unit (CVICU)", "Coronary Care Unit (CCU)", "Intensive Care Unit (ICU)",
             "Medical Intensive Care Unit (MICU)", "Medical/Surgical Intensive Care Unit (MICU/SICU)",
             "Neuro Surgical Intensive Care Unit (Neuro SICU)", "Surgical Intensive Care Unit (SICU)",
             "Trauma SICU (TSICU)", "Neuro Intermediate", "Neuro Stepdown"}
MAXH = 720

P = pd.read_pickle(WORK / "patients_split.pkl")            # 含 split、approx_year、intime、outtime、end_h、died 等
P["T"] = P.end_h
P["trunc"] = P["T"] > MAXH
P["K"] = np.where(P.trunc, MAXH - 1, np.ceil(P["T"]).astype(int) - 1)
if os.environ.get("SIM_SPLIT") == "group":                # 敏感性分析 1（方案第 10 节）：按 anchor_year_group 划分
    P["split"] = P.split_by_group
    P["int_split"] = np.select([(P.split == "dev") & P.anchor_year_group.isin(["2008 - 2010", "2011 - 2013"]),
                                (P.split == "dev")], ["internal_train", "internal_val"], default=P.split)
else:
    P["int_split"] = np.select([(P.split == "dev") & (P.approx_year <= 2014), (P.split == "dev")],
                               ["internal_train", "internal_val"], default=P.split)

# ---------------------------------------------------------------------------
# 逐小时表
# ---------------------------------------------------------------------------
H = pd.read_pickle(FW / "hourly.pkl")
H = H.merge(P[["K", "trunc", "died"]], left_on="stay_id", right_index=True)
H = H[H.k <= H.K].sort_values(["stay_id", "k"]).reset_index(drop=True)
N = H.N_unk0 | H.N_gap6 | H.N_drop2 | H.N_sed | H.N_rass | H.N_seiz
if os.environ.get("SIM_READY") == "v02":                   # 敏感性分析 2（方案第 10 节）：用 v0.2 代替 v0.3（变量名沿用 met03）
    H["met03"] = ~(H.R | H.C | N | H.M)
else:
    H["met03"] = ~(H.R | H.C | N | H.M | H.CBP | H.P)
grp = (H.met03 != H.groupby("stay_id").met03.shift()).cumsum()
H["run_h"] = H.groupby([H.stay_id, grp]).cumcount()
H["confirmed_now"] = H.met03 & (H.run_h >= 6)
H["ever_conf"] = H.groupby("stay_id").confirmed_now.cummax()
fc = H[H.confirmed_now].groupby("stay_id").k.min()
H["h_since_conf"] = np.where(H.ever_conf, H.k - H.stay_id.map(fc), np.nan)

W = pd.read_pickle(WORK / "windows_train.pkl")[["stay_id", "window_idx", "fstate"]]
H["win_avail"] = H.k // 6 - 1
H = H.merge(W.rename(columns={"window_idx": "win_avail"}), on=["stay_id", "win_avail"], how="left")
H["fstate"] = np.where(H.win_avail < 0, -1, H.fstate)
H = H.join(P[["intime", "subtype", "age_at_adm_capped", "reperfusion", "split", "int_split", "approx_year"]],
           on="stay_id")
t = H.intime + pd.to_timedelta(H.k, unit="h")
H["clock_block"] = (t.dt.hour // 4).astype(int)
H["weekend"] = t.dt.dayofweek >= 5
H["inter"] = H.unit_type == "神经中间监护/过渡病房"

last = (H.k == H.K) & ~H.trunc
H["y_death"] = last & H.died
nxt_unit = H.groupby("stay_id").unit_type.shift(-1)
H["y_dest"] = np.select([last & ~H.died, ~last & nxt_unit.notna() & (nxt_unit != H.unit_type)], [2, 1], default=0)
H["in_C1"] = ~H.y_death                                         # 死亡所在行不进入 C1
H["y_met_next"] = H.groupby("stay_id").met03.shift(-1)          # 只在 k < K 时有值
H["in_D3"] = H.k < H.K
keep = ["stay_id", "k", "K", "split", "int_split", "approx_year", "fstate", "met03", "run_h", "confirmed_now",
        "ever_conf", "h_since_conf", "unit_type", "inter", "clock_block", "weekend", "subtype", "age_at_adm_capped",
        "reperfusion", "y_death", "y_dest", "in_C1", "y_met_next", "in_D3", "died", "trunc", "careunit"]
H[keep].to_pickle(WORK / "v1_hourly.pkl")

# ---------------------------------------------------------------------------
# D2-0 与 D2
# ---------------------------------------------------------------------------
W = W.join(P[["T", "subtype", "age_at_adm_capped", "reperfusion", "split", "int_split"]], on="stay_id")
unit_at = H[["stay_id", "k", "inter"]]
d20 = W[(W.window_idx == 0) & (W["T"] > 6)].copy()
d20 = d20.merge(unit_at[unit_at.k == 0][["stay_id", "inter"]], on="stay_id", how="left")
# 入科来源（ICU 前最后所在单元，与路径分析 10_path_analysis.py 的归类相同）
tr0 = pd.read_csv(f"{B}/hosp/transfers.csv.gz", usecols=["hadm_id", "careunit", "intime"], parse_dates=["intime"])
tr0 = tr0[tr0.hadm_id.isin(set(P.hadm_id))].sort_values(["hadm_id", "intime"])
g0 = {h: g for h, g in tr0.groupby("hadm_id")}


def entry(h, t0):
    g = g0.get(h)
    if g is None:
        return "direct"
    x = g[g.intime < t0 - pd.Timedelta(minutes=1)]
    if not len(x) or pd.isna(x.careunit.iloc[-1]):
        return "direct"
    u = str(x.careunit.iloc[-1])
    if "Emergency" in u:
        return "ED"
    if any(k in u for k in ["PACU", "Operating", "Procedure", "Cath", "Interventional"]):
        return "OR"
    if u in ICU_UNITS:
        return "prior_ICU"
    return "ward"


P["entry"] = [entry(h, t0) for h, t0 in zip(P.hadm_id, P.intime)]
d20 = d20.join(P[["entry"]], on="stay_id")
d20.to_pickle(WORK / "v1_d20.pkl")

W = W.sort_values(["stay_id", "window_idx"])
W["prev_state"] = W.groupby("stay_id").fstate.shift()
d2 = W[(W.window_idx >= 1) & (W["T"] > 6 * (W.window_idx + 1))].copy()   # 窗口 w 完整：第 6(w+1) 个整点仍在 ICU
d2["k_eff"] = 6 * (d2.window_idx + 1)
d2 = d2[d2.k_eff <= d2.stay_id.map(P.K)]                               # 30 天删失：生效整点须在风险集内（≤ 第 719 行）
d2 = d2.merge(unit_at.rename(columns={"k": "k_eff"}), on=["stay_id", "k_eff"], how="left")
d2.to_pickle(WORK / "v1_d2.pkl")

# ---------------------------------------------------------------------------
# 离开后（P1a、P1b）
# ---------------------------------------------------------------------------
alive = P[~P.died & ~P.trunc].copy()
adm = pd.read_csv(f"{B}/hosp/admissions.csv.gz", usecols=["hadm_id", "dischtime", "deathtime", "discharge_location"],
                  parse_dates=["dischtime", "deathtime"]).set_index("hadm_id")
alive["dischtime"] = alive.hadm_id.map(adm.dischtime)
alive["deathtime"] = alive.hadm_id.map(adm.deathtime)
alive["discharge_location"] = alive.hadm_id.map(adm.discharge_location)
icu = pd.read_csv(f"{B}/icu/icustays.csv.gz", usecols=["hadm_id", "intime"], parse_dates=["intime"])
re = icu[icu.hadm_id.isin(set(alive.hadm_id))].merge(alive[["hadm_id", "outtime"]].reset_index(), on="hadm_id")
re = re[re.intime > re.outtime].groupby("stay_id").intime.min()
alive["readmit_time"] = re.reindex(alive.index)
hrs = lambda x: (x - alive.outtime).dt.total_seconds() / 3600   # noqa: E731
alive["d_dis"], alive["d_death"], alive["d_re"] = hrs(alive.dischtime), hrs(alive.deathtime), hrs(alive.readmit_time)
alive["direct_leave"] = alive.d_dis <= 1
last_row = H.groupby("stay_id").tail(1).set_index("stay_id")
alive["exit_fstate"] = last_row.fstate.reindex(alive.index)
alive["exit_ready"] = np.select([last_row.ever_conf.reindex(alive.index) & last_row.met03.reindex(alive.index),
                                 last_row.ever_conf.reindex(alive.index).astype(bool)], ["ready", "relapsed"],
                                default="never")
alive["exit_inter"] = last_row.inter.reindex(alive.index)
alive["after_hours"] = alive.outtime.dt.hour.map(lambda h: h >= 19 or h < 7)
alive["log_los"] = np.log(alive["T"].clip(lower=1))
alive.to_pickle(WORK / "v1_p1a.pkl")

ih = alive[~alive.direct_leave].copy()
neg = (ih[["d_re", "d_death", "d_dis"]] < 0).any(axis=1)
n_neg = int(neg.sum())
ih = ih[~neg]
rows, ties = [], 0
PRI = {"death": 0, "readmit": 1, "discharge": 2}
CODE = {"readmit": 1, "death": 2, "discharge": 3}
for sid, r in ih.iterrows():
    ev = [(r.d_re, "readmit"), (r.d_death, "death"), (r.d_dis, "discharge")]
    ev = [(d, n) for d, n in ev if pd.notna(d) and d <= 72]
    if ev:
        dmin = min(d for d, _ in ev)
        tied = [n for d, n in ev if d == dmin]
        ties += len(tied) > 1
        name = sorted(tied, key=PRI.get)[0]
        hE = max(0, int(np.ceil(dmin)) - 1)
        for h in range(hE + 1):
            rows.append((sid, h, CODE[name] if h == hE else 0))
    else:
        for h in range(72):
            rows.append((sid, h, 0))
post = pd.DataFrame(rows, columns=["stay_id", "h", "y_post"])   # 0 都没有 1 再入ICU 2 院内死亡 3 出院
post = post.join(ih[["exit_fstate", "exit_ready", "exit_inter", "after_hours", "log_los", "subtype",
                     "age_at_adm_capped", "reperfusion", "split", "int_split"]], on="stay_id")
post.to_pickle(WORK / "v1_p1b.pkl")
P.to_pickle(WORK / "v1_patients.pkl")
pd.Series({"p1b_excluded_negative_delta": n_neg, "p1b_exact_ties": ties}).to_csv(WORK / "v1_build_notes.csv")
print("完成", len(H), len(d20), len(d2), len(alive), len(post), "排除", n_neg, "同时", ties)
