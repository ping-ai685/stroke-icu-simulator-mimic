"""
单中心 ICU 路径模拟器 · 第 2 步：建模用数据表（只整理数据，不估计任何模块）

按《第四篇_模拟器分析方案》第 1–2 节：
  - 过滤态：第一篇的冻结模型（分类发射加 1e-4 下限），用每个 6 小时窗口及以前的数据计算（前向算法）
  - 逐小时表：每个整点可用的过滤态（上一个已结束窗口的状态；入 ICU 后前 6 小时记为"入科初期"）、
    v0.3 条件满足状态及其已持续时间、readiness 是否已确认及确认后的时间、单元类型、钟点、周末，
    以及各模块的结局：死亡（D1）、条件状态的变化（D3）、去向（C1）
  - 窗口表：病情状态的转移（D2）
  - 离开后表：去向（P1a）与离开后 72 小时的逐小时事件（P1b）
  - 年代划分：近似真实入院年（≤2016 建模；2017–2019 验证；≥2020 附加验证）

输出（个体层面，仅本机）：simulator/work/hourly_train.pkl、windows_train.pkl、post_exit.pkl、patients_split.pkl
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
sys.path.insert(0, str(ROOT / "03_code"))
import project_paths  # noqa: E402
from pomegranate_patches import CategoricalMixed  # noqa: E402,F401

B = project_paths.MIMIC_IV
HERE = Path(__file__).resolve().parent
WORK = HERE / "work"; WORK.mkdir(exist_ok=True)
SIMW = HERE.parent / "work"
FW = HERE.parent / "paper4_formal" / "work"
FP = ROOT / "07_paper2_eicu/frozen_params"
T1 = ROOT / "04_outputs/tables"
EPS = 1e-4
ICU_UNITS = {"Cardiac Vascular Intensive Care Unit (CVICU)", "Coronary Care Unit (CCU)", "Intensive Care Unit (ICU)",
             "Medical Intensive Care Unit (MICU)", "Medical/Surgical Intensive Care Unit (MICU/SICU)",
             "Neuro Surgical Intensive Care Unit (Neuro SICU)", "Surgical Intensive Care Unit (SICU)",
             "Trauma SICU (TSICU)", "Neuro Intermediate", "Neuro Stepdown"}
torch.set_grad_enabled(False)

P = pd.read_pickle(FW / "patients.pkl").set_index("stay_id")
H = pd.read_pickle(FW / "hourly.pkl")

# ---------------------------------------------------------------------------
# 1. 年代划分
# ---------------------------------------------------------------------------
pat = pd.read_csv(f"{B}/hosp/patients.csv.gz", usecols=["subject_id", "anchor_year"])
P = P.join(pat.set_index("subject_id"), on="subject_id")
P["approx_year"] = P.anchor_year_group.str[:4].astype(int) + 1 + (P.intime.dt.year - P.anchor_year)
P["split"] = np.select([P.approx_year <= 2016, P.approx_year <= 2019], ["dev", "test"], default="test2020")
P["split_by_group"] = np.where(P.anchor_year_group.isin(["2008 - 2010", "2011 - 2013", "2014 - 2016"]), "dev",
                               np.where(P.anchor_year_group == "2017 - 2019", "test", "test2020"))

# ---------------------------------------------------------------------------
# 2. 过滤态（窗口层面）
# ---------------------------------------------------------------------------
print("过滤态 ...", flush=True)
raw = pd.read_csv(SIMW / "timewindow_raw_full_stay.csv").sort_values(["stay_id", "window_idx"]).reset_index(drop=True)
raw = raw[raw.stay_id.isin(P.index)]
FEATS = pd.read_csv(FP / "feature_order.csv").feature.tolist()
imp = pd.read_csv(FP / "imputation_medians_train.csv").set_index("variable")
zp = pd.read_csv(FP / "zscore_params_train.csv", index_col=0)
for v in [f[:-2] for f in FEATS if f.endswith("_z")]:
    raw[v] = raw.groupby("stay_id")[v].ffill(limit=int(imp.loc[v, "ffill_limit_windows"]))
    raw[v] = raw[v].fillna(imp.loc[v, "train_median_post_ffill"])
    raw[f"{v}_z"] = (raw[v] - zp.loc[v, "mean"]) / zp.loc[v, "std"]
lbl = pd.read_csv(T1 / "state_label_mapping.csv").set_index("raw_state")["canonical_state"].to_dict()
model = torch.load(ROOT / ".model_k4.pt", weights_only=False)
for dist in model.distributions:
    for sub in dist.distributions[-4:]:
        p = torch.clamp(sub.probs.detach().clone(), EPS, 1 - EPS)
        p = p / p.sum(dim=-1, keepdim=True)
        sub.probs = torch.nn.Parameter(p, requires_grad=False)
        sub._log_probs = torch.log(p)
fs = []
for sid, g in raw.groupby("stay_id", sort=False):
    X = torch.tensor(g[FEATS].to_numpy(dtype=np.float32))
    if len(X) == 1:
        la = (torch.stack([d.log_probability(X) for d in model.distributions], 1) + model.starts).numpy()
    else:
        la = model.forward(X.unsqueeze(0))[0].numpy()
    fs.append(la.argmax(1))
raw["fstate"] = pd.Series(np.concatenate(fs), index=raw.index).map(lbl)   # 0–3，与第一篇的规范编号一致
W = raw[["stay_id", "window_idx", "fstate"]].copy()

# ---------------------------------------------------------------------------
# 3. 逐小时表
# ---------------------------------------------------------------------------
print("逐小时表 ...", flush=True)
N = H.N_unk0 | H.N_gap6 | H.N_drop2 | H.N_sed | H.N_rass | H.N_seiz
H["met03"] = ~(H.R | H.C | N | H.M | H.CBP | H.P)
H = H.sort_values(["stay_id", "k"]).reset_index(drop=True)
grp = (H.met03 != H.groupby("stay_id").met03.shift()).cumsum()
H["run_h"] = H.groupby([H.stay_id, grp]).cumcount()                     # 当前状态已持续的小时数（0 起）
H["confirmed_now"] = H.met03 & (H.run_h >= 6)
H["ever_conf"] = H.groupby("stay_id").confirmed_now.cummax()
first_conf = H[H.confirmed_now].groupby("stay_id").k.min()
H["h_since_conf"] = np.where(H.ever_conf, H.k - H.stay_id.map(first_conf), np.nan)
# 可用的过滤态：上一个已结束窗口
H["win_avail"] = H.k // 6 - 1
H = H.merge(W.rename(columns={"window_idx": "win_avail"}), on=["stay_id", "win_avail"], how="left")
H["fstate"] = H.fstate.fillna(-1).astype(int)                           # -1 = 入科初期（尚无完整窗口）
H = H.join(P[["intime", "died", "truncated", "K", "subtype", "age_at_adm_capped", "reperfusion", "split", "split_by_group"]],
           on="stay_id")
t = H.intime + pd.to_timedelta(H.k, unit="h")
H["clock_block"] = (t.dt.hour // 4).astype(int)
H["weekend"] = t.dt.dayofweek >= 5
H["inter"] = H.unit_type == "神经中间监护/过渡病房"
# 结局
# 超过 30 天被截断者：最后一个整点是删失，不是事件
last = (H.k == H.K) & ~H.truncated
H["y_death"] = last & H.died
nxt_unit = H.groupby("stay_id").unit_type.shift(-1)
H["y_dest"] = np.select([last & ~H.died, ~last & nxt_unit.notna() & (nxt_unit != H.unit_type)], [2, 1], default=0)
nxt_met = H.groupby("stay_id").met03.shift(-1)
H["y_met_next"] = nxt_met
keep = ["stay_id", "k", "split", "split_by_group", "fstate", "met03", "run_h", "confirmed_now", "ever_conf",
        "h_since_conf", "unit_type", "inter", "clock_block", "weekend", "subtype", "age_at_adm_capped", "reperfusion",
        "y_death", "y_dest", "y_met_next", "K", "died", "truncated", "careunit"]
H[keep].to_pickle(WORK / "hourly_train.pkl")

# 窗口表（D2）
W = W.join(P[["subtype", "age_at_adm_capped", "split", "split_by_group"]], on="stay_id")
W["fstate_next"] = W.groupby("stay_id").fstate.shift(-1)
u_at = H[H.k % 6 == 0][["stay_id", "k", "inter"]].assign(window_idx=lambda x: x.k // 6)
W = W.merge(u_at[["stay_id", "window_idx", "inter"]], on=["stay_id", "window_idx"], how="left")
W.to_pickle(WORK / "windows_train.pkl")

# ---------------------------------------------------------------------------
# 4. 离开后表（P1a、P1b）
# ---------------------------------------------------------------------------
print("离开后表 ...", flush=True)
alive = P[~P.died & ~P.truncated].copy()
adm = pd.read_csv(f"{B}/hosp/admissions.csv.gz", usecols=["hadm_id", "deathtime"], parse_dates=["deathtime"]) \
    .set_index("hadm_id")
tr = pd.read_csv(f"{B}/hosp/transfers.csv.gz", usecols=["hadm_id", "eventtype", "careunit", "intime"],
                 parse_dates=["intime"])
tr = tr[tr.hadm_id.isin(set(alive.hadm_id))].sort_values(["hadm_id", "intime"])
trg = {h: g for h, g in tr.groupby("hadm_id")}


def destination(h, t0):
    g = trg.get(h)
    if g is None:
        return "direct_discharge"
    x = g[(g.intime >= t0 - pd.Timedelta(minutes=1)) & ~g.careunit.isin(ICU_UNITS)]
    if not len(x) or x.eventtype.iloc[0] == "discharge":
        return "direct_discharge"
    return "ward"


alive["dest"] = [destination(h, t0) for h, t0 in zip(alive.hadm_id, alive.outtime)]
icu = pd.read_csv(f"{B}/icu/icustays.csv.gz", usecols=["hadm_id", "intime"], parse_dates=["intime"])
icu = icu[icu.hadm_id.isin(set(alive.hadm_id))]
re = icu.merge(alive[["hadm_id", "outtime"]].reset_index(), on="hadm_id")
re = re[re.intime > re.outtime].groupby("stay_id").intime.min()
alive["readmit_h"] = (re.reindex(alive.index) - alive.outtime).dt.total_seconds() / 3600
dt = alive.hadm_id.map(adm.deathtime)
alive["death_h"] = np.where(dt > alive.outtime, (dt - alive.outtime).dt.total_seconds() / 3600, np.nan)
lastrow = H.groupby("stay_id").tail(1).set_index("stay_id")
alive["exit_fstate"] = lastrow.fstate.reindex(alive.index)
alive["exit_ready"] = np.select([lastrow.ever_conf.reindex(alive.index) & lastrow.met03.reindex(alive.index),
                                 lastrow.ever_conf.reindex(alive.index).astype(bool)],
                                ["ready", "relapsed"], default="never")
alive["exit_inter"] = lastrow.inter.reindex(alive.index)
alive["after_hours"] = alive.outtime.dt.hour.map(lambda h: h >= 19 or h < 7)
alive["log_los"] = np.log(alive.end_h.clip(lower=1))
rows = []
for sid, r in alive.iterrows():
    ev_r = r.readmit_h if r.readmit_h <= 72 else np.inf
    ev_d = r.death_h if r.death_h <= 72 else np.inf
    end = min(ev_r, ev_d, 72)
    for h in range(int(np.floor(end)) + (0 if end == 72 else 1)):
        y = 0
        if h == int(np.floor(end)) and end < 72:
            y = 1 if ev_r <= ev_d else 2
        rows.append((sid, h, y))
post = pd.DataFrame(rows, columns=["stay_id", "h", "y_post"])          # 0 无事件；1 再入 ICU；2 院内死亡
post = post.join(alive[["exit_fstate", "exit_ready", "exit_inter", "after_hours", "log_los", "dest", "subtype",
                        "age_at_adm_capped", "reperfusion", "split", "split_by_group"]], on="stay_id")
post.to_pickle(WORK / "post_exit.pkl")
P.join(alive[["dest", "readmit_h", "death_h", "exit_fstate", "exit_ready"]]).to_pickle(WORK / "patients_split.pkl")

# 概况（只有汇总数字）
print("划分：", P.split.value_counts().to_dict())
print("逐小时表：", len(H), "行；过滤态分布：", H.fstate.value_counts().sort_index().to_dict())
print("去向结局：", H.y_dest.value_counts().to_dict(), "；死亡事件：", int(H.y_death.sum()))
print("离开后表：", len(post), "行；事件：", post.y_post.value_counts().to_dict(),
      "；去向：", alive.dest.value_counts().to_dict())
