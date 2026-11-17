"""
单中心 ICU 路径模拟器 · 端到端试运行（只检查程序，不作任何结论）

只用内部训练段的病人：随机 300 人拟合全部模块（只用 L1），另外 50 人从入 ICU 起各模拟 200 次。
不接触内部验证段、时间验证集和附加验证集。
检查：模块能拟合、概率有效、模拟引擎按第 2.1 节运行、逻辑检查全部为 0、运行时间。
输出：outputs/e2e_test.md
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

import simlib as S

HERE = Path(__file__).resolve().parent
WORK, OUT = HERE / "work", HERE / "outputs"
RNG = np.random.default_rng(20261106)

P = pd.read_pickle(WORK / "v1_patients.pkl")
P["age10"] = P.age_at_adm_capped / 10
P["reperf"] = P.reperfusion.fillna(False).astype(bool)
ids = P[P.int_split == "internal_train"].index.to_numpy()
RNG.shuffle(ids)
fit_ids, sim_ids = set(ids[:300]), ids[300:350]

H = pd.read_pickle(WORK / "v1_hourly.pkl")
H = H[H.stay_id.isin(fit_ids)].copy()
H = H.join(P[["age10", "reperf"]], on="stay_id")
H["conf_cat"] = S.conf_cat(H.h_since_conf)
H["reperf_early"] = H.reperf & (H.k < 24)
H["met03"] = H.met03.astype(bool)
H["fstate"] = H.fstate.astype(int)

t0 = time.time()
mods = {}
mods["D1"] = S.L1Module(S.FORMULAS["D1"]).fit(H, H.y_death.astype(int))
d3 = H[H.in_D3]
a = d3[~d3.met03]; b = d3[d3.met03]
mods["D3a"] = S.L1Module(S.FORMULAS["D3"]).fit(a, a.y_met_next.astype(bool).astype(int))
mods["D3b"] = S.L1Module(S.FORMULAS["D3"]).fit(b, (~b.y_met_next.astype(bool)).astype(int))
c1 = H[H.in_C1]
mods["C1"] = S.L1Module(S.FORMULAS["C1"]).fit(c1, c1.y_dest)
D20 = pd.read_pickle(WORK / "v1_d20.pkl")
D20 = D20[D20.stay_id.isin(fit_ids)].join(P[["age10", "reperf", "entry"]].rename(columns={"entry": "entry_p"}),
                                           on="stay_id")
D20["entry"] = D20.entry_p
mods["D20"] = S.L1Module(S.FORMULAS["D20"]).fit(D20, D20.fstate.astype(int))
D2 = pd.read_pickle(WORK / "v1_d2.pkl")
D2 = D2[D2.stay_id.isin(fit_ids)].join(P[["age10"]], on="stay_id").rename(columns={"k_eff": "k"})
mods["D2"] = {s: S.L1Module(S.FORMULAS["D2"]).fit(D2[D2.prev_state == s], D2[D2.prev_state == s].fstate.astype(int))
              for s in range(4)}
A = pd.read_pickle(WORK / "v1_p1a.pkl")
A = A[A.index.isin(fit_ids)].copy()
A["age10"] = A.age_at_adm_capped / 10
A["exit_fstate"] = A.exit_fstate.astype(int)
mods["P1a"] = S.L1Module(S.FORMULAS["P1a"]).fit(A, A.direct_leave.astype(int))
PB = pd.read_pickle(WORK / "v1_p1b.pkl")
PB = PB[PB.stay_id.isin(fit_ids)].copy()
PB["age10"] = PB.age_at_adm_capped / 10
PB["reperf"] = PB.reperfusion.fillna(False).astype(bool)
PB["exit_fstate"] = PB.exit_fstate.astype(int)
mods["P1b"] = S.L1Module(S.FORMULAS["P1b"]).fit(PB, PB.y_post)
t_fit = time.time() - t0
coef_bad = 0
for m in [mods["D1"], mods["D3a"], mods["D3b"], mods["C1"], mods["D20"], mods["P1a"], mods["P1b"]] + list(mods["D2"].values()):
    coef_bad += int((~np.isfinite(m.m.coef_)).sum() + (~np.isfinite(m.m.intercept_)).sum())

# 概率有效性
prob_bad = 0
for name, m in [("D1", mods["D1"]), ("C1", mods["C1"])]:
    p = m.proba(c1.head(5000))
    prob_bad += int((np.abs(p.sum(1) - 1) > 1e-8).sum() + ((p < 0) | (p > 1)).sum())

# 模拟
Hall = pd.read_pickle(WORK / "v1_hourly.pkl")
first = Hall[Hall.k == 0].set_index("stay_id")
rows, t1 = [], time.time()
for sid in sim_ids:
    p = P.loc[sid]
    pat = {"age10": p.age10, "subtype": p.subtype, "reperf": p.reperf, "entry": p.entry, "intime": p.intime}
    f = first.loc[sid]
    start = {"k": 0, "fstate": -1, "met03": bool(f.met03), "run_h": 0, "ever_conf": False, "first_conf_k": np.nan,
             "inter": bool(f.inter)}
    r = S.simulate(pat, start, mods, 200, np.random.default_rng(int(sid) % 2**32))
    r["stay_id"] = sid
    rows.append(r)
sim = pd.concat(rows)
t_sim = time.time() - t1

# 逻辑检查
lc = {
    "终点类型只有死亡 / 离开 / 删失": int((~sim.end_type.isin(["death", "exit", "censor"])).sum()),
    "删失者终点为第 719 行": int(((sim.end_type == "censor") & (sim.end_k != 719)).sum()),
    "确认 readiness 不早于第 6 个整点": int((sim.ever_conf & (sim.first_conf_k < 6)).sum()),
    "确认 readiness 不晚于终点": int((sim.ever_conf & (sim.first_conf_k > sim.end_k + 1)).sum()),
    "只有离开者有离开后结局": int(((sim.end_type != "exit") & (sim.post != "na")).sum()
                       + ((sim.end_type == "exit") & (sim.post == "na")).sum()),
    "模块概率有效（和为 1、在 0–1 之间）": prob_bad,
    "模块系数均为有限值（数值库警告是否为误报）": coef_bad,
}
lc = pd.Series(lc, name="违规数").to_frame()

# 粗略对照（只为发现程序错误，不作结论）
obs = P.loc[sim_ids]
Hs = Hall[Hall.stay_id.isin(sim_ids)]
obs_conf = Hs.groupby("stay_id").ever_conf.max().mean()
cmp = pd.DataFrame({
    "模拟": [f"{(sim.end_type == 'death').mean():.1%}", f"{(sim.end_k + 1).median():.0f}",
             f"{sim.ever_conf.mean():.1%}", f"{(sim.post == 'direct_leave').mean() / max((sim.end_type == 'exit').mean(), 1e-9):.1%}"],
    "观察（这 50 人）": [f"{obs.died.mean():.1%}", f"{obs['T'].median():.0f}", f"{obs_conf:.1%}",
                    f"{(pd.read_pickle(WORK / 'v1_p1a.pkl').reindex(sim_ids).direct_leave.mean()):.1%}"]},
    index=["ICU 内死亡比例", "ICU 停留中位数（小时）", "曾确认 readiness 的比例", "离开者中直接离开医院的比例"])

lines = ["# 端到端试运行（只检查程序，不作结论）", "",
         f"- 内部训练段随机 300 人拟合全部模块（L1），另 50 人从入 ICU 起各模拟 200 次；未接触任何验证数据。",
         f"- 拟合用时 {t_fit:.1f} 秒；模拟用时 {t_sim:.1f} 秒（平均每人 {t_sim / len(sim_ids):.2f} 秒 / 200 次）。",
         f"- C1 训练中的类别：{list(mods['C1'].classes_)}（0 留下 / 1 转单元 / 2 离开）；P1b 类别：{list(mods['P1b'].classes_)}", "",
         "## 逻辑检查", "", lc.to_markdown() if hasattr(lc, "to_markdown") and False else
         "\n".join(["| 检查 | 违规数 |", "|---|---|"] + [f"| {i} | {v} |" for i, v in lc["违规数"].items()]), "",
         "## 粗略对照（样本很小，只用于发现明显的程序错误）", "",
         "\n".join(["| | 模拟 | 观察 |", "|---|---|---|"] + [f"| {i} | {r['模拟']} | {r['观察（这 50 人）']} |"
                                                       for i, r in cmp.iterrows()]), ""]
(OUT / "e2e_test.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
