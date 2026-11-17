"""
验证快速引擎（纯工程优化）与原引擎完全一致，并测速度。只用内部训练段，与 30 号试运行同样的 300 / 50 人划分。
输出：outputs/fast_engine_check.md
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

import simlib as S

HERE = Path(__file__).resolve().parent
WORK, OUT = HERE / "work", HERE / "outputs"
T = S.load_tables(WORK)
P = T["P"]
rng = np.random.default_rng(20261106)
ids = P[P.int_split == "internal_train"].index.to_numpy()
rng.shuffle(ids)
fit_ids, sim_ids = set(ids[:300]), ids[300:350]
mods, baselines, data = S.fit_all_l1(T, fit_ids)
fm = S.fast_modules(mods, baselines)

# 1. 模块层面：概率是否相同
rows = []
for name in ["D1", "D3a", "D3b", "C1", "D20", "P1a", "P1b"]:
    df = data[name].sample(min(5000, len(data[name])), random_state=1).reset_index(drop=True)
    p_old = mods[name].proba(df)
    feats = {c: df[c].to_numpy() for c in df.columns}
    p_new = fm[name].proba(feats, len(df))
    rows.append({"模块": name, "比较行数": len(df), "最大绝对差": float(np.abs(p_old - p_new).max())})
for s in range(4):
    df = data["D2"][data["D2"].prev_state == s].head(3000).reset_index(drop=True)
    p_old = mods["D2"][s].proba(df)
    p_new = fm["D2"][s].proba({c: df[c].to_numpy() for c in df.columns}, len(df))
    rows.append({"模块": f"D2（当前状态 {s}）", "比较行数": len(df), "最大绝对差": float(np.abs(p_old - p_new).max())})
t1 = pd.DataFrame(rows)

# 2. 引擎层面：同一随机种子，结果是否逐行相同
Hall = T["H"]
first = Hall[Hall.k == 0].set_index("stay_id")
same, total, t_old, t_new = 0, 0, 0.0, 0.0
for sid in sim_ids[:10]:
    p = P.loc[sid]
    pat = {"age10": p.age10, "subtype": p.subtype, "reperf": p.reperf, "entry": p.entry, "intime": p.intime}
    f = first.loc[sid]
    start = {"k": 0, "fstate": -1, "met03": bool(f.met03), "run_h": 0, "ever_conf": False, "first_conf_k": np.nan,
             "inter": bool(f.inter)}
    a = time.time(); r_old = S.simulate(pat, start, mods, 200, np.random.default_rng(int(sid))); t_old += time.time() - a
    a = time.time(); r_new = S.simulate_fast(pat, start, fm, 200, np.random.default_rng(int(sid))); t_new += time.time() - a
    cols = ["end_type", "end_k", "ever_conf", "relapse6", "post"]
    same += int((r_old[cols].astype(str).values == r_new[cols].astype(str).values).all(1).sum())
    total += len(r_old)

# 3. 速度：50 人 × 1,000 次
a = time.time()
for sid in sim_ids:
    p = P.loc[sid]
    pat = {"age10": p.age10, "subtype": p.subtype, "reperf": p.reperf, "entry": p.entry, "intime": p.intime}
    f = first.loc[sid]
    start = {"k": 0, "fstate": -1, "met03": bool(f.met03), "run_h": 0, "ever_conf": False, "first_conf_k": np.nan,
             "inter": bool(f.inter)}
    S.simulate_fast(pat, start, fm, 1000, np.random.default_rng(int(sid)))
t_fast1000 = (time.time() - a) / len(sim_ids)

lines = ["# 快速引擎验证（纯工程优化）", "", "## 1. 模块概率：快速版 vs 原版", "",
         "\n".join(["| 模块 | 比较行数 | 最大绝对差 |", "|---|---|---|"] +
                   [f"| {r.模块} | {r.比较行数} | {r.最大绝对差:.2e} |" for r in t1.itertuples()]), "",
         "## 2. 模拟结果：同一随机种子", "",
         f"10 人 × 200 次 = {total} 条模拟路径，逐条完全相同的：{same}（{same / total:.2%}）", "",
         "## 3. 速度", "",
         f"- 10 人 × 200 次：原引擎 {t_old:.1f} 秒，快速引擎 {t_new:.1f} 秒（快 {t_old / max(t_new, 1e-9):.0f} 倍）",
         f"- 快速引擎 1,000 次 / 人：平均 {t_fast1000:.2f} 秒"]
(OUT / "fast_engine_check.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
