"""
单中心 ICU 路径模拟器 · 在建模集内部为每个模块选择算法（分析方案 v1.0 第 4.3 节）

- 除 P1b 外：各候选在内部训练段（≤ 2014）内完成调参、早停和校准，拟合后在内部验证段（2015–2016）上比较一次
- P1b：整个建模集上的嵌套交叉验证（外层 5 折）
- 不接触时间验证集和附加验证集
输出：outputs/module_selection.md、work/sel_*.pkl（内部训练段拟合的各候选，供系统检查使用）
"""
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

import selection_lib as L
import simlib as S

HERE = Path(__file__).resolve().parent
WORK, OUT = Path(os.environ.get("SIM_WORK", HERE / "work")), Path(os.environ.get("SIM_OUT", HERE / "outputs"))
T = S.load_tables(WORK)
P = T["P"]
split = P.int_split
H = T["H"].copy()
H["split"] = H.stay_id.map(split)


def tag(df, key="stay_id"):
    s = df[key].map(split) if key in df else df.index.map(split)
    return pd.Series(np.asarray(s), index=df.index)


A = T["A"].copy()
A["stay_id"] = A.index
d3 = H[H.in_D3]
UNITS = {
    "D1": (H, H.y_death.astype(int), "D1", True),
    "D2-0": (T["D20"], T["D20"].fstate.astype(int), "D20", False),
    **{f"D2（当前状态 {s}）": (T["D2"][T["D2"].prev_state == s], T["D2"][T["D2"].prev_state == s].fstate.astype(int),
                            "D2", False) for s in range(4)},
    "D3a 不满足→满足": (d3[~d3.met03], d3[~d3.met03].y_met_next.astype(bool).astype(int), "D3", False),
    "D3b 满足→不满足": (d3[d3.met03], (~d3[d3.met03].y_met_next.astype(bool)).astype(int), "D3", False),
    "C1 去向": (H[H.in_C1], H[H.in_C1].y_dest, "C1", False),
    "P1a 离开后去向": (A, A.direct_leave.astype(int), "P1a", False),
}
L2KEY = {"D1": "D1", "D20": "D20", "D2": "D2", "D3": "D3", "C1": "C1", "P1a": "P1a", "P1b": "P1b"}


def candidates(fkey, mono):
    return [L.L1(S.FORMULAS[fkey]), L.L2(S.FORMULAS[fkey] + L.L2_EXTRA[L2KEY[fkey]]),
            L.L3(S.FORMULAS[fkey], monotone_age=mono)]


def row_loss(y, Pm, classes):
    idx = np.searchsorted(classes, np.asarray(y))
    return -np.log(np.clip(Pm[np.arange(len(idx)), idx], 1e-15, 1))


def stability(groups, loss_sel, loss_l1, n=200, seed=L.SEED1):
    df = pd.DataFrame({"g": groups, "d": loss_sel - loss_l1, "one": 1}).groupby("g").sum()
    rng = np.random.default_rng(seed)
    v = df.to_numpy()
    out = [v[i, 0].sum() / v[i, 1].sum() for i in (rng.integers(0, len(v), len(v)) for _ in range(n))]
    return np.percentile(out, [2.5, 97.5])


def describe(c):
    if c.name == "L2":
        return f"α={c.alpha}, λ={c.lam:.2e}"
    if c.name == "L3":
        cal = f"T={c.T:.3f}" if c.ncls > 2 else "Platt"
        return f"叶子={c.nl}, 最少样本={c.mc}, 树={c.iters}, 校准 {cal}"
    return "—"


rows, fitted = [], {}
for name, (df, y, fkey, mono) in UNITS.items():
    t0 = time.time()
    s = tag(df)
    tr, va = (s == "internal_train").to_numpy(), (s == "internal_val").to_numpy()
    dtr, dva = df[tr].reset_index(drop=True), df[va].reset_index(drop=True)
    ytr, yva = np.asarray(y)[tr], np.asarray(y)[va]
    gtr = dtr.stay_id.to_numpy()
    classes = np.unique(ytr)
    ref = pd.Series(ytr).value_counts().idxmax()
    res, losses, cands = {}, {}, {}
    for c in candidates(fkey, mono):
        c.fit(dtr, ytr, gtr)
        Pm = L.expand(c.proba(dva), c.classes_, classes)
        if not set(np.unique(yva)) <= set(classes):                  # 验证段出现训练段没有的类别：评价用全部类别
            allc = np.union1d(classes, np.unique(yva))
            Pm = L.expand(Pm, classes, allc)
            res[c.name] = L.evaluate(yva, np.clip(Pm, 1e-15, 1), allc, ref)
            losses[c.name] = row_loss(yva, np.clip(Pm, 1e-15, 1), allc)
        else:
            res[c.name] = L.evaluate(yva, Pm, classes, ref)
            losses[c.name] = row_loss(yva, Pm, classes)
        cands[c.name] = c
    pick, why = L.choose(res)
    ci = stability(dva.stay_id.to_numpy(), losses[pick], losses["L1"]) if pick != "L1" else (np.nan, np.nan)
    fitted[name] = {"pick": pick, "fkey": fkey, "mono": mono,
                    "spec": {"L2": (cands["L2"].alpha, cands["L2"].lam),
                             "L3": (cands["L3"].nl, cands["L3"].mc, cands["L3"].iters)}}
    for k, (ll, sl) in res.items():
        rows.append({"模块": name, "候选": k, "训练行数": int(tr.sum()), "验证行数": int(va.sum()),
                     "验证对数损失": round(ll, 5),
                     "校准斜率": "; ".join(f"{c}:{v:.2f}" for c, v in sl.items()),
                     "合格": "是" if L.eligible(sl) else "否", "参数": describe(cands[k]),
                     "选中": "★" if k == pick else "", "理由": why if k == pick else "",
                     "相对L1对数损失差95%区间": f"{ci[0]:.5f} 至 {ci[1]:.5f}" if k == pick and pick != "L1" else ""})
    print(f"{name}: 选 {pick}（{why}），用时 {time.time() - t0:.0f} 秒", flush=True)

# ---------------------------------------------------------------------------
# P1b：整个建模集上的嵌套交叉验证
# ---------------------------------------------------------------------------
t0 = time.time()
PB = T["PB"]
dev = PB[PB.stay_id.map(split).isin(["internal_train", "internal_val"])].reset_index(drop=True)
y = dev.y_post.to_numpy()
classes = np.unique(y)
ref = pd.Series(y).value_counts().idxmax()
of = L.folds(dev.stay_id.to_numpy(), L.SEED1)
oof = {k: np.zeros((len(dev), len(classes))) for k in ("L1", "L2", "L3")}
chosen_params = {"L2": [], "L3": []}
for i in range(5):
    tr, va = of != i, of == i
    for c in candidates("P1b", False):
        c.fit(dev[tr].reset_index(drop=True), y[tr], dev[tr].stay_id.to_numpy())
        oof[c.name][va] = L.expand(c.proba(dev[va].reset_index(drop=True)), c.classes_, classes)
        if c.name == "L2":
            chosen_params["L2"].append((c.alpha, c.lam))
        if c.name == "L3":
            chosen_params["L3"].append((c.nl, c.mc, c.iters))
res = {k: L.evaluate(y, np.clip(v, 1e-15, 1), classes, ref) for k, v in oof.items()}
pick, why = L.choose(res)
mode = {k: pd.Series(v).value_counts().index[0] for k, v in chosen_params.items()}
for k, (ll, sl) in res.items():
    rows.append({"模块": "P1b 离开后 72 小时（嵌套交叉验证）", "候选": k, "训练行数": int(len(dev)), "验证行数": int(len(dev)),
                 "验证对数损失": round(ll, 5), "校准斜率": "; ".join(f"{c}:{v:.2f}" for c, v in sl.items()),
                 "合格": "是" if L.eligible(sl) else "否",
                 "参数": {"L1": "—", "L2": f"出现最多：α={mode['L2'][0]}, λ={mode['L2'][1]:.2e}",
                        "L3": f"出现最多：叶子={mode['L3'][0]}, 最少样本={mode['L3'][1]}, 树={mode['L3'][2]}"}[k],
                 "选中": "★" if k == pick else "", "理由": why if k == pick else "", "相对L1对数损失差95%区间": ""})
fitted["P1b"] = {"pick": pick, "mode_params": mode}
print(f"P1b: 选 {pick}（{why}），用时 {time.time() - t0:.0f} 秒", flush=True)

with open(WORK / "sel_fitted.pkl", "wb") as fh:
    pickle.dump(fitted, fh)
R = pd.DataFrame(rows)
R.to_csv(OUT / "module_selection.csv", index=False)
cols = ["模块", "候选", "验证对数损失", "校准斜率", "合格", "选中", "理由", "参数", "相对L1对数损失差95%区间"]
lines = ["# 模块的算法选择（建模集内部；分析方案 v1.0 第 4.3 节）", "",
         "除 P1b 外：内部训练段（近似入院年代 ≤ 2014）调参与校准，内部验证段（2015–2016）比较一次。"
         "P1b：整个建模集上的嵌套交叉验证。未接触时间验证集。", "",
         "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for _, r in R.iterrows()]
(OUT / "module_selection.md").write_text("\n".join(lines), encoding="utf-8")
print("完成")
