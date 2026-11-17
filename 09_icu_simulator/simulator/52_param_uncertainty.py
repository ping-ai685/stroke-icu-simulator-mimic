"""
单中心 ICU 路径模拟器 · 参数不确定性（分析方案 v1.0 第 4.6 节）

对建模集按病人有放回抽样 200 次；每次用已选定的算法和超参数重新估计 ICU 内的全部模块（不重新调参；
LightGBM 的树数量与校准器固定为最终模型的值），再对时间验证集中按固定种子分层抽取的 300 个病人·时点
（0 / 24 / 48 / 72 小时各 75 个）各模拟 1,000 次，得到预测概率的 95% 参数不确定性区间。

实施细节（见实施记录）：只在 300 个病人·时点上计算（全部 5,543 个需 100 小时以上）；P1 模块不影响 ICU 内预测，
沿用最终模型；重抽样中重复抽到的病人，其各行按抽中次数重复。
输出：outputs/param_uncertainty.md；work/pu_results.npz
"""
import copy
import pickle
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import final_lib as F  # noqa: E402
import selection_lib as L  # noqa: E402
import simlib as S  # noqa: E402

WORK, OUT = HERE / "work", HERE / "outputs"
NB, NSIM, HOR, NJOBS = 200, 1000, [24, 48, 72], 14
_C = {}


def setup():
    if "T" not in _C:
        T = S.load_tables(WORK)
        fm, raw = F.build("A", T)
        _C.update(T=T, fm=fm, raw=raw)
    return _C["T"], _C["fm"], _C["raw"]


def boot_fit(b):
    """第 b 次重抽样：返回引擎模块字典（ICU 内模块重新估计；P1 沿用最终模型）。"""
    T, fm_final, raw = setup()
    P = T["P"]
    dev = P.index[P.int_split.isin(F.DEV)].to_numpy()
    rng = np.random.default_rng(L.SEED1 + 1000 + b)
    cnt = pd.Series(rng.choice(dev, len(dev), replace=True)).value_counts()
    sel = pickle.load(open(WORK / "sel_fitted.pkl", "rb"))
    fm = {"D2": {}, "P1a": fm_final["P1a"], "P1b": fm_final["P1b"]}
    for name, (df, y, fkey, mono) in F.units(T).items():
        if name.startswith("P1"):
            continue
        sid = df.stay_id if "stay_id" in df else pd.Series(df.index, index=df.index)
        rep = sid.map(cnt).fillna(0).astype(int).to_numpy()
        d = df.iloc[np.repeat(np.arange(len(df)), rep)].reset_index(drop=True)   # 按位置重复，避免索引重复的问题
        yy = np.repeat(np.asarray(y), rep)
        pick, final_mod = raw[name]
        if pick == "L1":
            mod = S.L1Module(S.FORMULAS[fkey]).fit(d, yy)
            eng = S.FastModule(mod, d.iloc[0].to_dict())
        elif pick == "L2":
            a, lam = sel[name]["spec"]["L2"]
            c = L.L2(S.FORMULAS[fkey] + L.L2_EXTRA[fkey]).fit_fixed(d, yy, a, lam)
            eng = S.GenericModule(c, c.classes_)
        else:                                                           # L3：树数量与校准器固定为最终模型的值
            c = copy.copy(final_mod)
            X, _, _ = c._X(d, c.info)
            yi = np.searchsorted(c.classes_, yy)
            prm = dict(c.params, num_threads=1)
            c.b = lgb.train(prm, lgb.Dataset(X, yi), c.iters)
            eng = S.GenericModule(c, c.classes_)
        if name.startswith("D2（"):
            fm["D2"][int(name[-2])] = eng
        else:
            fm[F.ENGINE_KEY[name]] = eng
    return fm


def simulate_set(fm, tasks):
    T, _, _ = setup()
    P, H = T["P"], T["H"].set_index(["stay_id", "k"])
    out = np.zeros((len(tasks), 2 * len(HOR)))
    for i, (sid, k) in enumerate(tasks):
        p, r = P.loc[sid], H.loc[(sid, k)]
        pat = {"age10": p.age10, "subtype": p.subtype, "reperf": p.reperf, "entry": p.entry, "intime": p.intime}
        start = {"k": k, "fstate": int(r.fstate), "met03": bool(r.met03), "run_h": int(r.run_h),
                 "ever_conf": bool(r.ever_conf), "first_conf_k": (k - r.h_since_conf) if bool(r.ever_conf) else np.nan,
                 "inter": bool(r.inter)}
        res = S.simulate_fast(pat, start, fm, NSIM, np.random.default_rng(int(sid) * 100 + k), skip_first_update=k > 0)
        for j, h in enumerate(HOR):
            out[i, j] = ((res.end_type == "exit") & (res.end_k <= k + h - 1)).mean()
            out[i, len(HOR) + j] = ((res.end_type == "death") & (res.end_k <= k + h - 1)).mean()
    return out


def one_boot(b, tasks):
    t0 = time.time()
    fm = boot_fit(b)
    r = simulate_set(fm, tasks)
    return b, r, time.time() - t0


if __name__ == "__main__":
    t_all = time.time()
    T = S.load_tables(WORK)
    tv = pickle.load(open(WORK / "tv_predictions.pkl", "rb"))
    te = tv["te_lm"]
    rng = np.random.default_rng(L.SEED1)
    pick = pd.concat([te[te.k == k].sample(75, random_state=int(rng.integers(1e9))) for k in [0, 24, 48, 72]])
    tasks = list(zip(pick.stay_id, pick.k))
    simA = tv["sims"]["A"].set_index(["stay_id", "k"]).loc[tasks]
    point = np.column_stack([np.stack(simA[f"p{h}"])[:, 1] for h in HOR] + [np.stack(simA[f"p{h}"])[:, 2] for h in HOR])
    res = Parallel(n_jobs=NJOBS, verbose=5)(delayed(one_boot)(b, tasks) for b in range(NB))
    R = np.stack([r for _, r, _ in sorted(res, key=lambda x: x[0])])          # (NB, 300, 6)
    np.savez(WORK / "pu_results.npz", boots=R, point=point, tasks=np.array(tasks))
    lo, hi = np.percentile(R, 2.5, axis=0), np.percentile(R, 97.5, axis=0)
    width = hi - lo
    names = [f"{h} 小时存活离开" for h in HOR] + [f"{h} 小时 ICU 内死亡" for h in HOR]
    tab = pd.DataFrame({
        "最终模型预测（中位数）": np.median(point, 0).round(3),
        "区间宽度 中位数": np.median(width, 0).round(3),
        "区间宽度 第 90 百分位": np.percentile(width, 90, 0).round(3),
        "最终预测落在区间内的比例": ((point >= lo) & (point <= hi)).mean(0).round(3)}, index=names)
    lines = ["# 参数不确定性（分析方案 v1.0 第 4.6 节）", "",
             f"建模集按病人有放回抽样 {NB} 次；时间验证集中分层抽取 {len(tasks)} 个病人·时点，每次各模拟 {NSIM} 次。"
             f"总用时 {round((time.time() - t_all) / 3600, 1)} 小时。", "",
             "| 预测 | " + " | ".join(tab.columns) + " |", "|" + "---|" * (len(tab.columns) + 1)]
    lines += ["| " + i + " | " + " | ".join(str(v) for v in r) + " |" for i, r in tab.iterrows()]
    lines += ["", "说明：区间宽度 = 第 97.5 与第 2.5 百分位之差（概率单位）。这是**参数不确定性区间**，"
                  "反映规则从有限数据估计带来的误差；与病人未来结局的 80% 预测区间不同。"]
    (OUT / "param_uncertainty.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
