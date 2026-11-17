"""
单中心 ICU 路径模拟器 · 简化方案比较（分析方案 v1.0 第 6.3 节；技术设计 v0.3 第 5.3 节）

- 简化 A（去掉 readiness 层）：离开只取决于病情状态、钟点、单元等；C1 与 P1a / P1b 去掉全部 readiness 变量
  （C1：ever_conf、确认后时长、met03 及其交互；P1：exit_ready 及其交互）。D3 仍在引擎中运行，但其结果不再进入任何模块
- 简化 B（去向规则不含钟点、周末、单元）：C1 去掉 clock_block、weekend、inter 及含它们的交互
- 简化 C（readiness 一旦确认就不再失去）：训练数据中确认后的各小时 met03 = 满足，确认后的小时不再进入 D3 训练；
  离开时 readiness 只有"满足 / 从未确认"；引擎 never_lose=True；验证时点的起始 met03 同样处理
- 规则：变化了的模块用与第 4.3 节完全相同的算法选择规则重新选择（内部训练段调参校准、内部验证段比较一次；P1b 嵌套交叉验证），
  然后在整个建模集上按选定设定拟合。数据与公式都不变的模块，重新选择的结果与完整模拟器确定性相同，直接沿用
- 评价：与完整模拟器（配置 A）在同一时间验证集（2017–2019）、同一病人·时点、同一随机数种子上配对比较
输出：outputs/ablations.md、outputs/ablation_selection.md；work/abl_*.pkl（个体层面，仅本机）
"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import wasserstein_distance
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import final_lib as F  # noqa: E402
import selection_lib as L  # noqa: E402
import simlib as S  # noqa: E402

WORK, OUT = HERE / "work", HERE / "outputs"
NSIM, NCHUNK, HOR = 1000, 12, [24, 48, 72]
FO, XO = S.FORMULAS, L.L2_EXTRA
RL, RL_C = "levels=['ready','relapsed','never']", "levels=['ready','never']"


def rep(s, a, b):
    assert a in s, a
    return s.replace(a, b)


# ---------------------------------------------------------------------------
# 各简化方案的公式（由完整公式删减得到，逐项断言）
# ---------------------------------------------------------------------------
C1_A = FO["C1"]
for t in ["ever_conf + ", "C(conf_cat, levels=['none'] + " + str(S.CONF_LAB) + ") + ", "met03 + "]:
    C1_A = rep(C1_A, t, "")
X_C1_A = rep(rep(XO["C1"], " + ever_conf:C(clock_block, levels=[0,1,2,3,4,5])", ""), " + ever_conf:inter", "")
C1_B = FO["C1"]
for t in [" + C(clock_block, levels=[0,1,2,3,4,5])", " + weekend", " + inter"]:
    C1_B = rep(C1_B, t, "")
X_C1_B = rep(rep(rep(XO["C1"], " + ever_conf:C(clock_block, levels=[0,1,2,3,4,5])", ""), " + ever_conf:inter", ""),
             " + C(clock_block, levels=[0,1,2,3,4,5]):inter", "")
P1a_A = rep(FO["P1a"], " + C(exit_ready, levels=['ready','relapsed','never'])", "")
P1b_A = rep(FO["P1b"], " + C(exit_ready, levels=['ready','relapsed','never'])", "")
X_P1b_A = ""                                          # 预先列出的两项交互都含 exit_ready
P1a_C, P1b_C, X_P1b_C = rep(FO["P1a"], RL, RL_C), rep(FO["P1b"], RL, RL_C), rep(XO["P1b"], RL, RL_C).replace(RL, RL_C)

D3A, D3B, C1N, P1AN = "D3a 不满足→满足", "D3b 满足→不满足", "C1 去向", "P1a 离开后去向"
ABL = {
    "A": {"label": "简化 A（去掉 readiness 层）", "never_lose": False,
          "units": {C1N: (C1_A, X_C1_A), P1AN: (P1a_A, ""), "P1b": (P1b_A, X_P1b_A)}},
    "B": {"label": "简化 B（去向不含钟点、周末、单元）", "never_lose": False,
          "units": {C1N: (C1_B, X_C1_B)}},
    "C": {"label": "简化 C（readiness 确认后不再失去）", "never_lose": True,
          "units": {D3A: (FO["D3"], XO["D3"]), D3B: (FO["D3"], XO["D3"]), C1N: (FO["C1"], XO["C1"]),
                    P1AN: (P1a_C, ""), "P1b": (P1b_C, X_P1b_C)}},
}


def tables(abl):
    T = S.load_tables(WORK)
    if abl == "C":
        H = T["H"]
        ec = H.ever_conf.astype(bool)
        H.loc[ec, "met03"] = True
        H["in_D3"] = H.in_D3 & ~ec
        for k in ("A", "PB"):
            T[k]["exit_ready"] = T[k].exit_ready.replace({"relapsed": "ready"})
    return T


def units(T):
    """与 final_lib.units 相同的训练单元（简化 C 的 in_D3、met03 已在 tables() 中改写）。"""
    return F.units(T)


# ---------------------------------------------------------------------------
# 算法选择（与 40_select_modules.py 相同的规则）
# ---------------------------------------------------------------------------
def candidates(formula, extra, mono):
    return [L.L1(formula), L.L2(formula + extra), L.L3(formula, monotone_age=mono)]


def describe(c):
    if c.name == "L2":
        return f"α={c.alpha}, λ={c.lam:.2e}"
    if c.name == "L3":
        return f"叶子={c.nl}, 最少样本={c.mc}, 树={c.iters}, 校准 {'T=%.3f' % c.T if c.ncls > 2 else 'Platt'}"
    return "—"


def select_unit(name, df, y, formula, extra, mono, split):
    s = (df.stay_id.map(split) if "stay_id" in df else df.index.map(split))
    s = pd.Series(np.asarray(s))
    tr, va = (s == "internal_train").to_numpy(), (s == "internal_val").to_numpy()
    dtr, dva = df[tr].reset_index(drop=True), df[va].reset_index(drop=True)
    ytr, yva = np.asarray(y)[tr], np.asarray(y)[va]
    classes = np.unique(ytr)
    ref = pd.Series(ytr).value_counts().idxmax()
    res, cands = {}, {}
    for c in candidates(formula, extra, mono):
        c.fit(dtr, ytr, dtr.stay_id.to_numpy())
        Pm = L.expand(c.proba(dva), c.classes_, classes)
        if not set(np.unique(yva)) <= set(classes):
            allc = np.union1d(classes, np.unique(yva))
            res[c.name] = L.evaluate(yva, np.clip(L.expand(Pm, classes, allc), 1e-15, 1), allc, ref)
        else:
            res[c.name] = L.evaluate(yva, Pm, classes, ref)
        cands[c.name] = c
    pick, why = L.choose(res)
    rows = [{"模块": name, "候选": k, "验证对数损失": round(ll, 5),
             "校准斜率": "; ".join(f"{c}:{v:.2f}" for c, v in sl.items()), "合格": "是" if L.eligible(sl) else "否",
             "选中": "★" if k == pick else "", "理由": why if k == pick else "", "参数": describe(cands[k])}
            for k, (ll, sl) in res.items()]
    spec = {"L2": (cands["L2"].alpha, cands["L2"].lam), "L3": (cands["L3"].nl, cands["L3"].mc, cands["L3"].iters)}
    return {"pick": pick, "spec": spec}, rows


def select_p1b(PB, formula, extra, split):
    dev = PB[PB.stay_id.map(split).isin(F.DEV)].reset_index(drop=True)
    y = dev.y_post.to_numpy()
    classes = np.unique(y)
    ref = pd.Series(y).value_counts().idxmax()
    of = L.folds(dev.stay_id.to_numpy(), L.SEED1)
    oof = {k: np.zeros((len(dev), len(classes))) for k in ("L1", "L2", "L3")}
    chosen = {"L2": [], "L3": []}
    for i in range(5):
        tr, va = of != i, of == i
        for c in candidates(formula, extra, False):
            c.fit(dev[tr].reset_index(drop=True), y[tr], dev[tr].stay_id.to_numpy())
            oof[c.name][va] = L.expand(c.proba(dev[va].reset_index(drop=True)), c.classes_, classes)
            if c.name == "L2":
                chosen["L2"].append((c.alpha, c.lam))
            if c.name == "L3":
                chosen["L3"].append((c.nl, c.mc, c.iters))
    res = {k: L.evaluate(y, np.clip(v, 1e-15, 1), classes, ref) for k, v in oof.items()}
    pick, why = L.choose(res)
    mode = {k: pd.Series(v).value_counts().index[0] for k, v in chosen.items()}
    rows = [{"模块": "P1b（嵌套交叉验证）", "候选": k, "验证对数损失": round(ll, 5),
             "校准斜率": "; ".join(f"{c}:{v:.2f}" for c, v in sl.items()), "合格": "是" if L.eligible(sl) else "否",
             "选中": "★" if k == pick else "", "理由": why if k == pick else "",
             "参数": {"L1": "—", "L2": f"α={mode['L2'][0]}, λ={mode['L2'][1]:.2e}",
                    "L3": f"叶子={mode['L3'][0]}, 最少样本={mode['L3'][1]}, 树={mode['L3'][2]}"}[k]}
            for k, (ll, sl) in res.items()]
    return {"pick": pick, "mode_params": mode}, rows


# ---------------------------------------------------------------------------
# 最终拟合（整个建模集）
# ---------------------------------------------------------------------------
def fit_changed(pick, spec, formula, extra, mono, d, y):
    if pick == "L1":
        return S.FastModule(S.L1Module(formula).fit(d, y), d.iloc[0].to_dict())
    if pick == "L2":
        a, lam = spec["L2"]
        c = L.L2(formula + extra).fit_fixed(d, y, a, lam)
        return S.GenericModule(c, c.classes_)
    nl, mc, it = spec["L3"]
    c = L.L3(formula, monotone_age=mono).fit_fixed(d, y, d.stay_id.to_numpy(), nl, mc, it)
    return S.GenericModule(c, c.classes_)


_C = {}


def build_abl(abl, sel=None):
    if abl in _C:
        return _C[abl]
    T = tables(abl)
    fm, _ = F.build("A", T)                               # 不变的模块：与完整模拟器相同的设定
    sel = sel or pickle.load(open(WORK / f"abl_sel_{abl}.pkl", "rb"))
    split = T["P"].int_split
    U = units(T)
    for name, (formula, extra) in ABL[abl]["units"].items():
        if name == "P1b":
            d = T["PB"][T["PB"].stay_id.map(split).isin(F.DEV)].reset_index(drop=True)
            spec = {"L2": sel["P1b"]["mode_params"]["L2"], "L3": sel["P1b"]["mode_params"]["L3"]}
            fm["P1b"] = fit_changed(sel["P1b"]["pick"], spec, formula, extra, False, d, d.y_post.to_numpy())
            continue
        df, y, _, mono = U[name]
        s = df.stay_id.map(split) if "stay_id" in df else df.index.map(split)
        m = np.asarray(pd.Series(np.asarray(s)).isin(F.DEV))
        fm[F.ENGINE_KEY[name]] = fit_changed(sel[name]["pick"], sel[name]["spec"], formula, extra, mono,
                                             df[m].reset_index(drop=True), np.asarray(y)[m])
    _C[abl] = (fm, T)
    return _C[abl]


def run_chunk(abl, tasks):
    fm, T = build_abl(abl)
    nl = ABL[abl]["never_lose"]
    P, H = T["P"], T["H"].set_index(["stay_id", "k"])
    out = []
    for sid, k in tasks:
        p, r = P.loc[sid], H.loc[(sid, k)]
        pat = {"age10": p.age10, "subtype": p.subtype, "reperf": p.reperf, "entry": p.entry, "intime": p.intime}
        start = {"k": k, "fstate": int(r.fstate), "met03": bool(r.met03), "run_h": int(r.run_h),
                 "ever_conf": bool(r.ever_conf), "first_conf_k": (k - r.h_since_conf) if bool(r.ever_conf) else np.nan,
                 "inter": bool(r.inter)}                  # 简化 C：tables() 已把确认后的 met03 改为满足
        res = S.simulate_fast(pat, start, fm, NSIM, np.random.default_rng(int(sid) * 100 + k),
                              skip_first_update=k > 0, never_lose=nl)
        o = {"stay_id": sid, "k": k,
             "viol": int((~res.end_type.isin(["death", "exit", "censor"])).sum()
                         + ((res.end_type == "censor") & (res.end_k != 719)).sum()
                         + (res.ever_conf & (res.first_conf_k > res.end_k + 1)).sum()
                         + ((res.end_type != "exit") & (res.post != "na")).sum()
                         + ((res.end_type == "exit") & (res.post == "na")).sum())}
        for h in HOR:
            pe = float(((res.end_type == "exit") & (res.end_k <= k + h - 1)).mean())
            pdd = float(((res.end_type == "death") & (res.end_k <= k + h - 1)).mean())
            o[f"p{h}"] = np.array([1 - pe - pdd, pe, pdd])
        if k == 0:
            o["los"] = np.bincount(np.minimum(res.end_k.to_numpy() + 1, 720), minlength=721)
            o["relapse"] = (int((res.relapse6 & res.ever_conf).sum()), int(res.ever_conf.sum()))
        out.append(o)
    return out


def boot_diff(groups, a, b, n=1000, seed=L.SEED1):
    df = pd.DataFrame({"g": groups, "a": a, "b": b, "n": 1}).groupby("g").sum().to_numpy()
    rng = np.random.default_rng(seed)
    v = [(df[i, 0].sum() - df[i, 1].sum()) / df[i, 2].sum() for i in (rng.integers(0, len(df), len(df)) for _ in range(n))]
    return np.percentile(v, [2.5, 97.5])


def md(df):
    return "\n".join(["| " + str(df.index.name or "") + " | " + " | ".join(map(str, df.columns)) + " |",
                      "|" + "---|" * (len(df.columns) + 1)] +
                     ["| " + str(i) + " | " + " | ".join(str(v) for v in r) + " |" for i, r in df.iterrows()])


if __name__ == "__main__":
    t_all = time.time()
    for k, v in ABL.items():
        print(k, {n: f for n, (f, _) in v["units"].items()}, flush=True)
    tv = pickle.load(open(WORK / "tv_predictions.pkl", "rb"))
    te = tv["te_lm"].reset_index(drop=True)
    tasks = list(zip(te.stay_id, te.k))
    chunks = [tasks[i::NCHUNK] for i in range(NCHUNK)]
    for abl in ABL:
        t0 = time.time()
        sel_rows = []
        fsel = WORK / f"abl_sel_{abl}.pkl"
        if not fsel.exists():                                    # 1. 变化了的模块重新选择算法
            T = tables(abl)
            split, U, sel = T["P"].int_split, units(T), {}
            for name, (formula, extra) in ABL[abl]["units"].items():
                t1 = time.time()
                if name == "P1b":
                    sel[name], rows = select_p1b(T["PB"], formula, extra, split)
                else:
                    df, y, _, mono = U[name]
                    sel[name], rows = select_unit(name, df, y, formula, extra, mono, split)
                sel_rows += [{"简化方案": abl, **r} for r in rows]
                print(f"[{abl}] {name}: 选 {sel[name]['pick']}，用时 {time.time() - t1:.0f} 秒", flush=True)
            pickle.dump(sel, open(fsel, "wb"))
            pickle.dump(sel_rows, open(WORK / f"abl_selrows_{abl}.pkl", "wb"))
        fsim = WORK / f"abl_sims_{abl}.pkl"
        if not fsim.exists():                                    # 2. 时间验证集上模拟
            r = Parallel(n_jobs=NCHUNK)(delayed(run_chunk)(abl, ch) for ch in chunks)
            sims = pd.DataFrame([o for ch in r for o in ch]).set_index(["stay_id", "k"]).loc[tasks].reset_index()
            pickle.dump(sims, open(fsim, "wb"))
        print(f"[{abl}] 完成，用时 {(time.time() - t0) / 60:.0f} 分钟", flush=True)

    # 3. 评价（与完整模拟器配对比较）--------------------------------------------
    T = S.load_tables(WORK)
    P, H = T["P"], T["H"]
    sims = {"完整模拟器（配置 A）": tv["sims"]["A"]}
    for abl in ABL:
        sims[ABL[abl]["label"]] = pickle.load(open(WORK / f"abl_sims_{abl}.pkl", "rb"))
    g = te.stay_id.to_numpy()
    y = {h: te[f"y{h}"].to_numpy() for h in HOR}
    full = np.stack(sims["完整模拟器（配置 A）"]["p24"])[:, 1]
    se_full = (full - (y[24] == 1)) ** 2
    test_ids = P.index[P.int_split == "test"]
    adm = P.loc[sims["完整模拟器（配置 A）"].query("k == 0").stay_id]
    obs_los = np.minimum(adm["T"].to_numpy(), 720)
    rel = []
    for sid, gg in H[H.stay_id.isin(test_ids)].groupby("stay_id"):
        if not gg.ever_conf.any():
            continue
        a = gg[gg.k > gg.loc[gg.confirmed_now, "k"].min()]
        runs = cur = 0
        for v in ~a.met03.to_numpy():
            cur = cur + 1 if v else 0
            runs = max(runs, cur)
        rel.append(runs >= 6)
    obs_rel = float(np.mean(rel))
    rows, viol = [], 0
    for name, sm in sims.items():
        pr = {h: np.stack(sm[f"p{h}"]) for h in HOR}
        se = (pr[24][:, 1] - (y[24] == 1)) ** 2
        r = {"方案": name, "24h 存活离开 Brier": round(float(se.mean()), 4)}
        if name != "完整模拟器（配置 A）":
            ci = boot_diff(g, se, se_full)
            r["与完整模拟器之差（95% 区间）"] = f"{se.mean() - se_full.mean():+.4f}（{ci[0]:+.4f} 至 {ci[1]:+.4f}）"
            viol += int(sm.viol.sum())
        else:
            r["与完整模拟器之差（95% 区间）"] = "—"
        for h in HOR:
            pe = pr[h][:, 1]
            if h > 24:
                r[f"{h}h 存活离开 Brier"] = round(float(((pe - (y[h] == 1)) ** 2).mean()), 4)
            r[f"{h}h 存活离开 斜率"] = round(L.cal_slope((y[h] == 1).astype(int), np.clip(pe, 1e-6, 1 - 1e-6)), 2)
            r[f"{h}h 存活离开 预测−观察"] = round(float(pe.mean() - (y[h] == 1).mean()), 3)
        r["24h 存活离开 C"] = round(roc_auc_score(y[24] == 1, pr[24][:, 1]), 3)
        r["72h ICU死亡 Brier"] = round(float(((pr[72][:, 2] - (y[72] == 2)) ** 2).mean()), 4)
        dist = np.stack(sm[sm.k == 0].los).sum(0)
        r["停留时间 Wasserstein（小时）"] = round(wasserstein_distance(obs_los, np.arange(721), v_weights=dist), 1)
        c = np.cumsum(dist) / dist.sum()
        r["停留中位数 模拟 / 观察（小时）"] = f"{int(np.searchsorted(c, 0.5))} / {int(np.median(obs_los))}"
        rl = sm[sm.k == 0].relapse
        r["readiness 反复 模拟 / 观察"] = f"{sum(a for a, _ in rl) / max(sum(b for _, b in rl), 1):.3f} / {obs_rel:.3f}"
        rows.append(r)
    tab = pd.DataFrame(rows).set_index("方案")
    sel_rows = []
    for abl in ABL:
        sel_rows += pickle.load(open(WORK / f"abl_selrows_{abl}.pkl", "rb"))
    SR = pd.DataFrame(sel_rows)
    lines = ["# 简化方案比较（分析方案 v1.0 第 6.3 节）", "",
             f"时间验证集（2017–2019）{len(set(g)):,} 人、{len(te):,} 个病人·时点，每个病人·时点模拟 1,000 次；"
             "与完整模拟器同一病人·时点、同一随机数种子，配对比较（病人层面自助法 1,000 次）。"
             f"简化方案的逻辑检查违规数：{viol}。", "",
             "- 简化 A：去掉 readiness 层（C1、P1 不含任何 readiness 变量）",
             "- 简化 B：去向模块 C1 不含钟点、周末、单元类型",
             "- 简化 C：readiness 确认后不再失去（训练数据与引擎一致）",
             "- 变化了的模块按第 4.3 节同一规则重新选择算法，见 outputs/ablation_selection.md；其余模块数据与公式不变，沿用完整模拟器的选择", "",
             md(tab.T), "",
             "说明：Brier 之差为 简化 − 完整，正值表示简化方案更差。readiness 反复：从入 ICU 起模拟、曾确认者中失去 ≥ 6 小时的比例"
             "（简化 A 中 readiness 不影响任何模块，仅作记录；简化 C 按设计为 0）。",
             "", f"总用时 {(time.time() - t_all) / 3600:.1f} 小时。"]
    (OUT / "ablations.md").write_text("\n".join(lines), encoding="utf-8")
    cols = ["简化方案", "模块", "候选", "验证对数损失", "校准斜率", "合格", "选中", "理由", "参数"]
    (OUT / "ablation_selection.md").write_text("\n".join(
        ["# 简化方案中变化了的模块：算法选择（与第 4.3 节同一规则）", "", "| " + " | ".join(cols) + " |",
         "|" + "---|" * len(cols)] + ["| " + " | ".join(str(r[c]) for c in cols) + " |" for _, r in SR.iterrows()]),
        encoding="utf-8")
    print("\n".join(lines))
