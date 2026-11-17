"""
单中心 ICU 路径模拟器 · 系统层面的检查（分析方案 v1.0 第 4.5 节）：配置 A（逐模块选择）对 配置 B（全部 L1）

- 两种配置的模块都只用内部训练段拟合；在内部验证段病人上从 0、24、48、72 小时出发各模拟 1,000 次
- 必须通过的检查：逻辑检查为 0；概率有效；72 小时累积风险校准差 ≤ 0.05；readiness 反复比例相差 ≤ 10 个百分点；
  ICU 停留中位数相差 ≤ 20%。判定按方案第 4.5 节的四格表
- P1（离开后）不参与本检查；引擎中 P1 两种配置都用 L1
- 实施细节：readiness 反复比例的分母为"曾确认 readiness 者"（与第四篇正式分析 5.5 一致）；内部验证段在 72 小时内
  没有删失，观察到的累积发生率即为比例
输出：outputs/system_check.md
"""
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import selection_lib as L  # noqa: E402
import simlib as S  # noqa: E402

WORK, OUT = Path(os.environ.get("SIM_WORK", HERE / "work")), Path(os.environ.get("SIM_OUT", HERE / "outputs"))
NSIM, LANDMARKS, NCHUNK = 1000, [0, 24, 48, 72], 12

_CACHE = {}


def units(T):
    P = T["P"]
    split = P.int_split
    H = T["H"]
    d3 = H[H.in_D3]
    D2 = T["D2"]
    A = T["A"].copy()
    A["stay_id"] = A.index
    return {
        "D1": (H, H.y_death.astype(int), "D1", True),
        "D2-0": (T["D20"], T["D20"].fstate.astype(int), "D20", False),
        **{f"D2（当前状态 {s}）": (D2[D2.prev_state == s], D2[D2.prev_state == s].fstate.astype(int), "D2", False)
           for s in range(4)},
        "D3a 不满足→满足": (d3[~d3.met03], d3[~d3.met03].y_met_next.astype(bool).astype(int), "D3", False),
        "D3b 满足→不满足": (d3[d3.met03], (~d3[d3.met03].y_met_next.astype(bool)).astype(int), "D3", False),
        "C1 去向": (H[H.in_C1], H[H.in_C1].y_dest, "C1", False),
        "P1a 离开后去向": (A, A.direct_leave.astype(int), "P1a", False),
    }, split


ENGINE_KEY = {"D1": "D1", "D2-0": "D20", "D3a 不满足→满足": "D3a", "D3b 满足→不满足": "D3b", "C1 去向": "C1",
              "P1a 离开后去向": "P1a"}


def build(config):
    """在内部训练段上拟合某一配置的全部模块，返回引擎用的模块字典（每个进程只做一次）。"""
    if config in _CACHE:
        return _CACHE[config]
    T = S.load_tables(WORK)
    sel = pickle.load(open(WORK / "sel_fitted.pkl", "rb"))
    U, split = units(T)
    fm = {"D2": {}}
    for name, (df, y, fkey, mono) in U.items():
        tr = (df.stay_id.map(split) if "stay_id" in df else df.index.map(split)) == "internal_train"
        tr = np.asarray(tr)
        dtr, ytr = df[tr].reset_index(drop=True), np.asarray(y)[tr]
        pick = sel[name]["pick"] if (config == "A" and not name.startswith("P1")) else "L1"
        if pick == "L1":
            mod = S.L1Module(S.FORMULAS[fkey]).fit(dtr, ytr)
            eng = S.FastModule(mod, dtr.iloc[0].to_dict())
        elif pick == "L2":
            a, lam = sel[name]["spec"]["L2"]
            c = L.L2(S.FORMULAS[fkey] + L.L2_EXTRA[fkey]).fit_fixed(dtr, ytr, a, lam)
            eng = S.GenericModule(c, c.classes_)
        else:
            nl, mc, it = sel[name]["spec"]["L3"]
            c = L.L3(S.FORMULAS[fkey], monotone_age=mono).fit_fixed(dtr, ytr, dtr.stay_id.to_numpy(), nl, mc, it)
            eng = S.GenericModule(c, c.classes_)
        if name.startswith("D2（"):
            fm["D2"][int(name[-2])] = eng
        else:
            fm[ENGINE_KEY[name]] = eng
    PB = T["PB"]
    PBt = PB[PB.stay_id.map(split) == "internal_train"].reset_index(drop=True)
    fm["P1b"] = S.FastModule(S.L1Module(S.FORMULAS["P1b"]).fit(PBt, PBt.y_post), PBt.iloc[0].to_dict())
    _CACHE[config] = (fm, T)
    return _CACHE[config]


def run_chunk(config, tasks):
    fm, T = build(config)
    P, H = T["P"], T["H"].set_index(["stay_id", "k"])
    out = []
    for sid, k in tasks:
        p = P.loc[sid]
        r = H.loc[(sid, k)]
        pat = {"age10": p.age10, "subtype": p.subtype, "reperf": p.reperf, "entry": p.entry, "intime": p.intime}
        start = {"k": k, "fstate": int(r.fstate), "met03": bool(r.met03), "run_h": int(r.run_h),
                 "ever_conf": bool(r.ever_conf),
                 "first_conf_k": (k - r.h_since_conf) if bool(r.ever_conf) else np.nan, "inter": bool(r.inter)}
        res = S.simulate_fast(pat, start, fm, NSIM, np.random.default_rng(int(sid) * 100 + k),
                              skip_first_update=(k > 0))
        out.append({"stay_id": sid, "k": k,
                    "p_exit24": float(((res.end_type == "exit") & (res.end_k <= k + 23)).mean()),
                    "p_exit72": float(((res.end_type == "exit") & (res.end_k <= k + 71)).mean()),
                    "p_death72": float(((res.end_type == "death") & (res.end_k <= k + 71)).mean()),
                    "los": (res.end_k + 1).to_numpy() if k == 0 else None,
                    "conf": int(res.ever_conf.sum()), "relapse": int((res.relapse6 & res.ever_conf).sum()),
                    "viol": int((~res.end_type.isin(["death", "exit", "censor"])).sum()
                                + ((res.end_type == "censor") & (res.end_k != 719)).sum()
                                + (res.ever_conf & (res.first_conf_k > res.end_k + 1)).sum()
                                + ((res.end_type != "exit") & (res.post != "na")).sum()
                                + ((res.end_type == "exit") & (res.post == "na")).sum())})
    return out


if __name__ == "__main__":
    T = S.load_tables(WORK)
    P, H = T["P"], T["H"]
    val = P[P.int_split == "internal_val"]
    tasks = [(sid, k) for sid in val.index for k in LANDMARKS if val.loc[sid, "T"] > k]
    chunks = [tasks[i::NCHUNK] for i in range(NCHUNK)]

    # 观察值
    Hv = H[H.stay_id.isin(val.index)]
    obs = pd.DataFrame(tasks, columns=["stay_id", "k"]).join(val[["K", "died", "trunc", "T"]], on="stay_id")
    obs["y_exit24"] = (~obs.died & ~obs.trunc & (obs.K <= obs.k + 23)).astype(int)
    adm = val
    obs_exit72 = float((~adm.died & ~adm.trunc & (adm.K <= 71)).mean())
    obs_death72 = float((adm.died & (adm.K <= 71)).mean())
    obs_los_med = float(np.median(np.minimum(adm["T"], 720)))
    rel = []
    for sid, g in Hv.groupby("stay_id"):
        if not g.ever_conf.any():
            continue
        a = g[g.k > g.loc[g.confirmed_now, "k"].min()]
        un = ~a.met03.to_numpy()
        runs, cur = 0, 0
        for v in un:
            cur = cur + 1 if v else 0
            runs = max(runs, cur)
        rel.append(runs >= 6)
    obs_relapse = float(np.mean(rel))

    report, decide = [], {}
    for config in ["A", "B"]:
        t0 = time.time()
        res = Parallel(n_jobs=NCHUNK)(delayed(run_chunk)(config, ch) for ch in chunks)
        R = pd.DataFrame([r for ch in res for r in ch])
        M = R.merge(obs[["stay_id", "k", "y_exit24"]], on=["stay_id", "k"])
        brier = float(((M.p_exit24 - M.y_exit24) ** 2).mean())
        r0 = R[R.k == 0]
        pe72, pd72 = float(r0.p_exit72.mean()), float(r0.p_death72.mean())
        los_med = float(np.median(np.concatenate(r0.los.to_list())))
        relapse = float(R[R.k == 0].relapse.sum() / max(R[R.k == 0].conf.sum(), 1))
        checks = {
            "逻辑检查违规数为 0": int(R.viol.sum()) == 0,
            "概率有效": bool(((R[["p_exit24", "p_exit72", "p_death72"]] >= 0) & (R[["p_exit24", "p_exit72", "p_death72"]] <= 1)).all().all()
                          and ((R.p_exit72 + R.p_death72) <= 1 + 1e-9).all()),
            "72 小时累积风险校准（存活离开、ICU 内死亡）差 ≤ 0.05": abs(pe72 - obs_exit72) <= 0.05 and abs(pd72 - obs_death72) <= 0.05,
            "readiness 反复比例相差 ≤ 10 个百分点": abs(relapse - obs_relapse) <= 0.10,
            "ICU 停留中位数相差 ≤ 20%": abs(los_med - obs_los_med) / obs_los_med <= 0.20,
        }
        decide[config] = {"brier": brier, "pass": all(checks.values())}
        report.append((config, {"24 小时存活离开 Brier（四时点合并）": f"{brier:.4f}",
                                "72 小时存活离开：模拟 / 观察": f"{pe72:.3f} / {obs_exit72:.3f}",
                                "72 小时 ICU 内死亡：模拟 / 观察": f"{pd72:.3f} / {obs_death72:.3f}",
                                "readiness 反复（曾确认者中）：模拟 / 观察": f"{relapse:.3f} / {obs_relapse:.3f}",
                                "ICU 停留中位数（小时）：模拟 / 观察": f"{los_med:.0f} / {obs_los_med:.0f}",
                                "逻辑检查违规数": int(R.viol.sum()),
                                "病人·时点数": len(R), "用时（秒）": round(time.time() - t0)}, checks))
        print(config, "完成", round(time.time() - t0), "秒", flush=True)

    a, b = decide["A"], decide["B"]
    if a["pass"] and b["pass"]:
        final = "B" if a["brier"] > b["brier"] * 1.02 else "A"
        rule = f"A、B 都通过；A 的 Brier 比 B {'差超过' if final == 'B' else '没有差超过'} 2%（相对）→ 选 {final}"
    elif a["pass"]:
        final, rule = "A", "A 通过、B 不通过 → 选 A"
    elif b["pass"]:
        final, rule = "B", "A 不通过、B 通过 → 选 B"
    else:
        final, rule = "暂停", "A、B 都不通过 → 按方案暂停，不宣布任何配置合格"

    lines = ["# 系统层面的检查（分析方案 v1.0 第 4.5 节）", "",
             "两种配置的模块都只用内部训练段拟合；在内部验证段病人上从 0、24、48、72 小时出发各模拟 1,000 次。未接触时间验证集。", "",
             f"**判定：{rule}**", ""]
    for config, info, checks in report:
        lines += [f"## 配置 {config}{'（逐模块选择）' if config == 'A' else '（全部 L1）'}", "",
                  "| 指标 | 值 |", "|---|---|"] + [f"| {k} | {v} |" for k, v in info.items()] + \
                 ["", "| 必须通过的检查 | 结果 |", "|---|---|"] + \
                 [f"| {k} | {'通过' if v else '**未通过**'} |" for k, v in checks.items()] + [""]
    (OUT / "system_check.md").write_text("\n".join(lines), encoding="utf-8")
    pickle.dump({"final": final, "rule": rule, "decide": decide}, open(WORK / "system_check.pkl", "wb"))
    print("\n".join(lines))
