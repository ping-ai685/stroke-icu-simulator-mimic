"""70_prespecified_items.py 的并行工作函数（放在独立模块里，各进程自行拟合最终模型，避免传递 patsy 对象）。"""
from pathlib import Path

import numpy as np
import pandas as pd

import final_lib as F
import simlib as S

WORK = Path(__file__).resolve().parent / "work"
NSIM = 1000
_C = {}


# ---------------------------------------------------------------------------
# 2. 对调第 3、4 步的模拟引擎（其余与 simlib.simulate_fast 逐行相同）
# ---------------------------------------------------------------------------
def simulate_swapped(pat, start, fm, n, rng, maxh=720, skip_first_update=False):
    k = np.full(n, start["k"], int)
    fstate = np.full(n, start["fstate"], int)
    met = np.full(n, bool(start["met03"]))
    run = np.full(n, start["run_h"], int)
    ever = np.full(n, bool(start["ever_conf"]))
    fck = np.full(n, start["first_conf_k"] if start["ever_conf"] else np.nan, float)
    inter = np.full(n, bool(start["inter"]))
    active = np.ones(n, bool)
    end_type = np.full(n, "censor", dtype=object)
    end_k = np.full(n, -1, int)
    base = {"age10": pat["age10"], "subtype": pat["subtype"]}
    hours = pd.date_range(pat["intime"], periods=maxh + 1, freq="h")
    clock_all, wkend_all = np.asarray(hours.hour) // 4, np.asarray(hours.dayofweek) >= 5
    while active.any():
        idx = np.where(active)[0]
        kk = k[idx]
        k0 = int(kk[0])
        if k0 >= maxh:
            end_k[idx] = maxh - 1
            break
        if skip_first_update and k0 == start["k"]:
            pass
        elif k0 == 6:
            f = {**base, "inter": inter[idx], "reperf": pat["reperf"], "entry": pat["entry"]}
            fstate[idx] = S.sample(fm["D20"].proba(f, len(idx)), rng, fm["D20"].classes_)
        elif k0 >= 12 and k0 % 6 == 0:
            for s in range(4):
                sub = idx[fstate[idx] == s]
                if len(sub):
                    f = {**base, "k": k[sub], "inter": inter[sub]}
                    fstate[sub] = S.sample(fm["D2"][s].proba(f, len(sub)), rng, fm["D2"][s].classes_)
        h_since = np.where(ever[idx], kk - fck[idx], np.nan)
        cidx = np.where(np.isnan(h_since), 0, 1 + np.digitize(np.nan_to_num(h_since), [6, 24, 72]))
        f = {**base, "k": kk, "fstate": fstate[idx], "inter": inter[idx], "met03": met[idx], "ever_conf": ever[idx],
             "conf_cat#idx": cidx, "clock_block": clock_all[k0], "weekend": bool(wkend_all[k0]),
             "run_h": run[idx], "reperf_early": bool(pat["reperf"]) and k0 < 24}
        n_i = len(idx)
        # 第 4 步在前：去向
        dest = S.sample(fm["C1"].proba(f, n_i), rng, fm["C1"].classes_)
        lv = dest == 2
        end_type[idx[lv]], end_k[idx[lv]] = "exit", kk[lv]
        active[idx[lv]] = False
        rem = ~lv
        ir = idx[rem]
        if not len(ir):
            break
        fr = {key: (v[rem] if isinstance(v, np.ndarray) and v.shape == (n_i,) else v) for key, v in f.items()}
        # 第 3 步在后：未离开者的死亡
        dead = rng.random(len(ir)) < fm["D1"].proba(fr, len(ir))[:, 1]
        end_type[ir[dead]], end_k[ir[dead]] = "death", k[ir[dead]]
        active[ir[dead]] = False
        alive = ~dead
        st = ir[alive]
        mv = (dest[rem] == 1)[alive]
        inter[st[mv]] = ~inter[st[mv]]
        if len(st):
            fs = {key: (v[alive] if isinstance(v, np.ndarray) and v.shape == (len(ir),) else v) for key, v in fr.items()}
            m0 = met[st]
            p_on = fm["D3a"].proba(fs, len(st))[:, 1]
            p_off = fm["D3b"].proba(fs, len(st))[:, 1]
            u = rng.random(len(st))
            newm = np.where(m0, u >= p_off, u < p_on)
            run[st] = np.where(newm == m0, run[st] + 1, 0)
            met[st] = newm
            conf = met[st] & (run[st] >= 6)
            first = conf & ~ever[st]
            fck[st[first]] = k[st[first]] + 1
            ever[st] = ever[st] | conf
            k[st] += 1
    return pd.DataFrame({"end_type": end_type, "end_k": end_k})


def model():
    if "fm" not in _C:
        T = S.load_tables(WORK)
        fm = F.build("A", T)[0]
        for s, f in [(0, "D2状态0_elasticnet.npz"), (3, "D2状态3_elasticnet.npz")]:   # 用冻结系数（saga 不完全确定）
            z = np.load(WORK / "frozen" / f)
            fm["D2"][s].c.m.coef_, fm["D2"][s].c.m.intercept_ = z["coef"].copy(), z["intercept"].copy()
        _C.update(T=T, fm=fm)
    return _C["T"], _C["fm"]


def run_swapped(tasks):
    T, fm = model()
    P, H = T["P"], T["H"].set_index(["stay_id", "k"])
    out = []
    for sid, k in tasks:
        p, r = P.loc[sid], H.loc[(sid, k)]
        pat = {"age10": p.age10, "subtype": p.subtype, "reperf": p.reperf, "entry": p.entry, "intime": p.intime}
        start = {"k": k, "fstate": int(r.fstate), "met03": bool(r.met03), "run_h": int(r.run_h),
                 "ever_conf": bool(r.ever_conf), "first_conf_k": (k - r.h_since_conf) if bool(r.ever_conf) else np.nan,
                 "inter": bool(r.inter)}
        res = simulate_swapped(pat, start, fm, NSIM, np.random.default_rng(int(sid) * 100 + k), skip_first_update=k > 0)
        o = {"stay_id": sid, "k": k}
        for h in (24, 48, 72):
            pe = float(((res.end_type == "exit") & (res.end_k <= k + h - 1)).mean())
            pdd = float(((res.end_type == "death") & (res.end_k <= k + h - 1)).mean())
            o[f"p{h}"] = np.array([1 - pe - pdd, pe, pdd])
        out.append(o)
    return out


