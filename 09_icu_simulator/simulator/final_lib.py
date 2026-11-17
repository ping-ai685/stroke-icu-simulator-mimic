"""
最终模型：在指定集合上（最终 = 整个建模集）按已选定的算法和超参数拟合（确定性）。
供 50_freeze_check.py、51_temporal_validation.py 共用。
"""
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import selection_lib as L  # noqa: E402
import simlib as S  # noqa: E402

WORK = Path(os.environ.get("SIM_WORK", HERE / "work"))
DEV = ("internal_train", "internal_val")
ENGINE_KEY = {"D1": "D1", "D2-0": "D20", "D3a 不满足→满足": "D3a", "D3b 满足→不满足": "D3b", "C1 去向": "C1",
              "P1a 离开后去向": "P1a"}


def units(T):
    H, D2 = T["H"], T["D2"]
    d3 = H[H.in_D3]
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
    }


def fit_unit(pick, spec, fkey, mono, d, y):
    if pick == "L1":
        mod = S.L1Module(S.FORMULAS[fkey]).fit(d, y)
        return S.FastModule(mod, d.iloc[0].to_dict()), mod
    if pick == "L2":
        a, lam = spec["L2"]
        c = L.L2(S.FORMULAS[fkey] + L.L2_EXTRA[fkey]).fit_fixed(d, y, a, lam)
        return S.GenericModule(c, c.classes_), c
    nl, mc, it = spec["L3"]
    c = L.L3(S.FORMULAS[fkey], monotone_age=mono).fit_fixed(d, y, d.stay_id.to_numpy(), nl, mc, it)
    return S.GenericModule(c, c.classes_), c


def build(config, T, fit_splits=DEV):
    """config 'A'：按模块选择结果；'B'：全部 L1。返回 (引擎模块字典, 原始模型字典)。"""
    sel = pickle.load(open(WORK / "sel_fitted.pkl", "rb"))
    split = T["P"].int_split
    fm, raw = {"D2": {}}, {}
    for name, (df, y, fkey, mono) in units(T).items():
        s = df.stay_id.map(split) if "stay_id" in df else df.index.map(split)
        m = np.asarray(pd.Series(s).isin(fit_splits))
        d, yy = df[m].reset_index(drop=True), np.asarray(y)[m]
        pick = sel[name]["pick"] if config == "A" else "L1"
        eng, mod = fit_unit(pick, sel[name]["spec"], fkey, mono, d, yy)
        raw[name] = (pick, mod)
        if name.startswith("D2（"):
            fm["D2"][int(name[-2])] = eng
        else:
            fm[ENGINE_KEY[name]] = eng
    PB = T["PB"]
    d = PB[PB.stay_id.map(split).isin(fit_splits)].reset_index(drop=True)
    pick = sel["P1b"]["pick"] if config == "A" else "L1"
    if pick == "L3":
        nl, mc, it = sel["P1b"]["mode_params"]["L3"]
        eng, mod = fit_unit("L3", {"L3": (nl, mc, it)}, "P1b", False, d, d.y_post.to_numpy())
    elif pick == "L2":
        a, lam = sel["P1b"]["mode_params"]["L2"]
        eng, mod = fit_unit("L2", {"L2": (a, lam)}, "P1b", False, d, d.y_post.to_numpy())
    else:
        eng, mod = fit_unit("L1", {}, "P1b", False, d, d.y_post.to_numpy())
    fm["P1b"], raw["P1b"] = eng, (pick, mod)
    return fm, raw
