"""61_interpretability.py 的并行工作函数（放在独立模块里，各进程自行拟合最终模型并缓存，避免传递 patsy 对象）。"""
from pathlib import Path

import numpy as np

import final_lib as F
import simlib as S

WORK = Path(__file__).resolve().parent / "work"
NSIM = 1000
_C = {}


def model():
    if "fm" not in _C:
        T = S.load_tables(WORK)
        fm, raw = F.build("A", T)
        _C.update(T=T, fm=fm, raw=raw)
    return _C["T"], _C["fm"], _C["raw"]


def test_chunk(ids):
    T, fm, _ = model()
    P, H = T["P"], T["H"].set_index(["stay_id", "k"])
    out = []
    for sid in ids:
        p, r = P.loc[sid], H.loc[(sid, 0)]
        pat = {"age10": p.age10, "subtype": p.subtype, "reperf": p.reperf, "entry": p.entry, "intime": p.intime}
        start = {"k": 0, "fstate": int(r.fstate), "met03": bool(r.met03), "run_h": int(r.run_h),
                 "ever_conf": bool(r.ever_conf), "first_conf_k": np.nan, "inter": bool(r.inter)}
        res = S.simulate_fast(pat, start, fm, NSIM, np.random.default_rng(int(sid) * 100))
        ex = res[(res.end_type == "exit")]
        out.append({"stay_id": sid, "n_exit": len(ex), "n_exit_conf": int(ex.ever_conf.sum()),
                    "before": ex.first_conf_k[ex.ever_conf].to_numpy(),
                    "after": (ex.end_k + 1 - ex.first_conf_k)[ex.ever_conf].to_numpy()})
    return out
