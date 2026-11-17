"""
单中心 ICU 路径模拟器 · 共用库：特征构造、模块拟合（L1）、模拟引擎

特征构造在训练和模拟中共用同一套函数，保证二者一致（分析方案 v1.0 第 2.1 节）。
本文件目前只实现 L1（logistic / 多项 logistic + 自然样条），用于端到端试运行；
L2、L3 在正式估计时加入，接口相同。
"""
import numpy as np
import pandas as pd
from patsy import build_design_matrices, dmatrix
from sklearn.linear_model import LogisticRegression

CONF_BINS = [-np.inf, 6, 24, 72, np.inf]
CONF_LAB = ["c0_6", "c6_24", "c24_72", "c72p"]

# 各模块的公式（patsy）；cr() 为自然三次样条，df=4
FORMULAS = {
    "D1": "C(fstate, levels=[-1,0,1,2,3]) + cr(k, df=4) + age10 + C(subtype, levels=['AIS','ICH','SAH']) + inter",
    "D20": "age10 + C(subtype, levels=['AIS','ICH','SAH']) + inter + reperf + C(entry, levels=['ED','OR','direct','prior_ICU','ward'])",
    "D2": "cr(k, df=4) + age10 + C(subtype, levels=['AIS','ICH','SAH']) + inter",
    "D3": "C(fstate, levels=[-1,0,1,2,3]) + cr(k, df=4) + cr(run_h, df=4) + inter + C(subtype, levels=['AIS','ICH','SAH']) + reperf_early",
    "C1": "ever_conf + C(conf_cat, levels=['none'] + " + str(CONF_LAB) + ") + met03 + C(fstate, levels=[-1,0,1,2,3])"
          " + C(clock_block, levels=[0,1,2,3,4,5]) + weekend + inter + C(subtype, levels=['AIS','ICH','SAH']) + cr(k, df=4)",
    "P1a": "C(exit_fstate, levels=[-1,0,1,2,3]) + C(exit_ready, levels=['ready','relapsed','never']) + log_los"
           " + C(subtype, levels=['AIS','ICH','SAH']) + age10",
    "P1b": "cr(h, df=4) + C(exit_ready, levels=['ready','relapsed','never']) + C(exit_fstate, levels=[-1,0,1,2,3])"
           " + log_los + after_hours + exit_inter + C(subtype, levels=['AIS','ICH','SAH']) + age10 + reperf",
}


def conf_cat(h_since):
    h = np.asarray(h_since, float)
    out = np.full(h.shape, "none", dtype=object)
    m = ~np.isnan(h)
    out[m] = pd.cut(h[m], CONF_BINS, labels=CONF_LAB, right=False).astype(str)
    return out


class L1Module:
    """logistic / 多项 logistic（无惩罚）+ patsy 设计矩阵；记住样条结点以便预测时复用。"""

    def __init__(self, formula):
        self.formula = formula

    def fit(self, df, y):
        X = dmatrix(self.formula, df, return_type="dataframe", NA_action="raise")   # 不允许静默丢行
        self.info = X.design_info
        self.m = LogisticRegression(penalty=None, max_iter=5000)
        self.m.fit(X.values, np.asarray(y))
        self.classes_ = self.m.classes_
        return self

    def proba(self, df):
        X = np.asarray(build_design_matrices([self.info], df)[0])
        return self.m.predict_proba(X)


def sample(p, rng, classes):
    """按每行概率抽样，返回类别。"""
    c = p.cumsum(1)
    u = rng.random(len(p))[:, None]
    return np.asarray(classes)[(u > c).sum(1).clip(max=len(classes) - 1)]


# ---------------------------------------------------------------------------
# 模拟引擎（第 2.1 节六步）
# ---------------------------------------------------------------------------
def simulate(pat, start, mods, n, rng, maxh=720):
    """
    pat：病人固定特征（age10, subtype, reperf, entry, intime）
    start：起点状态 dict（k, fstate, met03, run_h, ever_conf, first_conf_k, inter）
    mods：已拟合模块 dict（D1, D20, D2[0..3], D3a, D3b, C1, P1a, P1b）
    返回每次模拟的结果 DataFrame。
    """
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
    relapse = np.zeros(n, bool)
    run_unmet_after = np.zeros(n, int)
    base = {"age10": pat["age10"], "subtype": pat["subtype"]}

    while active.any():
        idx = np.where(active)[0]
        kk = k[idx]
        if kk[0] >= maxh:                                     # 30 天删失
            end_k[idx] = maxh - 1
            break
        # (1) 状态
        if kk[0] == 6:
            d = pd.DataFrame({**base, "inter": inter[idx], "reperf": pat["reperf"], "entry": pat["entry"]})
            fstate[idx] = sample(mods["D20"].proba(d), rng, mods["D20"].classes_)
        elif kk[0] >= 12 and kk[0] % 6 == 0:
            for s in range(4):
                sub = idx[fstate[idx] == s]
                if len(sub):
                    d = pd.DataFrame({**base, "k": k[sub], "inter": inter[sub]})
                    fstate[sub] = sample(mods["D2"][s].proba(d), rng, mods["D2"][s].classes_)
        # (2) 输入
        t = pat["intime"] + pd.Timedelta(hours=int(kk[0]))
        h_since = np.where(ever[idx], kk - fck[idx], np.nan)
        d = pd.DataFrame({**base, "k": kk, "fstate": fstate[idx], "inter": inter[idx], "met03": met[idx],
                          "ever_conf": ever[idx], "conf_cat": conf_cat(h_since), "clock_block": t.hour // 4,
                          "weekend": t.dayofweek >= 5, "run_h": run[idx],
                          "reperf_early": bool(pat["reperf"]) and kk[0] < 24})
        # (3) 死亡
        dead = rng.random(len(idx)) < mods["D1"].proba(d)[:, 1]
        end_type[idx[dead]], end_k[idx[dead]] = "death", kk[dead]
        active[idx[dead]] = False
        alive = ~dead
        ia, da = idx[alive], d[alive].reset_index(drop=True)
        if not len(ia):
            break
        # (4) 去向
        dest = sample(mods["C1"].proba(da), rng, mods["C1"].classes_)
        lv = dest == 2
        end_type[ia[lv]], end_k[ia[lv]] = "exit", k[ia[lv]]
        active[ia[lv]] = False
        mv = dest == 1
        inter[ia[mv]] = ~inter[ia[mv]]
        # (5) 转出条件（留下或转单元者）
        st = ia[~lv]
        ds = da[~lv].reset_index(drop=True)
        if len(st):
            m0 = met[st]
            p_on = mods["D3a"].proba(ds)[:, 1]
            p_off = mods["D3b"].proba(ds)[:, 1]
            u = rng.random(len(st))
            newm = np.where(m0, u >= p_off, u < p_on)
            run[st] = np.where(newm == m0, run[st] + 1, 0)
            met[st] = newm
            conf = met[st] & (run[st] >= 6)
            first = conf & ~ever[st]
            fck[st[first]] = k[st[first]] + 1
            ever[st] = ever[st] | conf
            lost = ever[st] & ~met[st]
            run_unmet_after[st] = np.where(lost, run_unmet_after[st] + 1, 0)
            relapse[st] = relapse[st] | (run_unmet_after[st] >= 6)
            k[st] += 1

    res = pd.DataFrame({"end_type": end_type, "end_k": end_k, "ever_conf": ever, "first_conf_k": fck,
                        "relapse6": relapse, "exit_met": met, "exit_fstate": fstate, "exit_inter": inter})
    # (6) 离开后
    ex = res.end_type == "exit"
    res["post"] = "na"
    if ex.any():
        e = res[ex]
        ready = np.where(e.ever_conf & e.exit_met, "ready", np.where(e.ever_conf, "relapsed", "never"))
        los = np.log((e.end_k + 1).clip(lower=1))
        d = pd.DataFrame({**base, "exit_fstate": e.exit_fstate.values, "exit_ready": ready, "log_los": los.values})
        direct = rng.random(len(e)) < mods["P1a"].proba(d)[:, 1]
        post = np.where(direct, "direct_leave", "none72")
        ih = np.where(~direct)[0]
        exit_hours = np.array([(pat["intime"] + pd.Timedelta(hours=int(x))).hour for x in e.end_k.values])
        pending = ih.copy()
        for h in range(72):                                   # 所有仍在院内、尚无事件的模拟一起推进
            if not len(pending):
                break
            dd = pd.DataFrame({**base, "h": h, "exit_ready": ready[pending], "exit_fstate": e.exit_fstate.values[pending],
                               "log_los": los.values[pending],
                               "after_hours": (exit_hours[pending] >= 19) | (exit_hours[pending] < 7),
                               "exit_inter": e.exit_inter.values[pending], "reperf": pat["reperf"]})
            y = sample(mods["P1b"].proba(dd), rng, mods["P1b"].classes_)
            hit = y != 0
            post[pending[hit]] = np.vectorize({1: "readmit", 2: "death", 3: "discharge"}.get)(y[hit].astype(int)) \
                if hit.any() else post[pending[hit]]
            pending = pending[~hit]
        res.loc[ex, "post"] = post
    return res


# ---------------------------------------------------------------------------
# 快速版本（纯工程优化，不改变方法）
#
# L1 模块的线性预测值 = 截距 + 各项贡献之和（无交互项）。模拟中的时间变量都是整数，
# 分类变量取值有限，因此把每一项在全部可能取值上的贡献预先算成查找表（用同一个 patsy
# design_info 计算，数学上与原版完全相同），模拟时直接查表相加。
# ---------------------------------------------------------------------------
INT_DOMAIN = {"k": 721, "run_h": 721, "h": 72}


class FastModule:
    def __init__(self, l1, baseline_row):
        self.classes_ = l1.classes_
        info, m = l1.info, l1.m
        coef = m.coef_                                   # (1 或 n_class, p)
        self.binary = coef.shape[0] == 1 and len(l1.classes_) == 2
        self.intercept = m.intercept_.copy()
        self.tables, self.linear, self.int_offset = {}, {}, {}
        base = pd.DataFrame([baseline_row])
        for term, sl in info.term_slices.items():
            if term.name() == "Intercept":                  # patsy 的常数列也有系数（与 sklearn 截距共线），并入截距
                self.intercept = self.intercept + coef[:, sl].sum(1)
                continue
            factors = term.factors
            assert len(factors) == 1, "L1 无交互项"
            code = factors[0].code
            if code.startswith("C("):
                var = code[2:code.index(",")].strip()
            elif code.startswith("cr("):
                var = code[3:code.index(",")].strip()
            else:
                var = code.strip()
            if code.startswith("cr(") or var in INT_DOMAIN and not code.startswith("C("):
                dom = np.arange(INT_DOMAIN[var])
            elif code.startswith("C("):
                dom = pd.Index(eval(code[code.index("levels=") + 7:code.rindex(")")]))
            elif isinstance(baseline_row[var], (bool, np.bool_)):
                dom = pd.Index([False, True])
            else:
                self.linear[var] = coef[:, sl].sum(1)       # 连续变量的线性项
                continue
            df = pd.concat([base] * len(dom), ignore_index=True)
            df[var] = list(dom)
            X = np.asarray(build_design_matrices([info], df)[0])[:, sl]
            self.tables[var] = (pd.Index(dom), X @ coef[:, sl].T)   # (取值数, 1 或 n_class)
            d = np.asarray(list(dom))
            if d.dtype.kind in "iub" and len(d) and (np.diff(d.astype(int)) == 1).all():
                self.int_offset[var] = int(d[0])           # 连续整数（含 False/True）：下标 = 值 − 起点

    def eta(self, feats, n):
        e = np.tile(self.intercept, (n, 1)).astype(float)
        for var, (dom, tab) in self.tables.items():
            if var + "#idx" in feats:                       # 引擎预先算好的下标
                e += tab[feats[var + "#idx"]]
                continue
            v = feats[var]
            off = self.int_offset.get(var)
            if off is not None and not (np.isscalar(v) or np.ndim(v) == 0):
                vi = np.asarray(v)
                if vi.dtype.kind in "iub":
                    idx = vi.astype(int) - off
                    assert idx.min() >= 0 and idx.max() < len(dom), var
                    e += tab[idx]
                    continue
            if np.isscalar(v) or np.ndim(v) == 0:
                idx = dom.get_indexer([v])
                assert idx[0] >= 0, (var, v)
                e += tab[idx[0]]
            else:
                idx = dom.get_indexer(np.asarray(v))
                assert (idx >= 0).all(), (var, set(np.asarray(v)[idx < 0]))
                e += tab[idx]
        for var, c in self.linear.items():
            e += np.multiply.outer(np.broadcast_to(feats[var], (n,)).astype(float), c)
        return e

    def proba(self, feats, n):
        e = self.eta(feats, n)
        if self.binary:
            p1 = 1 / (1 + np.exp(-e[:, 0]))
            return np.column_stack([1 - p1, p1])
        e -= e.max(1, keepdims=True)
        p = np.exp(e)
        return p / p.sum(1, keepdims=True)


def fast_modules(mods, baselines):
    out = {k: FastModule(v, baselines[k]) for k, v in mods.items() if k != "D2"}
    out["D2"] = {s: FastModule(v, baselines["D2"]) for s, v in mods["D2"].items()}
    return out


def simulate_fast(pat, start, fm, n, rng, maxh=720, skip_first_update=False, never_lose=False):
    """与 simulate() 相同的六步和随机数使用顺序，只把模块概率改为查表计算。
    never_lose=True 仅供简化方案 C：readiness 一旦确认就不再失去（随机数使用顺序不变）。"""
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
    relapse = np.zeros(n, bool)
    run_unmet_after = np.zeros(n, int)
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
            pass                                               # 起点的状态已是实际观察值（第 2.1 节步骤 1 已发生）
        elif k0 == 6:
            f = {**base, "inter": inter[idx], "reperf": pat["reperf"], "entry": pat["entry"]}
            fstate[idx] = sample(fm["D20"].proba(f, len(idx)), rng, fm["D20"].classes_)
        elif k0 >= 12 and k0 % 6 == 0:
            for s in range(4):
                sub = idx[fstate[idx] == s]
                if len(sub):
                    f = {**base, "k": k[sub], "inter": inter[sub]}
                    fstate[sub] = sample(fm["D2"][s].proba(f, len(sub)), rng, fm["D2"][s].classes_)
        h_since = np.where(ever[idx], kk - fck[idx], np.nan)
        cidx = np.where(np.isnan(h_since), 0, 1 + np.digitize(np.nan_to_num(h_since), [6, 24, 72]))
        f = {**base, "k": kk, "fstate": fstate[idx], "inter": inter[idx], "met03": met[idx], "ever_conf": ever[idx],
             "conf_cat#idx": cidx, "clock_block": clock_all[k0], "weekend": bool(wkend_all[k0]),
             "run_h": run[idx], "reperf_early": bool(pat["reperf"]) and k0 < 24}
        n_i = len(idx)
        dead = rng.random(n_i) < fm["D1"].proba(f, n_i)[:, 1]
        end_type[idx[dead]], end_k[idx[dead]] = "death", kk[dead]
        active[idx[dead]] = False
        alive = ~dead
        ia = idx[alive]
        if not len(ia):
            break
        fa = {key: (v[alive] if isinstance(v, np.ndarray) and v.shape == (n_i,) else v) for key, v in f.items()}
        dest = sample(fm["C1"].proba(fa, len(ia)), rng, fm["C1"].classes_)
        lv = dest == 2
        end_type[ia[lv]], end_k[ia[lv]] = "exit", k[ia[lv]]
        active[ia[lv]] = False
        mv = dest == 1
        inter[ia[mv]] = ~inter[ia[mv]]
        st = ia[~lv]
        if len(st):
            fs = {key: (v[~lv] if isinstance(v, np.ndarray) and v.shape == (len(ia),) else v) for key, v in fa.items()}
            m0 = met[st]
            p_on = fm["D3a"].proba(fs, len(st))[:, 1]
            p_off = fm["D3b"].proba(fs, len(st))[:, 1]
            u = rng.random(len(st))
            newm = np.where(m0, u >= p_off, u < p_on)
            if never_lose:
                newm = newm | ever[st]
            run[st] = np.where(newm == m0, run[st] + 1, 0)
            met[st] = newm
            conf = met[st] & (run[st] >= 6)
            first = conf & ~ever[st]
            fck[st[first]] = k[st[first]] + 1
            ever[st] = ever[st] | conf
            lost = ever[st] & ~met[st]
            run_unmet_after[st] = np.where(lost, run_unmet_after[st] + 1, 0)
            relapse[st] = relapse[st] | (run_unmet_after[st] >= 6)
            k[st] += 1

    res = pd.DataFrame({"end_type": end_type, "end_k": end_k, "ever_conf": ever, "first_conf_k": fck,
                        "relapse6": relapse, "exit_met": met, "exit_fstate": fstate, "exit_inter": inter})
    ex = res.end_type == "exit"
    res["post"] = "na"
    if ex.any():
        e = res[ex]
        ready = np.where(e.ever_conf & e.exit_met, "ready", np.where(e.ever_conf, "relapsed", "never"))
        los = np.log((e.end_k + 1).clip(lower=1)).to_numpy()
        ne = len(e)
        f = {**base, "exit_fstate": e.exit_fstate.to_numpy(), "exit_ready": ready, "log_los": los}
        direct = rng.random(ne) < fm["P1a"].proba(f, ne)[:, 1]
        post = np.where(direct, "direct_leave", "none72").astype(object)
        exit_hours = np.asarray(hours.hour)[e.end_k.to_numpy()]
        pending = np.where(~direct)[0]
        for h in range(72):
            if not len(pending):
                break
            f = {**base, "h": h, "exit_ready": ready[pending], "exit_fstate": e.exit_fstate.to_numpy()[pending],
                 "log_los": los[pending], "after_hours": (exit_hours[pending] >= 19) | (exit_hours[pending] < 7),
                 "exit_inter": e.exit_inter.to_numpy()[pending], "reperf": pat["reperf"]}
            y = sample(fm["P1b"].proba(f, len(pending)), rng, fm["P1b"].classes_)
            hit = y != 0
            if hit.any():
                post[pending[hit]] = np.vectorize({1: "readmit", 2: "death", 3: "discharge"}.get)(y[hit].astype(int))
            pending = pending[~hit]
        res.loc[ex, "post"] = post
    return res


# ---------------------------------------------------------------------------
# 拟合全部 L1 模块（端到端试运行与验证共用）
# ---------------------------------------------------------------------------
def load_tables(work):
    P = pd.read_pickle(work / "v1_patients.pkl")
    P["age10"] = P.age_at_adm_capped / 10
    P["reperf"] = P.reperfusion.fillna(False).astype(bool)
    H = pd.read_pickle(work / "v1_hourly.pkl").join(P[["age10", "reperf"]], on="stay_id")
    H["conf_cat"] = conf_cat(H.h_since_conf)
    H["reperf_early"] = H.reperf & (H.k < 24)
    H["met03"] = H.met03.astype(bool)
    H["fstate"] = H.fstate.astype(int)
    D20 = pd.read_pickle(work / "v1_d20.pkl").drop(columns=["entry"], errors="ignore") \
        .join(P[["age10", "reperf", "entry"]], on="stay_id")
    D2 = pd.read_pickle(work / "v1_d2.pkl").join(P[["age10"]], on="stay_id").rename(columns={"k_eff": "k"})
    A = pd.read_pickle(work / "v1_p1a.pkl")
    A["age10"] = A.age_at_adm_capped / 10
    A["exit_fstate"] = A.exit_fstate.astype(int)
    PB = pd.read_pickle(work / "v1_p1b.pkl")
    PB["age10"] = PB.age_at_adm_capped / 10
    PB["reperf"] = PB.reperfusion.fillna(False).astype(bool)
    PB["exit_fstate"] = PB.exit_fstate.astype(int)
    return {"P": P, "H": H, "D20": D20, "D2": D2, "A": A, "PB": PB}


def fit_all_l1(T, ids):
    H = T["H"][T["H"].stay_id.isin(ids)]
    mods = {"D1": L1Module(FORMULAS["D1"]).fit(H, H.y_death.astype(int))}
    d3 = H[H.in_D3]
    a, b = d3[~d3.met03], d3[d3.met03]
    mods["D3a"] = L1Module(FORMULAS["D3"]).fit(a, a.y_met_next.astype(bool).astype(int))
    mods["D3b"] = L1Module(FORMULAS["D3"]).fit(b, (~b.y_met_next.astype(bool)).astype(int))
    c1 = H[H.in_C1]
    mods["C1"] = L1Module(FORMULAS["C1"]).fit(c1, c1.y_dest)
    D20 = T["D20"][T["D20"].stay_id.isin(ids)]
    mods["D20"] = L1Module(FORMULAS["D20"]).fit(D20, D20.fstate.astype(int))
    D2 = T["D2"][T["D2"].stay_id.isin(ids)]
    mods["D2"] = {s: L1Module(FORMULAS["D2"]).fit(D2[D2.prev_state == s], D2[D2.prev_state == s].fstate.astype(int))
                  for s in range(4)}
    A = T["A"][T["A"].index.isin(ids)]
    mods["P1a"] = L1Module(FORMULAS["P1a"]).fit(A, A.direct_leave.astype(int))
    PB = T["PB"][T["PB"].stay_id.isin(ids)]
    mods["P1b"] = L1Module(FORMULAS["P1b"]).fit(PB, PB.y_post)
    row = lambda df: df.iloc[0].to_dict()  # noqa: E731
    baselines = {"D1": row(c1), "D3a": row(a), "D3b": row(b), "C1": row(c1), "D20": row(D20), "D2": row(D2),
                 "P1a": row(A), "P1b": row(PB)}
    data = {"D1": H, "D3a": a, "D3b": b, "C1": c1, "D20": D20, "D2": D2, "P1a": A, "P1b": PB}
    return mods, baselines, data


class GenericModule:
    """把 selection_lib 的候选（L2 / L3，或任何有 proba(df) 的对象）包装成引擎接口 proba(feats, n)。"""
    CONF_LABELS = np.array(["none"] + CONF_LAB, dtype=object)

    def __init__(self, cand, classes):
        self.c, self.classes_ = cand, np.asarray(classes)

    def proba(self, feats, n):
        d = {}
        for key, v in feats.items():
            if key == "conf_cat#idx":
                d["conf_cat"] = self.CONF_LABELS[np.asarray(v)]
            else:
                d[key] = np.broadcast_to(v, (n,)) if (np.isscalar(v) or np.ndim(v) == 0) else v
        P = self.c.proba(pd.DataFrame(d))
        have = list(self.c.classes_)
        out = np.zeros((n, len(self.classes_)))
        for j, c in enumerate(have):
            out[:, list(self.classes_).index(c)] = P[:, j]
        return out
