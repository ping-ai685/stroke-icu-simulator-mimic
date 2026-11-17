"""
单中心 ICU 路径模拟器 · 模块的候选算法（L1 / L2 / L3）与选择规则（分析方案 v1.0 第 4.2–4.4 节）

实施细节（方案未写定之处，见实施记录）：
  - L2 的 λ 按 glmnet 定义（平均损失 + λ·惩罚），换算为 scikit-learn：C = 1 / (n_train · λ)
  - 按病人分折：病人随机排序（种子）后按序号对 5 取余
  - 多分类的校准斜率：参照类别取内部训练段中最常见的类别；每个非参照类别 c，以 1[y=c] 对 logit(p_c) 做 logistic 回归
  - L3 的连续时间变量用原值（不用样条），分类变量与 L1 用同一套独热编码
"""
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from patsy import build_design_matrices, dmatrix
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

import simlib as S

SEED1, SEED2 = 20261106, 20261107
LAMBDAS = np.logspace(-4, 1, 20)
ALPHAS = [0.0, 0.5, 1.0]
GRID_L3 = [(nl, mc) for nl in (7, 15, 31) for mc in (100, 200, 1000)]
N_JOBS = 15

# L2：L1 公式 + 事先列出的交互项
L2_EXTRA = {
    "D1": " + C(fstate, levels=[-1,0,1,2,3]):cr(k, df=4) + C(fstate, levels=[-1,0,1,2,3]):inter",
    "D20": "",
    "D2": " + cr(k, df=4):inter",
    "D3": " + C(fstate, levels=[-1,0,1,2,3]):cr(k, df=4) + C(fstate, levels=[-1,0,1,2,3]):inter",
    "C1": " + ever_conf:C(clock_block, levels=[0,1,2,3,4,5]) + ever_conf:inter"
          " + C(clock_block, levels=[0,1,2,3,4,5]):inter + C(fstate, levels=[-1,0,1,2,3]):cr(k, df=4)",
    "P1a": "",
    "P1b": " + C(exit_ready, levels=['ready','relapsed','never']):cr(h, df=4)"
           " + after_hours:C(exit_ready, levels=['ready','relapsed','never'])",
}


def l3_formula(f):
    for v in ("k", "run_h", "h"):
        f = f.replace(f"cr({v}, df=4)", v)
    return f


def folds(groups, seed, k=5):
    u = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(u)
    m = pd.Series(np.arange(len(u)) % k, index=u)
    return m.reindex(groups).to_numpy()


# ---------------------------------------------------------------------------
# 评价
# ---------------------------------------------------------------------------
def logit(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def cal_slope(y_bin, p):
    if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
        return np.nan
    lr = LogisticRegression(penalty=None, max_iter=1000).fit(logit(p)[:, None], y_bin)
    return float(lr.coef_[0, 0])


def evaluate(y, P, classes, ref):
    """返回 平均对数损失 与 各非参照类别的校准斜率（二分类只有一个）。"""
    y = np.asarray(y)
    ll = log_loss(y, P, labels=classes)
    if len(classes) == 2:
        slopes = {classes[1]: cal_slope((y == classes[1]).astype(int), P[:, 1])}
    else:
        slopes = {c: cal_slope((y == c).astype(int), P[:, i]) for i, c in enumerate(classes) if c != ref}
    return ll, slopes


def eligible(slopes):
    v = [s for s in slopes.values() if not np.isnan(s)]
    return len(v) > 0 and all(0.8 <= s <= 1.2 for s in v)


# ---------------------------------------------------------------------------
# 三种候选（统一接口：fit(df, y, groups) → 对象，.proba(df) → 概率矩阵，.classes_）
# ---------------------------------------------------------------------------
class L1:
    name = "L1"

    def __init__(self, formula):
        self.formula = formula

    def fit(self, df, y, groups=None):
        self.mod = S.L1Module(self.formula).fit(df, y)
        self.classes_ = self.mod.classes_
        return self

    def proba(self, df):
        return self.mod.proba(df)


def _cv_l2(X, y, f, labels, alpha, lam, i):
    tr, va = f != i, f == i
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
    m = L2._fit_one(X[tr], y[tr], alpha, lam, mu, sd)
    P = expand(m.predict_proba((X[va] - mu) / sd), m.classes_, labels)
    return alpha, lam, i, log_loss(y[va], P, labels=labels), int(va.sum())


class L2:
    name = "L2"

    def __init__(self, formula):
        self.formula = formula

    def _X(self, df, info=None):
        if info is None:
            X = dmatrix(self.formula, df, return_type="dataframe", NA_action="raise")
            return X.values[:, 1:], X.design_info                      # 去掉常数列，由模型截距承担
        return np.asarray(build_design_matrices([info], df)[0])[:, 1:], info

    @staticmethod
    def _fit_one(X, y, alpha, lam, mu, sd):
        Xs = (X - mu) / sd
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = LogisticRegression(penalty="elasticnet", l1_ratio=alpha, C=1.0 / (len(y) * lam), solver="saga",
                                   max_iter=5000, tol=1e-4)
            m.fit(Xs, y)
        return m

    def fit(self, df, y, groups):
        X, self.info = self._X(df)
        y = np.asarray(y)
        f = folds(groups, SEED1)
        labels = np.unique(y)

        res = Parallel(n_jobs=N_JOBS)(delayed(_cv_l2)(X, y, f, labels, a, l, i)
                                      for a in ALPHAS for l in LAMBDAS for i in range(5))
        r = pd.DataFrame(res, columns=["alpha", "lam", "fold", "ll", "n"])
        best = {}
        for a, g in r.groupby("alpha"):
            s = g.groupby("lam").ll.agg(["mean", "std"])
            se = s["std"] / np.sqrt(5)
            mn = s["mean"].idxmin()
            ok = s.index[s["mean"] <= s.loc[mn, "mean"] + se.loc[mn]]
            lam_1se = ok.max()                                          # 一个标准误内惩罚最强
            best[a] = (lam_1se, s.loc[lam_1se, "mean"])
        a_best = sorted(best, key=lambda a: (round(best[a][1], 12), -a))[0]   # 对数损失最低；相同取较大 α
        self.alpha, self.lam = a_best, best[a_best][0]
        self.cv = r
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-12
        self.m = self._fit_one(X, y, self.alpha, self.lam, self.mu, self.sd)
        self.classes_ = labels
        return self

    def fit_fixed(self, df, y, alpha, lam):
        """用已选定的 α、λ 在同一数据上重新拟合（确定性）。"""
        X, self.info = self._X(df)
        y = np.asarray(y)
        self.alpha, self.lam, self.classes_ = alpha, lam, np.unique(y)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-12
        self.m = self._fit_one(X, y, alpha, lam, self.mu, self.sd)
        return self

    def proba(self, df):
        X, _ = self._X(df, self.info)
        return expand(self.m.predict_proba((X - self.mu) / self.sd), self.m.classes_, self.classes_)


def expand(P, have, want):
    """把只含部分类别的概率矩阵扩展到全部类别（缺的类别为 0）。"""
    have, want = list(have), list(want)
    if have == want:
        return P
    out = np.zeros((P.shape[0], len(want)))
    for j, c in enumerate(have):
        out[:, want.index(c)] = P[:, j]
    return out


class L3:
    name = "L3"
    FIXED = dict(learning_rate=0.05, max_depth=-1, max_bin=255, feature_fraction=0.8, bagging_fraction=0.8,
                 bagging_freq=1, min_split_gain=0.0, reg_alpha=0.0, reg_lambda=1.0, deterministic=True,
                 num_threads=8, seed=SEED1, verbosity=-1)

    def __init__(self, formula, monotone_age=False):
        self.formula = l3_formula(formula)
        self.monotone_age = monotone_age

    def _X(self, df, info=None):
        if info is None:
            X = dmatrix(self.formula, df, return_type="dataframe", NA_action="raise")
            return X.values[:, 1:], X.design_info, list(X.columns[1:])
        return np.asarray(build_design_matrices([info], df)[0])[:, 1:], info, None

    def _params(self, nl, mc, ncls, cols):
        p = dict(self.FIXED, num_leaves=nl, min_child_samples=mc)
        p["objective"] = "binary" if ncls == 2 else "multiclass"
        if ncls > 2:
            p["num_class"] = ncls
        if self.monotone_age:
            p["monotone_constraints"] = [1 if c == "age10" else 0 for c in cols]
        return p

    def _train(self, X, yi, params, groups_tr, n_rounds=None):
        if n_rounds is None:                                           # 早停：训练部分内按病人留出 10%
            u = np.unique(groups_tr)
            rng = np.random.default_rng(SEED1)
            hold = set(rng.choice(u, max(1, int(round(0.1 * len(u)))), replace=False))
            es = np.isin(groups_tr, list(hold))
            dtr, dva = lgb.Dataset(X[~es], yi[~es]), lgb.Dataset(X[es], yi[es])
            b = lgb.train(params, dtr, 1000, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
            return b, b.best_iteration or 1000
        return lgb.train(params, lgb.Dataset(X, yi), n_rounds), n_rounds

    @staticmethod
    def _raw(b, X, ncls):
        r = b.predict(X, raw_score=True)
        return r if ncls > 2 else r.reshape(-1)

    def fit(self, df, y, groups):
        X, self.info, cols = self._X(df)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        yi = np.searchsorted(self.classes_, y)
        ncls = len(self.classes_)
        g = np.asarray(groups)
        f = folds(g, SEED1)
        res = []
        for nl, mc in GRID_L3:                                         # 第一轮：选参数和迭代次数
            prm = self._params(nl, mc, ncls, cols)
            lls, its = [], []
            for i in range(5):
                tr, va = f != i, f == i
                b, it = self._train(X[tr], yi[tr], prm, g[tr])
                P = self._p_from_raw(self._raw(b, X[va], ncls), ncls)
                lls.append(log_loss(yi[va], P, labels=range(ncls)))
                its.append(it)
            res.append((nl, mc, np.mean(lls), int(np.median(its))))
        r = pd.DataFrame(res, columns=["nl", "mc", "ll", "iters"])
        r = r.sort_values(["ll", "nl", "mc"], ascending=[True, True, False])
        self.nl, self.mc, _, self.iters = r.iloc[0]
        self.nl, self.mc, self.iters = int(self.nl), int(self.mc), int(self.iters)
        self.grid = r
        return self._finish(X, y, yi, g, ncls, cols)

    def fit_fixed(self, df, y, groups, nl, mc, iters):
        """用已选定的参数和迭代次数重新做第二轮（校准器）并拟合（确定性）。"""
        X, self.info, cols = self._X(df)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        yi = np.searchsorted(self.classes_, y)
        self.nl, self.mc, self.iters = nl, mc, iters
        return self._finish(X, y, yi, np.asarray(groups), len(self.classes_), cols)

    def _finish(self, X, y, yi, g, ncls, cols):
        self.params = self._params(self.nl, self.mc, ncls, cols)
        f2 = folds(g, SEED2)                                           # 第二轮：重新分折，折外预测拟合校准器
        oof = np.zeros((len(y), ncls)) if ncls > 2 else np.zeros(len(y))
        for i in range(5):
            tr, va = f2 != i, f2 == i
            b, _ = self._train(X[tr], yi[tr], self.params, g[tr], n_rounds=self.iters)
            oof[va] = self._raw(b, X[va], ncls)
        self._fit_cal(oof, yi, ncls)
        self.b, _ = self._train(X, yi, self.params, g, n_rounds=self.iters)
        self.ncls = ncls
        return self

    @staticmethod
    def _p_from_raw(r, ncls):
        if ncls == 2:
            p = 1 / (1 + np.exp(-r))
            return np.column_stack([1 - p, p])
        e = np.exp(r - r.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

    def _fit_cal(self, raw, yi, ncls):
        if ncls == 2:                                                  # Platt
            self.cal = LogisticRegression(penalty=None, max_iter=1000).fit(raw[:, None], yi)
        else:                                                          # 温度缩放
            def nll(t):
                return log_loss(yi, self._p_from_raw(raw / t, ncls), labels=range(ncls))
            self.T = minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x

    def proba(self, df):
        X, _, _ = self._X(df, self.info)
        r = self._raw(self.b, X, self.ncls)
        if self.ncls == 2:
            p = self.cal.predict_proba(r[:, None])[:, 1]
            return np.column_stack([1 - p, p])
        return self._p_from_raw(r / self.T, self.ncls)


ORDER = {"L1": 0, "L2": 1, "L3": 2}


def choose(results):
    """results：{name: (ll, slopes)}；按方案 4.3 第 3–5 步确定性选择。"""
    elig = {k: v for k, v in results.items() if eligible(v[1])}
    if not elig:
        return "L1", "无候选合格，按规则取 L1"
    best = min(elig, key=lambda k: elig[k][0])
    for k in sorted(elig, key=ORDER.get):                              # 更可解释者优先
        if ORDER[k] < ORDER[best] and (elig[k][0] - elig[best][0]) / elig[best][0] < 0.01:
            return k, f"{best} 对数损失最低，但 {k} 相差 < 1%，取更可解释者"
        if k == best:
            break
    return best, "合格者中对数损失最低"
