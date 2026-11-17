"""
单中心 ICU 路径模拟器 · 直接预测基线（分析方案 v1.0 第 6.1 节）：LR、GBM、GRU
在内部训练段调参与校准，内部验证段比较 24 小时存活离开概率的对数损失，确定最强的直接预测基线。
不接触时间验证集。

实施方式（方案未写定之处，见实施记录）：
  - 预测时点：入 ICU 后 0、24、48、72 小时，纳入该时点仍在 ICU 且存活者
  - LR、GBM：离散时间竞争风险——从预测时点起逐小时（j = 0…71）预测 留下 / 存活离开 / ICU 内死亡，
    协变量固定为预测时点的值，加上 j；由逐小时风险累积得到 24 / 48 / 72 小时的三类概率
  - GRU：入 ICU 至预测时点的逐小时序列（固定 72 步，前端补零 + 掩码），输出 24 / 48 / 72 小时三类概率
输出：outputs/direct_baselines.md；work/baselines_spec.pkl
"""
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss
from scipy.optimize import minimize_scalar

import selection_lib as L
import simlib as S

HERE = Path(__file__).resolve().parent
WORK, OUT = Path(os.environ.get("SIM_WORK", HERE / "work")), Path(os.environ.get("SIM_OUT", HERE / "outputs"))
FW = HERE.parent / "paper4_formal" / "work"
LMS, HOR = [0, 24, 48, 72], [24, 48, 72]
torch.manual_seed(L.SEED1)
torch.set_num_threads(8)

T = S.load_tables(WORK)
P, H = T["P"], T["H"]
org = pd.read_pickle(FW / "hourly.pkl")[["stay_id", "k", "R", "C"]].rename(columns={"R": "resp", "C": "pressor"})
H = H.merge(org, on=["stay_id", "k"], how="left")
H["hours_clock"] = H.clock_block

# ---------------------------------------------------------------------------
# 预测时点样本与结局
# ---------------------------------------------------------------------------
LMROW = H[H.k.isin(LMS)].copy()
LMROW["isplit"] = LMROW.stay_id.map(P.int_split)                      # K、died、trunc 已在逐小时表中
for h in HOR:
    ex = ~LMROW.died & ~LMROW.trunc & (LMROW.K <= LMROW.k + h - 1)
    de = LMROW.died & (LMROW.K <= LMROW.k + h - 1)
    LMROW[f"y{h}"] = np.select([ex, de], [1, 2], default=0)       # 1 存活离开；2 ICU 内死亡；0 仍在 ICU
LMROW = LMROW.reset_index(drop=True)
FEAT = "C(k, levels=[0,24,48,72]) + C(fstate, levels=[-1,0,1,2,3]) + met03 + cr(run_h, df=4) + ever_conf" \
       " + C(conf_cat, levels=['none','c0_6','c6_24','c24_72','c72p']) + inter + resp + pressor" \
       " + C(clock_block, levels=[0,1,2,3,4,5]) + weekend + age10 + C(subtype, levels=['AIS','ICH','SAH'])"


def hazard_rows(lm, predict=False):
    """每个预测时点展开为逐小时行。训练：到事件或 71 为止；预测：全部 0…71。"""
    reps = np.full(len(lm), 72) if predict else np.minimum(lm.K - lm.k, 71) + 1
    R = lm.loc[lm.index.repeat(reps)].copy()
    R["j"] = np.concatenate([np.arange(n) for n in reps])
    R["lm_id"] = np.repeat(lm.index.to_numpy(), reps)
    if not predict:
        ev = (R.k + R.j == R.K) & ~R.trunc
        R["yh"] = np.select([ev & ~R.died, ev & R.died], [1, 2], default=0)
    return R.reset_index(drop=True)


def cif_from_hazard(hz, n_lm):
    """hz：(n_lm*72, 3)，列为 留下 / 离开 / 死亡 的逐小时概率，按预测时点、j 排序。"""
    hz = hz.reshape(n_lm, 72, 3)
    surv = np.cumprod(np.concatenate([np.ones((n_lm, 1)), hz[:, :-1, 0]], 1), 1)
    cif_e, cif_d = np.cumsum(surv * hz[:, :, 1], 1), np.cumsum(surv * hz[:, :, 2], 1)
    out = {}
    for h in HOR:
        pe, pd_ = cif_e[:, h - 1], cif_d[:, h - 1]
        out[h] = np.column_stack([1 - pe - pd_, pe, pd_])
    return out


tr_lm = LMROW[LMROW.isplit == "internal_train"]
va_lm = LMROW[LMROW.isplit == "internal_val"]
results, spec = {}, {}

# ---------------------------------------------------------------------------
# LR
# ---------------------------------------------------------------------------
t0 = time.time()
lr = S.L1Module("cr(j, df=4) + " + FEAT).fit(hazard_rows(tr_lm), hazard_rows(tr_lm).yh)
hz = L.expand(lr.proba(hazard_rows(va_lm, predict=True)), lr.classes_, [0, 1, 2])
results["基线 1-LR"] = cif_from_hazard(hz, len(va_lm))
print("LR", round(time.time() - t0), "秒", flush=True)

# ---------------------------------------------------------------------------
# GBM（与 L3 相同的 9 组网格与两轮交叉拟合）
# ---------------------------------------------------------------------------
t0 = time.time()
Rtr = hazard_rows(tr_lm)
gbm = L.L3("j + " + FEAT).fit(Rtr, Rtr.yh.to_numpy(), Rtr.stay_id.to_numpy())
hz = L.expand(gbm.proba(hazard_rows(va_lm, predict=True)), gbm.classes_, [0, 1, 2])
results["基线 1-GBM"] = cif_from_hazard(hz, len(va_lm))
spec["GBM"] = (gbm.nl, gbm.mc, gbm.iters)
print("GBM", round(time.time() - t0), "秒", flush=True)

# ---------------------------------------------------------------------------
# GRU
# ---------------------------------------------------------------------------
SEQ_VARS = ["fstate", "met03", "run_h", "ever_conf", "inter", "resp", "pressor", "k", "clock_block", "weekend"]
Hs = H[H.stay_id.isin(LMROW.stay_id)].set_index(["stay_id", "k"], drop=False).sort_index()


def seq_matrix(lm, mu=None, sd=None):
    """每个预测时点：入 ICU 至该时点（含）最近 72 小时的逐小时变量；前端补零；加掩码与静态变量。"""
    X = np.zeros((len(lm), 72, len(SEQ_VARS) + 4 + 1), dtype=np.float32)
    for i, (sid, k) in enumerate(zip(lm.stay_id, lm.k)):
        g = Hs.loc[sid]
        g = g[(g.k <= k) & (g.k > k - 72)]
        v = g[SEQ_VARS].astype(float).to_numpy()
        n = len(v)
        X[i, -n:, :len(SEQ_VARS)] = v
        X[i, -n:, -1] = 1.0                                           # 掩码
    st = lm[["age10"]].to_numpy(float)
    sub = pd.get_dummies(lm.subtype).reindex(columns=["ICH", "SAH"], fill_value=0).to_numpy(float)
    X[:, :, len(SEQ_VARS):len(SEQ_VARS) + 1] = st[:, None, :]
    X[:, :, len(SEQ_VARS) + 1:len(SEQ_VARS) + 3] = sub[:, None, :]
    X[:, :, len(SEQ_VARS) + 3] = lm.k.to_numpy(float)[:, None] / 72
    nv = len(SEQ_VARS) + 4
    if mu is None:
        m = X[:, :, -1] > 0
        mu = np.array([X[:, :, c][m].mean() for c in range(nv)])
        sd = np.array([X[:, :, c][m].std() + 1e-6 for c in range(nv)])
    X[:, :, :nv] = np.where(X[:, :, -1:] > 0, (X[:, :, :nv] - mu) / sd, 0)
    return torch.tensor(X), mu, sd


class GRU(torch.nn.Module):
    def __init__(self, d_in, hid):
        super().__init__()
        self.g = torch.nn.GRU(d_in, hid, batch_first=True)
        self.drop = torch.nn.Dropout(0.2)
        self.heads = torch.nn.ModuleList([torch.nn.Linear(hid, 3) for _ in HOR])

    def forward(self, x):
        _, h = self.g(x)
        z = self.drop(h[-1])
        return torch.stack([hd(z) for hd in self.heads], 1)          # (n, 3 个期限, 3 类)


def train_gru(X, Y, hid, epochs=None, groups=None):
    torch.manual_seed(L.SEED1)
    model = GRU(X.shape[2], hid)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    if epochs is None:                                                # 训练部分内按病人留出 10% 早停
        u = np.unique(groups)
        rng = np.random.default_rng(L.SEED1)
        hold = np.isin(groups, rng.choice(u, max(1, int(round(0.1 * len(u)))), replace=False))
        Xt, Yt, Xe, Ye = X[~hold], Y[~hold], X[hold], Y[hold]
    else:
        Xt, Yt = X, Y
    best, best_ep, wait = np.inf, 0, 0
    g = torch.Generator().manual_seed(L.SEED1)
    for ep in range(epochs or 50):
        model.train()
        for idx in torch.randperm(len(Xt), generator=g).split(256):
            opt.zero_grad()
            out = model(Xt[idx])
            loss = sum(torch.nn.functional.cross_entropy(out[:, i], Yt[idx, i]) for i in range(len(HOR)))
            loss.backward()
            opt.step()
        if epochs is None:
            model.eval()
            with torch.no_grad():
                out = model(Xe)
                vl = float(sum(torch.nn.functional.cross_entropy(out[:, i], Ye[:, i]) for i in range(len(HOR))))
            if vl < best - 1e-6:
                best, best_ep, wait = vl, ep + 1, 0
            else:
                wait += 1
                if wait >= 5:
                    break
    return model, (best_ep if epochs is None else epochs)


def logits(model, X):
    model.eval()
    with torch.no_grad():
        return model(X).numpy()


def softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


t0 = time.time()
Xtr, mu, sd = seq_matrix(tr_lm)
Ytr = torch.tensor(tr_lm[[f"y{h}" for h in HOR]].to_numpy(), dtype=torch.long)
gtr = tr_lm.stay_id.to_numpy()
f1 = L.folds(gtr, L.SEED1)
cv = []
for hid in (32, 64):                                                  # 第一轮：隐藏单元与训练轮数
    lls, eps = [], []
    for i in range(5):
        tr, va = f1 != i, f1 == i
        m, ep = train_gru(Xtr[tr], Ytr[tr], hid, groups=gtr[tr])
        z = logits(m, Xtr[va])
        lls.append(np.mean([log_loss(Ytr[va][:, j].numpy(), softmax(z[:, j]), labels=[0, 1, 2]) for j in range(3)]))
        eps.append(ep)
    cv.append((hid, np.mean(lls), int(np.median(eps))))
hid, _, ep = sorted(cv, key=lambda r: (r[1], r[0]))[0]
f2 = L.folds(gtr, L.SEED2)                                            # 第二轮：折外预测拟合温度
oof = np.zeros((len(Xtr), 3, 3))
for i in range(5):
    tr, va = f2 != i, f2 == i
    m, _ = train_gru(Xtr[tr], Ytr[tr], hid, epochs=ep)
    oof[va] = logits(m, Xtr[va])
Tt = minimize_scalar(lambda t: np.mean([log_loss(Ytr[:, j].numpy(), softmax(oof[:, j] / t), labels=[0, 1, 2])
                                        for j in range(3)]), bounds=(0.05, 20), method="bounded").x
gru, _ = train_gru(Xtr, Ytr, hid, epochs=ep)
Xva, _, _ = seq_matrix(va_lm, mu, sd)
z = logits(gru, Xva) / Tt
results["基线 1-GRU"] = {h: softmax(z[:, j]) for j, h in enumerate(HOR)}
spec["GRU"] = (hid, ep, Tt)
print("GRU", round(time.time() - t0), "秒", flush=True)

# ---------------------------------------------------------------------------
# 比较与选择
# ---------------------------------------------------------------------------
rows = []
for name, pr in results.items():
    r = {"基线": name}
    for h in HOR:
        y = va_lm[f"y{h}"].to_numpy()
        p_exit = np.clip(pr[h][:, 1], 1e-15, 1 - 1e-15)
        r[f"{h}h 存活离开 对数损失"] = round(log_loss((y == 1).astype(int), p_exit, labels=[0, 1]), 5)
        r[f"{h}h 存活离开 Brier"] = round(float(((p_exit - (y == 1)) ** 2).mean()), 5)
        r[f"{h}h 校准斜率"] = round(L.cal_slope((y == 1).astype(int), p_exit), 2)
    rows.append(r)
R = pd.DataFrame(rows).set_index("基线")
best = R["24h 存活离开 对数损失"].idxmin()
spec["strongest"] = best
pickle.dump(spec, open(WORK / "baselines_spec.pkl", "wb"))
lines = ["# 直接预测基线：内部验证段比较（分析方案 v1.0 第 6.1 节）", "",
         f"内部训练段调参与校准；内部验证段 {len(va_lm):,} 个病人·时点（0、24、48、72 小时）。未接触时间验证集。", "",
         f"**最强的直接预测基线：{best}**（24 小时存活离开概率的对数损失最低）", "",
         "参数：GBM 叶子 / 最少样本 / 树 = " + str(spec["GBM"]) + "；GRU 隐藏单元 / 训练轮数 / 温度 = "
         + f"{spec['GRU'][0]} / {spec['GRU'][1]} / {spec['GRU'][2]:.3f}", "",
         "| 基线 | " + " | ".join(R.columns) + " |", "|" + "---|" * (len(R.columns) + 1)]
lines += ["| " + i + " | " + " | ".join(str(v) for v in r) + " |" for i, r in R.iterrows()]
(OUT / "direct_baselines.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
