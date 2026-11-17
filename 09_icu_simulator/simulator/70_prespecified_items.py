"""
单中心 ICU 路径模拟器 · 补做方案 v1.0 中预先规定、此前未执行的四项次要内容（2026-11-17，作者决定补做）

1. 基线 0（方案 6.1）：按卒中类型的经验停留时间分布。在每个预测时点，用拟合集中同一卒中类型、同一时点
   仍在 ICU 的病人·时点的经验比例，作为 24 / 48 / 72 小时存活离开、ICU 内死亡的概率。
   - 内部验证段：与 42_direct_baselines.py 同一指标（24 小时存活离开概率的对数损失），检查"最强直接基线"是否不变。
   - 时间验证集与 2020 年以后：与 51 / 53 相同的次要指标；与模拟器 A 的 Brier 差（同一自助法、同一种子）。
2. 逻辑检查（方案 5、7.3）：把每行内第 3 步（死亡 D1）与第 4 步（去向 C1）对调后重新模拟时间验证集
   （同一冻结模型、同一随机种子、每个病人·时点 1,000 次），报告离开与死亡概率的变化。只作报告。
3. 单元层面每日预计离开人数（方案 7.2 第 5 项）：MIMIC-IV 对每位病人的日期分别平移，不能把不同病人放在
   同一日历上，无法重建每日在科人数；改报最接近的可行量——每个预测时点在科病人 24 小时内预计离开人数之和
   与实际人数。
4. 罕见状态转移（方案 3.2）：建模集中观察到 < 30 次的 D2 转移，时间验证集中的预测与观察次数。

只输出汇总结果：outputs/prespecified_items.md。
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import log_loss, roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import baseline_lib as BL  # noqa: E402
import final_lib as F  # noqa: E402
import selection_lib as L  # noqa: E402
import simlib as S  # noqa: E402
import prespec_lib as PL  # noqa: E402

WORK, OUT = HERE / "work", HERE / "outputs"
DEV = ("internal_train", "internal_val")
NCHUNK = 12


def boot_diff(groups, a, b, n=1000, seed=L.SEED1):
    """与 51_temporal_validation.py 相同：病人层面自助法，(a − b) 平均差的 95% 区间。"""
    df = pd.DataFrame({"g": groups, "a": a, "b": b, "n": 1}).groupby("g").sum().to_numpy()
    rng = np.random.default_rng(seed)
    v = [(df[i, 0].sum() - df[i, 1].sum()) / df[i, 2].sum() for i in (rng.integers(0, len(df), len(df)) for _ in range(n))]
    return np.percentile(v, [2.5, 97.5])


# ---------------------------------------------------------------------------
# 1. 基线 0
# ---------------------------------------------------------------------------
def baseline0(fit, ev):
    out = {}
    for h in (24, 48, 72):
        tab = fit.groupby(["k", "subtype"])[f"y{h}"].agg(pe=lambda y: (y == 1).mean(), pd_=lambda y: (y == 2).mean())
        m = ev[["k", "subtype"]].merge(tab, left_on=["k", "subtype"], right_index=True, how="left")
        out[h] = np.column_stack([1 - m.pe - m.pd_, m.pe, m.pd_]).astype(float)
    return out


def metrics(pr, ev):
    r = {}
    for h in (24, 48, 72):
        y = ev[f"y{h}"].to_numpy()
        pe, pdd = pr[h][:, 1], pr[h][:, 2]
        r[f"{h}h alive-exit Brier"] = round(float(((pe - (y == 1)) ** 2).mean()), 4)
        r[f"{h}h alive-exit C"] = round(roc_auc_score(y == 1, pe), 3)
        r[f"{h}h alive-exit slope"] = round(L.cal_slope((y == 1).astype(int), np.clip(pe, 1e-6, 1 - 1e-6)), 2)
        r[f"{h}h alive-exit predicted − observed"] = round(float(pe.mean() - (y == 1).mean()), 3)
        r[f"{h}h ICU-death Brier"] = round(float(((pdd - (y == 2)) ** 2).mean()), 4)
        r[f"{h}h ICU-death C"] = round(roc_auc_score(y == 2, pdd), 3)
    return r


if __name__ == "__main__":
    t_all = time.time()
    LM = BL.LMROW
    md = ["# Pre-specified items run after the main analysis (simulator plan v1.0)", "",
          "Run on 17 November 2026 at the authors' decision; these items were pre-specified but had not been run. "
          "No other analysis was changed.", ""]

    # 1. Baseline 0 ---------------------------------------------------------
    itr, iva = LM[LM.isplit == "internal_train"], LM[LM.isplit == "internal_val"]
    p_iv = baseline0(itr, iva)
    yv = (iva.y24 == 1).astype(int).to_numpy()
    ll = log_loss(yv, np.clip(p_iv[24][:, 1], 1e-15, 1 - 1e-15), labels=[0, 1])
    br = float(((p_iv[24][:, 1] - yv) ** 2).mean())
    md += ["## 1. Baseline 0: empirical distribution by stroke subtype", "",
           "Probability at each landmark = share of development patient-landmarks with the same subtype and landmark "
           "that left alive / died within 24, 48, 72 hours.", "",
           f"Internal validation (fit on internal training, {len(itr)} rows; evaluated on {len(iva)} patient-landmarks): "
           f"24-h alive-exit log loss {ll:.5f}, Brier {br:.5f}. Strongest direct model was GBM with log loss 0.38046, "
           f"so the choice of comparator is {'unchanged' if ll > 0.38046 else 'CHANGED'}.", ""]
    dev = LM[LM.isplit.isin(DEV)]
    import pickle
    for label, split, fpred in [("Temporal validation 2017–2019", "test", "tv_predictions.pkl"),
                                ("Additional validation 2020 onwards", "test2020", "tv2020_predictions.pkl")]:
        ev = LM[LM.isplit == split].reset_index(drop=True)
        p0 = baseline0(dev, ev)
        mt = metrics(p0, ev)
        tv = pickle.load(open(WORK / fpred, "rb"))
        simA = tv["sims"]["A"].set_index(["stay_id", "k"]).loc[list(zip(ev.stay_id, ev.k))]
        pA = np.stack(simA.p24)[:, 1]
        y = (ev.y24 == 1).to_numpy().astype(float)
        seA, se0 = (pA - y) ** 2, (p0[24][:, 1] - y) ** 2
        ci = boot_diff(ev.stay_id.to_numpy(), seA, se0)
        md += [f"### {label} ({len(ev)} patient-landmarks)", "", "| Measure | Baseline 0 |", "|---|---|"]
        md += [f"| {k} | {v} |" for k, v in mt.items()]
        md += ["", f"Simulator A − baseline 0, 24-h alive-exit Brier: {seA.mean() - se0.mean():+.4f} "
               f"(95% CI {ci[0]:+.4f} to {ci[1]:+.4f}; 1,000 patient-level bootstrap samples, seed {L.SEED1}).", ""]
        if split == "test":
            # 3. closest feasible unit-level quantity
            md3 = ["## 3. Expected and observed exits (unit level)", "",
                   "Daily unit census cannot be reconstructed: MIMIC-IV shifts dates separately for each patient, so stays "
                   "cannot be placed on a common calendar. The closest feasible quantity is shown: at each landmark, the "
                   "number of patients in the ICU, the expected number leaving alive within 24 hours (sum of simulator A "
                   "probabilities) and the observed number (temporal validation 2017–2019).", "",
                   "| Landmark | Unit type | Patients | Expected exits | Observed exits |", "|---|---|---|---|---|"]
            ev["pA"] = pA
            for kk in (0, 24, 48, 72):
                for ut, mask in [("Standard ICU", ~ev.inter.astype(bool)), ("Intermediate / step-down", ev.inter.astype(bool))]:
                    s = ev[(ev.k == kk) & mask]
                    md3.append(f"| {kk} h | {ut} | {len(s)} | {s.pA.sum():.0f} | {int((s.y24 == 1).sum())} |")
            md3.append("")
            pA72 = {h: np.stack(simA[f"p{h}"]) for h in (24, 48, 72)}

    # 4. Rare transitions ---------------------------------------------------
    T, fm = PL.model()
    D2 = T["D2"].copy()
    D2["age10"] = D2.age_at_adm_capped / 10
    devD, teD = D2[D2.int_split.isin(DEV)], D2[D2.int_split == "test"]
    cnt = devD.groupby(["prev_state", "fstate"]).size()
    rare = cnt[cnt < 30]
    md4 = ["## 4. Rare condition-state transitions", "",
           "Transitions observed fewer than 30 times in the development set; predicted (sum of D2 probabilities) and "
           "observed counts in temporal validation 2017–2019. State codes: 0 renal dysfunction, 1 respiratory support, "
           "2 impairment with low support, 3 preserved.", "",
           "| From → to | Development count | Temporal validation: predicted | Temporal validation: observed |",
           "|---|---|---|---|"]
    for (a, b), c in rare.items():
        sub = teD[teD.prev_state == a]
        f = {"k": sub.k.to_numpy(), "age10": sub.age10.to_numpy(), "subtype": sub.subtype.to_numpy(),
             "inter": sub.inter.astype(bool).to_numpy()}
        pr = fm["D2"][int(a)].proba(f, len(sub))
        j = list(fm["D2"][int(a)].classes_).index(b)
        md4.append(f"| {int(a)} → {int(b)} | {int(c)} | {pr[:, j].sum():.1f} | {int((sub.fstate == b).sum())} |")
    md4 += ["", f"Transitions in the development set: {len(cnt)} types, {int(cnt.sum())} transitions; {len(rare)} types rare.", ""]

    # 2. Order swap ---------------------------------------------------------
    te = LM[LM.isplit == "test"].reset_index(drop=True)
    tasks = list(zip(te.stay_id, te.k))
    t0 = time.time()
    r = Parallel(n_jobs=NCHUNK)(delayed(PL.run_swapped)(tasks[i::NCHUNK]) for i in range(NCHUNK))
    sw = pd.DataFrame([o for ch in r for o in ch]).set_index(["stay_id", "k"]).loc[tasks]
    md2 = ["## 2. Logic check: death and destination steps swapped", "",
           f"Temporal validation 2017–2019, {len(te)} patient-landmarks, 1,000 simulations each, same frozen model and "
           f"random seeds as the main analysis; only the order of steps 3 (death) and 4 (destination) within each hour "
           f"was swapped. Reported only; the execution order of the main analysis is unchanged.", "",
           "| Horizon | Mean P(alive exit): main / swapped | Mean P(ICU death): main / swapped | Mean absolute difference, alive exit | Alive-exit Brier: main / swapped |",
           "|---|---|---|---|---|"]
    for h in (24, 48, 72):
        a, b = pA72[h], np.stack(sw[f"p{h}"])
        y = (te[f"y{h}"] == 1).to_numpy()
        md2.append(f"| {h} h | {a[:, 1].mean():.4f} / {b[:, 1].mean():.4f} | {a[:, 2].mean():.4f} / {b[:, 2].mean():.4f} | "
                   f"{np.abs(a[:, 1] - b[:, 1]).mean():.4f} | {((a[:, 1] - y) ** 2).mean():.4f} / {((b[:, 1] - y) ** 2).mean():.4f} |")
    md2 += ["", f"Run time {round((time.time() - t0) / 60)} minutes.", ""]

    (OUT / "prespecified_items.md").write_text("\n".join(md + md2 + md3 + md4), encoding="utf-8")
    print("done", round((time.time() - t_all) / 60), "min", flush=True)
