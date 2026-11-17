"""
单中心 ICU 路径模拟器 · 附加验证集（近似入院年 ≥ 2020，新冠时期；分析方案 v1.0 第 3 节表格：稳健性检验，单独报告）

与 51_temporal_validation.py 完全相同的最终模型（同一设定确定性重拟合，冻结检查与冻结包比对）和同一套指标，
只是评估集换成 test2020。不导出、不覆盖冻结包；不改动 2017–2019 的结果。
--- 以下为 51 的原说明 ---

1. 最终拟合：整个建模集（近似入院年代 ≤ 2016）上，配置 A（选定算法）、配置 B（全部 L1）、三个直接基线，
   超参数保持选定值，只重新拟合参数和校准器
2. 冻结检查：同一设定拟合两次，预测完全相同；导出系数、LightGBM 模型文件与清单
3. 时间验证（只做一次）：近似入院年代 2017–2019 的病人，0 / 24 / 48 / 72 小时，每个病人·时点模拟 1,000 次
   - 唯一主要比较：配置 A 对 最强直接基线（GBM），24 小时存活离开 Brier（四时点合并），病人层面自助法 95% 区间
   - 关键次要：ICU 停留时间分布的 Wasserstein 距离
   - 次要：三类结局 24 / 48 / 72 小时、校准、区分度、剩余时间、按单元类型与卒中类型分层、离开后 72 小时（P1）
参数不确定性（200 次）与简化方案比较另行运行。
输出：outputs/temporal_validation.md；work/frozen/（冻结包）；work/tv_*.pkl（个体层面预测，仅本机）
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from scipy.stats import wasserstein_distance
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import baseline_lib as BL  # noqa: E402
import final_lib as F  # noqa: E402
import selection_lib as L  # noqa: E402
import simlib as S  # noqa: E402

WORK, OUT = HERE / "work", HERE / "outputs"
FROZEN = WORK / "frozen"
SPLIT = "test2020"
NSIM, NCHUNK, HOR = 1000, 12, [24, 48, 72]
_C = {}


# ---------------------------------------------------------------------------
# 模拟（各进程各自做一次最终拟合并缓存）
# ---------------------------------------------------------------------------
def run_chunk(config, tasks):
    if config not in _C:
        T = S.load_tables(WORK)
        _C[config] = (F.build(config, T)[0], T)
    fm, T = _C[config]
    P, H = T["P"], T["H"].set_index(["stay_id", "k"])
    out = []
    for sid, k in tasks:
        p, r = P.loc[sid], H.loc[(sid, k)]
        pat = {"age10": p.age10, "subtype": p.subtype, "reperf": p.reperf, "entry": p.entry, "intime": p.intime}
        start = {"k": k, "fstate": int(r.fstate), "met03": bool(r.met03), "run_h": int(r.run_h),
                 "ever_conf": bool(r.ever_conf), "first_conf_k": (k - r.h_since_conf) if bool(r.ever_conf) else np.nan,
                 "inter": bool(r.inter)}
        res = S.simulate_fast(pat, start, fm, NSIM, np.random.default_rng(int(sid) * 100 + k), skip_first_update=k > 0)
        rem = (res.end_k + 1 - k).to_numpy()
        o = {"stay_id": sid, "k": k, "rem_q": np.percentile(rem, [10, 50, 90]),
             "term_death": float((res.end_type == "death").mean()),
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


def brier(p, y):
    return float(((p - y) ** 2).mean())


def boot_diff(groups, a, b, n=1000, seed=L.SEED1):
    """病人层面自助法：(a − b) 的平均差的 95% 区间；a、b 为每个病人·时点的平方误差。"""
    df = pd.DataFrame({"g": groups, "a": a, "b": b, "n": 1}).groupby("g").sum().to_numpy()
    rng = np.random.default_rng(seed)
    v = [(df[i, 0].sum() - df[i, 1].sum()) / df[i, 2].sum() for i in (rng.integers(0, len(df), len(df)) for _ in range(n))]
    return np.percentile(v, [2.5, 97.5])


if __name__ == "__main__":
    t_all = time.time()
    T = S.load_tables(WORK)
    P = T["P"]
    LM = BL.LMROW
    dev_lm = LM[LM.isplit.isin(F.DEV)].reset_index(drop=True)
    te_lm = LM[LM.isplit == SPLIT].reset_index(drop=True)

    # 1–2. 最终拟合与冻结检查 ---------------------------------------------------
    fmA, rawA = F.build("A", T)
    fmA2, _ = F.build("A", T)
    Hd = T["H"][T["H"].stay_id.map(P.int_split) == SPLIT].head(20000)
    feats = {c: Hd[c].to_numpy() for c in Hd.columns}
    feats["conf_cat#idx"] = np.where(Hd.h_since_conf.isna(), 0, 1 + np.digitize(Hd.h_since_conf.fillna(0), [6, 24, 72]))
    freeze_diff = max(float(np.abs(fmA[k].proba(feats, len(Hd)) - fmA2[k].proba(feats, len(Hd))).max())
                      for k in ["D1", "D3a", "D3b", "C1"])
    spec_b = pickle.load(open(WORK / "baselines_spec.pkl", "rb"))
    torch.manual_seed(L.SEED1)
    base = BL.fit_baselines(dev_lm, spec_b)
    manifest = {"建模集病人数": int(P.int_split.isin(F.DEV).sum()), "时间验证集病人数": int((P.int_split == SPLIT).sum()),
                "模块": {k: v[0] for k, v in rawA.items()},
                "软件": {"lightgbm": __import__("lightgbm").__version__, "sklearn": __import__("sklearn").__version__,
                       "torch": torch.__version__, "python": sys.version.split()[0]},
                "冻结检查：两次拟合预测最大差": freeze_diff}
    # 与冻结包比对：LightGBM 模型文件逐字相同、系数相同（证明用的是同一个冻结模型）
    same = []
    for name, (pick, mod) in rawA.items():
        tag = name.split("（")[0].split(" ")[0] + (name[name.find("状态"):-1].replace(" ", "") if "状态" in name else "")
        if pick == "L3":
            same.append(mod.b.model_to_string() == open(FROZEN / f"{tag}_lightgbm.txt").read())
        else:
            z = np.load(FROZEN / f"{tag}_{'elasticnet' if pick == 'L2' else 'logistic'}.npz")
            same.append(bool(np.array_equal(z["coef"], mod.m.coef_) and np.array_equal(z["intercept"], mod.m.intercept_)))
    same.append(base["GBM"].b.model_to_string() == open(FROZEN / "baseline_GBM_lightgbm.txt").read())
    sd = torch.load(FROZEN / "baseline_GRU.pt")
    same.append(all(torch.equal(sd[k], v) for k, v in base["GRU"][0].state_dict().items()))
    manifest["与冻结包一致的模型数"] = f"{sum(same)} / {len(same)}"
    print("最终拟合与冻结完成", round(time.time() - t_all), "秒；冻结检查差", freeze_diff, flush=True)

    # 3. 时间验证 -------------------------------------------------------------
    tasks = list(zip(te_lm.stay_id, te_lm.k))
    chunks = [tasks[i::NCHUNK] for i in range(NCHUNK)]
    sims = {}
    for config in ["A", "B"]:
        t0 = time.time()
        r = Parallel(n_jobs=NCHUNK)(delayed(run_chunk)(config, ch) for ch in chunks)
        sims[config] = pd.DataFrame([o for ch in r for o in ch]).set_index(["stay_id", "k"]).loc[tasks].reset_index()
        print(config, "模拟完成", round(time.time() - t0), "秒", flush=True)
    bp = BL.predict_baselines(base, te_lm)
    pickle.dump({"sims": sims, "baselines": bp, "te_lm": te_lm[["stay_id", "k", "y24", "y48", "y72", "inter"]]},
                open(WORK / "tv2020_predictions.pkl", "wb"))

    preds = {"模拟器 A": {h: np.stack(sims["A"][f"p{h}"]) for h in HOR},
             "模拟器 B（全部 L1）": {h: np.stack(sims["B"][f"p{h}"]) for h in HOR},
             "基线 LR": bp["LR"], "基线 GBM": bp["GBM"], "基线 GRU": bp["GRU"]}
    g = te_lm.stay_id.to_numpy()

    # 主要比较
    y24e = (te_lm.y24 == 1).to_numpy().astype(float)
    seA = (preds["模拟器 A"][24][:, 1] - y24e) ** 2
    seG = (preds["基线 GBM"][24][:, 1] - y24e) ** 2
    ci = boot_diff(g, seA, seG)
    primary = {"模拟器 A Brier": seA.mean(), "基线 GBM Brier": seG.mean(), "差（A − GBM）": seA.mean() - seG.mean(),
               "95% 区间": ci,
               "结论": "模拟器更好" if ci[1] < 0 else ("模拟器更差" if ci[0] > 0 else "无法区分（区间包含 0）")}

    # 次要：各方案各期限
    rows = []
    for name, pr in preds.items():
        r = {"方案": name}
        for h in HOR:
            y = te_lm[f"y{h}"].to_numpy()
            pe, pdd = pr[h][:, 1], pr[h][:, 2]
            r[f"{h}h 存活离开 Brier"] = round(brier(pe, y == 1), 4)
            r[f"{h}h 存活离开 C"] = round(roc_auc_score(y == 1, pe), 3)
            r[f"{h}h 存活离开 斜率"] = round(L.cal_slope((y == 1).astype(int), np.clip(pe, 1e-6, 1 - 1e-6)), 2)
            r[f"{h}h 存活离开 预测−观察"] = round(float(pe.mean() - (y == 1).mean()), 3)
            r[f"{h}h ICU死亡 Brier"] = round(brier(pdd, y == 2), 4)
            r[f"{h}h ICU死亡 C"] = round(roc_auc_score(y == 2, pdd), 3) if (y == 2).any() else np.nan
        rows.append(r)
    sec = pd.DataFrame(rows).set_index("方案")

    # 分层：24 小时存活离开 Brier
    strat = []
    for label, m in [("正规 ICU", ~te_lm.inter.astype(bool)), ("中间监护 / 过渡", te_lm.inter.astype(bool))] + \
                    [(st, te_lm.stay_id.map(P.subtype) == st) for st in ["AIS", "ICH", "SAH"]] + \
                    [(f"时点 {k} 小时", te_lm.k == k) for k in [0, 24, 48, 72]]:
        m = m.to_numpy()
        strat.append({"分层": label, "病人·时点": int(m.sum()),
                      **{n: round(brier(preds[n][24][m, 1], y24e[m]), 4) for n in preds}})
    strat = pd.DataFrame(strat).set_index("分层")

    # 关键次要：停留时间分布
    adm = sims["A"][sims["A"].k == 0]
    obs_los = np.minimum(P.loc[adm.stay_id, "T"].to_numpy(), 720)
    distA = np.stack(adm.los).sum(0)
    distB = np.stack(sims["B"][sims["B"].k == 0].los).sum(0)
    support = np.arange(721)
    wA = wasserstein_distance(obs_los, support, v_weights=distA)
    wB = wasserstein_distance(obs_los, support, v_weights=distB)
    obs72 = np.minimum(obs_los, 72)

    def cut72(dist):
        d = dist[:73].astype(float).copy()
        d[72] += dist[73:].sum()
        return d
    w72 = {"模拟器 A": wasserstein_distance(obs72, np.arange(73), v_weights=cut72(distA)),
           "模拟器 B（全部 L1）": wasserstein_distance(obs72, np.arange(73), v_weights=cut72(distB))}
    te0 = te_lm[te_lm.k == 0]
    for nm in ["LR", "GBM"]:
        R = BL.hazard_rows(te0, predict=True)
        hz = L.expand(base[nm].proba(R), base[nm].classes_, [0, 1, 2]).reshape(len(te0), 72, 3)
        surv = np.cumprod(np.concatenate([np.ones((len(te0), 1)), hz[:, :-1, 0]], 1), 1)
        ev = (surv * (hz[:, :, 1] + hz[:, :, 2])).sum(0)                      # 各小时的终止概率（合计）
        d = np.zeros(73); d[1:73] = ev; d[72] += len(te0) - ev.sum()
        w72[f"基线 {nm}"] = wasserstein_distance(obs72, np.arange(73), v_weights=np.clip(d, 0, None))
    qs = [10, 25, 50, 75, 90]

    def wq(dist):
        c = np.cumsum(dist) / dist.sum()
        return [int(np.searchsorted(c, q / 100)) for q in qs]
    los_tab = pd.DataFrame({"观察": np.percentile(obs_los, qs).round().astype(int), "模拟器 A": wq(distA),
                            "模拟器 B（全部 L1）": wq(distB)}, index=[f"第 {q} 百分位" for q in qs])

    # 剩余时间（模拟器 A）
    A_ = sims["A"].merge(te_lm[["stay_id", "k"]], on=["stay_id", "k"])
    actual = P.loc[A_.stay_id, "T"].to_numpy() - A_.k.to_numpy()
    q = np.stack(A_.rem_q)
    died = P.loc[A_.stay_id, "died"].to_numpy()
    trunc = P.loc[A_.stay_id, "trunc"].to_numpy()
    ok = ~trunc
    rem = pd.DataFrame({"终止方式": ["存活离开", "ICU 内死亡"],
                        "人·时点": [int((ok & ~died).sum()), int((ok & died).sum())],
                        "预测中位数的平均绝对误差（小时）": [round(float(np.abs(q[ok & ~died, 1] - actual[ok & ~died]).mean()), 1),
                                            round(float(np.abs(q[ok & died, 1] - actual[ok & died]).mean()), 1)],
                        "80% 预测区间覆盖率": [round(float(((actual >= q[:, 0]) & (actual <= q[:, 2]))[ok & ~died].mean()), 3),
                                        round(float(((actual >= q[:, 0]) & (actual <= q[:, 2]))[ok & died].mean()), 3)]}
                       ).set_index("终止方式")

    # P1：离开时的预测（实际存活离开的验证集病人）
    Ae = T["A"][T["A"].int_split == SPLIT].copy()
    pa = fmA["P1a"].proba({"age10": Ae.age10.to_numpy(), "subtype": Ae.subtype.to_numpy(),
                           "exit_fstate": Ae.exit_fstate.to_numpy(), "exit_ready": Ae.exit_ready.to_numpy(),
                           "log_los": Ae.log_los.to_numpy()}, len(Ae))[:, 1]
    PBt = T["PB"][T["PB"].int_split == SPLIT]
    ih = Ae[~Ae.direct_leave]
    rows_h = ih.loc[ih.index.repeat(72)].copy()
    rows_h["h"] = np.tile(np.arange(72), len(ih))
    rows_h["reperf"] = rows_h.reperfusion.fillna(False).astype(bool)
    fe = {c: rows_h[c].to_numpy() for c in ["age10", "subtype", "h", "exit_ready", "exit_fstate", "log_los",
                                           "after_hours", "exit_inter", "reperf"]}
    fe["exit_fstate"] = fe["exit_fstate"].astype(int)
    hz = fmA["P1b"].proba(fe, len(rows_h))
    cls = list(fmA["P1b"].classes_)
    hz = hz.reshape(len(ih), 72, len(cls))
    surv = np.cumprod(np.concatenate([np.ones((len(ih), 1)), hz[:, :-1, cls.index(0)]], 1), 1)
    cif_r = (surv * hz[:, :, cls.index(1)]).sum(1)
    cif_d = (surv * hz[:, :, cls.index(2)]).sum(1)
    yo = PBt.groupby("stay_id").y_post.max().reindex(ih.index)
    p1 = pd.DataFrame({
        "预测平均": [round(float(pa.mean()), 3), round(float(cif_r.mean()), 3), round(float(cif_d.mean()), 3)],
        "观察": [round(float(Ae.direct_leave.mean()), 3), round(float((yo == 1).mean()), 3), round(float((yo == 2).mean()), 3)],
        "C 统计量": [round(roc_auc_score(Ae.direct_leave, pa), 3), round(roc_auc_score(yo == 1, cif_r), 3),
                   round(roc_auc_score(yo == 2, cif_d), 3)]},
        index=["P1a 直接离开医院", "P1b 72 小时内再入 ICU（住院期间）", "P1b 72 小时内院内死亡"])

    viol = int(sims["A"].viol.sum() + sims["B"].viol.sum())
    relA = sims["A"][sims["A"].k == 0].relapse
    relapse_sim = sum(a for a, _ in relA) / max(sum(b for _, b in relA), 1)

    def md(df):
        return "\n".join(["| " + str(df.index.name or "") + " | " + " | ".join(map(str, df.columns)) + " |",
                          "|" + "---|" * (len(df.columns) + 1)] +
                         ["| " + str(i) + " | " + " | ".join(str(v) for v in r) + " |" for i, r in df.iterrows()])

    lines = ["# 附加验证集结果：近似入院年 ≥ 2020（新冠时期；稳健性检验，单独报告）", "",
             f"建模集（近似入院年代 ≤ 2016）{manifest['建模集病人数']:,} 人上最终拟合；附加验证集（≥ 2020）"
             f"{manifest['时间验证集病人数']:,} 人、{len(te_lm):,} 个病人·时点，每个病人·时点模拟 1,000 次。",
             f"冻结检查：同一设定两次拟合的预测最大差 {freeze_diff:.1e}；与冻结包一致的模型 {manifest['与冻结包一致的模型数']}。逻辑检查违规数：{viol}。", "",
             "## 唯一主要比较：模拟器（配置 A） 对 最强直接基线（GBM）", "",
             f"- 24 小时存活离开 Brier（四时点合并）：模拟器 A {primary['模拟器 A Brier']:.4f}；基线 GBM {primary['基线 GBM Brier']:.4f}",
             f"- 差（A − GBM）：{primary['差（A − GBM）']:+.4f}，95% 区间 {ci[0]:+.4f} 至 {ci[1]:+.4f}（病人层面自助法 1,000 次）",
             f"- **结论：{primary['结论']}**", "",
             "## 关键次要：ICU 停留时间分布", "",
             f"- 全程（截至 720 小时）Wasserstein 距离：模拟器 A {wA:.1f} 小时；模拟器 B {wB:.1f} 小时（直接基线只预测到 72 小时，无法给出全程分布）",
             "- 截至 72 小时的 Wasserstein 距离（各方案可比）：" + "；".join(f"{k} {v:.2f}" for k, v in w72.items()), "",
             md(los_tab), "",
             "## 次要：各方案、各期限", "", md(sec.T), "",
             "## 次要：24 小时存活离开 Brier 的分层", "", md(strat), "",
             "## 次要：至 ICU 停留终止的剩余时间（模拟器 A）", "", md(rem), "",
             f"readiness 反复（从入 ICU 起模拟，曾确认者中）：模拟器 A {relapse_sim:.3f}", "",
             "## 次要：离开 ICU 时的预测（P1，模拟器 A 的模块）", "", md(p1), "",
             f"总用时 {round((time.time() - t_all) / 60)} 分钟。"]
    (OUT / "test2020_validation.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
