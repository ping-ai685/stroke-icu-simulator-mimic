"""
第四篇 · 可解释性输出（分析方案 v1.0 第 8 节）——全部基于最终模型（建模集拟合，配置 A）

规则层面
  1. L1 / L2 模块的比值比表（L1：无惩罚 logistic 的点估计；多项模型相对参照类别；样条项只画图不列数；
     L2：标准化系数换算回原单位）→ 补充表
  2. 效应图：C1 去向——已达到 / 未达到转出条件时，不同钟点下一小时离开 ICU 的概率（正规 ICU 与中间监护）
  3. L3 模块（D2 状态 1、D3a、D3b、P1b）：LightGBM 自带的 TreeSHAP（pred_contrib）；各变量平均 |SHAP|；
     与同一模块 L1 的方向对照 = 两者逐行变量贡献的 Spearman 相关（多分类取各类平均）；负相关者列出讨论
个体层面
  4. 三个"典型病人"（虚构的特征组合，不是真实病人）从入 ICU 起模拟 1,000 次的路径：逐小时的
     在 ICU 未达到条件 / 在 ICU 已达到条件 / 已存活离开 / 已死亡 的比例；剩余时间拆分为"达到条件之前"和"达到之后到离开"
  5. 时间验证集（2017–2019）入 ICU 时点：模拟的两段时间 对 观察到的两段时间（按首个单元类型，汇总）
适用范围
  6. 适用范围标记规则（文字表）
输出：outputs/paper/（图由 63_plot_interpretability.py 绘制）、tableS_coefficients.md/.csv、tableS_shap.md、
      time_split.md、applicability.md
"""
import inspect
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from patsy import build_design_matrices  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import final_lib as F  # noqa: E402
import interp_lib as IL  # noqa: E402
import selection_lib as L  # noqa: E402
import simlib as S  # noqa: E402

WORK, OUT = HERE / "work", HERE / "outputs" / "paper"
OUT.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 300})
NSIM = 1000
VAR_EN = {"fstate": "Condition state", "k": "Hours in ICU", "run_h": "Hours in current readiness run",
          "inter": "Intermediate unit", "subtype": "Stroke subtype", "reperf_early": "Reperfusion, first 24 h",
          "age10": "Age (per 10 y)", "h": "Hours since ICU exit", "exit_ready": "Readiness at exit",
          "exit_fstate": "Condition state at exit", "log_los": "log ICU length of stay", "after_hours": "Exit 19:00–07:00",
          "exit_inter": "Exit from intermediate unit", "reperf": "Reperfusion therapy", "ever_conf": "Ever ready",
          "conf_cat": "Hours since readiness", "met03": "Readiness met now", "clock_block": "Hour of day (4-h block)",
          "weekend": "Weekend", "entry": "Admission source"}


def var_of(name):
    for p in ("C(", "cr("):
        if name.startswith(p):
            return name[len(p):name.index(",")].strip()
    return name.split("[")[0].strip()


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def dev_data(T, name):
    split = T["P"].int_split
    if name == "P1b":
        PB = T["PB"]
        d = PB[PB.stay_id.map(split).isin(F.DEV)].reset_index(drop=True)
        return d, d.y_post.to_numpy(), "P1b"
    df, y, fkey, _ = F.units(T)[name]
    s = df.stay_id.map(split) if "stay_id" in df else df.index.map(split)
    m = np.asarray(pd.Series(np.asarray(s)).isin(F.DEV))
    return df[m].reset_index(drop=True), np.asarray(y)[m], fkey


# ---------------------------------------------------------------------------
# 带逐小时记录的模拟（由 simulate_fast 的源码生成，只增加记录，不改变任何计算和随机数顺序）
# ---------------------------------------------------------------------------
_src = inspect.getsource(S.simulate_fast)
_src = _src.replace("def simulate_fast(pat, start, fm, n, rng, maxh=720, skip_first_update=False, never_lose=False):",
                    "def simulate_trace(pat, start, fm, n, rng, maxh=720, skip_first_update=False, never_lose=False, rec=None):")
_anchor = "            k[st] += 1\n"
assert _src.count(_anchor) == 1
_src = _src.replace(_anchor, _anchor + "        rec.append((k0 + 1, int((active & ~met).sum()), int((active & met).sum()),"
                                       " int((end_type == 'exit').sum()), int((end_type == 'death').sum())))\n")
_ns = dict(vars(S))
exec(_src, _ns)
simulate_trace = _ns["simulate_trace"]

if __name__ == "__main__":
    T, fm, raw = IL.model()

    # 1. 比值比表 ------------------------------------------------------------
    rows = []
    for name, (pick, mod) in raw.items():
        if pick == "L3":
            continue
        cls = list(mod.classes_)
        coef = mod.m.coef_
        cols = mod.info.column_names
        if pick == "L2":
            cols = cols[1:]
            coef = coef / mod.sd
        for j, c in enumerate(cols):
            if c == "Intercept" or c.startswith("cr("):
                continue
            if coef.shape[0] == 1:
                rows.append({"Module": name, "Algorithm": pick, "Term": c, "Outcome class": cls[1], "OR": np.exp(coef[0, j])})
            else:
                for i, k in enumerate(cls[1:], 1):
                    rows.append({"Module": name, "Algorithm": pick, "Term": c, "Outcome class": f"{k} vs {cls[0]}",
                                 "OR": np.exp(coef[i, j] - coef[0, j])})
    OR = pd.DataFrame(rows)
    OR["OR"] = OR.OR.map(lambda v: f"{v:.2f}" if 0.01 <= v <= 100 else f"{v:.2e}")
    OR.to_csv(OUT / "tableS_coefficients.csv", index=False)
    (OUT / "tableS_coefficients.md").write_text(
        "# Supplementary table. Odds ratios of the L1 / L2 modules (final model, development set)\n\n"
        "L1: unpenalised logistic, point estimates; multinomial modules relative to the first class. L2: ridge / elastic-net "
        "coefficients converted to the original unit (penalised; interpret direction only). Spline terms are shown as effect plots.\n\n"
        "| " + " | ".join(OR.columns) + " |\n|" + "---|" * len(OR.columns) + "\n" +
        "\n".join("| " + " | ".join(map(str, r)) + " |" for r in OR.itertuples(index=False)), encoding="utf-8")
    print("比值比表", len(OR), "行", flush=True)

    # 2. C1 效应图的数据 --------------------------------------------------------
    pick, c1 = raw["C1 去向"]
    assert pick == "L1"
    dC1, _, _ = dev_data(T, "C1 去向")
    fmode = int(dC1.fstate.mode()[0])
    grid_rows = []
    for inter in (False, True):
        for ready in (True, False):
            for cb in range(6):
                grid_rows.append({"ever_conf": ready, "conf_cat": "c6_24" if ready else "none", "met03": ready,
                                  "fstate": fmode, "clock_block": cb, "weekend": False, "inter": inter, "subtype": "AIS",
                                  "k": 48, "_ready": ready})
    G = pd.DataFrame(grid_rows)
    G["p_exit"] = c1.proba(G)[:, list(c1.classes_).index(2)] * 100
    G.drop(columns=["_ready"]).to_csv(OUT / "fig6a_data.csv", index=False)

    # 3. SHAP 与方向对照 ---------------------------------------------------------
    rng = np.random.default_rng(20261106)
    shap_rows = []
    for name, (pick, mod) in raw.items():
        if pick != "L3":
            continue
        d, y, fkey = dev_data(T, name)
        l1 = S.L1Module(S.FORMULAS[fkey]).fit(d, y)
        idx = rng.choice(len(d), min(40000, len(d)), replace=False)
        ds = d.iloc[idx].reset_index(drop=True)
        X, _, _ = mod._X(ds, mod.info)
        cols = mod.info.column_names[1:]
        ncls = mod.ncls
        con = mod.b.predict(X, pred_contrib=True)
        con = con.reshape(len(ds), 1, -1) if ncls == 2 else con.reshape(len(ds), ncls, -1)
        con = con[:, :, :-1]                                               # 去掉期望值列
        Xl = np.asarray(build_design_matrices([l1.info], ds)[0])
        cl = l1.m.coef_                                                    # 二分类 (1, p)；多分类 (K, p)
        for v in sorted({var_of(c) for c in cols}, key=lambda v: list(VAR_EN).index(v) if v in VAR_EN else 99):
            sc = [i for i, c in enumerate(cols) if var_of(c) == v]
            sh = con[:, :, sc].sum(2)                                      # (n, 类别数)
            lsl = [sl for t, sl in l1.info.term_slices.items() if t.name() != "Intercept" and var_of(t.name()) == v][0]
            lc = Xl[:, lsl] @ cl[:, lsl].T                                # (n, 1 或 K)
            if ncls > 2:                                                   # 两者都是对称的 softmax 参数化，逐类比较
                lc = lc - lc.mean(1, keepdims=True)
                sh = sh - sh.mean(1, keepdims=True)
            rs = []
            for j in range(sh.shape[1]):
                if np.std(sh[:, j]) > 0 and np.std(lc[:, j]) > 0:
                    rs.append(spearmanr(sh[:, j], lc[:, j]).correlation)
            shap_rows.append({"Module": name, "Variable": VAR_EN.get(v, v), "Mean |SHAP|": np.abs(sh).mean(),
                              "Direction agreement with L1 (Spearman)": np.mean(rs) if rs else np.nan})
        print("SHAP", name, flush=True)
    SH = pd.DataFrame(shap_rows)
    SH["Rank"] = SH.groupby("Module")["Mean |SHAP|"].rank(ascending=False).astype(int)
    SH = SH.sort_values(["Module", "Rank"])
    SH.to_csv(OUT / "tableS_shap_data.csv", index=False)
    (OUT / "tableS_shap.md").write_text(
        "# Supplementary table. LightGBM modules: TreeSHAP importance and agreement with the L1 model\n\n"
        "Mean |SHAP| on the log-odds scale (multiclass: average over classes after centring), development rows "
        "(random sample ≤40,000). Agreement = Spearman correlation between the variable's SHAP contribution and its "
        "contribution in the L1 model of the same module, row by row; negative values mean opposite directions and are "
        "discussed. SHAP values describe the model, not causal effects.\n\n"
        "| Module | Rank | Variable | Mean \\|SHAP\\| | Agreement with L1 |\n|---|---|---|---|---|\n" +
        "\n".join(f"| {r['Module']} | {r['Rank']} | {r['Variable']} | {r['Mean |SHAP|']:.3f} | {r['Direction agreement with L1 (Spearman)']:.2f} |" for _, r in SH.iterrows()),
        encoding="utf-8")

    # 4. 典型病人（虚构特征组合）的路径 ------------------------------------------
    t0 = pd.Timestamp("2150-03-04 14:00")                                    # 周二下午入 ICU（日期为虚构）
    PROTO = [("Ischaemic stroke, 72 y, thrombolysis/thrombectomy,\nstandard ICU",
              {"age10": 7.2, "subtype": "AIS", "reperf": True, "entry": "ED", "intime": t0}, False),
             ("Intracerebral haemorrhage, 65 y,\nstandard ICU", {"age10": 6.5, "subtype": "ICH", "reperf": False,
                                                                "entry": "ED", "intime": t0}, False),
             ("Ischaemic stroke, 75 y, no reperfusion,\nneuro intermediate unit",
              {"age10": 7.5, "subtype": "AIS", "reperf": False, "entry": "ED", "intime": t0}, True)]
    traces, summ = [], []
    for i, (lab, pat, inter) in enumerate(PROTO):
        start = {"k": 0, "fstate": -1, "met03": False, "run_h": 0, "ever_conf": False, "first_conf_k": np.nan, "inter": inter}
        rec = []
        res = simulate_trace(pat, start, fm, NSIM, np.random.default_rng(1000 + i), rec=rec)
        chk = S.simulate_fast(pat, start, fm, NSIM, np.random.default_rng(1000 + i))
        assert res.equals(chk), "记录版与原引擎结果不同"
        tr = pd.DataFrame(rec, columns=["hour", "icu_not_ready", "icu_ready", "exited", "died"]).set_index("hour")
        traces.append(tr)
        ex = res[res.end_type == "exit"]
        exc = ex[ex.ever_conf]
        summ.append({"Profile": lab.replace("\n", " "),
                     "P(alive exit ≤72 h)": ((res.end_type == "exit") & (res.end_k <= 71)).mean(),
                     "P(ICU death ≤72 h)": ((res.end_type == "death") & (res.end_k <= 71)).mean(),
                     "Median LOS (h)": float(np.median(res.end_k + 1)),
                     "Exits ready at some point (%)": 100 * len(exc) / max(len(ex), 1),
                     "Median h to readiness": float(np.median(exc.first_conf_k)) if len(exc) else np.nan,
                     "Median h readiness→exit": float(np.median(exc.end_k + 1 - exc.first_conf_k)) if len(exc) else np.nan})
    PS = pd.DataFrame(summ)
    print(PS.round(2).to_string(), flush=True)

    # 5. 时间验证集：两段时间（模拟 对 观察） --------------------------------------
    P = T["P"]
    ids = P.index[P.split == "test"].to_numpy()
    r = Parallel(n_jobs=12)(delayed(IL.test_chunk)(ids[i::12]) for i in range(12))
    R = [o for ch in r for o in ch]
    H = T["H"]
    fc = H[H.confirmed_now].groupby("stay_id").k.min()
    tp = P.loc[ids]
    first_inter = H[(H.k == 0) & H.stay_id.isin(ids)].set_index("stay_id").inter.reindex(ids)
    alive = tp[~tp.died & ~tp.trunc]
    rows_ts = []
    for lab, sel in [("All", np.ones(len(ids), bool)), ("First unit: standard ICU", ~first_inter.to_numpy().astype(bool)),
                     ("First unit: intermediate", first_inter.to_numpy().astype(bool))]:
        sids = set(ids[sel])
        a = alive[alive.index.isin(sids)]
        conf = a.index.isin(fc.index)
        ob_b = fc.reindex(a.index[conf]).to_numpy()
        ob_a = a["T"].to_numpy()[conf] - ob_b
        sb = np.concatenate([o["before"] for o in R if o["stay_id"] in sids])
        sa = np.concatenate([o["after"] for o in R if o["stay_id"] in sids])
        ne = sum(o["n_exit"] for o in R if o["stay_id"] in sids)
        nc = sum(o["n_exit_conf"] for o in R if o["stay_id"] in sids)
        q = lambda x: f"{np.median(x):.0f} ({np.percentile(x, 25):.0f}–{np.percentile(x, 75):.0f})"  # noqa: E731
        rows_ts.append({"Group": lab, "Alive exits, n": len(a),
                        "Ready before exit: observed / simulated (%)": f"{100 * conf.mean():.1f} / {100 * nc / ne:.1f}",
                        "Hours to readiness: observed": q(ob_b), "simulated": q(sb),
                        "Hours readiness→exit: observed": q(ob_a), "simulated ": q(sa)})
    TS = pd.DataFrame(rows_ts)
    (OUT / "time_split.md").write_text(
        "# Remaining time split into 'before readiness' and 'after readiness to exit' (temporal validation set, from ICU admission)\n\n"
        "Observed: alive ICU exits (not truncated). Simulated: simulated alive exits, 1,000 per patient, pooled. "
        "Median (IQR), hours.\n\n" +
        "| " + " | ".join(TS.columns) + " |\n|" + "---|" * len(TS.columns) + "\n" +
        "\n".join("| " + " | ".join(map(str, r)) + " |" for r in TS.itertuples(index=False)) + "\n\n" +
        "## Prototype profiles (fictional combinations of characteristics, not real patients)\n\n" +
        "| " + " | ".join(PS.columns) + " |\n|" + "---|" * len(PS.columns) + "\n" +
        "\n".join("| " + " | ".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in r) + " |" for r in PS.itertuples(index=False)),
        encoding="utf-8")
    print(TS.to_string(), flush=True)

    # 6. 适用范围标记 ------------------------------------------------------------
    (OUT / "applicability.md").write_text("""# Applicability flags (SAP v1.0 §8)

Every prediction is flagged as outside the model's reliable range when any of the following holds:

| Flag | Rule |
|---|---|
| Unit not in the list | The current unit is not one of the listed ICU or neuro intermediate / step-down units |
| Beyond 30 days | The patient has been in the ICU for more than 720 hours |
| Missing inputs | An input is missing beyond the pre-specified missing-data rules (e.g., no condition state available after hour 6) |
| Off-landmark | The prediction time is not one of the four pre-specified landmarks (0, 24, 48, 72 h); predictions at other hours are produced by the same engine but were not validated |
| Post-exit beyond 72 h | Post-exit outcomes beyond 72 hours, and a second ICU stay after readmission, are not simulated |
""", encoding="utf-8")

    # 图 6 与补充 SHAP 图由 63_plot_interpretability.py 绘制（读取本脚本输出的表，重跑三个典型病人的模拟）
    print("完成", flush=True)
