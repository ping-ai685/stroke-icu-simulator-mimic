"""
第四篇论文 · 表 1 与图 2–5（只输出汇总结果与图；个体层面数据不离开本机）

表 1：各划分的病人特征（建模集 / 时间验证集 / 附加验证集 / 全部）
图 2：ICU 时间的逐小时分解（按单元类型 × 住院阶段；中间监护所占小时的年代变化）——来自 paper4_formal/outputs 的汇总表
图 3：转出条件的动态（达到 v0.3 的累积发生率；再次失去时未满足的条件；达到与离开的钟点）——同上
图 4：时间验证（2017–2019）：24 / 48 / 72 小时存活离开的校准（模拟器 对 GBM，按预测十分位分组）；ICU 停留时间分布
图 5：简化方案比较（与完整模拟器配对的 Brier 差及 95% 区间；24 小时区分度；停留时间 Wasserstein 距离）
输出：outputs/paper/table1.md、table1.csv、fig2–fig5（.pdf + .png）
"""
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wasserstein_distance  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import simlib as S  # noqa: E402

WORK, OUT = HERE / "work", HERE / "outputs" / "paper"
OUT.mkdir(parents=True, exist_ok=True)
FO = HERE.parent / "paper4_formal" / "outputs"

# 配色：dataviz 参考调色板前三个类别色（已用验证脚本检查，三色两两通过）；文字用墨色，不用系列色
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 300})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def panel(ax, letter):
    ax.text(-0.22, 1.06, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", color=INK, va="bottom")


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


T = S.load_tables(WORK)
P = T["P"]
GROUPS = [("Development (≤2016)", P.split == "dev"), ("Temporal validation (2017–2019)", P.split == "test"),
          ("Additional validation (≥2020)", P.split == "test2020"), ("All", P.split.notna())]

# ---------------------------------------------------------------------------
# 表 1
# ---------------------------------------------------------------------------
INTER = {"Neuro Intermediate", "Neuro Stepdown"}
A = T["A"]
post = T["PB"].groupby("stay_id").y_post.max()


def q(x, d=0):
    a, b, c = np.percentile(x, [50, 25, 75])
    return f"{a:.{d}f} ({b:.{d}f}–{c:.{d}f})"


def pct(m, n=None):
    m = np.asarray(m)
    return f"{int(m.sum()):,} ({100 * m.mean():.1f})"


rows = {}
for name, m in GROUPS:
    p = P[m]
    a = A[A.index.isin(p.index)]
    pb = post.reindex(a.index[~a.direct_leave])
    r = {"ICU stays, n": f"{len(p):,}",
         "Age, years, median (IQR)": q(p.age_at_adm_capped),
         "Female, n (%)": pct(p.gender == "F"),
         "Stroke subtype, n (%)": "",
         "  Ischaemic stroke": pct(p.subtype == "AIS"),
         "  Intracerebral haemorrhage": pct(p.subtype == "ICH"),
         "  Subarachnoid haemorrhage": pct(p.subtype == "SAH"),
         "Reperfusion therapy (ischaemic stroke), n (%)": pct(p[p.subtype == "AIS"].reperf),
         "Source of ICU admission, n (%)": "",
         **{f"  {lab}": pct(p.entry == k) for k, lab in
            [("ED", "Emergency department"), ("ward", "Hospital ward"), ("OR", "Operating room"),
             ("direct", "Direct / other"), ("prior_ICU", "Another ICU stay")]},
         "First unit: neuro intermediate / step-down, n (%)": pct(p.first_careunit.isin(INTER)),
         "ICU length of stay, hours, median (IQR)": q(np.minimum(p["T"], 720)),
         "Stay truncated at 30 days, n (%)": pct(p.trunc),
         "Died in the ICU, n (%)": pct(p.died),
         "Alive ICU exits, n": f"{len(a):,}",
         "  Direct hospital discharge (≤1 h), n (%)": pct(a.direct_leave),
         "  ICU readmission within 72 h, n (%)": pct(pb == 1) if len(pb) else "",
         "  In-hospital death within 72 h, n (%)": pct(pb == 2) if len(pb) else "",
         "Patient-hours in the ICU, n": f"{int(T['H'].stay_id.isin(p.index).sum()):,}"}
    rows[name] = r
tab1 = pd.DataFrame(rows)
tab1.index.name = "Characteristic"
tab1.to_csv(OUT / "table1.csv")
note = ("Percentages for post-exit outcomes use alive ICU exits as the denominator (readmission and death: exits "
        "not directly discharged). Stays are ICU stays; development = approximate admission year ≤2016. "
        "ICU length of stay is truncated at 720 h.")
(OUT / "table1.md").write_text("# Table 1. Characteristics of ICU stays by data split\n\n" +
                               "| " + tab1.index.name + " | " + " | ".join(tab1.columns) + " |\n|" + "---|" * (len(tab1.columns) + 1) + "\n" +
                               "\n".join("| " + i + " | " + " | ".join(r) + " |" for i, r in tab1.iterrows()) +
                               "\n\n" + note + "\n", encoding="utf-8")

# ---------------------------------------------------------------------------
# 图 2：ICU 时间分解
# ---------------------------------------------------------------------------
ux = pd.read_csv(FO / "5_1_unit_x_phase.csv", index_col=[0, 1])
mu = pd.read_csv(FO / "main_by_unit_type.csv", index_col=0)
era = pd.read_csv(FO / "posthoc_unit_share_by_era.csv", index_col=0)
num = lambda s: float(str(s).split(" ")[0])  # noqa: E731
PH_EN = ["Day 1", "Days 2–3", "Days 4–7", "Days 8–14", "Days 15–30"]
UNITS = [("正规ICU", "Standard ICU"), ("神经中间监护/过渡病房", "Neuro intermediate / step-down")]
SEG = [("T1", "T1 organ-support needs", BLUE), ("T2", "T2 stroke-specific needs", ORANGE),
       ("T3", "T3 not explained by observed needs", AQUA)]

fig = plt.figure(figsize=(7.2, 3.6))
gs = fig.add_gridspec(1, 3, width_ratios=[2.3, 2.3, 1.5], wspace=0.38)
for j, (zh, en) in enumerate(UNITS):
    ax = fig.add_subplot(gs[0, j])
    labs = ["All days"] + PH_EN
    vals = {s: [num(mu.loc[zh, s])] + [num(ux.loc[(zh, ph), f"{s} 小时口径%"]) for ph in ux.loc[zh].index] for s, _, _ in SEG}
    x = np.arange(len(labs)) + np.r_[0, np.ones(5) * 0.4]
    bottom = np.zeros(len(labs))
    for s, lab, c in SEG:
        v = np.array(vals[s])
        ax.bar(x, v, 0.72, bottom=bottom, color=c, edgecolor="white", linewidth=1, label=lab)
        if s == "T3":
            for xi, b, vi in zip(x, bottom, v):
                ax.text(xi, b + vi / 2, f"{vi:.0f}", ha="center", va="center", fontsize=7, color=INK)
        bottom += v
    ax.set_xticks(x, labs, rotation=50, ha="right", rotation_mode="anchor", fontsize=7)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Share of ICU patient-hours (%)" if j == 0 else "")
    ax.set_title(f"{en}\n({int(mu.loc[zh, '人数']):,} stays)", fontsize=8, color=INK)
    grid(ax)
    panel(ax, "ab"[j])
ax = fig.add_subplot(gs[0, 2])
eras = [e.replace(" - ", "–") for e in era.index]
ax.bar(np.arange(len(eras)), era["神经中间监护/过渡病房"], 0.62, color=INK2, edgecolor="white")
for i, v in enumerate(era["神经中间监护/过渡病房"]):
    ax.text(i, v + 1.5, f"{v:.0f}", ha="center", fontsize=7, color=INK)
ax.set_xticks(np.arange(len(eras)), eras, rotation=50, ha="right", rotation_mode="anchor", fontsize=7)
ax.set_ylim(0, 60)
ax.set_ylabel("Hours in neuro intermediate units (%)")
ax.set_title("Unit mix by\nanchor-year group", fontsize=8, color=INK)
grid(ax)
panel(ax, "c")
h, l = fig.axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.09), fontsize=7.5)
save(fig, "fig2_icu_time_decomposition")

# ---------------------------------------------------------------------------
# 图 3：转出条件的动态
# ---------------------------------------------------------------------------
cif = pd.read_csv(FO / "5_4_cif_v03.csv", index_col=0)
rel = pd.read_csv(FO / "5_5_relapse.csv", index_col=0)["值"]
hod = pd.read_csv(FO / "5_6_hour_of_day.csv", index_col=0)
fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.7), gridspec_kw={"wspace": 1.0, "width_ratios": [1.15, 1.0, 1.25]})
ax = axs[0]
hrs = [8, 24, 48, 72, 168]
for key, lab, c in [("首个单元 中间监护/过渡", "Intermediate", ORANGE), ("全部", "All stays", INK2),
                    ("首个单元 正规ICU", "Standard ICU", BLUE)]:
    y = cif.loc[key, [f"{h}小时" for h in hrs]].to_numpy(float)
    ax.plot(hrs, y, "-o", color=c, lw=2, ms=4, label=lab)

ax.set_xticks([0, 24, 48, 72, 168], ["0", "24", "48", "72", "168"])
ax.set_xlabel("Hours since ICU admission")
ax.set_ylabel("Reached stroke-extended readiness (%)")
ax.set_ylim(0, 100)
ax.legend(fontsize=6.5, loc="lower right", handlelength=1.5, title="First unit", title_fontsize=6.5)
grid(ax)
panel(ax, "a")
ax = axs[1]
cause = [("呼吸", "Respiratory support"), ("代谢", "Metabolic"), ("意识：持续镇静", "Continuous sedation"),
         ("升压药", "Vasopressor"), ("意识：确认的GCS下降", "Confirmed GCS drop"), ("意识：GCS间隔>6h", "GCS gap >6 h"),
         ("降压药泵入", "IV antihypertensive"), ("意识：躁动", "Agitation"), ("意识：癫痫", "Seizure")]
v = [rel[f"不满足的小时中：{k}%"] for k, _ in cause]
yy = np.arange(len(cause))[::-1]
ax.barh(yy, v, 0.62, color=BLUE, edgecolor="white")
for yi, vi in zip(yy, v):
    ax.text(vi + 1, yi, f"{vi:.0f}", va="center", fontsize=7, color=INK)
ax.set_yticks(yy, [e for _, e in cause], fontsize=7)
ax.set_xlabel("Unmet hours after first readiness (%)")
ax.set_xlim(0, 62)
ax.set_title(f"Lost ≥6 h in {rel.iloc[1]:.1f}% of stays", fontsize=7.5, color=INK, loc="left")
grid(ax, "x")
ax.text(-0.95, 1.06, "b", transform=ax.transAxes, fontsize=10, fontweight="bold", color=INK, va="bottom")
ax = axs[2]
for col, lab, c in [("首次达到v0.3", "First reached readiness", BLUE), ("实际离开ICU", "Left the ICU", ORANGE)]:
    y = 100 * hod[col] / hod[col].sum()
    ax.plot(hod.index, y, "-", color=c, lw=2, label=lab)
ax.set_xticks([0, 6, 12, 18, 23], ["00", "06", "12", "18", "23"])
ax.set_xlabel("Hour of day")
ax.set_ylabel("Share of events (%)")
ax.set_ylim(0, 22)
ax.set_yticks([0, 5, 10, 15, 20])
ax.legend(fontsize=6.5, loc="upper left")
grid(ax)
panel(ax, "c")
save(fig, "fig3_readiness_dynamics")

# ---------------------------------------------------------------------------
# 图 4：时间验证
# ---------------------------------------------------------------------------
tv = pickle.load(open(WORK / "tv_predictions.pkl", "rb"))
te = tv["te_lm"].reset_index(drop=True)
simA = tv["sims"]["A"]
fig, axs = plt.subplots(1, 4, figsize=(7.4, 2.3), gridspec_kw={"wspace": 0.75})
for j, h in enumerate([24, 48, 72]):
    ax = axs[j]
    y = (te[f"y{h}"] == 1).to_numpy()
    ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls="--")
    for p, lab, c, mk in [(np.stack(simA[f"p{h}"])[:, 1], "Simulator", BLUE, "o"),
                          (tv["baselines"]["GBM"][h][:, 1], "Direct GBM", ORANGE, "s")]:
        b = pd.qcut(p, 10, labels=False, duplicates="drop")
        g = pd.DataFrame({"p": p, "y": y, "b": b}).groupby("b").mean()
        ax.plot(g.p, g.y, "-" + mk, color=c, lw=2, ms=4, label=lab)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title(f"Alive exit ≤{h} h", fontsize=7.5, color=INK)
    ax.set_xlabel("Predicted")
    if j == 0:
        ax.set_ylabel("Observed")
        ax.legend(fontsize=6.5, loc="upper left")
    grid(ax, "both")
    panel(ax, "abc"[j])
ax = axs[3]
adm = simA[simA.k == 0]
obs = np.sort(np.minimum(P.loc[adm.stay_id, "T"].to_numpy(), 720))
dist = np.stack(adm.los).sum(0)
cdf = np.cumsum(dist) / dist.sum()
ax.plot(obs, np.arange(1, len(obs) + 1) / len(obs), color=INK, lw=2, label="Observed")
ax.plot(np.arange(721), cdf, color=BLUE, lw=2, label="Simulated")
ax.set_xlim(0, 400)
ax.set_ylim(0, 1)
ax.set_box_aspect(1)
ax.set_xlabel("ICU length of stay (h)")
ax.set_title("LOS, cumulative", fontsize=7.5, color=INK)
ax.legend(fontsize=6.5, loc="lower right", bbox_to_anchor=(1.05, 0.0), handlelength=1.2)
grid(ax, "both")
panel(ax, "d")
save(fig, "fig4_temporal_validation")

# ---------------------------------------------------------------------------
# 图 5：简化方案
# ---------------------------------------------------------------------------
g = te.stay_id.to_numpy()
y24 = (te.y24 == 1).to_numpy().astype(float)
obs_los = np.minimum(P.loc[adm.stay_id, "T"].to_numpy(), 720)


def boot(groups, a, b, n=1000, seed=20261106):
    d = pd.DataFrame({"g": groups, "a": a, "b": b, "n": 1}).groupby("g").sum().to_numpy()
    rng = np.random.default_rng(seed)
    v = [(d[i, 0].sum() - d[i, 1].sum()) / d[i, 2].sum() for i in (rng.integers(0, len(d), len(d)) for _ in range(n))]
    return np.percentile(v, [2.5, 97.5])


sims = {"Full simulator": simA}
for k, lab in [("A", "A: no readiness layer"), ("B", "B: no clock / weekend / unit"), ("C", "C: no readiness relapse")]:
    sims[lab] = pickle.load(open(WORK / f"abl_sims_{k}.pkl", "rb"))
se_full = (np.stack(simA.p24)[:, 1] - y24) ** 2
res = []
for lab, sm in sims.items():
    p = np.stack(sm.p24)[:, 1]
    se = (p - y24) ** 2
    lo, hi = boot(g, se, se_full) if lab != "Full simulator" else (0, 0)
    d0 = sm[sm.k == 0]
    res.append({"label": lab, "diff": se.mean() - se_full.mean(), "lo": lo, "hi": hi, "c": roc_auc_score(y24, p),
                "w": wasserstein_distance(obs_los, np.arange(721), v_weights=np.stack(d0.los).sum(0))})
R = pd.DataFrame(res)
R.to_csv(OUT / "fig5_data.csv", index=False)
fig, axs = plt.subplots(1, 3, figsize=(7.2, 1.9), gridspec_kw={"wspace": 0.12, "width_ratios": [1.4, 1, 1]})
yy = np.arange(len(R))[::-1]
ax = axs[0]
ax.axvline(0, color=INK2, lw=0.8, ls="--")
for yi, r in zip(yy, R.itertuples()):
    ax.plot([r.lo * 1000, r.hi * 1000], [yi, yi], color=BLUE, lw=2)
    ax.plot(r.diff * 1000, yi, "o", color=BLUE if r.label != "Full simulator" else INK, ms=5)
ax.set_yticks(yy, R.label)
ax.set_xlabel("Δ 24-h Brier vs full (×10⁻³)\n(positive = worse)")
grid(ax, "x")
panel(ax, "a")
for j, (col, xl, fmt) in enumerate([("c", "24-h C statistic", "{:.3f}"), ("w", "LOS Wasserstein (h)", "{:.1f}")]):
    ax = axs[j + 1]
    ax.barh(yy, R[col], 0.55, color=[INK2] + [BLUE] * 3, edgecolor="white")
    for yi, v in zip(yy, R[col]):
        ax.text(v, yi, " " + fmt.format(v), va="center", fontsize=7, color=INK)
    ax.set_yticks(yy, [])
    ax.set_xlim(0.6 if col == "c" else 0, (0.74 if col == "c" else 42))
    ax.set_xlabel(xl)
    grid(ax, "x")
    panel(ax, "bc"[j])
save(fig, "fig5_ablations")
print(tab1.to_string())
print(R.round(4).to_string())
