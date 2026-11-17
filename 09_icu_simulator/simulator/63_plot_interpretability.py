"""第四篇 · 图 6 与补充 SHAP 图（读取 61_interpretability.py 的输出；三个典型病人的路径在此重跑，与 61 同一种子）"""
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import interp_lib as IL  # noqa: E402
import simlib as S  # noqa: E402

spec = importlib.util.spec_from_file_location("interp61", HERE / "61_interpretability.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)                       # 只执行顶层定义（含 simulate_trace），不执行主程序
OUT = M.OUT
BLUE, ORANGE, AQUA, INK, INK2, GRID = M.BLUE, M.ORANGE, M.AQUA, M.INK, M.INK2, M.GRID

T, fm, raw = IL.model()
G = pd.read_csv(OUT / "fig6a_data.csv")
G["_ready"] = G.ever_conf
SH = pd.read_csv(OUT / "tableS_shap_data.csv")
t0 = pd.Timestamp("2150-03-04 14:00")
PROTO = [("Ischaemic stroke, 72 y, reperfused\nstandard ICU", {"age10": 7.2, "subtype": "AIS", "reperf": True, "entry": "ED", "intime": t0}, False),
         ("Intracerebral haemorrhage, 65 y\nstandard ICU", {"age10": 6.5, "subtype": "ICH", "reperf": False, "entry": "ED", "intime": t0}, False),
         ("Ischaemic stroke, 75 y, no reperfusion\nneuro intermediate unit", {"age10": 7.5, "subtype": "AIS", "reperf": False, "entry": "ED", "intime": t0}, True)]
traces, summ = [], []
for i, (lab, pat, inter) in enumerate(PROTO):
    start = {"k": 0, "fstate": -1, "met03": False, "run_h": 0, "ever_conf": False, "first_conf_k": np.nan, "inter": inter}
    rec = []
    res = M.simulate_trace(pat, start, fm, M.NSIM, np.random.default_rng(1000 + i), rec=rec)
    traces.append(pd.DataFrame(rec, columns=["hour", "icu_not_ready", "icu_ready", "exited", "died"]).set_index("hour"))
    exc = res[(res.end_type == "exit") & res.ever_conf]
    summ.append((float(np.median(exc.first_conf_k)), float(np.median(exc.end_k + 1 - exc.first_conf_k))))

fig = plt.figure(figsize=(7.4, 5.2))
top = fig.add_gridspec(1, 2, left=0.08, right=0.98, top=0.93, bottom=0.6, width_ratios=[1.6, 1], wspace=0.75)
bot = fig.add_gridspec(1, 3, left=0.08, right=0.98, top=0.42, bottom=0.1, wspace=0.3)
ax = fig.add_subplot(top[0, 0])
xs = np.arange(6) * 4 + 2
for inter, c, ul in [(False, BLUE, "Standard ICU"), (True, ORANGE, "Intermediate unit")]:
    for ready, ls, rl in [(True, "-", "ready"), (False, "--", "not ready")]:
        g = G[(G.inter == inter) & (G._ready == ready)]
        ax.plot(xs, g.p_exit, ls, marker="o", ms=3.5, lw=2, color=c, label=f"{ul}, {rl}")
ax.set_xticks(np.arange(0, 25, 4), [f"{h:02d}" for h in range(0, 25, 4)])
ax.set_xlim(0, 24)
ax.set_ylim(0, 11)
ax.set_xlabel("Hour of day (4-hour blocks)")
ax.set_ylabel("P(leave ICU in next hour), %")
ax.set_title("Destination module (C1)", fontsize=8, color=INK, loc="left")
ax.legend(fontsize=6.3, loc="upper left", ncol=2, columnspacing=1)
ax.grid(axis="y", color=GRID, lw=0.6)
ax.set_axisbelow(True)
ax.text(-0.16, 1.05, "a", transform=ax.transAxes, fontsize=10, fontweight="bold")
ax = fig.add_subplot(top[0, 1])
t = SH[SH.Module.str.startswith("D3a")].sort_values("Mean |SHAP|")
ax.barh(np.arange(len(t)), t["Mean |SHAP|"], 0.6, color=BLUE, edgecolor="white")
ax.set_yticks(np.arange(len(t)), t.Variable, fontsize=6.8)
ax.set_xlabel("Mean |SHAP| (log-odds)")
ax.set_title("Readiness gained (D3a, LightGBM)", fontsize=8, color=INK, loc="left")
ax.grid(axis="x", color=GRID, lw=0.6)
ax.set_axisbelow(True)
ax.text(-1.05, 1.05, "b", transform=ax.transAxes, fontsize=10, fontweight="bold")
parts = [("icu_not_ready", "In ICU, not ready", BLUE), ("icu_ready", "In ICU, ready", AQUA),
         ("exited", "Left ICU alive", ORANGE), ("died", "Died in ICU", INK2)]
for i, (tr, (lab, _, _), (tb, ta)) in enumerate(zip(traces, PROTO, summ)):
    ax = fig.add_subplot(bot[0, i])
    tr = tr.reindex(range(1, 169)).ffill()
    ax.stackplot(tr.index, *[100 * tr[c].to_numpy() / M.NSIM for c, _, _ in parts], colors=[c for _, _, c in parts],
                 labels=[l for _, l, _ in parts], edgecolor="white", linewidth=0.3)
    ax.set_xlim(0, 168)
    ax.set_ylim(0, 100)
    ax.set_xticks([0, 24, 48, 72, 120, 168])
    ax.set_xlabel("Hours since ICU admission")
    if i == 0:
        ax.set_ylabel("Simulated paths, %")
    else:
        ax.set_yticklabels([])
    ax.set_title(lab, fontsize=6.8, color=INK, loc="left")
    ax.text(0.97, 0.04, f"to readiness {tb:.0f} h\nready → exit {ta:.0f} h", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=6.3, color=INK, bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.5))
    ax.text(-0.2 if i == 0 else -0.06, 1.17, "cde"[i], transform=ax.transAxes, fontsize=10, fontweight="bold")
h, l = ax.get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.53, 0.0), fontsize=7)
fig.savefig(OUT / "fig6_interpretability.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig6_interpretability.png", bbox_inches="tight", dpi=300)
plt.close(fig)

mods = list(dict.fromkeys(SH.Module))
names = {m: m.replace("D2（当前状态 1）", "D2, from state 1").replace("D3a 不满足→满足", "D3a readiness gained")
         .replace("D3b 满足→不满足", "D3b readiness lost").replace("P1b", "P1b after ICU exit") for m in mods}
fig, axs = plt.subplots(2, 2, figsize=(7.2, 4.6), gridspec_kw={"wspace": 1.0, "hspace": 0.6})
for ax, m in zip(axs.ravel(), mods):
    t = SH[SH.Module == m].sort_values("Mean |SHAP|")
    neg = t["Direction agreement with L1 (Spearman)"].fillna(1) < 0
    ax.barh(np.arange(len(t)), t["Mean |SHAP|"], 0.6, color=np.where(neg, ORANGE, BLUE), edgecolor="white")
    ax.set_yticks(np.arange(len(t)), t.Variable, fontsize=6.5)
    ax.set_title(names[m], fontsize=7.5, color=INK, loc="left")
    ax.set_xlabel("Mean |SHAP| (log-odds)", fontsize=7)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
fig.text(0.5, -0.02, "Blue: same direction as the logistic (L1) model of the same module; orange: opposite direction.",
         ha="center", fontsize=6.8, color=INK2)
fig.savefig(OUT / "figS_shap.pdf", bbox_inches="tight")
fig.savefig(OUT / "figS_shap.png", bbox_inches="tight", dpi=300)
print("完成")
