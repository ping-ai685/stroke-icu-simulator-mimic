"""第四篇 · 图 1：研究设计（人群与按年代划分）与模拟器的逐小时结构。只用汇总数字。输出 outputs/paper/fig1_design.*"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs" / "paper"
INK, INK2 = "#0b0b0b", "#52514e"
TINT = {"cond": "#cde2fb", "dec": "#fbd8c9", "ready": "#c6eedf", "post": "#ebeae6", "data": "#f4f3f0"}
EDGE = {"cond": "#2a78d6", "dec": "#eb6834", "ready": "#1baf7a", "post": INK2, "data": INK2}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5})

fig = plt.figure(figsize=(7.4, 4.9))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 66)
ax.axis("off")


def box(x, y, w, h, text, kind, size=7.2, weight="normal", align="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", facecolor=TINT[kind],
                                edgecolor=EDGE[kind], linewidth=1))
    ax.text(x + (w / 2 if align == "center" else 1.2), y + h / 2, text, ha=align, va="center", fontsize=size,
            color=INK, weight=weight, linespacing=1.35)


def arrow(x1, y1, x2, y2, style="-|>", rad=0.0, color=INK2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=9, color=color, lw=1,
                                 connectionstyle=f"arc3,rad={rad}"))


ax.text(1, 64, "a", fontsize=11, weight="bold")
ax.text(4, 64, "Population and temporal split", fontsize=8.5, weight="bold", color=INK)
box(2, 55.5, 36, 6, "MIMIC-IV v3.1 stroke ICU stays\n(cohort of our earlier study)  n = 6,368", "data")
arrow(20, 55.2, 20, 51.6)
box(22, 51.2, 17, 3.3, "Excluded: death at or before\nICU admission (n = 2)", "data", size=6.2)
box(2, 44, 36, 6, "6,366 ICU stays · 847,383 patient-hours\nhourly stroke-extended readiness\n6-hour condition states", "data")
arrow(20, 43.7, 20, 41.4)
ax.text(20, 40.2, "Approximate admission year", ha="center", fontsize=7, color=INK2)
segs = [("≤2014\nInternal\ntraining\n1,877", 2, 7.6, "cond"), ("2015–16\nInternal\nvalidation\n1,254", 11.1, 7.6, "cond"),
        ("2017–19\nTemporal\nvalidation\n1,714", 20.2, 7.6, "dec"), ("≥2020\nAdditional\nvalidation\n1,521", 29.3, 8.7, "post")]
for t, x, w, k in segs:
    box(x, 27.5, w, 11, t, k, size=6.4)
ax.annotate("", xy=(1.8, 25.6), xytext=(19, 25.6), arrowprops=dict(arrowstyle="-", color=INK2, lw=1))
ax.text(10.4, 23.6, "Development (≤2016):\nall selection, tuning", ha="center", va="top", fontsize=6.3, color=INK)
ax.text(24, 23.6, "Used\nonce", ha="center", va="top", fontsize=6.5, color=INK)
ax.text(33.6, 23.6, "COVID-era\nrobustness", ha="center", va="top", fontsize=6.3, color=INK)
box(2, 2.5, 36, 14.5, "Comparators and checks\n"
    "• Direct prediction: LR, GBM, GRU (no paths)\n"
    "• Simulator with all-logistic modules\n"
    "• Ablations: no readiness layer; no clock /\n   weekend / unit in destination; no relapse\n"
    "• Parameter uncertainty: 200 bootstraps", "data", size=6.6, align="left")

ax.text(42, 64, "b", fontsize=11, weight="bold")
ax.text(45, 64, "Simulator: one hour of a simulated path (fixed order)", fontsize=8.5, weight="bold", color=INK)
X0, W = 43, 33.5
steps = [(54.5, "1  Condition state (D2-0 at hour 6, then D2\nevery 6 h): 4 HMM states from our earlier study", "cond"),
         (45.5, "2  Death in the ICU  (D1)", "cond"),
         (36.5, "3  Destination  (C1): stay · move to the other\nunit type · leave the ICU", "dec"),
         (27.5, "4  Readiness gained / lost  (D3a / D3b)\nconfirmed after 6 h continuously met", "ready")]
for y, t, k in steps:
    box(X0, y, W, 6.3, t, k, size=6.8, align="left")
for y1, y2 in [(54.2, 52.1), (45.2, 43.1), (36.2, 34.1)]:
    arrow(X0 + 10, y1, X0 + 10, y2)
arrow(X0 + 0.2, 30.6, X0 + 0.2, 57.6, rad=-0.16)
ax.text(X0 - 3.0, 44, "next hour", fontsize=6.5, color=INK2, va="center", ha="center", rotation=90)
box(81, 36.5, 17, 6.3, "Died in the ICU", "post", size=6.8)
arrow(X0 + W, 48.6, 81, 41.5, rad=-0.15)
box(81, 21.5, 17, 12.5, "After ICU exit\nP1a direct hospital\ndischarge (≤1 h)\nP1b hourly to 72 h:\nreadmission · death ·\ndischarge", "post", size=6.5)
arrow(X0 + W, 38.5, 81, 30, rad=0.1)
ax.text(77.5, 35.6, "leave", fontsize=6.5, color=INK2)
box(X0, 2.5, 55, 17.5, "Inputs at each hour: condition state; readiness met now / ever / hours since;\n"
    "hours in ICU; unit type; hour of day (4-h blocks); weekend; age; stroke subtype;\n"
    "reperfusion\n\n"
    "Output, from any hour (validated at 0, 24, 48, 72 h), 1,000 simulated paths:\n"
    "• probabilities of alive exit / ICU death / still in ICU at 24, 48, 72 h\n"
    "• distribution of remaining ICU time, split into before and after readiness\n"
    "• post-exit outcomes within 72 h", "data", size=6.6, align="left")
for k, lab in [("cond", "Patient condition"), ("ready", "Readiness to leave"), ("dec", "Care decision"),
               ("post", "Terminal / post-exit")]:
    pass
for x, k, lab in [(43.3, "cond", "Patient condition"), (56.5, "ready", "Transfer readiness"), (70.0, "dec", "Care decision")]:
    ax.add_patch(FancyBboxPatch((x, 22.4), 1.8, 1.4, boxstyle="round,pad=0.1", facecolor=TINT[k], edgecolor=EDGE[k], lw=1))
    ax.text(x + 2.5, 23.1, lab, fontsize=6.2, va="center", color=INK)
fig.savefig(OUT / "fig1_design.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig1_design.png", bbox_inches="tight", dpi=300)
