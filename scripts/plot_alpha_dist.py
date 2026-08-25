#!/usr/bin/env python
# 用法: /root/venvs/kernel-opt/bin/python scripts/plot_alpha_dist.py
# 从 data/raw/EXP-002/awq_g128.json(raw 只读)重算 per-layer best-α 分布,
# 产出 figures/fig1_awq_alpha_dist.png(纯 CPU;图禁手改,STANDARDS §6)。
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data/raw/EXP-002/awq_g128.json"
OUT = ROOT / "figures/fig1_awq_alpha_dist.png"

# 配色(单系列):series-1 蓝 + 文本 token(dataviz 规范,白底)
BLUE, INK, INK2, SURFACE = "#2a78d6", "#0b0b0b", "#52514e", "#ffffff"

for name in ("Noto Sans CJK SC", "Noto Sans Mono CJK SC"):
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False

d = json.loads(SRC.read_text())
alphas = [e["alpha"] for e in d["per_layer"]]
n = len(alphas)
med = statistics.median(alphas)
grid = [round(i * 0.05, 2) for i in range(20)]  # AWQ 搜索网格 0–0.95
counts = [alphas.count(a) for a in grid]

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=220)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
ax.bar(grid, counts, width=0.04, color=BLUE, zorder=3)
ax.axvline(med, color=INK2, lw=1, ls="--", zorder=2)
ax.annotate(f"中位 {med:.2f}", xy=(med, max(counts)), xytext=(med + 0.02, max(counts) - 1),
            color=INK2, fontsize=9)
ax.annotate("强 outlier 层 ×2 顶到 0.95", xy=(0.95, alphas.count(0.95)),
            xytext=(0.62, 8), color=INK2, fontsize=9,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
ax.set_title(f"AWQ 保护强度按层自适应:α 中位 {med:.2f},主体 0.15–0.45,两层顶到 0.95",
             color=INK, fontsize=11, pad=12)
ax.set_xlabel("per-layer best α(搜索网格 0–0.95,步 0.05,无量纲)", color=INK, fontsize=9.5)
ax.set_ylabel("Linear 层数(个)", color=INK, fontsize=9.5)
ax.set_xticks([round(x * 0.1, 1) for x in range(10)])
ax.tick_params(colors=INK2, labelsize=8.5)
ax.grid(axis="y", color="#e6e5e2", lw=0.7, zorder=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color("#d8d7d3")
fig.text(0.01, 0.005,
         f"源数据 data/raw/EXP-002/awq_g128.json(n={n} Linear 层,EXP-002 §5)· "
         "RTX 4090 单卡 · 单轮 · scripts/plot_alpha_dist.py",
         color=INK2, fontsize=7)
fig.tight_layout(rect=(0, 0.03, 1, 1))
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, facecolor=SURFACE)
print(f"wrote {OUT} (n={n}, median={med})")
