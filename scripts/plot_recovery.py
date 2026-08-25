#!/usr/bin/env python
# 用法: /root/venvs/kernel-opt/bin/python scripts/plot_recovery.py
# 从 data/raw/EXP-00{1,2,3}/ 各臂 JSON(raw 只读)复算恢复率,
# 产出 figures/fig2_recovery_rates.png(纯 CPU;图禁手改,STANDARDS §6)。
# 恢复率定义:
#   W4A16 赛道: (PPL_RTN − PPL_arm) / (PPL_RTN − PPL_fp16)      —— 收回 RTN 缺口
#   W8A8  赛道: (PPL_naive − PPL_arm) / (PPL_naive − PPL_fp16)  —— 收回 naive 缺口
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw"
OUT = ROOT / "figures/fig2_recovery_rates.png"
GEN_DATE = "2026-08-25"

font_manager.fontManager.addfont("/usr/share/fonts/truetype/arphic/uming.ttc")
plt.rcParams["font.family"] = font_manager.FontProperties(
    fname="/usr/share/fonts/truetype/arphic/uming.ttc").get_name()
plt.rcParams["axes.unicode_minus"] = False

# 配色:主(我方)/ 次强调 / 中性(基线为 0% 轴,不占条)
MAIN, ACCENT, NEUTRAL = "#1a6fb8", "#0f4c81", "#999999"
INK, INK2, SURFACE = "#1a1a1a", "#555555", "#ffffff"


def ppl(exp: str, name: str) -> float:
    return json.loads((RAW / exp / f"{name}.json").read_text())["ppl"]


fp16 = ppl("EXP-001", "fp16_g128")
rtn = ppl("EXP-001", "rtn_g128")
gptq = ppl("EXP-001", "gptq_g128")
awq = ppl("EXP-002", "awq_g128")
stack = ppl("EXP-002", "awq_gptq_g128")
naive = ppl("EXP-003", "naive")
sq75 = ppl("EXP-003", "smooth_a0.75")

rec_w4 = lambda p: 100.0 * (rtn - p) / (rtn - fp16)
rec_w8 = lambda p: 100.0 * (naive - p) / (naive - fp16)

# 自上而下:W4A16 三臂(按恢复率降序)+ 分隔 + W8A8 一臂
arms = [
    ("AWQ+GPTQ 叠加",              rec_w4(stack), ACCENT,  f"PPL {stack:.4f}"),
    ("GPTQ(二阶误差补偿)",        rec_w4(gptq),  MAIN,    f"PPL {gptq:.4f}"),
    ("AWQ(一阶 best-scale)",      rec_w4(awq),   NEUTRAL, f"PPL {awq:.4f}"),
    ("SmoothQuant α=0.75(W8A8)",  rec_w8(sq75),  MAIN,    f"PPL {sq75:.4f}"),
]

fig, ax = plt.subplots(figsize=(8.6, 4.4), dpi=220)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

ys = [3, 2, 1, -0.35]  # W8A8 臂与 W4A16 组隔开
for y, (label, v, c, sub) in zip(ys, arms):
    ax.barh(y, v, height=0.55, color=c, zorder=3)
    ax.text(v + 1.2, y, f"{v:.1f}%", va="center", ha="left",
            color=INK, fontsize=11, fontweight="bold")
    ax.text(-1.5, y - 0.36, sub, va="center", ha="right", color=INK2, fontsize=7.5)

ax.set_yticks(ys)
ax.set_yticklabels([a[0] for a in arms], fontsize=10, color=INK)
ax.axhline(0.35, color="#cccccc", lw=0.8, ls=(0, (4, 3)), zorder=1)
ax.text(99, 3.42, "W4A16:收回 RTN 缺口(RTN 14.1154 → fp16 11.9152)",
        ha="right", color=INK2, fontsize=8)
ax.text(99, 0.15, "W8A8:收回 naive 缺口(naive 12.1227 → fp16 11.9152)",
        ha="right", color=INK2, fontsize=8)

ax.set_xlim(0, 100)
ax.set_ylim(-1.05, 3.75)
ax.set_xlabel("恢复率(% · 各赛道内以 fp16=100%、基线=0% 归一)",
              color=INK, fontsize=9.5)
ax.set_title("二阶补偿是恢复主力:GPTQ 收回 RTN 缺口 61.6%,一阶 AWQ 31.9%,叠加仅再 +1.0pp",
             color=INK, fontsize=11.5, pad=12)
ax.grid(axis="x", color="#e6e5e2", lw=0.7, zorder=0)
ax.tick_params(colors=INK2, labelsize=8.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color("#d8d7d3")

fig.text(0.01, 0.02,
         f"源数据 data/raw/EXP-001~003/ 各臂 JSON(2026-08-23,EXP-001/002/003)· 生成 {GEN_DATE} · scripts/plot_recovery.py\n"
         "PPL 为确定性 greedy scoring,单轮可复算(EXP-001 §6),无 3 轮 std 可画;"
         "EXP-003 记录将 48.5% 取整表述为 48%(EXP-003 §6)。",
         color=INK2, fontsize=6.5)
fig.tight_layout(rect=(0, 0.07, 1, 1))
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, facecolor=SURFACE)
print(f"wrote {OUT}")
print({a[0]: round(a[1], 2) for a in arms})
