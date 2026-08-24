# LLM_Quantization — 三种量化方法从零实现,一张表对照

> ⚠ 本仓当前**无 git 远端**:待用户创建 github.com/lyell0710/LLM_Quantization
> 后推送(不代建远端);在此之前所有里程碑仅本地 commit,铁律 5 的 push 项挂起。

量化工作的**唯一家**(2026-08-23 起):①三种方法从零实现同协议对照
(GPTQ/AWQ/SmoothQuant,本仓 src/ + EXP-001~003);②既有 AutoAWQ 侧实践
LLMQT_Example **连历史整体并入** `llmqt_example/`(框架源码 + 289QS 课程
数据:论文/slides/OPT/Qwen2-1.5B 的 PPL 与 latency JSON)。学习方法论:[docs/HOW_TO_LEARN_A_QUANT_METHOD.md](docs/HOW_TO_LEARN_A_QUANT_METHOD.md);
明日讲解提纲:[docs/talk/quant_walkthrough.md](docs/talk/quant_walkthrough.md);
工程准则:/root/standards/CORE.md。

## 主对照表(Qwen2.5-0.5B · wikitext-2 PPL,窗2048/步1536,同 298302 计分 token)

**W4A16 赛道**(INT4 per-group g=128 非对称,同一量化网格):

| 臂 | PPL | Δ vs fp16 | 收回 RTN 损失 | 出处 |
|---|---|---|---|---|
| fp16 基线 | 11.9152 | — | — | EXP-001 |
| RTN(对照) | 14.1154 | +2.2002 | — | EXP-001 |
| AWQ(α 网格 best-scale) | 13.4127 | +1.4975 | 31.9% | EXP-002 |
| **GPTQ**(二阶补偿) | **12.7600** | +0.8448 | **61.6%** | EXP-001 |
| **AWQ+GPTQ**(叠加) | **12.7376** | +0.8224 | **62.6%** | EXP-002 |

**W8A8 赛道**(W: per-输出行对称 INT8;A: per-token 对称 INT8):

| 臂 | PPL | Δ vs fp16 | 出处 |
|---|---|---|---|
| naive W8A8 | 12.1227 | +0.2075 | EXP-003 |
| smooth α=0.25 | 12.2332 | +0.3180(迁移不足反伤权重) | EXP-003 |
| smooth α=0.50 | 12.0394 | +0.1242 | EXP-003 |
| **smooth α=0.75** | **12.0221** | **+0.1069** | EXP-003 |

一句话读表:**GPTQ 的二阶补偿收回 RTN 损失的 61.6%,AWQ 一阶启发式收回
31.9%,二者叠加 62.6%(正交但小模型上高度重叠);0.5B 激活 outlier 温和,
SmoothQuant 收益存在但幅度反证其价值随模型规模增长。**

## 既有实践数据(llmqt_example/289QS,AutoAWQ 框架,**异协议**,不与主表混排)

AWQ-INT4 vs fp16(HF eval 协议,绝对值不与本仓主表比较):
Qwen2-1.5B PPL 8.933 vs 8.474;OPT-125m 25.18 vs 23.69
→ `llmqt_example/289QS/results/*.json`;latency/吞吐与图见同目录 figures/。

## 怎么跑

```bash
V=/root/venvs/v0.25.1/bin/python
$V scripts/run_w4a16.py --mode {fp16|rtn|gptq|awq|awq_gptq} --group-size 128 --out <json>
$V scripts/run_w8a8.py  --mode {fp16|naive|smooth} [--alpha 0.5] --out <json>
# 批量: scripts/run_all.sh(EXP-001 三臂) / scripts/run_all2.sh(EXP-002/003)
```

## EXP 索引

| 编号 | slug | 日期 | 状态 | 关键数字(指针) |
|---|---|---|---|---|
| [EXP-001](records/EXP-001_gptq_from_scratch.md) | gptq_from_scratch | 2026-08-23 | 完成 | GPTQ 收回 RTN 损失 61.6%(12.76/14.12/11.92 → data/raw/EXP-001/) |
| [EXP-002](records/EXP-002_awq_and_stack.md) | awq_and_stack | 2026-08-23 | 完成 | AWQ 31.9%;AWQ+GPTQ 62.6%;per-layer α 中位 0.30(→ data/raw/EXP-002/) |
| [EXP-003](records/EXP-003_smoothquant_w8a8.md) | smoothquant_w8a8 | 2026-08-23 | 完成 | smooth α=.75 收回 naive W8A8 缺口 48%(12.02/12.12 → data/raw/EXP-003/) |

## 措辞红线表

| 红线 | 当前 | 说明 |
|---|---|---|
| PPL 绝对值 | 限定 | 协议自定义,只作臂间相对比较,不与文献绝对值对比 |
| "61.6%/31.9%/48% 恢复" | ✅ 可用 | 控制变量对照(各自唯一差异=核心机制开关),EXP-001/002/003 §5 |
| "正交可叠加" | 限定 | 方向成立,0.5B 上增益 +1.0pp——**不得说"显著提升"**(EXP-002 §6) |
| SmoothQuant 收益 | 限定 | 须带"0.5B outlier 温和"语境;不得外推大模型幅度(EXP-003 §6) |
| AWQ 数字 | 限定 | 本仓为 per-linear 简化 + 无 clip(EXP-002 §2);与 AutoAWQ 完整实现不同口径 |
| per-layer loss 诊断值 | 🚫 不进表 | 量纲未标定(EXP-001 §7) |
| "从零实现" | ✅ 可用 | 三方法算法主循环全部自写;数值锚点对齐参考实现(theory §5 各文) |

## 结构

```
src/        gptq.py(二阶补偿) awq.py(best-scale 搜索) smoothquant.py(迁移+W8A8)
            quant_linear.py(INT4 打包 + pack↔fake 断言)
scripts/    run_w4a16.py / run_w8a8.py / run_all*.sh
records/    EXP-001~003(八节)     data/raw/EXP-00{1,2,3}/(provenance 首字段;
            EXP-001 三臂 sha=worktree,勘注见 EXP-001 §7——代码即 274acb2 所提交内容)
docs/theory 01_gptq / 02_awq / 03_smoothquant(五节,实证全回填)
docs/talk/  quant_walkthrough.md(讲解提纲)
figures/    fig1_awq_alpha_dist.png(scripts/plot_alpha_dist.py 从 EXP-002 raw 重算)
ENV.md      异地复现指南(llmqt_example/ENV.md 为其子范围)
llmqt_example/  并入的 LLMQT_Example 全史(LLMQT 框架 / llmqt_eval / 289QS)
```
