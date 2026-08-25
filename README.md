# LLM_Quantization — GPTQ / AWQ / SmoothQuant 从零实现,同协议一张表对照

> ⚠ 本仓当前**无 git 远端**:待用户创建 github.com/lyell0710/LLM_Quantization
> 后推送(不代建远端);在此之前所有里程碑仅本地 commit,铁律 5 的 push 项挂起。

本仓证明:**三种主流训练后量化方法的核心算法我都亲手实现过**,并在同一模型、
同一量化网格、同一 PPL 协议下用控制变量对照量化了各自机制的真实贡献——
W4A16 与 W8A8 双赛道,fake quant 与 real quant(INT4 打包)双链路,
每个数字可从 `data/raw/` 一键复算。

## 🎯 结果一览

| 结果 | 数字 | 指针 |
|---|---|---|
| **GPTQ** 从零实现:二阶误差补偿收回 RTN 质量损失 | **61.6%**(PPL 14.1154→**12.7600**,fp16 11.9152) | [EXP-001](records/EXP-001_gptq_from_scratch.md) · `data/raw/EXP-001/` |
| **AWQ** 从零实现(per-linear 简化、无 clip):一阶 best-scale 收回 | **31.9%**(13.4127) | [EXP-002](records/EXP-002_awq_and_stack.md) · `data/raw/EXP-002/` |
| **AWQ+GPTQ 叠加**:正交方向成立,0.5B 上高度重叠 | **62.6%**(12.7376,vs 单独 GPTQ 仅 +1.0pp) | [EXP-002](records/EXP-002_awq_and_stack.md) |
| **SmoothQuant W8A8**(α=0.75)收回 naive 缺口(0.5B 激活 outlier 温和的语境下) | **48%**(12.0221 vs 12.1227) | [EXP-003](records/EXP-003_smoothquant_w8a8.md) · `data/raw/EXP-003/` |
| **real quant 闭环**:INT4 两枚一字节打包,pack↔fake 逐元素断言 | 168/168 层通过,最大误差 ≤7.3e-4 | [EXP-001](records/EXP-001_gptq_from_scratch.md) §5 |

> 协议:Qwen2.5-0.5B · wikitext-2 PPL(窗 2048/步 1536,三臂同 298302 计分
> token)。PPL 为自定义协议,**只作协议内臂间相对比较**,不与文献绝对值对比
> (见下方红线表)。

<details>
<summary><b>完整主对照表</b>(逐臂 PPL,点开)</summary>

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

</details>

## 📊 图表

![恢复率对照](figures/fig2_recovery_rates.png)

> GPTQ 的二阶补偿是恢复主力(收回 RTN 缺口 61.6%),AWQ 一阶启发式 31.9%,
> 叠加仅再 +1.0pp——源:`data/raw/EXP-001` ~ EXP-003 各臂 JSON(EXP-001/002/003
> §5),`scripts/plot_recovery.py` 复算(EXP-003 记录取整表述为 48%)。

![AWQ α 分布](figures/fig1_awq_alpha_dist.png)

> AWQ 保护强度按层自适应:per-layer best-α 中位 0.30,主体 0.15–0.45,
> 两强 outlier 层顶到 0.95——源:`data/raw/EXP-002/awq_g128.json`(EXP-002 §5),
> `scripts/plot_alpha_dist.py`。

## 🔬 代码导览

GPTQ 的灵魂在两处:**Cholesky 求 H⁻¹ 的上三角因子**,以及**逐列量化 +
两级误差补偿**(块内立即、块间延迟)。节选自 [src/gptq.py](src/gptq.py)
`GPTQ.quantize()`(`# ←` 注释为导览所加):

```python
# Hinv 的上三角 Cholesky 因子:对角元即 [H^-1]_jj^(1/2) 的行主元
Hinv = torch.linalg.cholesky(H)
Hinv = torch.cholesky_inverse(Hinv)
Hinv = torch.linalg.cholesky(Hinv, upper=True)
...
for i in range(i2 - i1):                     # ← 块内逐列量化
    w = W1[:, i]
    dq, q = self.quantizer.quantize(w, scales[:, g], zeros[:, g])
    Q1[:, i] = dq
    if use_hessian:                          # ← False 即退化为 RTN 对照臂
        err = (w - dq) / Hinv1[i, i]         # ← 量化误差按 [H^-1] 对角归一
        W1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
        Err1[:, i] = err                     # ← 立即补偿到块内未量化列
...
Q[:, i1:i2] = Q1
if use_hessian and i2 < self.columns:
    W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]     # ← 块间 lazy update:批量补偿后续列
```

`use_hessian` 这个开关就是 EXP-001 的实验设计本身:RTN 臂与 GPTQ 臂共用同一
`GroupQuantizer` 与量化网格,唯一差异 = 补偿开关,61.6% 的恢复量因此可
**完全归因于二阶补偿机制**。INT4 打包与 pack↔fake 断言见
[src/quant_linear.py](src/quant_linear.py);AWQ 的 α 网格搜索见
[src/awq.py](src/awq.py),SmoothQuant 迁移见 [src/smoothquant.py](src/smoothquant.py)。

原理笔记(五节体:结论/机制/本仓实证/面试 Q&A/延伸):
[01_gptq](docs/theory/01_gptq.md) ·
[02_awq](docs/theory/02_awq.md) ·
[03_smoothquant](docs/theory/03_smoothquant.md)

## 🚀 复现

```bash
V=/root/venvs/v0.25.1/bin/python
$V scripts/run_w4a16.py --mode {fp16|rtn|gptq|awq|awq_gptq} --group-size 128 --out <json>
$V scripts/run_w8a8.py  --mode {fp16|naive|smooth} [--alpha 0.5] --out <json>
# 批量: scripts/run_all.sh(EXP-001 三臂) / scripts/run_all2.sh(EXP-002/003)
# 图(从 raw 重算,禁手改):
/root/venvs/kernel-opt/bin/python scripts/plot_recovery.py
/root/venvs/kernel-opt/bin/python scripts/plot_alpha_dist.py
```

## 🗂 仓库结构

```
src/        gptq.py(二阶补偿) awq.py(best-scale 搜索) smoothquant.py(迁移+W8A8)
            quant_linear.py(INT4 打包 + pack↔fake 断言)
scripts/    run_w4a16.py / run_w8a8.py / run_all*.sh / plot_*.py
records/    EXP-001~003(八节)     data/raw/EXP-00{1,2,3}/(provenance 首字段;
            EXP-001 三臂 sha=worktree,勘注见 EXP-001 §7——代码即 274acb2 所提交内容)
docs/theory 01_gptq / 02_awq / 03_smoothquant(五节,实证全回填)
docs/talk/  quant_walkthrough.md(讲解提纲)
figures/    fig1_awq_alpha_dist.png / fig2_recovery_rates.png(全部脚本生成)
ENV.md      异地复现指南(llmqt_example/ENV.md 为其子范围)
llmqt_example/  并入的 LLMQT_Example 全史(LLMQT 框架 / llmqt_eval / 289QS)
```

学习方法论:[docs/HOW_TO_LEARN_A_QUANT_METHOD.md](docs/HOW_TO_LEARN_A_QUANT_METHOD.md);
讲解提纲:[docs/talk/quant_walkthrough.md](docs/talk/quant_walkthrough.md);
工程准则:/root/standards/CORE.md。

## 🧾 实验台账

| 编号 | slug | 日期 | 状态 | 关键数字(指针) |
|---|---|---|---|---|
| [EXP-001](records/EXP-001_gptq_from_scratch.md) | gptq_from_scratch | 2026-08-23 | 完成 | GPTQ 收回 RTN 损失 61.6%(12.76/14.12/11.92 → data/raw/EXP-001/) |
| [EXP-002](records/EXP-002_awq_and_stack.md) | awq_and_stack | 2026-08-23 | 完成 | AWQ 31.9%;AWQ+GPTQ 62.6%;per-layer α 中位 0.30(→ data/raw/EXP-002/) |
| [EXP-003](records/EXP-003_smoothquant_w8a8.md) | smoothquant_w8a8 | 2026-08-23 | 完成 | smooth α=.75 收回 naive W8A8 缺口 48%(12.02/12.12 → data/raw/EXP-003/) |

### 既有实践数据(llmqt_example/289QS,AutoAWQ 框架,**异协议**,不与主表混排)

AWQ-INT4 vs fp16(HF eval 协议,绝对值不与本仓主表比较):
Qwen2-1.5B PPL 8.933 vs 8.474;OPT-125m 25.18 vs 23.69
→ `llmqt_example/289QS/results/*.json`;latency/吞吐与图见同目录 figures/。

## 🧭 措辞红线表与方法论

| 红线 | 当前 | 说明 |
|---|---|---|
| PPL 绝对值 | 限定 | 协议自定义,只作臂间相对比较,不与文献绝对值对比 |
| "61.6%/31.9%/48% 恢复" | ✅ 可用 | 控制变量对照(各自唯一差异=核心机制开关),EXP-001/002/003 §5 |
| "正交可叠加" | 限定 | 方向成立,0.5B 上增益 +1.0pp——**不得说"显著提升"**(EXP-002 §6) |
| SmoothQuant 收益 | 限定 | 须带"0.5B outlier 温和"语境;不得外推大模型幅度(EXP-003 §6) |
| AWQ 数字 | 限定 | 本仓为 per-linear 简化 + 无 clip(EXP-002 §2);与 AutoAWQ 完整实现不同口径 |
| per-layer loss 诊断值 | 🚫 不进表 | 量纲未标定(EXP-001 §7) |
| "从零实现" | ✅ 可用 | 三方法算法主循环全部自写;数值锚点对齐参考实现(theory §5 各文) |

**诚实度文化(本仓的差异化卖点,如实展示)**:① 所有 raw 结果首行/首字段带
provenance(env/sha/cmd/date/gpu/driver),raw 不可变——发现口径问题不改历史,
以勘注留痕(EXP-001 §7 两则勘注 + dirty 拒跑工装杜绝复发);② 每个主张配
对照/反例臂:RTN 与 naive 是"唯一差异=机制开关"的控制变量对照,α=0.25 是
主动保留的反例(迁移不足反伤权重);③ 进 README 的关键数字要求 ≥3 轮取
mean/std,本仓 PPL 为确定性 greedy scoring,按 EXP-001 §6 以"单轮可复算"
显式注明豁免——措辞红线表约束对外每一句量化主张,证据不足即降级。

## 🔗 相关仓

- [vllmExperience](https://github.com/lyell0710/vllmExperience) —— vLLM serving
  侧证据仓;本仓 GPTQ 离线算法与其 EXP-016(GPTQ-Int4+Marlin 在线部署)构成
  "离线算法 → 在线部署"完整链(EXP-001 §8)。
- [Kernel_Optimazation](https://github.com/lyell0710/Kernel_Optimazation) ——
  CUDA kernel 优化证据仓;本仓的控制变量/反例臂方法论与其同源。
- [llm-engine](https://github.com/lyell0710/llm-engine) —— 推理引擎实践仓。
- LLMQT_Example —— 既有 AutoAWQ 侧实践,已连完整 git 历史并入本仓
  `llmqt_example/`(异协议数据单列,见上)。
