# 01 · 三种训练后量化方法合讲:均匀量化基础 → GPTQ → AWQ → SmoothQuant

> 深度讲义。配套代码 `src/`，配套数据 `data/raw/EXP-00{1,2,3}/`，全部数字带 EXP 锚，可从 raw 复算。阅读顺序即讲解顺序：先把"量化"这件事本身讲透， 再逐个拆三种方法的机制、代码与实验证据。凡属论文/官方文档的论断一律给出处（标题 + arXiv/DOI 编号 + 章节或公式编号，文档给 URL 路径 + 小节名）； 凡属本讲义自己补出的推导或折算，行内标注"本讲义推导"；检索不到、无法逐字核对的说法标注"未核实"，不进任何主张。

## 目录

- [1. 这一篇回答什么问题](#1-这一篇回答什么问题)
  - [1.1 本篇要建立的七条能力](#11-本篇要建立的七条能力)
  - [1.2 符号与口径约定](#12-符号与口径约定)
  - [1.3 本篇引用的一级文献(作者全名、章节与"读它解决什么疑问"见 §8.4)](#13-本篇引用的一级文献作者全名章节与读它解决什么疑问见-84)
- [2. 直觉与第一性原理](#2-直觉与第一性原理)
  - [2.1 三条贯穿全篇的公理](#21-三条贯穿全篇的公理)
  - [2.2 三方法的坐标图:各自动了哪个自由度](#22-三方法的坐标图各自动了哪个自由度)
  - [2.3 三层误差必须分开](#23-三层误差必须分开)
- [3. 完整推导与机制](#3-完整推导与机制)
  - [3.1 均匀量化:scale、zero-point、粒度谱系](#31-均匀量化scalezero-point粒度谱系)
  - [3.2 RTN 为什么不够:一笔误差账](#32-rtn-为什么不够一笔误差账)
  - [3.3 GPTQ:从 OBD/OBS 到逐列补偿公式](#33-gptq从-obdobs-到逐列补偿公式)
  - [3.4 AWQ:显著通道判据与等价缩放证明](#34-awq显著通道判据与等价缩放证明)
  - [3.5 SmoothQuant:迁移强度 α 与逐通道误差分析](#35-smoothquant迁移强度-α-与逐通道误差分析)
  - [3.6 INT4 打包与反量化的数值细节](#36-int4-打包与反量化的数值细节)
  - [3.7 PPL 作为量化评价指标的局限](#37-ppl-作为量化评价指标的局限)
- [4. 代码逐段走读](#4-代码逐段走读)
- [5. 实验数据怎么读](#5-实验数据怎么读)
  - [5.1 主对照表](#51-主对照表)
  - [5.2 恢复率怎么算(列算式)](#52-恢复率怎么算列算式)
  - [5.3 这个实验设计防了哪些坑](#53-这个实验设计防了哪些坑)
  - [5.4 图怎么读](#54-图怎么读)
  - [5.5 α 分布的完整直方图(从 raw 复算)](#55-α-分布的完整直方图从-raw-复算)
  - [5.6 时间账:从 raw per-layer 复算的成本结构](#56-时间账从-raw-per-layer-复算的成本结构)
  - [5.7 存储账与两种压缩比](#57-存储账与两种压缩比)
  - [5.8 数字背后的机理账](#58-数字背后的机理账)
  - [5.9 哪些数字能外推,哪些不能](#59-哪些数字能外推哪些不能)
- [6. 误区与边界](#6-误区与边界)
- [7. 连环追问](#7-连环追问)
- [8. 工业对照与延伸](#8-工业对照与延伸)
  - [8.1 论文/文档怎么说 vs 本项目实测:逐条对照](#81-论文文档怎么说-vs-本项目实测逐条对照)
  - [8.2 与生产实现的差距各在哪一层](#82-与生产实现的差距各在哪一层)
  - [8.3 硬件语义:这些约束如何决定代码写法](#83-硬件语义这些约束如何决定代码写法)
  - [8.4 延伸阅读(每条一句话说明它能解决什么疑问)](#84-延伸阅读每条一句话说明它能解决什么疑问)

## 1. 这一篇回答什么问题

GPTQ、AWQ、SmoothQuant 这三种主流训练后量化（PTQ）方法，各自到底在优化什么、差在哪一阶、贡献几何？本篇从均匀量化的第一性原理讲起，把三者的核心公式逐步推完，再对照本仓从零实现的真实代码与同协议对照实验（EXP-001/002/003）逐段走读。读完你应当能：①白板手推 GPTQ 的逐列补偿公式（含 Cholesky 技巧为什么成立）； ②解释 61.6%/31.9%/48% 三个恢复率各自的控制变量设计与含义；③答上 "为什么叠加只 +1.0pp""容差为什么要幅值感知"这类三层追问。

### 1.1 本篇要建立的七条能力

1. **量化基本功**：给定位宽 b、粒度 g 与一段权重，能写出 scale/zero-point 的求法、码域边界、存储开销（bit/权重），并说清每一步的边界条件。
2. **二阶推导能力**：能从 OBD 的对角近似出发，经 OBS 的拉格朗日闭式解、 OBQ 的"删→量化"替换，一路推到 GPTQ 的逐列补偿式与 Cholesky 重排。
3. **等价变换能力**：能证明 AWQ/SmoothQuant 的缩放是数学恒等，并说清恒等变换把什么挪走了、代价挂在谁头上、上界与下界各是多少。
4. **粒度—硬件对应能力**：能从 INT8 GEMM 的整数累加语义反推出"激活为什么只能 per-token 不能 per-channel"，并解释这条约束如何决定每个 `amax(dim=...)` 的轴。
5. **数值账能力**：能把机制误差、存储降精、未标定诊断量三层分开，知道每层该用什么判据、容差该按什么量纲定。
6. **口径能力**：任何数字出口都带模型/协议/简化定语，知道 31.9% 只属于 "per-linear 简化 + 无 clip"口径，48% 只属于"0.5B outlier 温和"语境。
7. **计时与归因能力**：能从原始 per-layer 数据反推成本结构，并识别异步执行下"逐算子墙钟"这类不可信的测量。

### 1.2 符号与口径约定

模型侧取值全部取自本机 HF cache 的 `config.json`(Qwen/Qwen2.5-0.5B)：

| 符号 | 含义 | Qwen2.5-0.5B 取值 |
|---|---|---|
| L | 层数 `num_hidden_layers` | 24 |
| d_model | `hidden_size` | 896 |
| d_ff | `intermediate_size` | 4864 |
| H / KVH | Q 头数 / KV 头数 | 14 / 2 |
| D | head_dim = d_model / H | 64 |
| V | `vocab_size` | 151936 |
| tie | `tie_word_embeddings` | true（lm_head 复用 embedding 表） |
| b | 权重位宽 | 4（W4A16 赛道）/ 8（W8A8 赛道） |
| g | group_size | 128 |
| N_calib | 校准 token 数 | 128 段 × 2048 = 262144 |
| N_score | PPL 计分 token 数 | 298302（各臂逐位相同） |

量化侧符号：$s$=scale（格距），$z$=zero-point，$q$=整数码，$\hat w$=反量化值， $\Delta$=网格步长（等于 $s$，行文中在讨论"组的步长"时写 $\Delta$）， $H = 2XX^\top$=层 Hessian，$U$=$H^{-1}$ 的上三角 Cholesky 因子， $\alpha$=AWQ/SmoothQuant 各自的指数超参（两者含义不同，见 §3.4 与 §3.5）。

**七类 Linear 的形状表**（量化范围 = 每层 7 个 Linear，lm_head 不量化； 形状由上表推出，本讲义推导）：

| Linear | (out, in) | 列数 = in | 组数 = in/128 | H 尺寸（in², fp32） |
|---|---|---|---|---|
| self_attn.q_proj | (896, 896) | 896 | 7 | 3.06 MiB |
| self_attn.k_proj | (128, 896) | 896 | 7 | 3.06 MiB |
| self_attn.v_proj | (128, 896) | 896 | 7 | 3.06 MiB |
| self_attn.o_proj | (896, 896) | 896 | 7 | 3.06 MiB |
| mlp.gate_proj | (4864, 896) | 896 | 7 | 3.06 MiB |
| mlp.up_proj | (4864, 896) | 896 | 7 | 3.06 MiB |
| mlp.down_proj | (896, 4864) | 4864 | 38 | 90.25 MiB |

三个立刻能读出来的事实：①k/v 的 out 是 128 = KVH×D = 2×64，GQA 的痕迹； ②所有 in 都能被 128 整除，没有"残缺组"，也没有奇数列（nibble 打包的前提， `src/quant_linear.py:33` 的断言因此永不触发）；③down_proj 的 H 比其余六个大一个量级（90.25 MiB vs 3.06 MiB），单层七个并存 108.6 MiB——这是 `GPTQ.free()` 必须逐 Linear 释放的原因（`src/gptq.py:193-197`），24 层不释放则峰值到 2.5 GiB 量级（本讲义推导）。

### 1.3 本篇引用的一级文献(作者全名、章节与"读它解决什么疑问"见 §8.4)

剪枝二阶理论 LeCun et al. "Optimal Brain Damage"(NIPS 1989)与 Hassibi & Stork "Optimal Brain Surgeon"(NIPS 1992);OBQ arXiv:2208.11580; GPTQ arXiv:2210.17323;AWQ arXiv:2306.00978;SmoothQuant arXiv:2211.10438； 激活 outlier 的规模相变 LLM.int8() arXiv:2208.07339；量化噪声模型 Bennett(BSTJ 27(3):446-472, 1948)；数值线性代数 Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM 2002 第 10 章； 评价指标局限 arXiv:2407.09141；生产内核 MARLIN arXiv:2408.11743； 官方文档 NVIDIA cuBLAS / CUDA C++ Programming Guide / PTX ISA / Ada 白皮书。

## 2. 直觉与第一性原理

**没有量化，世界会怎样。** LLM 推理的 decode 阶段每生成一个 token，都要把全部权重从显存读一遍——权重带宽是第一瓶颈（实测证据见 vllm/experiments#EXP-014《D1 MoE decode 分解》，转引于 `docs/talk/quant_walkthrough.md` §0）。 fp16 权重每参数 16 bit；若能压到 4 bit，同一张卡上 decode 的权重读取量即降为 1/4，显存占用同理。这就是 W4A16 赛道的物理动机。W8A8 赛道另有所图： 激活也换成 INT8，连算力路径（INT8 TensorCore）一起换，prefill/大 batch 受益。两个赛道压的东西不同、快的 regime 不同——这是全篇的骨架。

**类比：有限刻度的尺子。** 量化就是把连续实数逼到一把只有 16 个刻度（INT4）的尺子上，每个数只能记"离它最近的刻度是第几格"。scale 是格距， zero-point 是"第 0 格对准哪里"。类比在两处失效：①尺子量身高，误差就是误差； 量化误差却要再过一次矩阵乘，被激活放大——同样大小的误差，落在不同输入通道上伤害差几个数量级，这是三种方法全部故事的起点；②量身高不能换坐标系， 量化却可以先给数乘个 s 再量、事后除回（数学恒等）——AWQ 与 SmoothQuant 的全部操作空间都在这个"类比里不存在"的自由度上。

**再给 GPTQ 一个类比：逐件装箱。** 往一个不规则的箱子里逐件放行李，每放一件都会留下缝隙（量化误差）；聪明的做法是每放完一件，立刻调整还没放的行李的摆法，把缝隙吃掉。失效点：行李之间没有"相关性矩阵"，而权重列之间有——校准集激活的二阶矩 XXᵀ 精确刻画了"调整哪些列、按什么比例调整"最划算， 这正是 GPTQ 用 H⁻¹ 做的事。

### 2.1 三条贯穿全篇的公理

- **公理 A（度量必须取在输出空间）**：优化目标只有与推理在乎的量一致才有意义。RTN 最小化 $\|W-\hat W\|^2$，GPTQ 最小化 $\|WX-QX\|^2$，AWQ 的网格搜索直接用输出 MSE 打分，SmoothQuant 的 α 也只能用端到端质量来选。三种方法的分歧全在"怎么逼近输出误差"，没有一种回到权重空间。
- **公理 B（等价变换是免费的自由度，但只是搬家）**： $y = (X\,\mathrm{diag}(s)^{-1})(\mathrm{diag}(s)W^\top)$ 对任意正 $s$ 严格成立（逐元素验证即可），所以量化前后可以任意换坐标系。但恒等变换不消灭难度：它只把难度从一侧搬到另一侧，或在一组通道之间重新分配。**任何声称 "缩放让误差凭空变小"的说法都缺了后半句：代价挂在谁头上。**
- **公理 C（每个误差都要有账）**：一个数值实现里的每一处残差都必须能被归因到唯一来源。本仓 pack↔fake 断言之所以能立，正是因为反量化路径上只剩一处降精（scale 的 half 存储），残差因此可以被精确定界（§3.6.2）。

### 2.2 三方法的坐标图:各自动了哪个自由度

一次 Linear 的量化，可动的自由度只有四个：**取整规则**、**网格参数（scale/zero）**、**坐标系（逐通道缩放）**、**未量化权重的取值**。三种方法各占其一或其二：

| 方法 | 取整规则 | 网格参数 | 坐标系 | 未量化权重 | 用到的统计阶数 |
|---|---|---|---|---|---|
| RTN | round-to-nearest | 组内 min-max | 不动 | 不动 | 零阶（不看数据） |
| GPTQ | round-to-nearest | 组内 min-max（取自补偿后权重） | 不动 | **改**（逐列补偿） | 二阶（$XX^\top$） |
| AWQ | round-to-nearest | 组内 min-max（在 W·s 域重算） | **改**(per-input-channel s) | 不动 | 一阶（absmean） |
| SmoothQuant | round-to-nearest | 权重 per-行 / 激活 per-token | **改**(per-input-channel s) | 不动 | 一阶（actmax） |

读这张表能立刻回答两个常见问题：①**为什么 AWQ 和 GPTQ 可以叠加**——一个动坐标系、一个动未量化权重，列不冲突；②**为什么 SmoothQuant 不属于 W4A16 赛道**——它动的坐标系是为了救*激活*的量化，而 W4A16 的激活根本不量化， 迁移过去等于白白撑大权重侧的 scale。

### 2.3 三层误差必须分开

| 层 | 现象 | 判据 | 本仓实例 |
|---|---|---|---|
| 机制误差 | 量化网格本身放不下这些数 | 端到端质量评测（PPL/下游任务） | RTN +2.2002 PPL(EXP-001 §5) |
| 存储误差 | 码存下来再取回时的降精 | 逐元素断言 + 幅值感知容差 | pack↔fake ≤7.3e-4（EXP-001 口径） |
| 诊断量 | 量纲未标定的中间量 | 只能做相对比较，不进表格 | 逐列 loss 1e-20~1e-15(EXP-001 §7) |

把三层混成一个"误差阈值"是最常见的错误。本仓的做法是三层各有独立判据： PPL 管机制，逐元素断言管存储，诊断量只作回归监控——**三条证据互不依赖， 任何一条塌了另外两条还能站住**，这是"控制变量"之外的第二重设计。

## 3. 完整推导与机制

### 3.1 均匀量化:scale、zero-point、粒度谱系

#### 3.1.1 映射与反映射:每一步凭什么合法

b bit 非对称均匀量化把实数 w 映为整数码 q 再映回：

$q = \mathrm{clamp}(\mathrm{round}(w/s) + z,\ 0,\ 2^b{-}1)$,  $\hat w = (q - z)\cdot s$

- 为什么 $s = (w_{max}-w_{min})/(2^b-1)$：$2^b$ 个刻度铺满区间，格距 = 区间长度 /（格数−1），两端各占一格。这是"端点对齐"约定，$w_{min}$ 落码 0、 $w_{max}$ 落码 $2^b-1$，两端精确可表示。另一种约定（除以 $2^b$）让格心对齐、两端各留半格；两种都自洽，但**混用会让 scale 差 $2^b/(2^b-1)$ 倍**，INT4 下 6.7%，足以让臂间比较失真。本仓统一端点对齐（`src/gptq.py:45` 的 `maxq = 2**bits - 1`）。
- 为什么要 zero-point $z = \mathrm{round}(-w_{min}/s)$：权重组的 min/max 通常不关于 0 对称，对称量化会浪费半边码域；z 把网格平移到贴合实际区间。取 round 是因为 z 必须是整数——否则 dequant 的 $(q-z)s$ 里 z 要存浮点， 元数据翻倍且失去整数减法。round 引入的偏移最多半格，被 clamp 吸收。
- 为什么要强制 $0 \in [w_{min}, w_{max}]$（`src/gptq.py:51-52` 的 clamp）： 保证实数 0 精确可表示，且全正/全负组的 z 不越码域。以全正组为例，不 clamp 时 $z = \mathrm{round}(-w_{min}/s) < 0$，而码域下界是 0，dequant 系统性偏移； clamp 后 $w_{min}\le 0 \Rightarrow z \ge 0$ 必然成立。
- 两道防除零：全零组把 $w_{max}$ 顶成 1(`src/gptq.py:54`)，否则 $s=0$ 让 $w/s$ 直接 NaN；再加 `clamp(min=1e-8)`(`src/gptq.py:55`)兜住近似全零组的下溢。**数学上退化、数值上必须显式处理**，是这类代码的常态。
- 两条形状前提：本仓所有 in ∈ {896, 4864} 都被 128 整除（无残缺组；残缺组由 `src/awq.py:55` 的 `min((g+1)*group_size, cols)` 处理，但其极值统计样本更少、格距更抖，换模型时要重查），且都是偶数（两码一字节的前提， `src/quant_linear.py:33` 的断言）。

#### 3.1.2 舍入误差 $s^2/12$ 的成立条件与失效边界

round 到最近刻度，单点误差落在 $[-s/2, s/2]$。"均方误差 $s^2/12$"这个常用结论来自把量化误差建模为该区间上的**均匀分布**，方差 $\int_{-s/2}^{s/2} e^2 \cdot \frac{1}{s}\,\mathrm{d}e = s^2/12$。这个模型的原始出处是 Bennett 的量化噪声谱分析(Bell System Technical Journal 27(3)：446-472, 1948)；其严格成立需要"量化误差与信号统计独立且在格内均匀" 这类条件（Widrow 的量化定理给出充分条件：信号特征函数带限）。

**在本仓的设定下，这三个条件全部被破坏，所以 $s^2/12$ 只能当量级参照：**

1. **端点误差恒为 0**：min-max 网格下组内最大值与最小值恰落在网格端点， 它们的量化误差精确为 0(在 round(z) 的偏移之内)，不是均匀分布。
2. **s 与数据强相关**：$s$ 由组内极值决定，极值大的组格距也大——误差与数据不独立。
3. **权重不是白噪声**：同一输入通道上的权重跨输出行高度相关，这正是 GPTQ 能用 $H$ 做补偿的前提。

**一个用得上的量级折算（本讲义推导）。** 设组内 128 个权重近似 iid $\mathcal{N}(0,\sigma^2)$。128 个标准正态样本极大值的期望约 2.58σ（数量级参照，精确值随阶统计量表而异），故极差约 5.16σ， $s \approx 5.16\sigma/15 = 0.344\sigma$，舍入 RMS 误差 $\approx s/\sqrt{12} = 0.099\sigma$——即**约 10% 的相对权重扰动**。换成 per-tensor（整层 802816 个元素，q_proj 尺寸）：极大值期望约 4.8σ， $s\approx 9.6\sigma/15 = 0.64\sigma$，RMS $\approx 0.185\sigma$。两者之比 $0.099/0.185 \approx 0.54$：**g=128 相对 per-tensor 把 RTN 的权重 RMS 误差砍掉约一半**，代价是每 128 个权重多存 32 bit 元数据。这就是"g=128 是行业默认"背后可算的那笔账——不是玄学，是极值统计与元数据开销的交点。（前提：iid 高斯，真实权重不满足；此处只用于说明量级与单调方向。）

#### 3.1.3 粒度谱系与两种压缩比口径

**粒度谱系**（scale/zero 共享到什么范围）：per-tensor（整层 1 套）→ per-channel/per-输出行（out 套）→ per-group（本仓 g=128，每输出行每 128 个输入维一套）→ 极限是 per-element（等于不量化，信息全搬进 scale）。粒度越细， 网格越贴合局部分布、误差越小，代价是元数据存储与 kernel 复杂度。本仓 W4A16 的存储账：码 4 bit + fp16 scale/zero 各 16 bit 摊到 128 个权重， $4 + 32/128 = 4.25$ bit/权重（`src/quant_linear.py:19`）。激活侧另有一条谱系： per-tensor 静态 → per-token 动态（本仓 W8A8 用）→ per-channel——最后一档与 INT8 GEMM 的数学不相容，见 §3.5.2，这是 SmoothQuant 存在的理由。

**压缩比有两种口径，不能混说（本讲义推导，数据源为 config.json 与 `scripts/run_w4a16.py:46-48` 的量化范围）。** 先把参数数出来：每层 7 个 Linear 合计 $14{,}909{,}440$ 个参数（q/o 各 802,816，k/v 各 114,688，gate/up/down 各 4,358,144），24 层共 **357,826,560** 个被量化参数；tie 的 embedding 表 $151{,}936\times896 = 136{,}134{,}656$ 个参数**不量化**（lm_head 复用它， `scripts/run_w4a16.py:45` 按惯例排除），占全模型约 27.6%。

| 口径 | fp16 | INT4-g128 | 压缩比 |
|---|---|---|---|
| 只算被量化的 7 类 Linear | 682.5 MiB | 181.3 MiB | **3.76×** |
| 全模型（含未量化的 embedding） | 942.2 MiB | 440.9 MiB | **2.14×** |

$3.76 = 16/4.25$ 是位宽比，与模型无关；$2.14$ 只属于这个模型——**词表大、层少的小模型，未量化的 embedding 会把全模型压缩比腰斩**。面试里被问 "W4A16 能省多少显存"，只答 4× 是错的：必须先问"哪些张量在量化范围内"。（注：上表按纯权重字节折算，不含 KV cache、激活缓冲与 CUDA context。）

### 3.2 RTN 为什么不够:一笔误差账

#### 3.2.1 加权误差恒等式与它的三个前提

RTN(round-to-nearest)= 上面的公式直接逐元素套用，优化目标是权重误差 $\|W-\hat W\|^2$。但推理在乎的是**输出**。设某输出行权重误差向量为 $\delta = \hat w - w$，输入为 x，则输出误差为 $\delta^\top x$，在校准分布上：

$\mathbb{E}[(\delta^\top x)^2] = \mathbb{E}[\delta^\top x x^\top \delta] = \delta^\top\, \mathbb{E}[xx^\top]\, \delta$

逐步说明每一步凭什么：①标量转置等于自身，$(\delta^\top x)^2 = (\delta^\top x)(x^\top\delta)$； ②矩阵乘法结合律给出 $\delta^\top(xx^\top)\delta$，与是否随机无关； ③取期望时 $\delta$ 可提出——**前提是 $\delta$ 与 $x$ 独立**，量化离线完成、不依赖具体 token，该前提成立；但本仓 sequential 标定下深层的 $x$ 已带着浅层量化误差，存在跨层间接相关，这正是"逐层顺序标定"与"用 fp16 激活标定"结果不同的原因（`scripts/run_w4a16.py:189-192`）；④$\mathbb{E}[xx^\top]$ 是有限样本的经验期望，样本不足时近奇异，见 §3.3.9 的阻尼。

**含义**：权重误差被激活二阶矩**加权**。若某输入通道的 $\mathbb{E}[x_j^2]$ 比别的通道大 100 倍，该通道上同样大小的 $\delta_j$ 伤害大 100 倍。RTN 对此完全无感——它把每个通道的误差压到同样的 $s^2/12$， 等于在错误的度量下做了最优。

#### 3.2.2 各向异性有多强:文献与本仓的两条侧证

论文侧：LLM.int8()(arXiv:2208.07339)§4 报告 transformer 隐状态里出现 "magnitudes up to 20x larger than in other dimensions" 的特征维，并在 §4.2 给出规模相变的观察——"the emergence of large magnitude features across all layers of a transformer occurs suddenly between 6B and 6.7B parameters"， 相变后"all transformer layers and 75% of all sequence dimensions are affected"。SmoothQuant(arXiv:2211.10438)§3 给出的量级更大："The scale of outliers in activations is ~100× larger than most of the activation values."

本仓侧：我们没有直接测 outlier 幅值倍数（**本仓未测**），但有两条间接证据。 ①AWQ 的 per-layer best-α 分布（EXP-002《AWQ 从零实现 + AWQ×GPTQ 叠加》§5，可从 `data/raw/EXP-002/awq_g128.json` 复算）中位 0.30、主体 0.15–0.45，而有两层顶到网格上限 0.95——**同一模型内部各层的各向异性强度相差极大**； ②W8A8 naive 臂只掉 +0.2075 PPL（EXP-003《SmoothQuant 从零实现》§5），说明 0.5B 上激活 outlier 远没到让 per-token INT8 崩掉的程度。与 LLM.int8() 的相变结论并排看，结论一致： **0.5B 在相变之下**。

实测代价：同一 INT4-g128 网格下，RTN 把 Qwen2.5-0.5B 的 wikitext-2 PPL 从 11.9152 打到 14.1154（+2.2002；EXP-001《GPTQ 从零实现》§5，`data/raw/EXP-001/rtn_g128.json`， 单轮确定性评测，下同）。三种方法都是对这笔"加权误差账"的不同回应： GPTQ 在固定网格内**重新分配**误差（二阶），AWQ **重塑**误差落点的难度分布（一阶），SmoothQuant 把难度在激活/权重两侧**搬家**（W8A8 专属）。

### 3.3 GPTQ:从 OBD/OBS 到逐列补偿公式

#### 3.3.1 OBD:三重近似与对角 Hessian

OBD(LeCun， Denker & Solla， "Optimal Brain Damage"， NIPS 1989)要回答"删掉哪个权重最不痛"。它把损失二阶展开后叠三重简化（论文以 quadratic / extremum / diagonal 命名；**公式编号未核实**——原始 PDF 为扫描件，无法逐字核对）： **quadratic** 丢掉三阶以上项，合法条件是扰动足够小（删一个权重的扰动并不小， 这是最脆弱的一环）；**extremum** 设训练已收敛，梯度项 $g^\top\delta w\approx 0$ 可丢；**diagonal** 只留 $H$ 的对角，于是"删多个权重的损失 = 各自单独删的损失之和"，合法条件是权重之间不相关。三项一叠得 saliency $s_k = h_{kk}u_k^2/2$。**OBD 的问题在第三项**：它假定删掉一个权重后其余权重不动，于是根本没有"补偿"这一步。

#### 3.3.2 OBS:去掉对角假设,拉格朗日给出闭式补偿

OBS(Hassibi & Stork， "Second Order Derivatives for Network Pruning： Optimal Brain Surgeon"， NIPS 1992)保留完整 $H$，把"删第 q 个权重"写成约束， 其余权重可自由调整来补偿：

$\Delta E \approx \Delta E_0 + g^\top \delta w + \tfrac12 \delta w^\top H \delta w$， 约束 $e_q^\top \delta w + w_q = 0$（把第 q 个权重打到 0），解得 $\delta w = -\dfrac{w_q}{[H^{-1}]_{qq}} H^{-1} e_q$，代价 $L_q = \dfrac{w_q^2}{2[H^{-1}]_{qq}}$。

论文对 OBD 的批评正是本篇反复用到的直觉：**off-diagonal 项决定了"补偿往哪儿传"，丢掉它就等于放弃补偿**。注意 $[H^{-1}]_{qq}$ 是**逆的对角元**而不是 $1/h_{qq}$——两者只有在 $H$ 对角时才相等，混同它们是这条推导线上最经典的错误。

#### 3.3.3 OBQ:把"删"换成"量化"

OBQ(Frantar， Singh & Alistarh， "Optimal Brain Compression"， arXiv:2208.11580)的替换只有一处：约束从"$w_q$ 打到 0"改成"$w_q$ 打到最近的网格点 $\mathrm{quant}(w_q)$"。GPTQ 论文 §3(Background)把它写成（该文的 Equation 2）：

$w_q = \arg\min_{w_q}\ \dfrac{(\mathrm{quant}(w_q)-w_q)^2}{[H_F^{-1}]_{qq}}$,  $\delta_F = -\dfrac{w_q - \mathrm{quant}(w_q)}{[H_F^{-1}]_{qq}}\cdot (H_F^{-1})_{:,q}$

以及删列后 $H^{-1}$ 的高斯消元更新（该文 Equation 3）。下标 $F$ 是"尚未量化的列集合"，**每量化一列 $F$ 就缩一个，$[H_F^{-1}]$ 必须重算**——朴素做法 $O(d_{row}\cdot d_{col}^3)$，这正是 GPTQ 要解决的成本。

#### 3.3.4 第 0 步:层内目标是精确二次型,不是泰勒截断

取一个输出行 $w \in \mathbb{R}^d$（行与行独立，下面解释），量化后 $\hat w = w + \delta$，层输出误差

$L(\delta) = \|\hat w^\top X - w^\top X\|^2 = \|\delta^\top X\|^2 = \delta^\top (XX^\top)\, \delta = \tfrac{1}{2}\,\delta^\top H\,\delta,\quad H = 2XX^\top$

为什么可以这么做：**这不是泰勒展开**——层内目标对 $\delta$ 本来就是二次的， 一步都没截断。于是 §3.3.1 的三重近似里，quadratic 与 extremum 两项在层内目标上是**精确**的，只有"层输出误差能代表最终损失"这一层代理仍是近似。$H$ 与行无关（只含 X），所以全部输出行共享同一个 H、同一次分解——这是 GPTQ 能把整层向量化的根基，也是相对 OBQ 的第一个数量级红利：**每个 Linear 只分解一次， 不是每行一次**。因子 2 来自 $\partial^2\|\delta^\top X\|^2/\partial\delta^2 = 2XX^\top$，代码把它写进 `src/gptq.py:93` 的 `math.sqrt(2.0 / self.nsamples)`——注意 2 在**开方之内**， 因为要作用在 $X$ 上再自乘。

#### 3.3.5 第 1-2 步:单列约束优化,每步一行理由

量化第 j 列：把 $w_j$ 固定到网格点 $q_j$，记 $\varepsilon = w_j - q_j$，即约束 $\delta_j = -\varepsilon$；其余坐标自由，用来吸收伤害。问题： $\min_\delta \tfrac{1}{2}\delta^\top H \delta$ s.t. $e_j^\top \delta = -\varepsilon$。

为什么可以这么建模：量化是硬约束（码必须落网格），但尚未量化的坐标还在连续域，可以连续优化补救——"牺牲是强制的，补偿是自由的"。为什么只约束一个坐标而不是同时约束一批：一批也有闭式解（GPTQ 论文 §4 的 Equation 4-5 即块版本）， 但块内各列的取整会互相影响（第 i 列补偿后第 i+1 列的值和网格位置都变了）， 逐列串行才能让每次取整看到最新权重。**GPTQ 的块化只批量化更新的写回， 不批量化取整决策**(§3.3.8)。

拉格朗日求解：

1. $\mathcal{L} = \tfrac{1}{2}\delta^\top H\delta + \lambda(e_j^\top\delta + \varepsilon)$—— 等式约束的标准拉格朗日化。
2. $\partial\mathcal{L}/\partial\delta = H\delta + \lambda e_j = 0 \Rightarrow \delta = -\lambda H^{-1} e_j$—— H 正定（阻尼保证，见 §3.3.9）故可逆，凸问题一阶条件即全局最优。
3. 代回约束：$e_j^\top\delta = -\lambda\,[H^{-1}]_{jj} = -\varepsilon \Rightarrow \lambda = \varepsilon / [H^{-1}]_{jj}$—— $e_j^\top H^{-1} e_j$ 就是 $H^{-1}$ 的第（j，j） 元。
4. 最优补偿：$\delta^* = -\dfrac{\varepsilon}{[H^{-1}]_{jj}}\, H^{-1} e_j$—— 方向是 $H^{-1}$ 第 j 列，强度按其对角元归一。
5. 代回目标：$\Delta L = \tfrac{1}{2}\delta^{*\top} H \delta^* = \dfrac{\varepsilon^2}{2\,[H^{-1}]_{jj}}$—— 与 OBS 的 $L_q$ 同形，把 $w_q$ 换成量化残差 $\varepsilon$。这就是代码里那个"量纲未标定、只作诊断"的逐列 loss(EXP-001 §7)。

第 5 步值得多说一句：$\Delta L$ 随 $[H^{-1}]_{jj}$ **增大而减小**——该值大意味着方向"软"（激发弱），量化代价小；小则是"硬"方向，同样的 $\varepsilon$ 代价更大。act-order(§3.3.10)排的正是这个量。

#### 3.3.6 第 3 步:剩余子问题的 H 是 Schur 补("子矩阵的逆"≠"逆的子矩阵")

OBQ 每次挑 $\Delta L$ 最小的列，每列都要对"尚未量化的列集 F"重求 $[H_F^{-1}]$。这里的经典易错点必须点破：**$[H_F]^{-1}$（先取子矩阵再求逆） 不等于 $[H^{-1}]_F$（先求逆再取子矩阵）**。两者的关系正是块矩阵逆恒等式：

$[H_F]^{-1} = [H^{-1}]_{F} - [H^{-1}]_{F,F^c}\,([H^{-1}]_{F^c})^{-1}\,[H^{-1}]_{F^c,F}$

即：对 $H^{-1}$ 关于已量化块 $F^c$ 取 Schur 补。**为什么成立**：把 $H$ 与 $H^{-1}$ 都按 $(F, F^c)$ 分块，写出 $HH^{-1}=I$ 的四个块方程消去交叉块即得（标准结果）。**成立条件**：$[H^{-1}]_{F^c}$ 可逆——$H$ 正定 ⇒ $H^{-1}$ 正定 ⇒ 任意主子阵正定 ⇒ 可逆。阻尼（§3.3.9）保证的就是这个"正定"。

GPTQ 论文 §4 的 Step 1(Arbitrary Order Insight)说明：按固定列序量化精度几乎不掉，理由是"any fixed order may perform well， especially on large models"——贪心排序的收益要与"越晚量化剩下可调权重越少"相抵。固定序把运行时从 $O(d_{row}\cdot d_{col}^3)$ 降到 $O(\max\{d_{row}\cdot d_{col}^2, d_{col}^3\})$，即 $\min\{d_{row}, d_{col}\}$ 倍加速。**注意这是经验性论断**（论文用 "may perform well" 的措辞），不是定理；仓库后来又把 act-order 加回来，恰说明 outlier 严重时顺序仍要紧。

#### 3.3.7 第 4 步:Cholesky 逆上三角为什么"恰好可用"

设 $H^{-1} = U^\top U$（U 上三角，即代码第三步 `cholesky(Hinv, upper=True)`）。按前缀/后缀分块 $U = \begin{pmatrix} U_{11} & U_{12} \\ 0 & U_{22} \end{pmatrix}$，直接乘开：

$H^{-1} = \begin{pmatrix} U_{11}^\top U_{11} & U_{11}^\top U_{12} \\ U_{12}^\top U_{11} & U_{12}^\top U_{12} + U_{22}^\top U_{22} \end{pmatrix}$

对后缀块取 Schur 补（为什么可以：$U_{11}$ 是正定矩阵的 Cholesky 因子， 对角非零必可逆）：

$U_{12}^\top U_{12} + U_{22}^\top U_{22} - U_{12}^\top U_{11}(U_{11}^\top U_{11})^{-1}U_{11}^\top U_{12} = U_{22}^\top U_{22}$

中间那一步的抵消值得写全： $U_{11}(U_{11}^\top U_{11})^{-1}U_{11}^\top = U_{11}U_{11}^{-1}U_{11}^{-\top}U_{11}^\top = I$， 所以第三项恰好等于 $U_{12}^\top U_{12}$，与第一项对消。

所以 $[H_F^{-1}] = U_{22}^\top U_{22}$：**一次上三角分解，右下角每个后缀块自动就是对应剩余列集的逆**。取该块的（1,1） 元与第一行（$U_{22}$ 上三角，其第一列只有对角元非零）：

$[H_F^{-1}]_{jj} = U_{jj}^2,\qquad [H_F^{-1}]_{j,\,j:} = U_{jj}\cdot U_{j,\,j:}$

代入第 2 步的 $\delta^*$，补偿公式化简为：

$\delta^*_{j:} = -\dfrac{\varepsilon}{U_{jj}^2}\cdot U_{jj}\,U_{j,j:} = -\dfrac{\varepsilon}{U_{jj}}\cdot U_{j,j:}$

这正是 `src/gptq.py:170-171` 那两行。一次 $O(d^3)$ 分解替代 d 次重求逆； "上"三角而非"下"，是因为固定序从左到右、被冻结的是前缀、自由的是后缀， 所有后缀块沿上三角因子的行依次排开（若量化序反向则用下三角）。补偿只向右传播：左侧列的整数码已定、不可再动，右侧列还有自由度——误差传播方向必然"向未量化侧"。

**为什么论文选 Cholesky 而不是继续用高斯消元递推。** GPTQ 论文 §4 Step 3 给的理由是数值不是速度：反复套用消元更新式会累积误差，"the matrix $H_F^{-1}$ becomes indefinite， which can cause the algorithm to aggressively update remaining weights in incorrect directions"，而且在十亿参数以上的模型里"almost certainly occurs for at least a few layers"。论文同时点破两条路同源：对称矩阵的逐行消去"essentially corresponds to taking a Cholesky decomposition"。**Cholesky 不是另一个算法，是同一个算法的数值稳定实现**， 顺带一次算完所有需要的行。数值依据是 Cholesky 对对称正定矩阵后向稳定且 **不需要选主元**（Higham， 2nd ed.， SIAM 2002， 第 10 章；计算因子满足 $\hat R^\top \hat R = A + \Delta A$ 且 $\|\Delta A\|$ 被 $\|A\|$ 乘以维度相关小常数与机器精度所界——**定理编号未核实**，只用定性结论）。代码里 `torch.cholesky_inverse` 走"由因子求逆"而非通用 `inverse` (`src/gptq.py:128`)，同一条理由。

#### 3.3.8 lazy batch update:等价性与访存账

GPTQ 论文 §4 的 Step 2(Lazy Batch-Updates)按 **B=128 列**成块处理，论文称其带来"an order of magnitude speedup for very large models in practice"。本仓 `blocksize` 默认 128 且被断言等于 group_size(`src/gptq.py:136`)。

**先证等价。** 逐列全宽更新是：对每个 j，$W_{:,j+1:} \mathrel{-}= \varepsilon_j u_{j,j+1:}$， 其中 $u = U$ 的第 j 行、$\varepsilon_j$ 是归一化后的误差。把 j 在一块 $[i_1, i_2)$ 内累加，对块外列 $c \ge i_2$ 的总更新是 $\sum_{j=i_1}^{i_2-1}\varepsilon_j U_{j,c}$，恰是矩阵乘 $\mathrm{Err}_1 \cdot U_{i_1:i_2,\ i_2:}$ 的第 c 列（`src/gptq.py:184`）。**逐列累加与一次 GEMM 逐位不同（浮点求和序不同）， 但数学上严格相同**——这是"数学结果与逐列全宽更新严格相同"这句注释的准确含义，写讲义时不要把它说成"逐位相同"。

**再算访存。** 朴素做法：每列对全宽 $W$ 做 rank-1 更新，读写 $O(\mathrm{rows}\times \mathrm{in})$ 字节，共 in 次 → 总访存 $O(\mathrm{rows}\times \mathrm{in}^2)$。以 down_proj（896×4864，fp32 工作副本） 为例：单次全宽读写 $2\times896\times4864\times4$ B $= 34.9$ MB，乘 4864 次 $\approx 170$ GB；而 4090 的显存带宽规格值是 1008 GB/s（NVIDIA Ada GPU Architecture 白皮书 Appendix A），**光这一层就要 0.17 s 的纯带宽下限， 24 层就是 4 s 起**，且这还只是 down_proj 一个 Linear（本讲义推导）。块化后块内更新只碰 $\mathrm{rows}\times B$ 的块（0.92 MB），块外更新每 128 列才做一次 → 总访存降到约 $1/B$ 量级。**这就是 lazy batch update 的全部动机： 不是省算力，是省带宽。**

#### 3.3.9 阻尼 λ:三重作用与 1% 的来历

校准样本有限或输入通道强共线时 H 近奇异，$H^{-1}$ 在弱激发方向的元素爆炸， 补偿量随之爆炸——本想救误差，反把权重毁了。加 $\lambda = \text{percdamp}\cdot\mathrm{mean}(\mathrm{diag}\,H)$ 的 ridge (`src/gptq.py:117-118`，percdamp=0.01)有三重效果：

①**保证正定**：$H + \lambda I$ 的最小特征值至少 $\lambda$，Cholesky 必成功， §3.3.6 的 Schur 补前提随之成立；②**压回病态方向**：补偿强度 $\propto [H_F^{-1}]$，加 ridge 后弱激发方向的逆元被压到 $\le 1/\lambda$； ③**对整体量级自适应**：取"相对平均对角"而非绝对常数，这一条要与 `add_batch` 的运行均值配合才成立——H 已归一为 $(2/N)\sum xx^\top$，对角量级只反映激活强度，不随 token 数漂移。

**1% 是谁定的。** GPTQ 论文 §4 Step 3 明确写"add $\lambda$ = 1% of average diagonal value to H's diagonal"，本仓同设（`src/gptq.py:115-116`）。这个魔法数属于"**论文默认值 + 本仓沿用**"，不是本仓扫出来的——本仓没做 percdamp 扫描， 被追问"0.005 或 0.05 会怎样"的正确答法是"没测过，但方向可推：太小则病态层分解失败或补偿爆炸，太大则 $H \to \lambda I$、退化向 RTN"。

#### 3.3.10 act-order:动机、代价,以及本仓为什么留位不开

GPTQ 仓库（IST-DASLab/gptq）README 提供 `--act-order`，即"quantizing columns in order of decreasing activation size"，并给出实测：OPT-66B 上 4bit 由 9.55 降到 9.34、3bit 由 14.16 降到 9.95(Wiki2 PPL)。3bit 那一档的改善幅度说明：**在 outlier 严重的模型与更低位宽上，列序远不是"随便挑"**。

动机可以从 §3.3.5 直接读出来：第 j 列量化后，能吸收误差的自由集是它右边的所有列。**越靠右的列，能补偿它的自由度越少；最右一列的误差完全无人补偿。** 所以把"硬方向"(diag(H) 大、$[H^{-1}]_{jj}$ 小)排到最前面，让它享有最大的补偿容量，是自然的贪心。

代价在部署侧：act-order 打乱了列的物理顺序，而 per-group 的 scale/zero 是按**物理连续 128 列**存的。要么在推理时带一个 `g_idx` 置换（每次反量化多一次间接寻址），要么用仓库提供的 `--static-groups` 先把组划分固定下来——README 明确写道，与 `--static-groups` 合用时 act-order 就"does not require any inference changes (that may cause slowdown)"。

**本仓为什么不开：** 本仓的 `blocksize == group_size` 断言（`src/gptq.py:136`）把"组边界"和"补偿块边界"绑死，而 act-order 会让同一组的列在补偿顺序上不再连续——两者直接冲突，要开 act-order 就必须先实现 static-groups。EXP-001 §7 把它记为未测项。这是一个诚实的实现边界，不是疏漏：**开关没开就不能声称有它的收益**。

### 3.4 AWQ:显著通道判据与等价缩放证明

#### 3.4.1 判据:为什么看激活不看权重

AWQ 论文 §3.1(Improving LLM Quantization by Preserving 1% Salient Weights)先做了一个消融：把 1% 的权重留在 FP16、其余走 INT3-g128，比较三种挑选依据。论文 Table 1（OPT 系列，WikiText PPL）给出的对照极其干净：

| 模型 | FP16 | RTN(INT3-g128) | 保 1%（按激活幅值） | 保 1%（按权重幅值） | 保 1%（随机） |
|---|---|---|---|---|---|
| OPT-1.3B | 14.62 | 119.00 | 16.91 | 98.55 | 109.38 |
| OPT-6.7B | 10.86 | 23.54 | 11.39 | 22.37 | 24.23 |
| OPT-13B | 10.13 | 46.04 | 10.43 | 48.96 | 42.00 |

读法：**按权重幅值挑，和随机挑几乎一样没用**；按激活幅值挑，一步把 OPT-6.7B 从 23.54 拉回 11.39（FP16 是 10.86）。论文据此主张 "To identify salient weight channels， we should refer to the activation distribution， not weights"（摘要）。这与 §3.2.1 的加权误差恒等式完全一致： 伤害是 $\delta^\top \mathbb{E}[xx^\top]\delta$，权重自己多大根本不在式子里。 **但混精度不好落地**——1% 的 FP16 权重散落在 INT3 矩阵里，kernel 要处理两种数据类型和不规则寻址。AWQ 的后半段就是在回答：能否不用混精度，只靠一个逐通道缩放拿到同样的保护？

#### 3.4.2 缩放降低误差的证明,以及它的成立条件

AWQ 不改取整规则，改坐标系：对通道 j 的权重先乘 $s_j>1$ 再量化，运行时等价除回：

$y = XW^\top = (X\,\mathrm{diag}(s)^{-1})\,(\mathrm{diag}(s)\,W^\top)$—— 数学恒等，只改变误差落在哪里。

论文 §3.2 把误差写成（该文的表述） $\mathrm{Err}(Q(w)x) = \Delta\cdot\mathrm{RoundErr}(w/\Delta)\cdot x$， 缩放后 $\mathrm{Err}(Q(w\cdot s)(x/s)) = \Delta'\cdot\mathrm{RoundErr}(ws/\Delta')\cdot x\cdot(1/s)$， 两者之比为 $\dfrac{\Delta'}{\Delta}\cdot\dfrac{1}{s}$。

逐步拆开这个式子凭什么成立：①$\mathrm{RoundErr}(\cdot)\in[-0.5,0.5]$ 是 **以格为单位**的取整残差，绝对误差 = 格距 × 残差，这一步是定义不是近似； ②论文把缩放前后的 $\mathrm{RoundErr}$ 视为**同一量级**，这是**近似**，偏差就是 §3.1.2 列的那些（端点误差为 0、残差与数据相关），论文没给界； ③剩下全部戏码在 $\Delta'/\Delta$：若组的极值不由被放大的通道决定， $\Delta'\approx\Delta$，误差比就是 $1/s$——**整整缩小 s 倍**。

论文 Table 2(OPT-6.7B)给出了第 3 点的经验统计，只放大**单个**显著通道：

| s | 组中 Δ′≠Δ 的比例 | 平均 Δ′/Δ | 平均（Δ′/Δ）·(1/s) | Wiki-2 PPL |
|---|---|---|---|---|
| 1 | 0% | 1 | 1 | 23.54 |
| 1.25 | 2.8% | 1.005 | 0.804 | 12.87 |
| 1.5 | 4.4% | 1.013 | 0.676 | 12.48 |
| 2 | 8.2% | 1.038 | 0.519 | 11.92 |
| 4 | 21.2% | 1.213 | 0.303 | 12.36 |

关键读点两个：①$s=2$ 时 $\Delta'/\Delta$ 只涨到 1.038 而 $1/s=0.5$，净收益 0.519——**放大是划算的**；②$s=4$ 时误差比继续降到 0.303，PPL 却由 11.92 反弹到 12.36——**误差比不是质量**，过度放大伤到了同组其他通道，而这个损失不在那个比值里。这张表本身就是"保护是有代价的转移"的证据。

#### 3.4.3 从"单通道放大"到"全向量 s":论文没写的那一步(本讲义推导)

Table 2 的前提是**只放大一个通道**。而 AWQ 真正部署的算法是 $s = s_X^\alpha$——**每个通道都被乘上各自的 $s_j$**(论文 §3.2 的 Eq.(4)-(5) 与 grid search)。两者不是同一件事：全向量缩放下组的极值几乎必然改变，"$\Delta'/\Delta\approx1$" 不再成立。论文用网格搜索直接优化输出 MSE，算法本身是对的；但**"为什么缩放有效"的那段论证严格来说只覆盖了单通道情形**。本讲义把中间步补出来。

设组 $g$ 内原始极值给出步长 $\Delta_g$，缩放后 $\Delta'_g$ 由 $\max_{j\in g}|W_j s_j|$ 决定。对任意 $j\in g$：

$\min_{k\in g} s_k \cdot \Delta_g \ \le\ \Delta'_g \ \le\ \max_{k\in g} s_k \cdot \Delta_g$

（左右两边分别把 $s_k$ 换成组内最小/最大值，再提出常数即得。）于是通道 j 的**有效**量化误差 $\mathrm{err}_j \approx \Delta'_g/(2 s_j)$ 被夹在：

$\dfrac{\Delta_g}{2}\cdot\dfrac{\min_k s_k}{s_j} \ \le\ \mathrm{err}_j \ \le\ \dfrac{\Delta_g}{2}\cdot\dfrac{\max_k s_k}{s_j}$

两个立刻能用的推论：

- **对组内最被保护的通道**($s_j = \max_k s_k$)：上界退化为 $\Delta_g/2$， 即**它的误差绝不会比 RTN 更差**；下界是 $(\Delta_g/2)\cdot(\min_k s_k/\max_k s_k)$， 即最好能改善"组内 s 的展布"倍。
- **对组内最不被保护的通道**($s_j = \min_k s_k$)：误差最坏放大到 $(\Delta_g/2)\cdot(\max_k s_k/\min_k s_k)$。

**收益上界与代价上界由同一个数控制：组内 s 的展布 $\rho_g = \max_{k\in g}s_k/\min_{k\in g}s_k$。** 而 $s_j = \mathrm{absmean}(X_j)^\alpha$ 给出 $\rho_g = \left(\dfrac{\max_k a_k}{\min_k a_k}\right)^{\alpha}$（$a$ 记 absmean）——**α 就是展布的指数**。α=0 时 $\rho=1$（退化为 RTN， 无收益也无代价），α 增大则收益与代价同步放大。这条式子把"α 是权衡旋钮"从定性说法变成了一个可以写在白板上的单调关系，也解释了为什么本仓的 per-layer best-α 会随层的激活分布形状而变（§5.5）。

#### 3.4.4 几何归一为什么是必要条件,不是风格选择

参考实现 `mit-han-lab/llm-awq` 的 `awq/quantize/auto_scale.py` 里有一行 `scales = scales / (scales.max() * scales.min()).sqrt()`，本仓 `src/awq.py:115` 与之同形。为什么这一步不能省：

若不归一，$s = a^\alpha$ 的整体量级随 α 变化：典型值是 $\bar a^\alpha$， α 越大整个 $s$ 向量越大（$\bar a>1$）或越小（$\bar a<1$）。**这会让每组的 min-max 范围整体变粗或变细**，α 档之间比较的就不再是"幅值再分配"，而混进了 "网格整体缩放"。归一后 $\sqrt{s_{max}s_{min}}=1$，即 $\log s$ 关于 0 对称； 此时 α 只改变 §3.4.3 的展布 $\rho$，不改整体量级——各档才可比。

**归一的数值陷阱。** 参考实现的这一行在真实模型上曾产生 inf（llm-awq 仓库 issue #96：某模型上 `scales.min()` 为 0 导致除零）。本仓在取幂之前先做 `base = self.absmean.clamp(min=1e-4)`(`src/awq.py:102`)，把死通道（absmean=0）顶到 1e-4，从源头堵掉 $0^\alpha = 0$ → $s_{min}=0$ → inf 这条路径。这是一处"本仓比参考实现多一道防护"的地方，值得在面试里主动说。

#### 3.4.5 打分对象必须是"有效权重"

**best-scale 目标函数**（EXP-002 §1 口径）：

$\min_s\ \|\,Q(W\cdot\mathrm{diag}(s))\,\mathrm{diag}(s)^{-1}X - WX\,\|^2,\quad s_j = \mathrm{absmean}(X_j)^\alpha,\ \alpha \in \{0, 0.05, \dots, 0.95\}$

三个设计决定，各有理由：①打分在**输出空间**（MSE 对 ref = XW^⊤），不在权重空间——优化目标必须与推理在乎的东西一致，这是全篇的主旋律； ②重要性代理用 absmean 不用 absmax——量通道的系统性幅值，不被单个极端 token 绑架；参考实现同口径（`get_act_scale` 为 `x.abs().view(-1, x.shape[-1]).mean(0)`）；③s 做几何归一（§3.4.4）。

**为什么必须除回 s 再打分。** 代码是 `w_eff = fq / s.unsqueeze(0)` (`src/awq.py:120`)。若直接拿 `fq`（即 $Q(W\cdot s)$）对 ref 算 MSE，比较的是两个不同坐标系里的输出，$Q(Ws)X$ 与 $WX$ 天然差一个 $\mathrm{diag}(s)$， α 越大差得越多，搜索会永远选 α=0。**这个 bug 不报错，只让 AWQ 静默退化成 RTN**——PPL 只告诉你"AWQ 没用"，不告诉你哪错了。

**为什么网格必须含 α=0，又为什么不含 α=1。** α=0 即 $s\equiv1$(RTN)， 是搜索的保底选项：**打分口径下结果不会劣于 RTN 起点**；EXP-002 的分布里恰有 1 层选中 α=0，保底被真实用到。论文搜索区间 [0,1]、grid size 20；本仓 `alpha = i / self.n_grid`，i∈[0,20)，取到 0.95 为止（`src/awq.py:109`）。 α=1 即 $s=a$，展布 $\rho$ 取最大值，常常过度。副作用是**顶到 0.95 的层究竟是"最优恰在 0.95"还是"被边界截断"，数据分不出来**——EXP-002 §7 记为可能欠搜索，§5.5 用 raw 分布给出区分判据。

#### 3.4.6 per-linear 简化的代价与折叠约束(限定语教学)

本仓为臂间可比取 per-linear 独立 s + per-linear MSE 打分，且未实现 clip； 参考实现是**块级 MSE**（`loss = (org_out - out).float().pow(2).mean().item()`， 其中 `org_out`/`out` 是整个 decoder block 的输出）+ **共享输入组共享 s**（q/k/v 传进同一个 `linears2scale` 列表）+ **weight clip**（`auto_clip.py` 按 `max_val = org_max_val * (1 - i_s / n_grid)` 扫 10 档收缩比，目标同样是输出 MSE）。EXP-002 §2 已诚实标注这三处差异。

三处简化各自的方向性影响（本讲义推导，**未做消融实验，不作定量主张**）： **per-linear vs 块级 MSE**——per-linear 打分看不到后续算子（softmax、门控） 对误差的放大或吸收，可能选出对本层最优、对块次优的 α；**独立 s vs 共享 s**——独立 s 的搜索空间更大、单看打分应当不劣，但它与折叠部署不兼容，买到的自由度在生产里花不出去；**无 clip**——clip 直接压缩组极值、缩小 $\Delta$， 与缩放互补，缺它相当于少一维搜索。

因此本仓 AWQ 的 31.9% 是"**per-linear 简化、无 clip** 口径下"的数字，不能拿去与参考实现的相对表现直接比——引用数字时这个定语不许丢。这同时是一个可讲的活例：简化的代价是可测的（EXP-002 §6）。

**部署侧的折叠约束。** 要让 $X\,\mathrm{diag}(s)^{-1}$ 不产生额外算子，只能把 $1/s$ 折叠进**产生这个 X 的那个算子**的权重里——pre-norm 结构下就是前置 RMSNorm(`ln.weight /= s`)。前提是**所有消费同一个 X 的 Linear 必须共用同一个 s**，否则一个 RMSNorm 没法同时满足几套。本模型里即 {q，k，v} 一组、 {gate，up} 一组。本仓走 fake-quant 评测形式（把 $Q(Ws)/s$ 写回 weight、激活不动，`src/awq.py:29-33`），与"runtime 给激活除 s"端到端等价，所以 per-linear 独立 s 在**评测口径下合法**；**但这不是部署形态**，拿去做零开销部署会立刻撞上共享约束。这条边界必须随数字一起说。

### 3.5 SmoothQuant:迁移强度 α 与逐通道误差分析

#### 3.5.1 W8A8 的病灶:激活 outlier 的三条性质

SmoothQuant 论文 §3(Review of Quantization Difficulty)给出三条观察： ①**幅值极端**——"The scale of outliers in activations is ~100× larger than most of the activation values."；②**按通道固定**——"Outliers appear in a small fraction of the channels. If one channel has an outlier， it persistently appears in all tokens"，同一通道跨 token 的幅值方差很小； ③**权重侧平坦**——权重分布均匀，INT8 下几乎无损，难度全在激活侧。

第 2 条是关键：outlier 是**模型属性**而非 token 属性，所以可以离线用校准集统计每通道幅值、一次性决定 $s$，不需要在线自适应——本仓 `collect_actmax` 正是这么做的（`scripts/run_w8a8.py:31-38`）。而 per-token 对称 INT8 的 scale 取该 token 全通道 absmax，被 outlier 独占后其余通道分辨率崩塌。 **那为什么不给激活做 per-channel？**

#### 3.5.2 为什么激活不能 per-channel:从 GEMM 代数到指令语义

INT8 GEMM 要求 scale 能从整数乘加中整体提出：

$y_{ij} = \sum_k x_{ik}w_{jk} \approx \sum_k (s^x_i\,qx_{ik})(s^w_j\,qw_{jk}) = s^x_i \cdot s^w_j \cdot \sum_k qx_{ik}\,qw_{jk}$

第二个等号成立的**唯一条件**是 $s^x$、$s^w$ 都不依赖 k。也就是说 **scale 必须呈行×列外积形状**；沿求和维（输入通道 k）的 per-channel scale 卡在 $\sum_k$ 里面，提不出来。论文 §3 的原话："scaling can only be performed along the outer dimensions of the matrix multiplication (i.e.， token dimension of activations T， output channel dimension of weights Co)， which can be applied after the matrix multiplication finishes."

**这条约束在硬件语义上是什么。** 整数张量核指令（PTX ISA 的 `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` 一族）在 k 维上做 s8×s8 乘加并累到 s32 寄存器；cuBLAS 的整数计算类型 `CUBLAS_COMPUTE_32I` 文档写 "This is the default 32-bit integer mode and uses compute and intermediate storage precisions of at least 32-bits"(§2.2.11 `cublasComputeType_t`)。 **整个 k 维归约在整数域一口气做完，中间没有插入浮点缩放的位置**——scale 只能在归约后作用于（行， 列） 两个外维。NVIDIA 给这类向量缩放的命名就叫 "Outer Vector Scaling"（cuBLAS 文档 §3.1.4.3，该小节列在 FP8 数据类型下； **INT8 路径是否有同名接口未核实**，此处只借其命名说明"外维"语义）。

代码上这条约束落成两个轴参数（`src/smoothquant.py:41-53`）：激活 `amax(dim=-1)`（每 token 一个 scale，token 是外维），权重 `amax(dim=1)`（每输出行一个 scale，输出行是外维）。**两个 dim 不是风格，是指令语义的直接投影**；写反任何一个，得到的"量化"在真实 INT8 kernel 上根本无法执行。于是结论：outlier 只能靠**迁移**消解，不能靠更细粒度硬扛。

#### 3.5.3 迁移公式与两侧幅值的逐步代入

恒等变换 $y = (X\,\mathrm{diag}(s)^{-1})(\mathrm{diag}(s)W^\top)$，取（SmoothQuant 论文 §4 的 Equation 4）

$s_j = \mathrm{actmax}_j^{\alpha} \,/\, \mathrm{wmax}_j^{1-\alpha}$

迁移后两侧的通道幅值（逐步代入）：

- 激活侧：$\mathrm{actmax}_j / s_j = \mathrm{actmax}_j\cdot \mathrm{actmax}_j^{-\alpha}\cdot\mathrm{wmax}_j^{1-\alpha} = (\mathrm{actmax}_j\,\mathrm{wmax}_j)^{1-\alpha}$
- 权重侧：$\mathrm{wmax}_j \cdot s_j = \mathrm{wmax}_j\cdot\mathrm{actmax}_j^{\alpha}\cdot\mathrm{wmax}_j^{\alpha-1} = (\mathrm{actmax}_j\,\mathrm{wmax}_j)^{\alpha}$

两侧都只是**同一个量** $p_j:= \mathrm{actmax}_j\,\mathrm{wmax}_j$ 的幂： 激活侧是 $p_j^{1-\alpha}$、权重侧是 $p_j^{\alpha}$。这个观察有个直接推论， 后面要用：**两侧的通道最大值由同一个 j 取到**（因为 $p^{1-\alpha}$ 与 $p^{\alpha}$ 都是 $p$ 的单调增函数），即 $M_x(\alpha) = P^{1-\alpha}$、$M_w(\alpha) = P^{\alpha}$，其中 $P = \max_j p_j$。

α=0.5 时两侧恰为几何均值 $\sqrt{\mathrm{actmax}_j\,\mathrm{wmax}_j}$——难度几何均衡；α 越大激活越轻、权重越重。论文 §5.5 的 Figure 10 给出 α 消融： "when α is too small (<0.4)， the activations are hard to quantize； when α is too large (>0.6)， the weights will be hard to quantize"，甜区 0.4–0.6， 默认 0.5；GLM-130B 因 outlier 更严重而取 0.75。

#### 3.5.4 α 的一阶最优条件与权重粒度项(本讲义推导)

论文只说"α 平衡两侧难度"，没给最优点的条件式。本讲义把它推出来，因为它恰好能解释本仓与论文的冲突（§8.1 第 10 条）。

**建模。** 量化误差按 §3.1.2 的独立均匀模型处理（前提与失效边界同 §3.1.2）。记 $\tilde x_j = x_j/s_j$、$\tilde W_{ij} = W_{ij}s_j$，输出 $y_i = \sum_j \tilde x_j \tilde W_{ij}$ 严格成立。一阶扰动：

$\Delta y_i \approx \sum_j \big(e^x_j\,\tilde W_{ij} + \tilde x_j\,e^W_{ij}\big)$

设 $e^x$、$e^W$ 零均值独立、方差分别 $\Delta_x^2/12$、$\Delta_W^2/12$，则

$\mathrm{Var}(\Delta y_i) \approx \tfrac{\Delta_x^2}{12}\sum_j \tilde W_{ij}^2 \;+\; \tfrac{\Delta_W^2}{12}\sum_j \tilde x_j^2$

**代入幅值。** 用"通道幅值 × 形状因子"的写法 $x_j = a_j u_j$、 $W_{ij} = w_j v_{ij}$（$a=\mathrm{actmax}$、$w=\mathrm{wmax}$，故 $|u|,|v|\le 1$）。由 §3.5.3，$\tilde x_j = p_j^{1-\alpha}u_j$、 $\tilde W_{ij} = p_j^{\alpha}v_{ij}$，而 $\Delta_x = P^{1-\alpha}/127$、$\Delta_W = P^{\alpha}V/127$，其中 $V$ 是权重量化粒度决定的"峰值利用率"因子（per-tensor 时 $V=\max_{ij}|v_{ij}|$， per-输出行时 $V_i=\max_j |v_{ij}|$，后者逐行取，必然 $\le$ 前者）。激活侧的同名因子被 per-token 动态 scale 逐 token 吸收，故此处取 1，不出现在式中——**两侧的不对称正是"粒度"这个变量进入结论的入口**。

把 $q_j:= p_j/P \in (0,1]$ 代进去，$P$ 会整体提出：

$12\cdot 127^2\cdot\mathrm{Var}(\Delta y_i) = P^2\Big[\underbrace{\sum_j q_j^{2\alpha}v_{ij}^2}_{S_1(\alpha)} \;+\; V^2\underbrace{\sum_j q_j^{2-2\alpha}u_j^2}_{S_2(\alpha)}\Big]$

**结构读出来的三件事：**

①**内点最优存在**：$q_j\le1$，故 $S_1$ 随 α 单调减、$S_2$ 随 α 单调增， 一减一增使和在（0,1） 内取到极小——这是"α 是天平"的数学形式，而不只是比喻。 ②**一阶条件**：$\sum_j (\ln q_j)\big[q_j^{2\alpha}v_{ij}^2 - V^2 q_j^{2-2\alpha}u_j^2\big] = 0$； $\alpha=0.5$ 是解当且仅当加权后两项恰好相等——**几何均衡只是一个特例，不是普遍解**。③**权重粒度进入的位置是 $V^2$**，它只乘在权重误差项上：**粒度越细 $V$ 越小，权重侧惩罚越轻，最优 α 越大**。

第 3 点正是本仓与论文冲突的解释：论文 Table 2 里 SmoothQuant-O1/O2/O3 的权重全是 **per-tensor**，而本仓是 **per-输出行对称 INT8** (`src/smoothquant.py:50-53`)，粒度细得多 → $V$ 小得多 → 最优 α 应当**大于** 论文的 0.4–0.6。本仓实测最优在 α=0.75（三点扫描的上界），方向与该推导一致（EXP-003 §5）。

**这条推导的边界必须说清**：(a) 独立均匀舍入模型的前提在 §3.1.2 已列明失效点；(b) 激活是 per-token 动态量化，每个 token 有自己的 max，这里用全局 $P$ 作代理；(c) $x_j=a_ju_j$ 的分解是理想化，真实激活的形状因子随 token 变化；(d) 该式给出的是**逐（行， token） 的最优 α**，而 SmoothQuant 用的是全局 α——所以现实里的最优 α 是各行各 token 的折中。(d) 同时说明了一个可做的后续实验：像 AWQ 那样把 α 做成 per-linear 搜索。

#### 3.5.5 α 扫描的形状与那个反例臂

本仓 α∈{0.25,0.5,0.75} 三点扫描给出 0.5B 上的形状：单调向好、最优点偏大（权重侧余量足），而 α=0.25 比不迁移还差——迁移不足以救激活、却已开始伤权重（EXP-003 §6）。用 §3.5.4 的式子看：α 太小时激活侧步长 $P^{1-\alpha}$ 仍被 outlier 撑着（$S_1$ 很大），而 $V^2S_2$ 已经开始上涨，**两头不讨好**。 **这个反例臂必须留在表里**：只报单调向好的臂，读者无法排除"α 随便设都行" 这个竞争假设；留一个明确更差的点，"α 是真实的权衡旋钮"才成为可证伪的主张。

### 3.6 INT4 打包与反量化的数值细节

#### 3.6.1 nibble 打包的位序契约

两枚 INT4 码合一字节：偶列进低 4 位、奇列进高 4 位（`src/quant_linear.py:35`），dequant 用 `& 0xF` / `>> 4` 拆开再交错写回（`src/quant_linear.py:46-50`）。这里有三条约定必须成对出现，错一处就静默出错：

①**配对方向**：`0::2`（偶列）配低 nibble、`1::2`（奇列）配高 nibble。写反的后果是 dequant 出的权重列序**两两对调**——forward 不报错，PPL 灾难性劣化。 ②**无符号语义**：qidx 是 `uint8` 且值域 [0,15]，`<< 4` 不溢出、`>> 4` 是逻辑右移；换成 `int8` 存则 `>> 4` 变算术右移、高位符号扩展，高 nibble 解错。 ③**列数偶数**：见 §3.1.1 最后一条。

生产内核在这一层还要多做一件本仓不做的事：按 mma fragment 的线程—元素映射 **预置换**权重，否则反量化后还要跨 lane 洗牌（Marlin 的 "bespoke quantization support"，arXiv:2408.11743）。本仓 forward 走通用 `torch.nn.functional.linear`，没有这个约束——**打包格式的自由度取决于下游 kernel，不取决于量化算法**。

#### 3.6.2 fp16 的 ulp 与那个容差公式的完整推导(本讲义推导)

IEEE 754 binary16:1 符号位 + 5 指数位 + 10 尾数位，含隐含位的精度 $p = 11$。对 $|w| \in [2^e, 2^{e+1})$，ulp $= 2^{e-10}$，故**相对 ulp** $\in (2^{-11}, 2^{-10}]$；round-to-nearest 的绝对误差不超过半个 ulp，即 $\le |w|\cdot 2^{-11}$。

现在数一数 pack↔fake 比较路径上有几次 fp16 舍入：①**fake 侧**——GPTQ 把 $Q$ 写回 `layer.weight`(fp16)，一次（`src/gptq.py:189`）；②**pack 侧的 scales**——存 half(`src/quant_linear.py:39`)，一次；③**pack 侧的 zeros**——也存 half，但 **z 是 [0,15] 的整数**，binary16 精确表示 2048 以内整数，**无损**； ④**反量化**——$(q-z)\cdot s$ 在 **float32** 域算（`src/quant_linear.py:57-58`）， 不引入新的 fp16 舍入。

于是最大绝对残差 $\le |w|\cdot 2^{-11} + |(q-z)s|\cdot 2^{-11} \approx |w|\cdot 2^{-10}$（第二项的 $(q-z)s$ 就是 $\hat w \approx w$）。取全矩阵上界即 $w_{max}\cdot 2^{-10}$——**这正是代码里的 `tol = max(1e-3, wmax * 2 ** -10)` (`scripts/run_w4a16.py:176-177`)**。

**所以这个魔法数是硬件格式决定的，不是扫出来的**：2 的指数来自 binary16 的尾数位数，系数 2（即 $2^{-11}\to 2^{-10}$）来自路径上**两次**独立的 fp16 存储。若哪天把 scale 改存 fp32，分子就该退回 $2^{-11}$；若中间再加一次 fp16 中转，就该放到 $3\cdot 2^{-11}$。**容差随实现路径变，不随经验变。**

**与实测对齐（本讲义推导）**：EXP-002 §7 记录 W·s 域的相对 ulp 为 1.7e-3， 反推该臂 $w_{max}\approx 1.7$；同臂实测最大 pack 误差 1.22e-3，落在 1.7e-3 的界内（占界的 72%）。EXP-001 口径下固定 1e-3 未被触发，反推该臂 $w_{max}\cdot 2^{-10}\le 1\mathrm{e}{-3}$ 即 $w_{max}\lesssim 1.0$；实测最大 7.3e-4，同样落在界内（约 75%）。两臂的"实测/上界"比接近，说明这个界既不松垮也没被击穿——**一个好的容差应当是紧的，而不是大到永远通过**。

#### 3.6.3 为什么反量化在 float32 域

`((q.float() - self.zeros.float()[:, g]) * self.scales.float()[:, g])` (`src/quant_linear.py:57-58`)刻意升到 fp32 再算。若留在 fp16 域，乘法本身又引入一次舍入，断言残差就变成"三次舍入的和"，**无法干净归因到"scale 的 half 存储"这一单一来源**，§3.6.2 那条界也就不再是紧的。这是公理 C（每个误差都要有账）在一行代码上的体现。

### 3.7 PPL 作为量化评价指标的局限

#### 3.7.1 本仓协议的六个参数与 PPL 的三条结构性局限

$\mathrm{PPL} = \exp\big(\tfrac{1}{N}\sum_t -\log p(x_t \mid x_{<t})\big)$。本仓协议（`scripts/run_w4a16.py:204-224`）的六个参数各守一样东西：窗口 2048 让每窗跑满上下文量级；步长 1536 使相邻窗重叠 512 token 只作上文不计分（若步长等于窗口，每窗前几个 token 几乎没有上文，NLL 被系统性抬高；代价是计算量增加 $2048/1536\approx1.33$ 倍）；尾部不足一窗即停，保证各臂计分集逐位相同（298302）；移位对齐使位置 t 预测 t+1、首 token 不计分；log_softmax 在 fp32 上做，防 fp16 下溢；无采样，单轮即可逐位复算。**这些参数一改绝对值就变**——所以本仓 PPL 只在协议内做臂间比较，不与文献绝对值对比。

三条结构性局限：

1. **只看 ground-truth token 的那一个概率。** 分布其余 151935 维怎么变， PPL 完全看不见。量化若把"第二名 token 的概率"从 0.3 打到 0.05，PPL 可能纹丝不动，而采样生成的行为已经变了。
2. **几何平均会互相抵消。** Dutta et al.("Accuracy is Not All You Need"， arXiv:2407.09141)指出 perplexity 可解读为 token 概率的几何平均的倒数， 部分 token 变差可被另一部分变好抵消；该文更进一步给出一个刺眼的构造： **加对称噪声不改变 PPL，而生成质量随噪声标准差下降**。
3. **口径敏感、跨实现不可比。** 窗口/步长/是否丢尾/tokenizer/是否 fp32 log_softmax 全都改数值。这是本仓反复强调"只作协议内臂间相对比较"的原因。

#### 3.7.2 该补什么:KL 与 flips

同一篇论文提出的两个补充指标正好对上上面三条：**KL 散度**（baseline 分布 vs 压缩后分布）覆盖整个分布而非单个 token；**flips**（答案由对变错或由错变对的比例）直接度量"用户能看见的行为变化"；论文报告两者相关性良好。

**本仓的诚实位置**：只有 PPL，没有 KL、flips 与下游任务。所以本仓能主张的只有"在这套确定性 PPL 协议内，机制 A 比机制 B 收回了更多缺口"，**不能**主张 "量化后模型对用户行为无影响"。补测入口很近：评测脚本已经在算 `log_softmax` 的全分布（`scripts/run_w4a16.py:218`），留下 fp16 臂的分布做参照即可。

## 4. 代码逐段走读

按 gptq 臂的执行顺序走 `scripts/run_w4a16.py` 与 `src/`，再补 AWQ/SmoothQuant 与 real-quant 链路。引文逐字拷贝自仓内现行代码，标注 文件：起-止行。每段固定四问：**这段在算法里担什么角色 / 关键行是哪几行 / 改错会怎样 / 它踩在哪条硬件或数值语义上**。

**第 1 段 · 校准输入截获**(`scripts/run_w4a16.py:72-91`)——逐层量化只需要第 0 层的输入，跑完整模型纯属浪费。用"哨兵异常"在第 0 层截断前向：

```python
    class Catcher(torch.nn.Module):
        def __init__(self, mod):
            super().__init__()
            self.mod = mod

        def forward(self, hidden_states, **kw):
            inps.append(hidden_states.detach())
            caches.append({k: v for k, v in kw.items()})
            raise RuntimeError("__captured__")  # 截获即中止,省掉后续层前向

    layer0 = model.model.layers[0]
    model.model.layers[0] = Catcher(layer0)
    for batch in calib:
        try:
            model(batch.to(dev))
        except RuntimeError as e:
            if "__captured__" not in str(e):
                raise  # 只吞哨兵,真实错误照常抛出
    model.model.layers[0] = layer0
    return inps, caches
```

角色：为逐层 sequential 量化备好第 0 层的（hidden_states， kwargs）。关键行 `raise RuntimeError("__captured__")` 抓到即弃剩余 23 层计算；except 里只吞哨兵——改成裸 `except: pass` 的话，真实 bug（如 OOM）会被静默吞掉，校准输入悄悄少一批，H 偏了都无从发现。后续每层量化完用量化后权重重算输出作为下层输入（`run_w4a16.py:189-199` 阶段 C）：下层的 H 见到的是前序误差已累积的真实分布， 改用 fp16 激活校准会系统性高估深层质量。

数值与显存语义：`caches` 存的是**原样的 kwargs**（rotary、mask 等），每层重放按批回填——position ids 或 mask 与捕获时不一致，H 统计的就不是同一个分布。`inps` 是 128 个（1, 2048, 896） 的 fp16 张量共 469 MB，阶段 C 再开一份 `new_inps`，峰值双份约 0.94 GB（本讲义推导）——这是 EXP-001 §2 那个 ~6GB 峰值里可解释的一大块，也是"校准集不能再放大 10 倍"的直接原因。

**第 2 段 · H 的累积**(`src/gptq.py:81-94`)——H = 2XX^⊤ 的运行均值形态：

```python
    @torch.no_grad()
    def add_batch(self, inp: torch.Tensor):
        # inp: (..., in) → (tokens, in);H 用移动平均保持数值尺度稳定:
        # 维护 H = (2/N)·Σ xxᵀ 而非裸 Σ——裸累加的量级随 token 数线性增长,
        # fp32 会"大数吃小数"。先把旧 H 缩到新总数占比,再加上按
        # sqrt(2/N) 预缩放的新批,恒等于全量均值(与到达顺序无关)。
        # 注:补偿量对 H 的整体缩放不变(H→cH ⇒ err×√c、Hinv 行×1/√c,
        # 乘积不变),故此归一只影响数值稳定与 percdamp 的相对含义,不改结果。
        inp = inp.reshape(-1, self.columns).float()
        n = inp.shape[0]
        self.H *= self.nsamples / (self.nsamples + n)
        self.nsamples += n
        inp = inp * math.sqrt(2.0 / self.nsamples)  # 2 来自 ∂²‖WX−QX‖²/∂W²
        self.H += inp.t() @ inp
```

角色：把约 26 万校准 token(128×2048，EXP-001 §2)的二阶统计压进（in，in） 矩阵。为什么均值而不是裸和：26 万 token 的外积裸累加量级线性膨胀，fp32 后来的批会"大数吃小数"；而补偿公式对 H 的整体缩放不变（注释里那行代数）， 归一只买数值稳定，不改结果。

**"缩放不变"这一步值得展开一次**（本讲义推导）：设 $H \to cH$($c>0$)，则 $H^{-1}\to H^{-1}/c$，其 Cholesky 上三角因子 $U \to U/\sqrt{c}$；于是 $\varepsilon/U_{jj}$ 放大 $\sqrt{c}$、$U_{j,j:}$ 缩小 $\sqrt{c}$，乘积不变——**补偿量与 c 无关**。唯一受影响的是 `percdamp` 的含义（它取"相对平均对角"），这正是两个设计必须配套的原因。

改错会怎样：漏掉 `self.H *=` 的旧值收缩，H 变成"越晚到的批权重越低"的错误加权；用 fp16 存 H 则病态层直接分解失败（`src/gptq.py:73-76` 把 dtype 写死， 不是可调项）。硬件账：down_proj 的 H 是 90.25 MiB、单层七个 108.6 MiB (§1.2)，不逐 Linear 释放则 24 层累计 2.5 GiB（本讲义推导）。

**第 3 段 · H 预处理与三步分解**(`src/gptq.py:109-118`、`126-129`)——死列、阻尼、再到 $H^{-1}$ 的上三角因子：

```python
            dead = torch.diag(H) == 0
            H[dead, dead] = 1.0
            W[:, dead] = 0.0
            # 阻尼:校准样本有限/通道强相关时 H 近奇异,H⁻¹ 在弱激发方向
            # 爆炸 → 补偿量爆炸反而毁掉权重。加 1% 平均对角的 ridge 保证
            # 正定可 Cholesky,同时抑制病态方向的过度补偿;取"相对平均
            # 对角"而非绝对常数,对 H 的整体量级自适应(percdamp=0.01 为
            # GPTQ 论文默认,EXP-001 同设)。
            damp = percdamp * torch.mean(torch.diag(H))
            H += torch.eye(self.columns, device=self.dev) * damp
```

```python
            # Hinv 的上三角 Cholesky 因子:对角元即 [H^-1]_jj^(1/2) 的行主元
            Hinv = torch.linalg.cholesky(H)
            Hinv = torch.cholesky_inverse(Hinv)  # 经 Cholesky 求逆,比直接 inv 稳
            Hinv = torch.linalg.cholesky(Hinv, upper=True)
```

角色：一次性算出 §3.3.7 的 U。死列（校准集中恒为零的输入通道）无二阶信息， 对角置 1 保可分解、权重清零省码域；三步分解即 分解 H → 由因子求 $H^{-1}$ → 对 $H^{-1}$ 再做上三角分解得 U。

**三步各自的必要性**：`cholesky(H)` 既是求逆的前置，也是"H 是否真正定"的运行时检查（失败即说明阻尼不够）；`cholesky_inverse` 而非 `torch.linalg.inv`，因为前者利用对称正定结构、避免通用 LU 的额外误差（§3.3.7）；`upper=True` 决定了后面按**行**取 $U_{j,j:}$ 的方向。

改错会怎样：少了阻尼，强共线层的 cholesky 直接抛错或补偿量爆炸；丢掉 `upper=True` 拿到下三角因子，行方向与"后缀自由集"错位、补偿方向整个转置， PPL 不报错但显著劣化——这类 bug 只有对照臂能抓出来。另一个易忽略的细节： `H[dead, dead] = 1.0` 是**花式索引的对角写法**，只写 $(k,k)$ 这些点；死列的整行整列本就全 0，把对角顶起来即可重新正定，误写成 `H[dead] = 1.0`（整行置 1）会破坏对称性，Cholesky 直接失效。

**第 4 段 · 逐列量化与两级补偿**(`src/gptq.py:155-172`、`177-184`)——算法心脏，blocksize=group_size 的对齐原因在此：

```python
            g = i1 // gs
            # 组参数取自当前(前序块误差已补偿到位的)权重——GPTQ+groups 标准行为
            scales[:, g], zeros[:, g] = self.quantizer.find_params(W1)

            for i in range(i2 - i1):
                w = W1[:, i]
                dq, q = self.quantizer.quantize(w, scales[:, g], zeros[:, g])
                Q1[:, i] = dq
                qidx[:, i1 + i] = q.to(torch.uint8)
                if use_hessian:
                    # OBQ 最优补偿 δ = −(w−dq)/[H_F⁻¹]_ii · [H_F⁻¹]_{i,i:},
                    # 用 U 行改写即下面两行(1/U[i,i] 进 err,方向在 U[i,i:])。
                    # 只向右传播(i:):列序固定后,右侧列尚未定码、还有自由
                    # 度吸收误差;已量化列的整数码不可再动——这就是误差
                    # 传播方向必然"向未量化侧"的原因。
                    err = (w - dq) / Hinv1[i, i]
                    W1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
                    Err1[:, i] = err
```

```python
            Q[:, i1:i2] = Q1
            # ── 阶段 3:块间 lazy 补偿 ──
            # 块内对 W1 已就地逐列更新;块外列则攒满一块后用一次 GEMM 批量
            # 补偿。不这样:每列对全宽 W 做 rank-1 更新,访存量 O(rows×in)
            # × in 次,纯带宽浪费——"块内立即 + 块间延迟"正是 GPTQ 对 OBQ
            # 的关键工程改造(数学结果与逐列全宽更新严格相同)。
            if use_hessian and i2 < self.columns:
                W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
```

角色：§3.3 推导的最终落地，`err = (w-dq)/Hinv1[i,i]` 与下一行合起来就是 $-(\varepsilon/U_{jj})\cdot U_{j,j:}$。**blocksize 必须等于 group_size**（`src/gptq.py:136` 断言）：组参数 find_params 在块头一次性确定，此刻该组所有列的"前序块误差"已通过块间 lazy 更新传播到位——组网格描述的正是即将被量化的那批数值；若 blocksize≠gs，组横跨块边界，scale/zero 取自"部分已补偿、部分未补偿"的混合权重，与实际被量化的数值系统性错位。`use_hessian=False` 时跳过全部补偿路径、其余一切共享——这个开关就是 EXP-001 的实验设计本身。

改错会怎样：把 `W1[:, i:]` 写成 `W1[:, i+1:]` 看似更合理（别改自己），实际等价（第 i 列的更新在赋值 Q1 之后不再被读）；但把块间更新的 `Hinv[i1:i2, i2:]` 切片切错一列，后续所有组的网格都建立在错位的权重上，误差逐块滚雪球。

**两级补偿的分工**（对应 §3.3.8）：块内 `W1[:, i:] -= ...` 是 rank-1 就地更新，只碰 $\mathrm{rows}\times B$ 的小块；块外 `W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]` 是一次 GEMM，把整块 128 列的误差一次推给右边所有列。两者数学等价于"逐列全宽更新"但访存差 $O(B)$ 倍；浮点求和序不同，所以**逐位不同、数学相同**——做对拍时这个区分要紧。性能语义：内层 `for i in range(i2 - i1)` 是 Python 循环，列数就是迭代次数，**成本 ∝ 列数而非参数量**，§5.6 用原始数据把这条验证到 3% 以内。

**第 5 段 · AWQ 的 α 网格搜索**(`src/awq.py:104-124`)——一阶方法的全部机关：

```python
        best = (None, float("inf"), 0.0)
        for i in range(self.n_grid):
            # α 网格 {0, 0.05, …, 0.95}:α=0 即 s≡1(退化为 RTN),网格必含
            # "不保护"选项,搜索结果不会劣于 RTN 起点;α=1 为纯激活幅值
            # 主导(常过度),不在网格内
            alpha = i / self.n_grid
            s = base.pow(alpha)
            # 面试点:几何归一 s/√(smax·smin) 让 log s 关于 0 对称。不归一
            # 时 s 整体 >1(或 <1)会系统性撑大(缩小)每组的 min-max 范围,
            # 各 α 的 MSE 差异混入"网格整体变粗/变细"的混杂因素,α 之间
            # 不再可比——归一后比较的才是"幅值再分配"本身。
            s = s / (s.max() * s.min()).sqrt()        # 几何归一,防网格漂移
            fq, *_ = group_fakequant(W * s.unsqueeze(0), self.group_size,
                                     self.quantizer)
            # 打分对象 = Q(W·s)/s 这一"有效权重":激活不动时它与部署形态
            # (激活除 s)端到端等价,MSE 即真实输出误差
            w_eff = fq / s.unsqueeze(0)
            loss = ((X @ w_eff.t()) - ref).pow(2).mean().item()
            if loss < best[1]:
                best = (s, loss, alpha)
        return W, best
```

角色：对每档 α 构造 s、fake-quant、按输出 MSE 打分取最优。三个关键行：α=0 必在网格内（结果不劣于 RTN 起点）；几何归一（§3.4.4）；打分对象是 `fq / s`——除回 s 的**有效权重**，与部署形态端到端等价。改错会怎样：打分时忘了除回 s 就是在比较两个不同坐标系的输出，α 越大 MSE 天然越大，搜索永远选 α=0，AWQ 静默退化为 RTN。量化内核用 `group_fakequant`(`src/awq.py:42-62`)，纯 RTN、与 GPTQ 臂共用同一 GroupQuantizer：收益只可能来自 s 预缩放，归因才干净。

成本语义：每档 α 一次 `(tok,in)×(in,out)` 打分 GEMM。以 down_proj 为例， `x_sample` 封顶 4096 token(`src/awq.py:70`)，单次 $2\times4096\times4864\times896 = 3.57\mathrm{e}10$ FLOP，20 档 $7.1\mathrm{e}11$， 24 层合计 $1.7\mathrm{e}13$ FLOP（本讲义推导）。这解释了 AWQ 为什么比 RTN 贵却比 GPTQ 便宜：它有 GEMM，但没有 26 万 token 的 $X^\top X$。

**第 6 段 · SmoothQuant 的迁移与双端 fake quant**(`src/smoothquant.py:80-101`)：

```python
        # wmax 沿 dim=0(跨输出行)取:s 是输入通道方向的量,必须与
        # actmax 同轴;clamp(1e-4) 三处都在防死通道/零权重列把 s 推向
        # 0 或 ∞(等价变换在数学上允许任意正 s,数值上不允许)
        wmax = W.abs().amax(dim=0).clamp(min=1e-4)     # 每输入通道
        s = actmax.clamp(min=1e-4).pow(alpha) / wmax.pow(1.0 - alpha)
        s = s.clamp(min=1e-4)
    else:
        s = torch.ones_like(actmax)
    # 权重半边:先乘 s(接下难度)再量化——此后 weight 里存的是
    # fakeq(W·s);若先量化再乘 s,迁移就发生在网格确定之后,完全无效
    linear.weight.data = fakeq_int8_per_row(W * s.unsqueeze(0)).to(
        linear.weight.dtype)

    # 预先取倒数并降到权重 dtype:每次 forward 只做一次逐元素乘,
    # 不在热路径重复除法/类型转换
    inv_s = (1.0 / s).to(linear.weight.dtype)

    def pre_hook(mod, args):
        # 激活半边:先除 s(x·inv_s)再 per-token 量化。顺序不可换——
        # 量化若在除 s 之前,scale 仍被原 outlier 撑爆,迁移白做
        x = args[0]
        return (fakeq_int8_per_token(x * inv_s),) + args[1:]
```

角色：§3.5 公式的落地，两侧各一半。两处"顺序不可换"是同一件事的镜像：权重先乘 s 再量化（网格在迁移后的分布上确定），激活先除 s 再量化（scale 在 outlier 已消解的分布上取）；哪边顺序反了迁移就在那边完全失效，且不报任何错， 只有 PPL 对照能抓到。`alpha < 0` 走 else 分支 s≡1，naive 臂与 smooth 臂共用同一改装管线，唯一差异 = 是否迁移（EXP-003 控制变量设计）。

三处 `clamp(min=1e-4)` 各防什么：`wmax` 的防某输入通道权重整列为 0 时 $s\to\infty$；`actmax` 的防死通道 $0^\alpha=0$ 让 $s\to0$；对 s 本身的是兜底。 **等价变换在数学上允许任意正 s，数值上不允许**——这句注释是整段题眼。硬件语义：两个 `amax` 的轴由 §3.5.2 的外积约束决定；`/127.0` 而非 `/128.0` 则是让码域对称到 [-127,127]——s8 按补码解释，-128 没有正的对应值，留着它会把 $\pm$ 不对称的舍入偏差带进累加。

**第 7 段 · INT4 打包与反量化**(`src/quant_linear.py:34-39`、`45-58`)—— real quant 闭环的存储半边：

```python
        # 偶列进低 4 位、奇列进高 4 位:与 dequant 的 0::2 / 1::2 切片互为逆
        packed = (qidx[:, 0::2] | (qidx[:, 1::2] << 4)).contiguous()
        self.register_buffer("qweight", packed)          # (out, cols/2) uint8
        # scale/zero 存 half:模拟真实部署的存储 dtype;pack_check 的非零
        # 误差(~1e-4 量级)即源于这次降精度,见 run_w4a16 的幅值感知容差
        self.register_buffer("scales", scales.half())    # (out, n_groups)
```

```python
    def dequant(self) -> torch.Tensor:
        lo = self.qweight & 0xF
        hi = self.qweight >> 4
        q = torch.empty(self.out_features, self.in_features,
                        device=self.qweight.device, dtype=torch.uint8)
        q[:, 0::2], q[:, 1::2] = lo, hi   # 交错写回原列序,与打包严格互逆
        # g[c] = c // group_size:每列查自己的组参数;scales[:, g] 是花式索引
        # gather 出 (out, in) 全尺寸矩阵——组粒度反量化的向量化写法
        g = torch.arange(self.in_features, device=q.device) // self.group_size
        # 面试点:(q−zero)·scale 在 float32 域算而非 half——若在 fp16 域,
        # 乘法舍入会叠进误差,pack_check 就无法把残差干净地归因于
        # "scale/zero 的存储精度"这一单一来源
        return ((q.float() - self.zeros.float()[:, g])
                * self.scales.float()[:, g])
```

角色：fake quant 只证明"数值网格受得了"，不证明"码真的能紧凑存下并原样取回"。两枚 INT4 码合一字节（偶列低 nibble、奇列高 nibble），dequant 的切片互逆还原；高低 nibble 写反的话 dequant 出的权重列序两两对调——forward 不报错，PPL 灾难性劣化，这类 bug 恰是下一段逐元素断言存在的理由。反量化在 float32 域做乘法，让断言残差**干净地归因于单一来源**（scale 的 half 存储）， 展开见 §3.6.2/§3.6.3。

`g = torch.arange(in_features) // group_size` 是组粒度反量化的向量化写法： 把（out， n_groups） 的 scale 表 gather 成（out， in） 全尺寸矩阵。代价是 **物化了一个与权重同样大的 fp32 中间张量**——教学实现可以接受，生产内核绝不会这么干（它们在寄存器里按需取组参数，见 §8.2）。

**第 8 段 · pack↔fake 断言与幅值感知容差**(`scripts/run_w4a16.py:172-178`)：

```python
                err = pack_check(linears[n].weight.data, res["qidx"],
                                 res["scales"], res["zeros"], group_size)
                # 容差幅值感知:fp16 尾数 10 位,权重经 fp16 存储的舍入
                # ~|w|·2^-10;W·s 域幅值可远超 1,固定 1e-3 会误报
                wmax = float(linears[n].weight.data.abs().max())
                tol = max(1e-3, wmax * 2 ** -10)  # fp16 相对 ulp,幅值感知
                assert err < tol, f"pack mismatch {err} (tol {tol})"
```

角色：对每个 Linear 断言 dequant(pack(qidx)) 与 fake-quant 权重逐元素一致。为什么断言能成立：给定同一（q， zero， scale），反量化是确定映射，打包只搬比特不碰数值，残差仅可能来自 half 存储的 ulp 级舍入（`src/quant_linear.py:65-77`）。容差公式的完整推导见 §3.6.2，容差工程的教训见 §6 误区四。

断言设计本身也有讲究：它比较的是**全矩阵最大绝对误差**(`.abs().max()`) 而非均值。均值会稀释掉一处严重错位——高低 nibble 写反时约一半元素错，均值仍可能不显眼，最大值却立刻爆表。**验证"确定性映射"用最大值，验证"统计性质"才用均值。**

**第 9 段 · 叠加臂的恒等拆分**(`scripts/run_w4a16.py:154-163`、`166-171`)——两个机制怎么在一条流水线上不打架：

```python
                def mk(g, n):
                    # 工厂函数按值绑定 (g, n),同上防闭包晚绑定;
                    # awq_gptq 时 hook 现场把激活换到 X/s 域再喂 H
                    if n in awq_s:
                        inv = (1.0 / awq_s[n]).half()
                        return lambda mod, args: g.add_batch(args[0] * inv)
                    return lambda mod, args: g.add_batch(args[0])
                _run_block(layer, inps, caches, [
                    m.register_forward_pre_hook(mk(gptq[n], n))
                    for n, m in linears.items()])
```

```python
                if n in awq_s:
                    # 临时切到 W·s 域做 GPTQ:qidx/scale/zero 描述的是该域
                    linears[n].weight.data = (
                        linears[n].weight.data.float()
                        * awq_s[n].unsqueeze(0)).half()
                res = gptq[n].quantize(use_hessian=(mode != "rtn"))
```

角色：awq_gptq 臂先搜出 s、再把 GPTQ 作用在 $W\cdot s$ 上。恒等拆分 $(Ws)(X/s)\equiv WX$ 要求**两个半边同步换域**：权重半边在 `linears[n].weight.data = ... * awq_s[n]` 换到 $W\cdot s$；激活半边必须在 hook 里现场换到 $X/s$ 再喂 `add_batch`，否则 H 描述的是原始 $X$ 的二阶矩而被量化的对象已是 $W\cdot s$——**补偿方向对错了坐标系**。关键行是 `inv = (1.0 / awq_s[n]).half()` 与 `lambda mod, args: g.add_batch(args[0] * inv)`； 把 `inv` 删掉不会报错，PPL 会介于 AWQ 与 GPTQ 之间，**看起来"还行"因此最难被发现**。

另一处易错：`mk(g, n)` 是工厂函数，按值绑定 `(g, n)`。若直接在列表推导里写闭包引用循环变量 `n`，Python 的晚绑定会让 7 个 hook 全指向最后一个 Linear——7 个 H 全变成 down_proj 的统计，同样静默（`scripts/run_w4a16.py:115-116` 对 AWQ 侧的同一问题也留了注释）。量化完再除回 s (`scripts/run_w4a16.py:179-183`)得到评测态权重 $Q_{gptq}(W\cdot s)/s$， 激活侧零改动——与 AWQ 单独臂形式一致，两臂因此可比。

**第 10 段 · PPL 评测的三处口径**(`scripts/run_w4a16.py:208-223`)：

```python
    nll, count, pos = 0.0, 0, 0
    # 尾部不足一窗即停:丢掉尾巴保证各臂计分 token 集完全一致(298302)
    while pos + WINDOW <= len(ids):
        window = torch.tensor(ids[pos:pos + WINDOW]).unsqueeze(0).to(dev)
        logits = model(window).logits.float()  # fp32 做 log_softmax,防 fp16 下溢
        # 滑窗去重:首窗全计分;其后每窗前 WINDOW−STRIDE=512 token 已在
        # 上一窗计过,只作上下文——每 token 恰计一次且至少带 512 token 上文
        score_from = 0 if pos == 0 else WINDOW - STRIDE
        # 移位对齐:位置 t 的 logits 预测 token t+1,故 logits 去尾一位、
        # target 去头一位(首 token 无人预测,不计分)
        lp = torch.log_softmax(logits[0, :-1], dim=-1)
        tgt = window[0, 1:]
        tok_nll = -lp.gather(1, tgt.unsqueeze(1)).squeeze(1)
        nll += tok_nll[score_from:].sum().item()
        count += tok_nll[score_from:].numel()
        pos += STRIDE
```

角色：把"同一计分 token 集"写进代码而非口头约定。三处口径逐一对应 §3.7.1： `while pos + WINDOW <= len(ids)` 丢尾；`score_from` 做滑窗去重； `logits.float()` 让 log_softmax 在 fp32 上做。`lp.gather(1, tgt.unsqueeze(1))` 取每个位置 ground-truth token 的对数概率——**PPL 的全部信息就是这一列数**， §3.7.1 的三条局限都源于此；想补 KL/flips，入口就在上一行的完整 `lp`。

改错会怎样：把 `score_from` 恒设为 0，重叠的 512 token 会被计两次，PPL 仍 "能算出来"，但计分 token 数不再是 298302，臂间比较失去共同基准；而且重叠段上文更长、NLL 更低，PPL 会系统性偏低。数值语义：`log_softmax` 必须在 fp32 上做——fp16 最小规格数约 6.1e-5,151936 维 softmax 的尾部概率轻易低于它， 直接取对数会得到 -inf 并污染整个 NLL 求和。

**第 11 段 · actmax 一次收齐**(`scripts/run_w8a8.py:44-58`)：

```python
            key = (li, n)
            stats[key] = torch.zeros(m.in_features, device=dev)

            def mk(k):
                # 工厂函数按值绑定 key,防闭包晚绑定(同 run_w4a16)
                def hook(mod, args):
                    x = args[0].reshape(-1, args[0].shape[-1]).float()
                    stats[k] = torch.maximum(stats[k], x.abs().amax(dim=0))
                return hook
            handles.append(m.register_forward_pre_hook(mk(key)))
    for batch in calib:
        model(batch.to(dev))
    for h in handles:
        h.remove()
    return stats
```

角色：W8A8 的标定形态，与 W4A16 的逐层 sequential 完全不同——这里挂满全部 $24\times7=168$ 个 pre-hook，跑一遍完整模型就收齐所有通道统计。

**为什么可以这么做**：$s$ 只依赖激活统计、不含任何量化误差项；而 GPTQ 的 $H$ 必须在"前序层已量化"的分布上收集，只能逐层来。这是两条标定路线的本质分野，不是实现偷懒。**为什么必须在改权重之前**：先改权重再收 actmax，收到的是"已被 INT8 量化过的激活分布"，$s$ 围绕变形后的分布来定，迁移强度失准（`scripts/run_w8a8.py:77` 把这条顺序写死）。

`torch.maximum(stats[k], x.abs().amax(dim=0))` 取的是**校准集内的最坏值**， 与 SmoothQuant 论文 §4 Equation 4 的 $\max(|X_j|)$ 同口径。用 max 而非分位数的理由：$s$ 要保证迁移后激活侧 absmax 可控，分位数会漏掉恰被 per-token absmax 命中的那个极端 token；代价是对单个异常样本敏感——**这是一个可以做消融、本仓未做的旋钮**。

## 5. 实验数据怎么读

### 5.1 主对照表

全部单轮确定性 greedy scoring；协议：Qwen2.5-0.5B，wikitext-2 PPL， 窗 2048/步 1536，各臂同 298302 计分 token；PPL 为自定义协议，只作协议内臂间相对比较，不与文献绝对值对比：

| 臂 | PPL | Δ vs fp16 | 恢复率 | 出处 |
|---|---|---|---|---|
| fp16 | 11.9152 | — | — | EXP-001,`data/raw/EXP-001/fp16_g128.json` |
| RTN INT4-g128 | 14.1154 | +2.2002 | 0%（定义基线） | EXP-001,`rtn_g128.json` |
| AWQ | 13.4127 | +1.4975 | 31.9% | EXP-002,`awq_g128.json` |
| GPTQ | 12.7600 | +0.8448 | 61.6% | EXP-001,`gptq_g128.json` |
| AWQ+GPTQ | 12.7376 | +0.8224 | 62.6% | EXP-002,`awq_gptq_g128.json` |
| naive W8A8 | 12.1227 | +0.2075 | 0%（W8A8 基线） | EXP-003,`naive.json` |
| smooth α=0.25 | 12.2332 | +0.3180 | 为负（反例臂） | EXP-003,`smooth_a0.25.json` |
| smooth α=0.50 | 12.0394 | +0.1242 | 40% | EXP-003,`smooth_a0.5.json` |
| smooth α=0.75 | 12.0221 | +0.1069 | 48% | EXP-003,`smooth_a0.75.json` |

### 5.2 恢复率怎么算(列算式)

定义：收回基线缺口的比例，W4A16 赛道以 RTN 为 0%、fp16 为 100%（`scripts/plot_recovery.py:5-7` 同定义）：

- GPTQ:$(14.1154-12.7600)/(14.1154-11.9152) = 1.3554/2.2002 = 61.6\%$
- AWQ:$(14.1154-13.4127)/2.2002 = 0.7027/2.2002 = 31.9\%$
- 叠加：$(14.1154-12.7376)/2.2002 = 1.3778/2.2002 = 62.6\%$
- smooth α=0.75（W8A8 赛道，以 naive 为基线）： $(12.1227-12.0221)/(12.1227-11.9152) = 0.1006/0.2075 = 48.5\%$， EXP-003 §6 取整表述为 48%（图 fig2 脚注同此说明）。

**为什么用"恢复率"而不是直接报 PPL 差。** PPL 差与基线绝对水平耦合：同样 +0.1，在 PPL 12 上和 PPL 30 上意义完全不同。恢复率把两个端点（自家基线与 fp16）固定住，得到 [0,1] 的无量纲比例，臂间可比。**代价是不能跨赛道读**： W8A8 的 48% 收的是 +0.2075 的小缺口，W4A16 的 61.6% 收的是 +2.2002 的大缺口。

### 5.3 这个实验设计防了哪些坑

①**控制变量到开关级**：RTN 与 GPTQ 共用同一 GroupQuantizer、同一 find_params、同一网格，唯一差异是 use_hessian 布尔值（`src/gptq.py:99`）； AWQ 的量化内核是纯 RTN，收益只可能来自 s。归因链条"数字差异 → 机制开关" 中间没有任何别的变量。

②**反例臂主动保留**：α=0.25 比 naive 更差不是失败，是 α 作为真实权衡旋钮的证据——只报单调向好的臂，读者无法排除"α 随便设都行"。

③**同一计分 token 集**：eval 尾部不足一窗即停（`scripts/run_w4a16.py:209-210`）， 各臂 298302 个 token 逐位相同；滑窗去重保证每 token 恰计分一次、且至少带 512 token 上文。

④**单轮的资格**：关键数字通常要求多轮 mean±std，本仓 PPL 为确定性 greedy scoring（无采样、seed 固定、fp32 log_softmax），同机重跑逐位一致，EXP-001 §6 据此显式豁免——"单轮"限定语因此必须随数字出现，这是豁免的对价。 ⑤**代码共享而非协议复述**：W8A8 脚本直接 `from run_w4a16 import ... eval_ppl`(`scripts/run_w8a8.py:27`)，两赛道的 PPL 协议一致性由**共用同一份代码**保证；口头约定会漂，import 不会。

### 5.4 图怎么读

fig2(`figures/fig2_recovery_rates.png`)：横轴恢复率 0-100%，各赛道内以自家基线为 0%、fp16 为 100% 归一——两赛道的条**不可跨读**（理由见 §5.2）。 fig1(`figures/fig1_awq_alpha_dist.png`)：横轴是 per-layer best-α（网格 0-0.95、步 0.05，无量纲），纵轴是 Linear 层数，n=168（24 层×7 Linear）；读点有三：中位 0.30、主体 0.15-0.45 说明多数层只要温和保护；两层顶到上限 0.95 是强 outlier 层被自动识别（"顶到上限"也提示可能欠搜索，EXP-002 §7）；α=0 恰 1 层——该层激活均匀，AWQ 无事可做。

### 5.5 α 分布的完整直方图(从 raw 复算)

把 `data/raw/EXP-002/awq_g128.json` 的 168 条 per-layer 记录按 α 计数（复算自 raw，与 fig1 同源）：

| α | 层数 | | α | 层数 |
|---|---|---|---|---|
| 0.00 | 1 | | 0.35 | 23 |
| 0.05 | 1 | | 0.40 | 20 |
| 0.10 | 1 | | 0.45 | 21 |
| 0.15 | 11 | | 0.50 | 3 |
| 0.20 | 20 | | 0.55 | 6 |
| 0.25 | 25 | | 0.60 | 2 |
| 0.30 | 32 | | 0.95 | 2 |

三条只有看完整直方图才能读出来的东西：

1. **主体极窄**：0.15–0.45 七档合计 152 层，占 90.5%——AWQ 在九成的 Linear 上只做"温和保护"，不是一个到处开大的方法。
2. **0.65–0.90 是空档**：六个档位一层都没有。若那两个 0.95 的层只是"稍微更需要保护"，本该看到过渡层。空档意味着这两层与其余 166 层**不是同一分布**， 而且顶在边界上——**"最优恰在 0.95"和"被边界截断"这两种解释，现有数据无法区分**。EXP-002 §7 记为可能欠搜索；这里给出判据：把网格上界放到 1.0 以上重搜那两层，看它们是否继续往外跑。
3. **α=0 只有 1 层**：保底选项被用到但极少——若 s 预缩放是无效操作，α=0 应当在很多层上胜出。

### 5.6 时间账:从 raw per-layer 复算的成本结构

`data/raw/EXP-001/{gptq,rtn}_g128.json` 的 `per_layer` 数组里每条都带 `sec`（单个 Linear 一次 `quantize()`+`pack_check()` 的墙钟）。按 name 汇总 24 层（复算自 raw；每条 sec 已被四舍五入到 2 位，故合计有 0.1 s 级舍入）：

| Linear | 列数（in） | 参数量 | GPTQ 臂 24 层合计 | RTN 臂 24 层合计 | GPTQ/RTN |
|---|---|---|---|---|---|
| q_proj | 896 | 802,816 | 38.17 s | 2.87 s | 13.3 |
| k_proj | 896 | 114,688 | 6.46 s | 2.63 s | 2.46 |
| v_proj | 896 | 114,688 | 6.46 s | 2.63 s | 2.46 |
| o_proj | 896 | 802,816 | 6.49 s | 2.64 s | 2.46 |
| gate_proj | 896 | 4,358,144 | 6.50 s | 2.65 s | 2.45 |
| up_proj | 896 | 4,358,144 | 6.52 s | 2.65 s | 2.46 |
| down_proj | 4864 | 4,358,144 | 35.51 s | 14.03 s | 2.53 |
| 逐 Linear 合计 |— |— | 106.11 s | 30.10 s |— |
| 臂总时长（JSON `quant_seconds`） |— |— | 135.4 s | 30.0 s |— |

**结论一：成本 ∝ 列数，不 ∝ 参数量。** gate_proj 与 down_proj 参数量完全相同（都是 4,358,144），RTN 臂耗时却是 2.65 s vs 14.03 s，比值 5.29；而它们的**列数**比是 4864/896 = 5.43，吻合到 3% 以内（本讲义推导）。q_proj 参数量只有 gate_proj 的 18%，耗时却相当（2.87 vs 2.65）——同样因为列数相同。 **机制**：逐列 Python 循环每列固定发起若干小 kernel，单列成本几乎与行数无关（行数只影响 kernel 内部并行度，而这些 kernel 远未打满 GPU）。GPTQ 臂上同一关系依然成立：35.51/6.50 = 5.46 vs 5.43。

**结论二：补偿路径把单列成本抬到约 2.5 倍。** 除 q_proj 外，GPTQ/RTN 的比值稳定在 2.45–2.53。多出来的这一倍半来自每列的 rank-1 更新与那次 `.item()` 同步。这是"二阶信息的定价"在时间侧的直接读数。

**结论三：Cholesky 不是 GPTQ 溢价的来源（常见误解）。** 三步分解的算术量约 $1.67 d^3$（分解 $d^3/3$ + 由因子求逆 $\approx d^3$ + 再分解 $d^3/3$）。 $d=4864$ 时约 $1.9\mathrm{e}11$ FLOP，$d=896$ 时约 $1.2\mathrm{e}9$；每层七个合计 $2.0\mathrm{e}11$，24 层共 $4.8\mathrm{e}12$ FLOP。按 RTX 4090 的 FP32 峰值 82.6 TFLOPS（NVIDIA Ada GPU Architecture 白皮书 Appendix A 口径） 折算，下限约 **0.06 s**——即便实际只跑到峰值的 5%，也不过 1 秒出头。 **GPTQ 比 RTN 多出的 105 秒，不在求逆上。**（本讲义推导）

**结论四：溢价在校准重放。** $H$ 累积的算术量为每层 $6\times 2\times 896^2\times 262144 + 2\times 4864^2\times 262144 = 1.49\mathrm{e}13$ FLOP， 24 层共 $3.58\mathrm{e}14$ FLOP，全部在 fp32 上做（`src/gptq.py:89`），同一峰值口径下的下限是 **4.3 s**。而臂总时长减去逐 Linear 合计是 $135.4-106.11=29.3$ s； 再算上结论五指出的、被错记到 q_proj 名下的约 31.7 s，重放与 sequential 前推合计约 **61 s**，占臂总时长的 45%。下限 4.3 s 与实际 61 s 的差距说明这段远不在峰值：全模型要跑 $24\times7\times128 = 21504$ 次独立小 GEMM，K 只有 2048， 启动开销与尾波占比高（本讲义推导）。

**结论五：q_proj 的 38.17 s 是计时假象，不是算法成本。** 它比同形状的 k/v/o/gate/up 多出约 31.7 s。三条证据指向"异步执行下的计时归属"： (a) 七个 Linear 里只有 q_proj 异常，而它是每层量化循环的**第一个**； (b) RTN 臂完全没有这个异常（2.87 vs 2.63，仅 +9%），而 RTN 臂与 GPTQ 臂的唯一结构差异正是 **RTN 不跑校准重放**(`scripts/run_w4a16.py:151`)； (c) 循环里第一处强制同步的调用出现在 q_proj 的量化内部，于是整层重放排队的 GPU 工作被记到了 q_proj 名下。

**但这条归因还有不闭合处，如实写出**：AWQ 臂同样跑重放，其 q_proj 只比同伴多约 0.26 s（1.81 vs 1.55；`data/raw/EXP-002/awq_g128.json` 复算），远达不到"整层重放全额计入"的预期。可能的解释是 AWQ 的 hook 只做 absmean 累加（带宽级），重放代价本就远小于 GPTQ 的 $X^\top X$；但本仓**没有做隔离测量来确认**，此处留作开放问题，不下定论。

**可操作结论**：未插 `torch.cuda.synchronize()` 的逐算子 `time.time()` 差不能当作算子成本；只有臂级总时长（以结果落盘前的同步为界）可信。本仓对外只报臂级时长（RTN 30 s、AWQ 43 s、GPTQ 135 s；EXP-001/002 §5），这个选择在方法上是对的——**per_layer 的 sec 字段只该当作回归监控，不该当作性能数据**。

### 5.7 存储账与两种压缩比

见 §3.1.3 的推导。三个必须一起说的数：**4.25 bit/权重**（格式属性）、 **3.76×**（被量化部分的压缩比，= 16/4.25）、**2.14×**（全模型压缩比，含未量化的 tie embedding）。引用时说清是哪一种，否则就是在误导。

**还能再压吗？** 三条路都在精度与元数据之间换：①zero 用 4 bit 整数存（z∈[0,15] 本来就只要 4 bit）→ 省 12/128 = 0.094 bit；②scale 二次量化 → 可省到 0.1 bit 量级；③加大 g(g=256 → 4.125 bit)→ 但 §3.1.2 的极值统计说明格距会变粗、误差上升。本仓都没做，这是后续实验的自然方向。

### 5.8 数字背后的机理账

**为什么叠加只 +1.0pp(62.6% vs 61.6%)？** AWQ 修的是"显著通道误差被放大"， GPTQ 的输出空间补偿修的是"误差总量在列间的最优分配"——机制上正交（一个改坐标系、一个改取整），但两者救的都是同一批对输出伤害最大的权重。0.5B 上 GPTQ 已把 AWQ 能救的大部分吸收掉，叠加的边际收益只剩 +1.0pp(EXP-002 §6)。这解释了工程实践中两者通常二选一。

**为什么 AWQ 便宜而 GPTQ 贵？** 时间账：RTN 30 s、AWQ 43 s、GPTQ 135 s (EXP-001/002 §5)。由 §5.6：GPTQ 的溢价花在校准重放（26 万 token 过每层收 $X^\top X$）与逐列补偿路径上；AWQ 的溢价只是 20 档 α × 4096 token 的打分 GEMM，不需要 26 万 token 的二阶统计。**二阶信息的定价，时间侧同样成立。**

**为什么 W8A8 各臂全面优于 W4A16 各臂？** 位宽 8 > 4，精度当然更好（EXP-003 §6）。两赛道的取舍在速度侧（W4 省 decode 权重带宽 vs INT8 换算力路径），不在质量侧——把两条赛道的 PPL 直接比大小是范畴错误。

### 5.9 哪些数字能外推,哪些不能

| 数字 | 能外推的部分 | 不能外推的部分 |
|---|---|---|
| 61.6% / 31.9% / 62.6% | 序关系（二阶 > 一阶，叠加 ≈ 二阶） | 具体百分比：单模型、单任务、单协议、同源校准 |
| 48%(smooth α=0.75) | α 是权衡旋钮、存在内点最优 | 幅度：0.5B outlier 温和，大模型上缺口与收益都更大 |
| 4.25 bit / 3.76× | 格式属性，与模型无关 | 2.14×：随词表占比变，换模型必须重算 |
| ≤7.3e-4 / 1.22e-3 | 上界公式 $w_{max}\cdot2^{-10}$(§3.6.2) | 具体数值：随权重幅值与实现路径变 |
| 30/43/135 s | 三者的相对量级关系 | 绝对秒数：单卡 4090、本实现、本校准规模 |
| α 中位 0.30 | "保护强度按层自适应"这一定性结论 | 中位数本身：随模型与校准集变 |

## 6. 误区与边界

**误区一："权重误差小，模型就好。"** 本仓的直接反证：RTN 与 GPTQ 在同一网格上，逐元素误差同为 $s^2/12$ 量级，PPL 却差 1.36(14.1154 vs 12.7600， EXP-001 §5)——差的全部是"误差往哪些列放"。文献侧的同一课来自 AWQ §3.1 Table 1：按权重幅值挑 1% 保 FP16，效果和随机挑几乎一样（OPT-6.7B 22.37 vs 24.23），按激活幅值挑则一步拉回 11.39。**度量必须取在输出空间。**

**误区二："两个好方法叠加，收益近似相加。"** 31.9% + 61.6% 远大于实测的 62.6%(EXP-002 §5)。正交性是机制层面的（改坐标系 ⊥ 改取整），不保证收益可加——两者保护的是同一批难量化权重，0.5B 上高度重叠。措辞上"正交可叠加"只能说**方向成立**(EXP-002 §6)，不得说"显著提升"：+1.0pp 的增益配不上这个词。

**误区三："SmoothQuant 加了就好，α 越大越保险。"** 本仓的反例臂：α=0.25 的 PPL 12.2332 比 naive 的 12.1227 还差（EXP-003 §5）——迁移不足以救激活、却已开始伤权重；§3.5.4 的式子给出机制（α 太小时 $S_1$ 仍被 outlier 撑着， 而 $V^2S_2$ 已经开始涨）。且 0.5B 上 naive 缺口本来只有 +0.2075。48% 这个数字必须带"0.5B outlier 温和"定语，不得外推大模型幅度（EXP-003 §6）。

**误区四："断言容差设个 1e-3 就稳了。"** 本仓被实测证伪过两次（EXP-002 §7）： AWQ+GPTQ 臂在 W·s 域做 pack 校验，权重幅值被 s 撑大后，fp16 存储的相对舍入（~|w|·2⁻¹⁰）在 wmax≈1.7 时就是 1.7e-3——正确实现被固定容差误杀两次。修复是幅值感知容差 max(1e-3， wmax·2⁻¹⁰)(`scripts/run_w4a16.py:176-177`)；本仓口径因此是两段式的：EXP-001 口径最大误差 ≤7.3e-4，EXP-002 臂在幅值感知容差下最大 1.22e-3(`src/quant_linear.py:16`)。**容差必须与被检对象的数值幅度挂钩**，浮点世界里"绝对误差阈值"几乎总是错的。§3.6.2 把它从"经验修补"提到 "可推导"：指数 $2^{-10}$ 来自 binary16 的尾数位数，系数 2 来自比较路径上两次独立的 fp16 存储。

**误区五："per-layer loss 可以当质量指标排层。"** GPTQ 的逐列损失 $\varepsilon^2/(2[H_F^{-1}]_{jj})$ 数值量级 1e-20~1e-15(EXP-001 §7)：量纲随 H 的运行均值归一化缩放，未标定，仅可作层间相对诊断，不进任何表格。量化正确性由 PPL 与 pack 断言独立支撑。**同理适用于 per_layer 的 `sec` 字段**(§5.6)。

**误区六："PPL 不掉就说明模型没坏。"** PPL 只看 ground-truth token 的那一个概率，且是几何平均——部分 token 变差可被另一部分变好抵消（Dutta et al.， arXiv:2407.09141）。该文更给出一个刺眼的反例：加对称噪声不改 PPL，而生成质量随噪声标准差下降。**本仓只有 PPL，所以本仓能主张的只有"机制 A 比机制 B 收回更多缺口"，不能主张"量化后模型行为不变"。** 要补的是 KL 与 flips，入口在 `scripts/run_w4a16.py:218` 的完整分布上（§3.7.2）。

**误区七："校准集随便选，反正只是估个统计量。"** 本仓校准取 wikitext-2 的 **train**、评测取 **test**——不重叠，但**同源**。AWQ 论文 §5.3(Figure 8(b)) 报告：校准与评测换成不同分布（PubMed↔Enron）时 AWQ 只涨 0.5–0.6 PPL，而 GPTQ 涨 2.3–4.9；GPTQ 原论文本身用 **C4**（128 段 × 2048 token）去评 WikiText， 是有意的跨域设置。**所以本仓的同源校准很可能对 GPTQ/AWQ 相对 RTN 的优势偏乐观**，而本仓没有跨域臂来排除。这是最该补的一个对照，也是引用 61.6% 时必须一起说出口的限定。

**误区八："GPTQ 慢是因为要算 Hessian 的逆。"** §5.6 结论三把这条算死了： 168 次三步 Cholesky 的算术量在 4090 峰值口径下只有 0.06 s 量级，而 GPTQ 比 RTN 多花了约 105 秒。真正的开销是校准重放（$3.58\mathrm{e}14$ FLOP 的 fp32 $X^\top X$）和逐列 Python 循环。**猜瓶颈之前先算量级**，这是这条误区唯一的解法。

**适用边界（引用这些数字时限定语不许丢）。** ①全部结论出自 Qwen2.5-0.5B 单模型、wikitext-2 单任务、自定义 PPL 协议——臂间相对比较可信，绝对值不跨协议比较；②AWQ 数字是 per-linear 简化 + 无 clip 口径；③blocksize= group_size 是本仓实现约束（EXP-001 §7），act_order 未开，通用化未做； ④α 网格 0.95 封顶，两层顶到上界可能欠搜索（§5.5 给了区分判据）； ⑤SmoothQuant 对全部 7 类 Linear 施加迁移，与原论文"仅 post-LN linears" 口径不同（EXP-003 §2，实现选择）；⑥W8A8 的 α 只扫了三点，上界未封； ⑦校准与评测同源（误区七）；⑧速度侧一律未测——`QuantLinear4.forward` 是 "先反量化、再 fp GEMM"的教学实现，本仓不作任何吞吐主张。

## 7. 连环追问

**Q1 非对称量化的 zero-point 到底买到了什么？** 权重组 min/max 不对称时， 对称网格有半边码域低利用；z 把网格平移贴合真实区间，代价是 dequant 多一次减法（`src/gptq.py:63`）。追一层：z 为什么必须取整？dequant 要走整数减法、 z 要以整数存。再追一层：round(z) 的半格偏移去哪了？被 clamp 与 scale 一起吸收，且对全部码等量偏移，不改变相对分辨率。

**Q2 H = 2XX^⊤ 是什么的 Hessian？为什么全部输出行共享？** 是层输出 MSE 对单个输出行权重的 Hessian；目标 $\|\delta^\top X\|^2$ 展开后只含 X 不含行号， 所以每个 Linear 只分解一次、全部行共享（§3.3.4）。追一层：2 从哪来？ $\partial^2\|\delta^\top X\|^2/\partial\delta^2 = 2XX^\top$；代码里它在 `math.sqrt(2.0/nsamples)` 的**开方之内**，因为要作用在 X 上再自乘。

**Q3 为什么 Cholesky 一次分解就够，OBQ 却要逐列重算？** OBQ 按"当前伤害最小"动态挑列，自由集 F 的变化无规律，$[H_F^{-1}]$ 只能重算。GPTQ 固定列序后 F 恒为后缀，块矩阵逆恒等式 + Cholesky 的 Schur 补结构让 U 的每一行恰好携带对应后缀的 $[H_F^{-1}]_{jj}$ 与方向（§3.3.6-3.3.7）。追一层： GPTQ 论文说选 Cholesky 的首要理由是什么？是**数值**不是速度——反复消元会让 $H_F^{-1}$ 变成不定矩阵，在十亿参数级模型上"almost certainly occurs for at least a few layers"(§4 Step 3)。

**Q4 阻尼 percdamp 调大调小各会怎样？** 太小：病态层 Cholesky 失败或弱激发方向补偿爆炸；太大：H → λI，补偿失去各向异性信息、退化向 RTN。1% 平均对角是论文默认值（§4 Step 3），本仓沿用；**本仓没做 percdamp 扫描**，被问到具体拐点时应当直说没测。

**Q5 为什么 blocksize 必须等于 group_size？** 组参数必须描述"即将被量化的那批数值"。块间 lazy 更新以块为单位传播误差；组横跨块边界时，组内一部分列的误差已传播、另一部分没有，find_params 见到的是混合态，scale/zero 与实际量化值系统性错位（`src/gptq.py:131-136` 断言与注释，EXP-001 §7）。追一层： 这条约束还带来什么副作用？它与 act-order 直接冲突——act-order 会打乱列序， 要共存必须先实现 static-groups(§3.3.10)。

**Q6 AWQ 的 s 为什么用 absmean 不用 absmax？为什么要几何归一？** absmax 被单个极端 token 绑架，absmean 量的是通道系统性幅值（参考实现同口径： `x.abs().view(-1, x.shape[-1]).mean(0)`）；几何归一让各 α 档比较的是"幅值再分配"本身（§3.4.4）。追一层：不归一会怎样？α 越大 s 整体越偏离 1，每组 min-max 范围被系统性撑大或缩小，α 之间不再可比。再追一层：归一本身有坑吗？ 有——参考实现曾因 `scales.min()` 为 0 产生 inf(llm-awq issue #96)，本仓在取幂前先 `clamp(min=1e-4)` 堵死（`src/awq.py:102`）。

**Q7 awq_gptq 叠加臂里，GPTQ 的 H 为什么必须取自 X/s？** 恒等拆分 $(W s)(X/s) \equiv WX$：权重半边进 GPTQ 量化，激活半边就必须同步换到 X/s 域——H 描述的是量化对象 W·s 所面对的激活分布。若 H 仍用原 X，补偿方向对错了坐标系（`scripts/run_w4a16.py:136-139` 注释）。追一层：这个 bug 的症状是什么？不报错，PPL 落在 AWQ 与 GPTQ 之间，**看起来"还行"因此最难被发现**——只有和单独 GPTQ 臂对照才看得出不对。

**Q8 为什么激活不能 per-channel 量化，权重却能 per-输出行？** INT8 GEMM 要求 scale 呈行×列外积：激活行方向是 token（per-token 可提出），列方向是输入通道（在求和维内，提不出）；权重的输出行在求和维外，天然可提（`src/smoothquant.py:31-35`）。追一层：这条约束在硬件上对应什么？整数张量核指令在 k 维上做 s8×s8→s32 的累加，整个归约在整数域一口气做完，中间没有插浮点缩放的位置（§3.5.2）；cuBLAS 的整数计算类型 `CUBLAS_COMPUTE_32I` 的文档措辞是"uses compute and intermediate storage precisions of at least 32-bits"。

**Q9 部署时 AWQ/SmoothQuant 的 s 怎么做到零开销？约束是什么？** s 折叠进前置 RMSNorm/Linear 的权重，runtime 无额外算子。约束：共享同一输入的 linears(q/k/v；gate/up)必须共享同一个 s——这是折叠的约束不是算法的； 本仓 fake-quant 评测形式下 per-linear s 合法（`src/awq.py:29-33`）。追一层： 那本仓的 s 能直接拿去部署吗？不能，要先改成共享 s 重搜（§3.4.6）。

**Q10 4.25 bit/权重怎么算出来的？还能再压吗？** 4 bit 码 + (16+16) bit scale/zero 摊到 g=128:32/128=0.25。再压的路子见 §5.7 的三条。追一层： 那 4.25 bit 对应多少倍压缩？**要看口径**——被量化部分 3.76×，全模型只有 2.14×，因为 tie 的 embedding 占了 27.6% 的参数且不量化（§3.1.3）。

**Q11（压力）61.6% 能外推到 7B、外推到下游任务吗？** 不能直接外推，这是单模型单任务单协议的数字。方向性论据可以给：大模型参数冗余更多、RTN 缺口相对更小，文献中 GPTQ 相对优势依模型而变；但本仓没有 7B 实测，任务面也只有 PPL。诚实答法："在我的控制变量协议内，二阶补偿收回 61.6%；跨模型跨任务的量级要重测，我的协议与脚本可直接换模型复跑。"

**Q12（压力）你的 AWQ 只有 31.9%，比社区口碑差，是不是实现错了？** 首先承认口径差异：本仓是 per-linear 简化 + 无 clip，参考实现是块级 MSE + 共享 s + clip，数字不同口径不可直接比（EXP-002 §2/§6）。其次给证据链：α 分布形状合理（中位 0.30、outlier 层自动顶格）、α=0 档保底存在、pack 断言全过——实现正确性有独立支撑。最后给可证伪路径：补 clip 与共享 s 的增量是现成的后续实验设计。"数字低"与"实现错"之间，隔着口径与消融两层证据。

**Q13 H 为什么必须 fp32，bf16 不行吗？** 两个理由：①累加——26 万 token 的外积在 bf16（尾数 7 位）上累加会严重丢位，运行均值也压不住；②条件数—— Cholesky 的后向误差被 $\|A\|$ 乘机器精度所界，精度越低病态层越容易在分解阶段就崩（§3.3.7）。代码把 dtype 写死（`src/gptq.py:73-76`），不是可调项。

**Q14 补偿只向右传播，那最左边的列是不是最吃亏？** 恰恰相反——第一列虽然没有任何前序补偿，但它右边有全部 d−1 列可以吸收它的误差；**最后一列才是最吃亏的**，它的量化误差完全无人补偿。这正是 act-order 的动机：把 "硬方向"排到最前面，让它享有最大的补偿容量（§3.3.10）。

**Q15 叠加臂的 pack 断言为什么要对 Q(W·s) 而不是最终权重？** 因为最终权重是 $Q(W\cdot s)/s$，除回 s 之后它**已经不落在整数网格上**了，而 qidx/scales/zeros 描述的是 $Q(W\cdot s)$ 这个本体。对最终权重做逐元素对齐在数学上就不成立（`src/awq.py:139-143` 的注释即此）。

**Q16 W4A16 的 12.76 和 W8A8 的 12.02 能直接比大小说 W8A8 更好吗？** 位宽不同，比"谁 PPL 低"是范畴错误。两条赛道压的东西不同（权重带宽 vs 算力路径），受益的 regime 也不同。可比的是各自赛道内的恢复率，以及在给定部署约束（显存 / batch 规模 / kernel 可用性）下的选型。

**Q17 zero-point 存 half 会不会引入误差？** 不会。z 是 [0,15] 的整数， binary16 能精确表示 2048 以内的整数，这一步无损。pack↔fake 的残差**只可能来自 scale 的 half 存储与 fake 权重的 fp16 写回**两处——这正是 §3.6.2 那条容差界能推出来的原因。

**Q18 SmoothQuant 的 α 能不能像 AWQ 那样按层搜？** 能，而且 §3.5.4 的推导说明最优 α 本来就随（行， token） 变化，全局单一 α 是折中。本仓只做了全局三点扫描（EXP-003）。要注意搜索目标该取什么：SmoothQuant 的迁移同时影响两侧， 打分必须把权重侧与激活侧的 fake quant 都包进去才公平。

**Q19（压力）校准集和评测集同源，你的数字是不是偏乐观？** 很可能是，而且本仓没有排除它（证据链见 §6 误区七）。**同源校准偏向依赖校准集的方法**， 61.6% 与 31.9% 都可能被抬高，且 GPTQ 被抬高的幅度可能更大。补法很直接： 换一个域（如 C4 或 PTB）的校准集重跑同样五臂，其余全不动。

**Q20（压力）只让你上一个方法到生产，选哪个？** 先反问部署约束：激活要不要量化（决定 W4A16 还是 W8A8 赛道）、目标 batch 规模（决定带宽瓶颈还是算力瓶颈）、有没有可用的融合内核。在 W4A16 且有 Marlin 类内核的前提下，本仓数据支持先上 GPTQ（恢复率最高、时间成本可接受），把 AWQ 留作校准集易变或跨域场景的备选（依据是 AWQ 论文 §5.3 的跨域鲁棒性，**不是**本仓实测——本仓没有跨域臂）。这个回答的关键不在选谁，在于把"选择依据"和"证据来自哪儿"分清楚。

## 8. 工业对照与延伸

### 8.1 论文/文档怎么说 vs 本项目实测:逐条对照

本节把"论文或官方文档说了什么"与"本仓在单卡 RTX 4090 上测到什么"并排放， 并诚实分析差异来源。差异不粉饰：多数来自规模、口径与本仓的实现简化， 少数是本仓根本没测的项——那就写"本仓未测"，不借文献充数。

| # | 来源与声称 | 本仓实测（EXP 锚） | 差异分析 |
|---|---|---|---|
| 1 | GPTQ 摘要：175B 模型约 4 GPU 小时量化到 3-4 bit，"negligible accuracy degradation" | Qwen2.5-0.5B INT4-g128 退化 **+0.8448** PPL(EXP-001 §5)，量化 135.4 s | 规模差 350 倍：小模型参数冗余少，同位宽退化本就更大。**不矛盾，是 regime 不同**；"negligible" 属于百亿级模型的语境，不能搬到 0.5B |
| 2 | GPTQ 摘要：端到端推理提速约 3.25×(A100)/4.5×(A6000) | 本仓**未测速度** | `QuantLinear4.forward` 是"先反量化再 fp GEMM"的教学实现，只闭合正确性；速度主张需要融合内核，不在本仓范围 |
| 3 | GPTQ §4 Step 2/Step 3：块大小 B=128；阻尼取平均对角的 1% | 本仓 blocksize=128、percdamp=0.01(`src/gptq.py:97`) | **数值锚逐一对齐**，这是"论文 → 实现"一致的一条闭环。注意本仓额外加了 blocksize==group_size 的断言，论文没有这条 |
| 4 | GPTQ §5 设置：校准用 C4 的 128 段 × 2048 token，评测 WikiText（有意跨域） | 本仓校准 wikitext-2 **train**、评测 wikitext-2 **test**（同源，分割不重叠） | **口径不同，且方向对本仓不利地乐观**：同源校准偏向依赖校准集的方法。本仓没有跨域臂，该风险未被排除（§6 误区七） |
| 5 | GPTQ 仓库 README：`--act-order` 在 OPT-66B 上 4bit 9.55→9.34、3bit 14.16→9.95 | 本仓 act_order **未开**(EXP-001 §7) | 未实现，不是不同结论。本仓的 blocksize==group_size 约束与 act-order 直接冲突，要共存须先做 `--static-groups`(§3.3.10) |
| 6 | AWQ §3.1 Table 1：OPT-6.7B INT3-g128，RTN 23.54；保 1% 按激活挑 → 11.39，按权重挑 → 22.37，随机 → 24.23 | 本仓**未做混精度消融** | 口径不同（论文是 1% 保 FP16 的混精度，本仓是全量化 + 缩放）。同向的间接证据：本仓 per-layer best-α 由激活统计自适应决定，两层顶格 0.95(EXP-002 §5) |
| 7 | AWQ §3.2 Table 2：**单通道**放大 s=2 时平均 Δ′/Δ=1.038、误差比 0.519；s=4 时误差比 0.303 但 PPL 反弹到 12.36 | 本仓用的是**全向量** $s=a^\alpha$ + 几何归一，Table 2 的前提不成立 | 这是论文"直觉证明"与"落地算法"之间的缝。本讲义 §3.4.3 补出了全向量情形的上下界，结论是收益与代价被同一个量（组内 s 展布 $\rho=\rho_a^\alpha$）控制 |
| 8 | AWQ 摘要：TinyChat 相对 HF FP16 提速 >3× | 本仓**未测速度** | 同第 2 条 |
| 9 | AWQ §5.3 Figure 8(b)：跨域校准时 AWQ 只涨 0.5–0.6 PPL，GPTQ 涨 2.3–4.9；AWQ 用 16 条校准序列即可（GPTQ 用 192 条） | 本仓两法都用 128×2048，**未做数据效率/跨域消融** | 本仓无法证实或证伪该对比。它同时是本仓 61.6% vs 31.9% 这个序关系的最大外部风险源（§6 误区七） |
| 10 | SmoothQuant §5.5 Figure 10：α 甜区 0.4–0.6，默认 0.5；<0.4 激活难量化，>0.6 权重难量化；GLM-130B 因 outlier 更重取 0.75 | 本仓 α=0.75 最优（12.0221）、0.5 次之（12.0394）、0.25 **劣于 naive**(12.2332 vs 12.1227)，EXP-003 §5 | **三条差异来源**：①权重粒度不同——论文 Table 2 的 O1/O2/O3 权重全是 per-tensor，本仓是 per-输出行，按 §3.5.4 的 $V^2$ 项，粒度越细最优 α 越大，方向一致；②模型规模，0.5B outlier 温和；③本仓 α 只有三点且 0.75 是上界，**单调性尾部未封** |
| 11 | SmoothQuant §3：激活 outlier 幅值约为普通值的 ~100×，且固定在少数通道 | 本仓**未测 outlier 幅值倍数** | 只有间接证据：naive W8A8 仅退化 +0.2075(EXP-003 §5)，说明 0.5B 远没到那个量级。不作定量主张 |
| 12 | LLM.int8() §4.2:6B 与 6.7B 之间发生相变，相变后所有 transformer 层与 75% 的序列维度被极端幅值特征影响；§4：幅值可达其他维度的 20× | 本仓模型 0.5B，远在相变之下 | **这条正是本仓 SmoothQuant 收益小的外部解释**：不是实现问题，是模型尺度不在该方法的主战场。把它写进限定语，比含糊说"收益有限"诚实得多 |
| 13 | SmoothQuant §3：INT8 kernel 的缩放"can only be performed along the outer dimensions of the matrix multiplication" | 本仓 `src/smoothquant.py:31-35` 的注释是同一论断；代码轴选择与之一致（`amax(dim=-1)` / `amax(dim=1)`） | **文档/论文与实现完全一致**。cuBLAS 对这类向量缩放的官方命名就是 "Outer Vector Scaling"（文档 §3.1.4.3，该小节列在 FP8 下；INT8 是否有同名接口**未核实**） |
| 14 | CUDA C++ Programming Guide：wmma 的子字节类型 `experimental::precision::u4/s4` 与 `b1` 的 XOR 变体已弃用，并在 sm_90 移除 | 本仓不写 kernel，INT4 只作**存储**格式，forward 反量化到 fp16 再走通用 GEMM | 这条解释了为什么 W4A16 的生产内核（Marlin，arXiv:2408.11743）也一律"INT4 存 → 寄存器内反量化 → FP16 张量核"，而不是喂 INT4 张量核。**本仓的教学实现与生产实现在这一点上路线相同，差的只是融合** |
| 15 | Ada 白皮书 Appendix A：RTX 4090 显存带宽 1008 GB/s、FP32 82.6 TFLOPS、L2 72 MB | 本仓所有带宽/算力折算**用的都是这些规格值**，未做微基准 | 这是 §3.3.8 与 §5.6 里全部"下限"折算的已知偏差来源：实测可达值通常低于规格值，故那些下限是**乐观下限**，真实下限更大。白皮书里的 INT8 Tensor TOPS 本讲义**未能逐字核对**，故不引用 |
| 16 | Dutta et al.(arXiv:2407.09141)：PPL 可解读为 token 概率几何平均的倒数，加对称噪声不改 PPL 而生成质量下降；建议同时报 KL 与 flips | 本仓**只有 PPL**，无 KL、无 flips、无下游任务 | 不矛盾，是覆盖面不足。本仓主张的措辞因此被限定为"收回缺口的比例"，而非"质量无损"（§3.7.2 给了补测入口） |

**这张表的读法**：16 条里只有第 3 条与第 13 条是严格的"文档—实现"对齐闭环； 第 10 条是**有机制解释的冲突**（权重粒度 + 规模，方向可推）；其余大多是 "不可比"或"本仓未测"。**论文的数字几乎从不能直接搬到你的设置上**，能搬的是机制、判据与那些不依赖规模的代数关系。

### 8.2 与生产实现的差距各在哪一层

- **GPTQ 侧**：GPTQ-for-LLaMa/AutoGPTQ 的算法主循环与本仓同构(H 滑动缩放、 percdamp、cholesky(inverse(H)， upper) 三处数值锚点逐一对齐，见 `docs/theory/01_gptq.md` §5)；差距在 act_order(§3.3.10)、 `--true-sequential`、更低位宽（W3/W2）与超大模型的分层调度。
- **推理侧**：本仓 `QuantLinear4.forward` 是"先反量化、再 fp GEMM"的教学实现，只为正确性闭环（`src/quant_linear.py:20-22`）；生产内核（vLLM 的 Marlin 家族）做 fused dequant-GEMM，INT4 权重直接进寄存器再融合反量化——速度差距在 kernel 层，不在算法层。serving 侧实测见 vllm/experiments#EXP-016《D4 FP8 vs W4A16 同卡对比》：W4A16(Marlin)decode 快 23-48%，但大 M (prefill)下反量化开销显形，FP8 在 c128 TTFT 反超——按 regime 选型。
- **AWQ 侧**：参考实现 `llm-awq/awq/quantize/auto_scale.py` 为**块级输出 MSE**（对整个 decoder block 算 `(org_out - out).pow(2).mean()`）+ **共享输入组共享 s**（q/k/v 传进同一个 `linears2scale`）+ **weight clip**（`auto_clip.py` 按 `max_val = org_max_val * (1 - i_s / n_grid)` 扫 10 档， 目标同为输出 MSE，对 q/k 跳过），三件本仓皆简化或未做——差距在搜索目标的粒度与 clip 这一维，不在恒等变换本身（EXP-002 §2）。
- **SmoothQuant 侧**：原实现把 s 折叠进 RMSNorm(`ln.weight /= s`)做零开销部署、仅对 post-LN linears 迁移，且论文 Table 2 的三档（O1/O2/O3）权重一律 per-tensor；本仓 hook 形式数学等价但有 runtime 开销，对全部 7 类 Linear 迁移，权重用 per-输出行（EXP-003 §2）。生产 W8A8 还会配静态激活 scale（O3，免每 token 求 absmax），本仓未做，是自然的后续对照。
- **评价侧**：生产流程会跑下游任务集与生成质量抽检，不是只看 PPL；本仓刻意没有跨进这一步（§3.7），所以结论只能是"机制贡献"的比较，不是"可上线"的判定。

### 8.3 硬件语义:这些约束如何决定代码写法

本仓不写 CUDA kernel，硬件语义因此不以内核代码出现，而以**对数据布局与算法形态的硬约束**出现。四条最要紧的（前三条的推导分别见 §3.5.2、§3.6.1、 §3.3.8，这里只汇总"约束 → 代码后果"这一层）：

1. **整数张量核在 k 维上做整数累加 → 激活只能 per-token。** PTX ISA 的整数 mma 一族（如 `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`）把 s8×s8 的乘积累到 s32，整个 k 维归约在整数域完成；cuBLAS 的 `CUBLAS_COMPUTE_32I` 文档写"uses compute and intermediate storage precisions of at least 32-bits"(§2.2.11)，NVIDIA 把归约后作用于两个外维的向量缩放直接叫 "Outer Vector Scaling"(§3.1.4.3)。**代码后果**： `src/smoothquant.py:41-53` 的两个 `amax` 轴参数不是风格选择；SmoothQuant 必须存在，也正是因为这条约束堵死了"给激活做 per-channel"这条省事的路。
2. **INT4 没有可用的张量核路径 → 4 bit 只是存储格式。** CUDA 的 wmma 子字节类型（`experimental::precision::u4/s4`、`b1` 的 XOR 变体）已弃用并在 sm_90 移除。**代码后果**：W4A16 整条链路都是"INT4 存 → 反量化到 fp16 → fp16 GEMM"，本仓 `QuantLinear4.forward` 与 Marlin 路线相同，差别只在反量化发生在全局内存还是寄存器里。这也解释了 §8.2 里 vllm/experiments#EXP-016 的 regime 分化：大 M 时反量化的固定开销摊不掉，收益就被吃掉。
3. **打包位序与下游 kernel 的 fragment 布局耦合。** 本仓走通用 `F.linear`， 自然列序打包就够（`src/quant_linear.py:35`）；生产内核要按 mma 的线程—元素映射预置换权重，否则反量化后还要跨 lane 洗牌（Marlin 的 "bespoke quantization support"）。**打包格式的自由度取决于下游 kernel，不取决于量化算法。**
4. **带宽与容量决定哪些中间量能留、哪些必须释放。** down_proj 的 H 是 90.25 MiB、单层七个 108.6 MiB，24 层不释放要 2.5 GiB(§1.2)；校准输入 `inps` 469 MB、阶段 C 双份约 0.94 GB（§4 第 1 段）。4090 的 24 GB 显存与 1008 GB/s 带宽（白皮书 Appendix A）决定了这些量必须逐 Linear 释放（`src/gptq.py:193-197`），也决定了 §3.3.8 的 lazy batch update 是**带宽优化而非算力优化**——朴素逐列全宽更新在 down_proj 一层就要 170 GB 读写， 规格带宽下 0.17 s 起（本讲义推导）。

### 8.4 延伸阅读(每条一句话说明它能解决什么疑问)

1. LeCun， Denker & Solla， "Optimal Brain Damage"， NIPS 1989。——想知道"用二阶信息判断权重重不重要"这条线的起点，以及 quadratic/extremum/diagonal 三重近似各自假设了什么，读这篇。
2. Hassibi & Stork， "Second Order Derivatives for Network Pruning： Optimal Brain Surgeon"， NIPS 1992。——想弄清 $\delta w = -\frac{w_q}{[H^{-1}]_{qq}}H^{-1}e_q$ 与 $L_q = \frac{w_q^2}{2[H^{-1}]_{qq}}$ 怎么从拉格朗日推出来、为什么是 **逆的对角元**而不是对角元的倒数，读这里。
3. Frantar， Singh & Alistarh， "Optimal Brain Compression"， arXiv:2208.11580。——想看清"把删权重换成量化权重"这一步替换的完整框架（它同时覆盖剪枝）， 读这篇；GPTQ 的全部数学都是它的工程化。
4. Frantar， Ashkboos， Hoefler & Alistarh， "GPTQ： Accurate Post-Training Quantization for Generative Pre-trained Transformers"， arXiv:2210.17323， §3 Background（OBQ 的 Eq.2-3）与 §4 The GPTQ Algorithm 的三步（Arbitrary Order / Lazy Batch-Updates / Cholesky Reformulation）与 Algorithm 1。——想确认"固定列序凭什么行""B=128 从哪来""为什么是 Cholesky 而不是继续消元"，这三节一次讲完；注意论文对固定列序用的是 "may perform well" 这种经验性措辞。
5. IST-DASLab/gptq 仓库 README 与 `gptq.py`。——想知道 `--act-order`、 `--static-groups`、`--true-sequential` 各解决什么问题、推理侧代价是什么， 读 README；`gptq.py` 是本仓三处数值锚点的对齐原件。
6. Lin et al.， "AWQ： Activation-aware Weight Quantization for LLM Compression and Acceleration"， arXiv:2306.00978，§3.1 Table 1、§3.2 Table 2 与 Eq.(4)-(5)、§5.3 Figure 8(b)。——"为什么按激活而不是按权重挑显著通道""缩放降低误差的证明覆盖了什么情形""跨域校准时 AWQ 和 GPTQ 差多少"，这三处各管一个。
7. mit-han-lab/llm-awq 的 `awq/quantize/auto_scale.py` 与 `auto_clip.py`。——想搞清本仓 per-linear 简化到底简化掉了什么（块级 MSE、共享输入组共享 s、 weight clip 的 10 档收缩搜索），对着这两个文件读最快；`auto_scale.py` 的几何归一那一行还带着一个真实的 inf 事故（仓库 issue #96）。
8. Xiao et al.， "SmoothQuant： Accurate and Efficient Post-Training Quantization for Large Language Models"， arXiv:2211.10438，§3 Review of Quantization Difficulty、§4 的 Eq.4、Table 2、§5.5 Figure 10。——想确认 "outlier 为什么按通道固定""迁移公式怎么来""α 甜区是多少、超出会怎样" "O1/O2/O3 的粒度组合分别是什么"，这四处一一对应；Table 2 是理解本仓 α=0.75 与论文 0.5 冲突的关键（粒度不同）。
9. Dettmers et al.， "LLM.int8()： 8-bit Matrix Multiplication for Transformers at Scale"， arXiv:2208.07339，§4 与 §4.2。——想知道"激活 outlier 到底多大、什么规模开始出现、影响多少层"，读这里；它是"为什么 0.5B 上 SmoothQuant 收益小"最有力的外部解释。
10. Dutta， Krishnan， Kwatra & Ramjee， "Accuracy is Not All You Need"， arXiv:2407.09141。——想知道"PPL 不掉是不是等于模型没变"以及该用什么替代（KL 与 flips），读这篇；它直接决定了本仓 PPL 结论的措辞边界。
11. Frantar， Castro， Chen， Hoefler & Alistarh， "MARLIN： Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models"， arXiv:2408.11743。——想知道"4 bit 权重在真实内核里怎么跑得快""为什么 batch 一大加速就衰减"，读这篇；它也是本仓 `QuantLinear4` 教学实现与生产实现之间那道鸿沟的准确描述。
12. Bennett， "Spectra of Quantized Signals"， Bell System Technical Journal 27(3)：446-472, 1948。——想知道 $s^2/12$ 这个到处被引用的量化噪声方差出自哪里、在什么假设下成立，读这篇；§3.1.2 的失效条件正是对照它的假设写的。
13. Higham， *Accuracy and Stability of Numerical Algorithms*， 2nd ed.， SIAM 2002，第 10 章 Cholesky Factorization。——想弄明白"为什么 Cholesky 不用选主元也稳""为什么 `cholesky_inverse` 比通用 `inv` 好"，读这一章。
14. NVIDIA 官方文档四处：PTX ISA 的 warp-level matrix instructions（整数 mma 的形状与 s32 累加语义）、CUDA C++ Programming Guide 的 wmma 小节（子字节类型的弃用说明）、cuBLAS 文档 §2.2.11 `cublasComputeType_t` 与 §3.1.4 Narrow Precision Data Types Usage、"NVIDIA Ada GPU Architecture" 白皮书 Appendix A(1008 GB/s、FP32 82.6 TFLOPS、L2 72 MB)。——前三处把 "激活为什么只能 per-token""INT4 为什么只能当存储格式"从"听说"变成 "有出处"；白皮书是本篇每一个"下限"折算的常数来源。
15. 本仓内部指针：`records/EXP-001~003`（含假设、阈值与开放问题）、 `data/raw/EXP-00{1,2,3}/*.json`（每个数字的原始出处，含 per-layer 的 α/loss/耗时/pack 误差）、`docs/theory/01_gptq.md` 至 `03_smoothquant.md`、 `docs/HOW_TO_LEARN_A_QUANT_METHOD.md`。——想复算本篇任何一个数字，或想知道某个结论当初是怎么被证伪或保留的，从这里进。
