---
topic: SmoothQuant
status: 完成(实证=EXP-003)
---

# 03 · SmoothQuant(激活难度向权重迁移的 W8A8)

## 1. 一句话结论

W8A8 的瓶颈不在权重而在**激活**:少数通道的系统性 outlier 把 per-token
INT8 的动态范围撑爆;SmoothQuant 用恒等变换 y=(X/s)·(s·Wᵀ) 把激活难度按
α 迁移给分布平坦、吃得下的权重侧,使两边都落回 INT8 可承受区间。

## 2. 机制(自己的话)

- **为什么激活难量化**:LLM 激活的 outlier 集中在**固定的少数通道**
  (与 token 无关),幅值可比其他通道大百倍;per-token 对称量化的 scale
  被 outlier 通道决定 → 其余通道分辨率骤降。per-channel 量化激活又与
  GEMM 的数学不相容(scale 无法从求和里提出来)——这就是迁移的动机。
- **迁移公式**:s_j = actmax_j^α / wmax_j^(1−α)。α 是"难度天平":
  α=0 不迁移(naive W8A8),α=1 全迁给权重(权重被打爆)。
  α 的最优点取决于激活/权重两侧的难度对比,典型 0.5 上下。
- **零开销折叠**:X/s 折进前置 RMSNorm/LayerNorm 的 weight(`ln.weight/=s`),
  s·W 直接写进 Linear——runtime 无额外算子。共享输入的 linears 须共享 s
  (与 AWQ 折叠同一约束)。本仓评测用 hook 形式(数学等价),不做折叠。
- **与 W4A16 组的边界**:GPTQ/AWQ 是 weight-only(激活 FP16,decode 省权重
  带宽);SmoothQuant 服务于 W8A8(算力路径也换 INT8,prefill/大 batch 受益,
  A8 还能省激活/KV 传输)。三者不是竞争关系,是不同 regime 的工具。

## 3. 本项目实证(EXP-003,Qwen2.5-0.5B,同 PPL 协议)

| 臂 | PPL(fp16=11.9152) |
|---|---|
| naive W8A8 | 12.1227(+0.21) |
| smooth α=0.25 | 12.2332(迁移不足反伤权重) |
| smooth α=0.50 | 12.0394(收回缺口 40%) |
| **smooth α=0.75** | **12.0221(收回 48%)** |

最有信息量的不是改善本身,而是幅度:0.5B 激活 outlier 温和(naive 仅
+0.21),可迁移的难度本来就少——反面印证"SmoothQuant 价值随模型规模/
outlier 严重度增长"(论文主战场是 OPT-13B+)。0.25→0.75 单调改善 =
本模型最优 α 偏大(权重侧余量足)。→ records/EXP-003

## 4. 面试追问 Q&A

- **Q: 为什么不直接 per-channel 量化激活?** GEMM 里 Σ_j x_j·w_ij 的
  per-j scale 无法在求和外提出(除非做逐通道反量化,失去 INT8 GEMM 意义);
  per-token(行方向)可以。权重则相反,per-输出行天然可提。
- **Q: actmax 用校准集统计,线上分布漂了怎么办?** outlier 通道位置是模型
  属性(训练所致),跨数据集稳定——这是 SmoothQuant 成立的经验基石;
  仍建议部署前用域内数据复核。
- **Q: α 怎么选?** 校准集上扫 PPL/任务分;激活 outlier 越猛 α 越大。
  本仓 α 三点扫描给出本模型的形状(§3)。
- **Q: 和 AWQ 公式那么像,能互换吗?** 不能:AWQ 的 s 服务于"权重 INT4
  内部谁吃亏"(激活不量化,方向是放大显著维权重);SmoothQuant 的 s 服务于
  "激活侧难度搬走"(方向是把激活缩小)。同一恒等式,两个相反的使用动机。
- **Q: W8A8 什么时候比 W4A16 快?** 算力受限段(prefill/大 batch,INT8
  TensorCore 双倍吞吐);decode 带宽受限段 W4A16 的 4-bit 权重更省
  ——与 vllm/experiments#EXP-016 的 regime 分化结论同构。

## 5. 延伸(源码/论文锚点)

- 论文:Xiao et al., *SmoothQuant*, arXiv:2211.10438(§4 迁移公式与折叠)。
- 既有实践:AutoAWQ 侧 OPT/Qwen2 端到端(`ln.weight /= s` 折叠已做过,
  本仓补的是此前缺失的效果数字)。
- 本仓实现:`src/smoothquant.py`(actmax 收集 / apply_w8a8 / per-token A8)。
