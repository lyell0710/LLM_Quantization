---
topic: AWQ
status: 完成(实证=EXP-002)
date: 2026-08-23
exp: EXP-002
---

# 02 · AWQ(激活感知的权重缩放量化)

## 1. 一句话结论

AWQ 不改取整规则，而是在量化**之前**给"激活大的输入维"的权重乘一个 per-channel scale s（量化后等价除回）：让少数决定输出的显著权重占据更细的量化格，其余权重稍微吃亏——用一次 α 网格搜索（本仓 n_grid=20）在两者间找最优折中。

## 2. 机制(自己的话)

- **观察**：~1% 的权重通道对输出贡献极大（它们对应的**激活**通道数值大）； RTN 对所有权重一视同仁，显著权重的量化误差被大激活放大。
- **等价变换**：y = X·Wᵀ = (X)·(diag(s)⁻¹·diag(s)·W)ᵀ。对 W·diag(s) 量化、推理时把 diag(s)⁻¹ 并进激活（或折叠进前置算子）——**数学恒等，只改变量化误差落在哪里**。s>1 的维：量化步长相对变小（该维权重被保护）。
- **搜索**：s_j = absmean(X_j)^α，α∈[0,1) 网格，目标 min ‖Q(W·s)·s⁻¹·X − W·X‖²（输出空间的 MSE，不是权重空间）； α=0 退化为 RTN，α→1 完全按激活强度保护。
- **与 GPTQ 的关系**：GPTQ 改"怎么取整"（二阶补偿），AWQ 改"在哪个坐标系里取整"（难度重分布）——**正交，可叠加**：先乘 s 再对 W·s 做 GPTQ（此时 GPTQ 的 H 必须取自 X/s，本仓 awq_gptq 臂即此拆分）。
- **部署折叠的约束（面试点）**：fake-quant 评测里每个 Linear 可各持一个 s（激活从未真正被除）；但要做零开销 runtime（s 折进前置 RMSNorm/Linear）， **共享同一输入的 linears(q/k/v；gate/up)必须共享同一个 s**——这是折叠的约束，不是算法的。本仓为臂间可比取 per-linear s，并如实标注。

## 3. 本项目实证(EXP-002,Qwen2.5-0.5B,INT4 g128,同 EXP-001 协议)

| 臂 | PPL | 收回 RTN 损失 |
|---|---|---|
| RTN 14.1154 → **AWQ** | **13.4127** | **31.9%** |
| GPTQ 12.7600 → **AWQ+GPTQ** | **12.7376** | **62.6%**（vs GPTQ 单独 61.6%） |

per-layer best α：中位数 0.30，主体 0.15–0.45，单层 α=0（激活均匀层）， 两层 0.95（强 outlier 层）——保护强度按层自适应。叠加增益 +1.0pp： "正交可叠加"方向成立，0.5B 上与 GPTQ 补偿重叠度高，幅度不夸大。 → records/EXP-002，data/raw/EXP-002/*.json

## 4. 面试追问 Q&A

- **Q： s 为什么用 absmean 而不是 absmax？** absmax 被 outlier 单点绑架； absmean 反映通道整体强度。AutoAWQ 亦有 max/mean 变体，同属启发式， 最终由网格搜索的输出 MSE 裁决。
- **Q： 为什么 AWQ 不用二阶信息还能打平 GPTQ？** 二者修的是不同的病： GPTQ 精修取整误差的**分配**，AWQ 重塑误差的**权重（难度）分布**； 显著通道主导输出时，保护它们的一阶启发式就能拿到大头。
- **Q： 网格搜到的 α 通常多大？** 经验 0.3–0.7 居多（本仓 per-layer α 分布见 EXP-002《AWQ 从零实现 + AWQ×GPTQ 叠加》raw）；α 贴 0 说明该层激活均匀、AWQ 无事可做。
- **Q： clip 是什么、为什么有用？** 收缩组内 max_val 再量化（牺牲极值精度换普遍步长变细），AutoAWQ 与 scale 搜索联用；本仓未实现（见 EXP-002 §7， 既有 AutoAWQ 实践中已做过 grid clip）。
- **Q： AWQ 和 SmoothQuant 一句话区分？** AWQ 是 W4A16 的**权重内部**难度重分布（激活不量化）；SmoothQuant 是 W8A8 的**激活→权重**难度迁移。公式形似（都是 per-channel s），目标物不同。

## 5. 延伸(源码/论文锚点)

- 论文：Lin et al., *AWQ: Activation-aware Weight Quantization*, arXiv:2306.00978（§3 salient channels 与 s 搜索）。
- 参考实现：AutoAWQ `quantizer.py`（scale/clip 网格；块级输出 MSE——本仓简化为 per-linear MSE，EXP-002 §7 注明）。
- 部署侧：W4A16 kernel 反量化融合（Marlin 家族），serving 实测 vllm/experiments#EXP-016《D4 FP8 vs W4A16 同卡对比》。
- 本仓实现：`src/awq.py`(_search / search_and_apply / search_scale_only)。
