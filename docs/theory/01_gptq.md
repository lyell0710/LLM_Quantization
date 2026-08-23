---
topic: GPTQ
status: 完成(实证=EXP-001)
---

# 01 · GPTQ(逐列二阶误差补偿量化)

## 1. 一句话结论

GPTQ 把"权重量化"从逐元素最近取整(RTN)升级为**逐列量化 + 用二阶信息把每列
的量化误差补偿到未量化列上**,以最小化层输出误差 ‖WX−QX‖²(而非权重误差
‖W−Q‖²),从而在 INT4 下把精度损失压到 RTN 的一小部分——代价只是一次校准
集前向和每层一次 Cholesky。

## 2. 机制(自己的话)

**目标错了,方法就错。** RTN 最小化的是权重本身的误差,但推理关心的是输出:
同样大小的权重扰动,落在"激活大的输入维"上伤害远大于"激活小的维"。
输入维的重要性由校准集的二阶统计 **H = 2·X·Xᵀ**(per-layer Hessian)刻画。

**OBQ → GPTQ 的工程化三步**:
1. *固定量化顺序*:OBQ 逐个挑"当前伤害最小"的权重,O(n³·) 不可扩展;
   GPTQ 发现按任意固定顺序(直接按列序)逐列量化,精度几乎不掉——
   于是同一层所有行可以共享同一套列序与 H⁻¹,批量处理。
2. *误差补偿闭式解*:量化第 j 列后,最优补偿是把误差按
   δ = −(w_j − q_j)/[H⁻¹]_jj · [H⁻¹]_{j,j>} 分摊到后续列
   ——实现里用 **H⁻¹ 的上三角 Cholesky 因子**一次算好,列循环里只做
   rank-1 更新(本仓 `src/gptq.py` 的 `Hinv` 与 `err` 路径)。
3. *块化 + 延迟全局更新*:128 列一块,块内即时补偿、块外批量补偿
   (`W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]`),把访存从 O(n²) 次小更新
   合并成 GEMM。
另加 **阻尼** λ=1%·mean(diag H) 防 H 病态(死列/共线激活)。

**与 per-group 的交互**(容易被问倒的点):组参数(scale/zero)必须取自
"前序误差已补偿进来的当前权重",而不是原始权重——本仓实现里组边界与补偿块
边界对齐(blocksize=group_size=128)正是为此。

**三种方法一张图**:
- RTN:min ‖W−Q‖²,无数据,秒级。
- GPTQ:min ‖WX−QX‖²,列序固定 + 二阶补偿,分钟级,**权重侧被动适应激活**。
- AWQ:不动取整,先用 per-channel scale 把"激活大的维"的权重放大再量化
  (**主动重塑量化难度分布**),与 GPTQ 正交(可叠加)。
- SmoothQuant:目标不同——为 W8A8 把**激活**的 outlier 难度迁移到权重侧;
  GPTQ/AWQ 是 weight-only(W4A16),激活保持 FP16,不需要它。

## 3. 本项目实证(EXP-001,Qwen2.5-0.5B,INT4 g128 非对称)

| 臂 | wikitext-2 PPL(窗2048/步1536,同 298302 计分 token) |
|---|---|
| fp16 | 11.9152 |
| RTN INT4-g128(同网格对照) | 14.1154(+2.20) |
| **GPTQ INT4-g128** | **12.7600(+0.84)** |

控制变量归因:两臂共用同一 GroupQuantizer/find_params,唯一差异=误差补偿
开关 → **GPTQ 的二阶补偿收回 RTN 质量损失的 61.6%**。全模型量化 135s
(单张 4090);real quant 打包↔fake 一致 ≤5.5e-4(168/168 层断言通过)。
→ records/EXP-001,data/raw/EXP-001/*.json

## 4. 面试追问 Q&A

- **Q: H 是什么的 Hessian?** 层输出 MSE 对该层权重(单行)的 Hessian
  = 2XXᵀ,与行无关——这就是全部行能共享一次 Cholesky 的原因。
- **Q: 为什么用 Cholesky 而不是直接存 H⁻¹?** 数值稳定 + 补偿只需要
  H⁻¹ 的行上三角部分;逐列更新 H⁻¹ 的朴素写法(OBS 公式)在 fp 下会
  累积误差,大模型上会炸(论文 §3 的 Step 2)。
- **Q: act_order(desc 重要性排序)是什么、为什么帮精度?** 按 diag(H)
  降序先量化"重要列",它们享受最完整的后续补偿空间;小模型收益小,
  低 bit/大模型明显。本仓默认关(与主流 config 一致),留了实现位。
- **Q: GPTQ vs AWQ 怎么选?** 同为 W4A16:GPTQ 需要逐层解方程但对
  outlier 权重更稳;AWQ 校准更便宜、对指令模型泛化口碑好;生产里
  常见"都试,PPL/任务分说话"。二者正交于 SmoothQuant(那是 W8A8 的事)。
- **Q: 量化完怎么跑得快?** 存 INT4 + scale/zero,推理时 kernel 内
  反量化融合进 GEMM(Marlin 类),decode 阶段权重读取带宽减半再减半
  ——衔接 vllm/experiments#EXP-016:W4A16(Marlin)decode 快 23–48%,
  但大 M(prefill)反量化开销显形,FP8 在 c128 TTFT 反超。
- **Q: 校准集选错会怎样?** H 偏了补偿方向就偏;域外校准可测出 PPL 退化
  (留作 EXP-002 的可选对照)。

## 5. 延伸(源码/论文锚点)

- 论文:Frantar et al., *GPTQ: Accurate Post-Training Quantization for
  Generative Pre-trained Transformers*, arXiv:2210.17323(§3 三步工程化)。
- 参考实现对照:GPTQ-for-LLaMa `gptq.py`(H 累积的滑动缩放、
  percdamp、cholesky(inverse(H), upper) 三处与本仓一致)。
- 部署侧:vLLM `AutoGPTQLinearMethod → MarlinLinearKernel`
  (vllm/experiments#EXP-016 日志原文);EPLB×GPTQ 的上游缺口
  `routed_experts.py:139-152`(vllm/experiments#EXP-017)。
- 本仓实现:`src/gptq.py`(算法)、`src/quant_linear.py`(INT4 打包与
  real-quant 前向,pack↔fake 一致性由 pack_check 断言)。
