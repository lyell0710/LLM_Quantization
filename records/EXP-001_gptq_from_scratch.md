# EXP-001 · GPTQ 从零实现:INT4-g128 三臂对照(fp16 / RTN / GPTQ)

> **一句话结论**：同一网格、同一评测协议下只换误差补偿这一个变量：GPTQ 相对 RTN **收回 61.6% 的量化损失**（PPL 14.1154 → 12.7600），这份恢复量可完全归因于二阶补偿机制本身。

## 0. 元信息

| 字段 | 值 |
|---|---|
| 日期 | 2026-08-23 |
| 环境 | env=v0.25.1-venv(torch 2.11.0+cu130, transformers 5.15.1);RTX 4090 ×1(cuda:0),driver 610.57.04 |
| 状态 | 完成 |
| 关联清单项 | 简历量化段升级（GPTQ 补洞）；docs/theory/01_gptq.md 实证节 |

## 1. 目的与假设

从零实现 GPTQ 关键路径（H 累积/Cholesky 逆/逐列误差补偿/per-group 非对称 INT4），验证可证伪假设（阈值跑前锁定）：**同一量化网格下，GPTQ 的 PPL 退化 < RTN 退化的一半**（即二阶补偿收回 >50% 的质量损失）。

## 2. 环境与配置

- 模型 Qwen/Qwen2.5-0.5B（base，本机 HF cache）；量化 24 层 × 7 Linear (q/k/v/o/gate/up/down)，lm_head 不量化（惯例）。
- 量化设定与既有 AWQ 实践对齐：INT4、per-group g=128、非对称 min-max； GPTQ 附加：percdamp=0.01、blocksize=group_size（组参数取自补偿后权重）、 sequential（逐层用量化后输出作为下层输入）、act_order 关。
- 校准：wikitext-2-raw train,128 段 × 2048 token,seed=3407。
- 对照设计（控制变量）：RTN 臂与 GPTQ 臂共用同一 GroupQuantizer / find_params / 量化网格，唯一差异 = 误差补偿开关（use_hessian）。
- 单卡显存峰值 ~6GB；三臂共用 scripts/run_all.sh，provenance 首行入 JSON。

## 3. 步骤

`bash scripts/run_all.sh` = 三次 `run_gptq.py --mode {fp16|rtn|gptq} --group-size 128`（勘注 2026-08-24：run_gptq.py 已于 EXP-002《AWQ 从零实现 + AWQ×GPTQ 叠加》收编时改名为 scripts/run_w4a16.py，见 §7）；每臂结束即评 PPL（wikitext-2 test，窗 2048/步 1536， 与 vllm/experiments#EXP-016《D4 FP8 vs W4A16 同卡对比》同族协议，三臂同计分 token 集 298302 tok）。

## 4. 原始数据

`data/raw/EXP-001/`：{fp16，rtn，gptq}_g128.json（首字段 provenance，含 per-layer loss/耗时/pack 校验）+ 对应 *_run.log。

## 5. 结果

| 臂 | PPL | Δ vs fp16 | 量化耗时 | pack↔fake 最大误差 |
|---|---|---|---|---|
| fp16 | 11.9152 | — | — | — |
| RTN INT4-g128 | 14.1154 | +2.2002 | 30 s | 7.3e-4 |
| **GPTQ INT4-g128** | **12.7600** | **+0.8448** | 135 s | 5.5e-4 |

**补偿贡献（本实验主结论）**：GPTQ 收回 RTN 质量损失的（14.1154−12.7600）/(14.1154−11.9152) = **61.6%**；假设（>50%）成立。

## 6. 分析与结论

- 唯一变量是误差补偿 → 61.6% 的恢复量可**完全归因于二阶补偿机制本身**（同一网格、同一 find_params、同一评测协议）。
- 绝对幅度符合文献预期：0.5B 小模型 INT4 退化本就偏大（参数冗余少）， GPTQ 后 +0.84 PPL 属正常区间；结论只作臂间相对比较，不跨协议引用绝对值。
- real quant 闭环：INT4 两枚一字节打包 → 反量化与 fake-quant 权重逐元素一致（≤fp16 舍入），存储格式实现被断言级验证（pack_check，168/168 层过）。
- PPL 评测为确定性计算（greedy scoring，无采样），单轮即可复算； GPTQ 过程由 seed 固定，整链可复现。

## 7. 异常、偏差与开放问题

- per-layer `loss`（论文 Σerr²/2[H⁻¹]jj 项）数值量级 1e-20~1e-15： 量纲随 H 的滑动归一化缩放，**未标定，仅可作层间相对诊断**，不进任何表格； 量化正确性由 PPL 与 pack 校验独立支撑，不依赖该诊断值。
- act_order、W3/W2 更低位宽、域外校准集敏感性均未测——留作 EXP-002 候选。
- 实现约束：blocksize 必须等于 group_size（组参数需取自补偿后权重）； 通用化（组内跨块）未做，记录在 src/gptq.py 断言处。
- 【勘注 2026-08-24】raw JSON provenance 的 sha=worktree：三臂运行时实验代码尚未 commit，run_all.sh 以 worktree 占位。代码即 274acb2 所提交内容（该 commit 首次入库 src/gptq.py 与 scripts/，与运行版本一致，git show 可核）。 raw 不改；run_all.sh 已改为 worktree dirty 即拒跑，杜绝复发。
- 【勘注 2026-08-24】§3 与 raw provenance cmd 中的 run_gptq.py 已于 EXP-002 收编时改名为 scripts/run_w4a16.py（同一入口扩为五臂，b11f0f2 rename 可溯）； 历史文件与 raw 不改，以本勘注为准。

## 8. 下游影响

- docs/theory/01_gptq.md §3 实证节回填本表；§4 Q&A 全部有实测背书。
- 简历量化段可新增 GPTQ bullet（数字：61.6% 恢复 / 12.76 vs 14.12 vs 11.92）， 并与 vllm/experiments#EXP-016（GPTQ-Int4+Marlin serving 侧）构成 "离线算法 → 在线部署"完整链。
- 学习方法论固化于 docs/HOW_TO_LEARN_A_QUANT_METHOD.md（6 步 + 三问自测）。
