# LEDGER — 状态与措辞账本(对内)

> **本文件是状态与措辞的唯一权威;README 为对外版,措辞以本表为准。**
> README 的读者假设是陌生面试官:不出现日期、勘误、审计、红线、台账、
> 待办类内部词汇;数字的测量条件(轮数/窗口/协议)作为自然定语保留。

## 仓库状态

- ⚠ 本仓当前**无 git 远端**:待用户创建
  github.com/lyell0710/LLM_Quantization 后推送(不代建远端);在此之前
  所有里程碑仅本地 commit,铁律 5 的 push 项挂起。

## 实验台账

| 编号 | 名称 | slug | 日期 | 状态 | 关键数字(指针) |
|---|---|---|---|---|---|
| [EXP-001](records/EXP-001_gptq_from_scratch.md) | GPTQ 从零实现:INT4-g128 三臂对照(fp16 / RTN / GPTQ) | gptq_from_scratch | 2026-08-23 | 完成 | GPTQ 收回 RTN 损失 61.6%(12.76/14.12/11.92 → data/raw/EXP-001/) |
| [EXP-002](records/EXP-002_awq_and_stack.md) | AWQ 从零实现 + AWQ×GPTQ 叠加(W4A16 赛道补全) | awq_and_stack | 2026-08-23 | 完成 | AWQ 31.9%;AWQ+GPTQ 62.6%;per-layer α 中位 0.30(→ data/raw/EXP-002/) |
| [EXP-003](records/EXP-003_smoothquant_w8a8.md) | SmoothQuant 从零实现:W8A8 赛道 + α 扫描 | smoothquant_w8a8 | 2026-08-23 | 完成 | smooth α=.75 收回 naive W8A8 缺口 48%(12.02/12.12 → data/raw/EXP-003/) |

## 待办 / backlog

- 待用户创建远端 github.com/lyell0710/LLM_Quantization 后首推全量并验证。
- 待 GPU 空闲:可选 EXP-004(clip / 块级共享 s / act_order)、
  EXP-005(α>0.75 尾部 / 静态激活量化)。
- 简历量化段只从 records 数字生成,措辞过本表红线。

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

## 勘误 / 审计留痕

- EXP-001 §7 两则勘注(2026-08-24):①raw provenance sha=worktree——三臂
  运行时代码尚未 commit,代码即 274acb2 所提交内容,git show 可核;
  ②run_gptq.py 已改名 scripts/run_w4a16.py(b11f0f2 rename 可溯)。
  raw 不改,以勘注为准。
- 2026-08-24 审计收尾批次:外部审计确认的 9 项问题逐条闭环
  (LAB_JOURNAL §4),零重跑、零改 raw 本体;run_all*.sh 已加固为
  UTC 前缀新文件 + 同名拒覆盖 + worktree dirty 拒跑。
- "关键数字 ≥3 轮 mean/std"要求:本仓 PPL 为确定性 greedy scoring,
  按 EXP-001 §6 以"单轮可复算"显式豁免;对外一律表述为「单轮」,
  不用"终端级证据"。

## 内部硬约定

- data/raw 不可变;发现口径问题不改历史,以勘注留痕;dirty 拒跑工装杜绝复发。
- 图禁手改,只由 scripts/plot_*.py 从 raw 重算(STANDARDS §6)。
  图脚注/标题不带日期(对外化,2026-08-25 起),源数据文件+硬件+轮数保留。
- 289QS / llmqt_eval **异协议**说明:PPL 为 HF eval 协议(不同模型、不同
  评测实现),绝对值不与主对照表比较、不混排。数据:AWQ-INT4 vs fp16——
  Qwen2-1.5B 8.933 vs 8.474;OPT-125m 25.18 vs 23.69
  → llmqt_example/289QS/results/*.json;latency/吞吐与图见同目录 figures/。
  llmqt_example/(2026-08-23 连完整 git 历史并入,merge 1feb83e)为只读快照,submit/ 交付原貌不清理。
- 工程准则:/root/standards/CORE.md;收尾自检 bash /root/standards/check.sh
  必须 0 FAIL。
