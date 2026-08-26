# EXP-003 · SmoothQuant 从零实现:W8A8 赛道 + α 扫描

> **一句话结论**：α 扫描给出的不只是「有效」：α=0.5 收回 naive 缺口的 40%、α=0.75 收回 48%，但 **0.5B 模型 naive W8A8 本身只退化 +0.21 PPL**——激活 outlier 温和时 SmoothQuant 的可发挥空间本就小。α=0.25 反而劣于 naive，说明 α 是天平不是免费午餐。

## 0. 元信息

| 字段 | 值 |
|---|---|
| 日期 | 2026-08-23 |
| 环境 | env=v0.25.1-venv;RTX 4090 ×1；同 EXP-001 协议 |
| 状态 | 完成 |
| 关联清单项 | 量化项目"数据补全"（此前 SmoothQuant 仅异机跑通无数字）；theory/03 实证节 |

## 1. 目的与假设

从零实现 SmoothQuant(s_j=actmax_j^α/wmax_j^{1−α})并给 W8A8 赛道补齐效果数字。可证伪假设（跑前锁定）：smooth(α=0.5)的 PPL 优于 naive W8A8（同 INT8 格式，唯一差异=是否迁移）。

## 2. 环境与配置

- W8A8 fake quant：权重 per-输出行对称 INT8（静态）；激活 per-token 对称 INT8（动态，forward pre-hook 实现）。lm_head 不量化。
- smooth 臂：actmax 由 128×2048 校准集全模型一遍收集；hook 形式评测 (x→fakeq(x/s)，W→fakeq(W·s))，数学上与 RMSNorm 折叠部署完全等价（折叠约束见 theory/03 §2）。本仓对全部 7 类 Linear 施加迁移（原论文只对 post-LN linears；实现选择，注明）。
- α ∈ {0.25, 0.5, 0.75} 三点扫描；naive = s≡1 同管线。

## 3. 步骤

`bash scripts/run_all2.sh` 后半：`run_w8a8.py --mode {naive|smooth --alpha A}`。

## 4. 原始数据

`data/raw/EXP-003/`:naive.json、smooth_a{0.25,0.5,0.75}.json（provenance 首字段）+ 各 run.log + manifest.txt。

## 5. 结果

| 臂 | PPL | Δ vs fp16(11.9152) |
|---|---|---|
| naive W8A8 | 12.1227 | +0.2075 |
| smooth α=0.25 | 12.2332 | +0.3180 |
| smooth α=0.50 | 12.0394 | +0.1242 |
| **smooth α=0.75** | **12.0221** | **+0.1069** |

## 6. 分析与结论

- 假设成立（α=0.5：收回 naive 缺口的 40%；α=0.75:48%），但**幅度是本实验最有信息量的部分**：0.5B 模型 naive W8A8 本身只退化 +0.21 PPL——激活 outlier 温和，SmoothQuant 的"可迁移难度"本来就少。这从反面印证论文叙事：**该方法的价值随模型规模/outlier 严重度增长**（原论文主战场是 OPT-13B+ 的百倍级 outlier）。
- α=0.25 反而劣于 naive：迁移不足以救激活、却已开始伤权重——α 是天平不是免费午餐；0.25→0.75 单调改善说明本模型的最优 α 偏大（权重侧余量足）。
- 与 W4A16 赛道对照（同协议）：W8A8 各臂全面优于 W4A16 各臂（位宽 8>4， 精度当然更好）；两赛道的取舍在速度侧（W4 省 decode 带宽 vs INT8 算力）， 见 vllm/experiments#EXP-016《D4 FP8 vs W4A16 同卡对比》的 regime 分化。

## 7. 异常、偏差与开放问题

- 未测 α>0.75（单调性尾部未封）；未做 per-tensor 静态激活量化对照（部署更常见的形态）；未复刻"仅 post-LN linears 迁移"的原论文口径。均可作 EXP-005 候选。
- fp16 基线复用 EXP-001《GPTQ 从零实现》（同模型同协议，评测确定性）。

## 8. 下游影响

- theory/03 §3 回填（含 α 单调性的解读）；README 主表 W8A8 段。
- 简历 SmoothQuant 段可补数字；talk 提纲收"小模型 outlier 温和"的分寸讲法。
