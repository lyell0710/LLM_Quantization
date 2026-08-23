# LLM_Quantization — 量化方法从零实现与对照实证

延续既有 AWQ / SmoothQuant 实践(AutoAWQ 侧,异机)的第三块拼图:
**从零实现 GPTQ** 并用控制变量对照量化"方法本身的贡献"。
学习方法论见 [docs/HOW_TO_LEARN_A_QUANT_METHOD.md](docs/HOW_TO_LEARN_A_QUANT_METHOD.md);
工程准则见 /root/standards/CORE.md。

## 怎么跑

```bash
/root/venvs/v0.25.1/bin/python scripts/run_gptq.py --mode {fp16|rtn|gptq} \
  --group-size 128 --out data/raw/EXP-001/<name>.json
# 或三臂一键: bash scripts/run_all.sh
```

## EXP 索引

| 编号 | slug | 日期 | 状态 | 关键数字(指针) |
|---|---|---|---|---|
| [EXP-001](records/EXP-001_gptq_from_scratch.md) | gptq_from_scratch | 2026-08-23 | 完成 | GPTQ 收回 RTN 质量损失 **61.6%**(PPL 12.76 vs 14.12 vs fp16 11.92,INT4-g128,Qwen2.5-0.5B → data/raw/EXP-001/*.json) |

## 当前关键数字(带指针)

- wikitext-2 PPL(窗2048/步1536,同计分 token 集):fp16 11.9152 /
  RTN 14.1154 / GPTQ 12.7600 → `data/raw/EXP-001/{fp16,rtn,gptq}_g128.json`
- real quant 闭环:INT4 打包↔fake-quant 最大误差 ≤7.3e-4(fp16 舍入级),
  168/168 层 pack_check 通过 → 同上 JSON `max_pack_err`
- 部署侧衔接(跨项目):GPTQ-Int4+Marlin serving 实测
  → `vllm/experiments#EXP-016`

## 措辞红线表

| 红线 | 当前 | 说明 |
|---|---|---|
| PPL 绝对值 | 限定 | 协议自定义(窗/步/拼接),只作臂间相对比较,不与文献绝对值对比 |
| "61.6% 恢复" | ✅ 可用 | 控制变量对照(唯一差异=补偿开关),EXP-001 §5 |
| per-layer loss 诊断值 | 🚫 不进表 | 量纲未标定(EXP-001 §7) |
| "从零实现" | ✅ 可用 | 算法主循环/H 累积/Cholesky 路径/打包全部自写;数值锚点与参考实现对齐(theory §5) |

## 结构

`src/gptq.py`(算法)· `src/quant_linear.py`(INT4 打包+real-quant 前向)·
`scripts/run_gptq.py`(端到端)· `docs/theory/01_gptq.md`(五节原理笔记)
