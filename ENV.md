# ENV — 异地复现指南(主线:三法从零实现)

本机实况以 /root/work/infra/machine/ENV_REGISTRY.md 为准;本文只讲复现主线
EXP-001~003 所需环境。版本指针 = records/EXP-001 §0 与各 raw JSON 的
provenance env 字段(不在此复制更多数字)。

## 主线环境(EXP-001~003 全部数据产自此)

- Python venv:`/root/venvs/v0.25.1`(实验时 torch 2.11.0+cu130、
  transformers 5.15.1、datasets;见 EXP-001 §0)
- GPU:NVIDIA RTX 4090 ×1,driver 610.57.04(raw provenance);
  显存峰值 ~6GB(EXP-001 §2),异地单卡 ≥8GB 即可
- 模型:Qwen/Qwen2.5-0.5B(base,HF cache)
- 数据集:Salesforce/wikitext · wikitext-2-raw-v1
  (train 做校准 128×2048,seed=3407;test 评 PPL,窗 2048/步 1536)

## 入口

```bash
V=/root/venvs/v0.25.1/bin/python
$V scripts/run_w4a16.py --mode {fp16|rtn|gptq|awq|awq_gptq} --group-size 128 --out <json>
$V scripts/run_w8a8.py  --mode {fp16|naive|smooth} [--alpha 0.5] --out <json>
bash scripts/run_all.sh    # EXP-001 三臂(dirty 拒跑;UTC 前缀新文件,拒覆盖)
bash scripts/run_all2.sh   # EXP-002/003(同上)
```

## 出图(纯 CPU,无需 GPU)

- matplotlib 环境:`/root/venvs/kernel-opt/bin/python`
- `scripts/plot_alpha_dist.py` → figures/fig1(从 data/raw/EXP-002 重算)

## 子范围

- `llmqt_example/`(AutoAWQ 侧既有实践,异机 RTX 4070 Laptop)自带环境
  说明:llmqt_example/ENV.md——与本文件相互独立,勿混用版本号。
