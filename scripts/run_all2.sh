#!/bin/bash
# EXP-002（AWQ 从零实现 + AWQ×GPTQ 叠加）(awq/awq_gptq)+ EXP-003（SmoothQuant 从零实现）(naive/smooth α 扫描)批跑。
# 铁律 3/4(2026-08-24 审计加固):UTC 前缀新文件 + 同名拒覆盖 + dirty 拒跑。
set -u
cd /root/projects/LLM_Quantization
V=/root/venvs/v0.25.1/bin/python
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || { echo "FATAL: 非 git 仓,无法记 provenance sha" >&2; exit 1; }
[ -z "$(git status --porcelain)" ] || { echo "FATAL: worktree dirty——先 commit 再跑(provenance sha 必须指向真实代码版本)" >&2; exit 1; }
SHA=$(git rev-parse --short=10 HEAD)
TS=$(date -u +%Y%m%dT%H%M)  # 同次测量共用 UTC 前缀(铁律 4)
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
prov() { echo "# provenance: env=v0.25.1-venv sha=$SHA cmd=\"$1\" date=$(date -u +%Y-%m-%dT%H:%M:%S+00:00) gpu=\"$GPU\" driver=$DRV"; }
guard() { for f in "$@"; do [ -e "$f" ] && { echo "FATAL: $f 已存在,拒绝覆盖 raw" >&2; exit 1; }; done; }

R2=data/raw/EXP-002; mkdir -p $R2
for MODE in awq awq_gptq; do
  OUT=$R2/${TS}_${MODE}_g128.json; LOG=$R2/${TS}_${MODE}_run.log
  guard "$OUT" "$LOG"
  $V scripts/run_w4a16.py --mode $MODE --out "$OUT" \
    --provenance "$(prov "run_w4a16.py --mode $MODE --group-size 128")" \
    > "$LOG" 2>&1
  echo "ARM_${MODE}_DONE rc=$?"
done

R3=data/raw/EXP-003; mkdir -p $R3
OUT=$R3/${TS}_naive.json; LOG=$R3/${TS}_naive_run.log
guard "$OUT" "$LOG"
$V scripts/run_w8a8.py --mode naive --out "$OUT" \
  --provenance "$(prov "run_w8a8.py --mode naive")" > "$LOG" 2>&1
echo "ARM_w8a8_naive_DONE rc=$?"
for A in 0.25 0.5 0.75; do
  OUT=$R3/${TS}_smooth_a${A}.json; LOG=$R3/${TS}_smooth_a${A}_run.log
  guard "$OUT" "$LOG"
  $V scripts/run_w8a8.py --mode smooth --alpha $A --out "$OUT" \
    --provenance "$(prov "run_w8a8.py --mode smooth --alpha $A")" > "$LOG" 2>&1
  echo "ARM_smooth_${A}_DONE rc=$?"
done
echo "QUANT2_ALL_DONE"
