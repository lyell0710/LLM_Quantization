#!/bin/bash
# EXP-001（GPTQ 从零实现）三臂批跑(fp16/rtn/gptq)。
# 铁律 3/4(2026-08-24 审计加固):
#   - raw 不覆盖:输出一律 UTC 前缀新文件,同名已存在则拒跑;
#   - provenance sha 必须指向真实代码版本:worktree dirty 拒跑(EXP-001 §7 勘注教训)。
set -u
cd /root/projects/LLM_Quantization
V=/root/venvs/v0.25.1/bin/python
RAW=data/raw/EXP-001
mkdir -p $RAW
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || { echo "FATAL: 非 git 仓,无法记 provenance sha" >&2; exit 1; }
[ -z "$(git status --porcelain)" ] || { echo "FATAL: worktree dirty——先 commit 再跑(provenance sha 必须指向真实代码版本)" >&2; exit 1; }
SHA=$(git rev-parse --short=10 HEAD)
TS=$(date -u +%Y%m%dT%H%M)  # 同次测量共用 UTC 前缀(铁律 4)
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
for MODE in fp16 rtn gptq; do
  OUT=$RAW/${TS}_${MODE}_g128.json
  LOG=$RAW/${TS}_${MODE}_run.log
  for f in "$OUT" "$LOG"; do
    [ -e "$f" ] && { echo "FATAL: $f 已存在,拒绝覆盖 raw" >&2; exit 1; }
  done
  PROV="# provenance: env=v0.25.1-venv sha=$SHA cmd=\"run_w4a16.py --mode $MODE --group-size 128\" date=$(date -u +%Y-%m-%dT%H:%M:%S+00:00) gpu=\"$GPU\" driver=$DRV"
  $V scripts/run_w4a16.py --mode $MODE --out "$OUT" --provenance "$PROV" > "$LOG" 2>&1
  echo "ARM_${MODE}_DONE rc=$?"
done
echo "QUANT_ALL_DONE"
