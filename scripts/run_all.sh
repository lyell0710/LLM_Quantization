#!/bin/bash
cd /root/projects/LLM_Quantization
V=/root/venvs/v0.25.1/bin/python
RAW=data/raw/EXP-001
mkdir -p $RAW
SHA=$(git rev-parse --short=10 HEAD 2>/dev/null || echo worktree)
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
for MODE in fp16 rtn gptq; do
  PROV="# provenance: env=v0.25.1-venv sha=$SHA cmd=\"run_w4a16.py --mode $MODE --group-size 128\" date=$(date -u +%Y-%m-%dT%H:%M:%S+00:00) gpu=\"$GPU\" driver=$DRV"
  $V scripts/run_w4a16.py --mode $MODE --out $RAW/${MODE}_g128.json --provenance "$PROV" > $RAW/${MODE}_run.log 2>&1
  echo "ARM_${MODE}_DONE rc=$?"
done
echo "QUANT_ALL_DONE"
