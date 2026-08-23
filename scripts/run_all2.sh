#!/bin/bash
cd /root/projects/LLM_Quantization
V=/root/venvs/v0.25.1/bin/python
SHA=$(git rev-parse --short=10 HEAD)
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
prov() { echo "# provenance: env=v0.25.1-venv sha=$SHA cmd=\"$1\" date=$(date -u +%Y-%m-%dT%H:%M:%S+00:00) gpu=\"$GPU\" driver=$DRV"; }

R2=data/raw/EXP-002; mkdir -p $R2
for MODE in awq awq_gptq; do
  $V scripts/run_w4a16.py --mode $MODE --out $R2/${MODE}_g128.json \
    --provenance "$(prov "run_w4a16.py --mode $MODE --group-size 128")" \
    > $R2/${MODE}_run.log 2>&1
  echo "ARM_${MODE}_DONE rc=$?"
done

R3=data/raw/EXP-003; mkdir -p $R3
$V scripts/run_w8a8.py --mode naive --out $R3/naive.json \
  --provenance "$(prov "run_w8a8.py --mode naive")" > $R3/naive_run.log 2>&1
echo "ARM_w8a8_naive_DONE rc=$?"
for A in 0.25 0.5 0.75; do
  $V scripts/run_w8a8.py --mode smooth --alpha $A --out $R3/smooth_a${A}.json \
    --provenance "$(prov "run_w8a8.py --mode smooth --alpha $A")" > $R3/smooth_a${A}_run.log 2>&1
  echo "ARM_smooth_${A}_DONE rc=$?"
done
echo "QUANT2_ALL_DONE"
