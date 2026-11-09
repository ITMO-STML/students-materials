#!/usr/bin/env bash
set -euo pipefail
cd /home/jovyan/iudin/w2v-steering
run_dir="logs/runs/2026-10-19/w2v_voxenes_original_english_v1"
mkdir -p "$run_dir"
exec /home/jovyan/.pixi/bin/pixi run --as-is -e default python -u eval.py \
  evaluation=voxenes evaluation.extract=true evaluation.device=auto \
  "steering.resume_dir=$run_dir" "hydra.run.dir=$run_dir" "$@" >> "$run_dir/pipeline.log" 2>&1
