#!/usr/bin/env bash
set -euo pipefail
cd /home/jovyan/iudin/w2v-steering
run_dir="logs/runs/2026-10-31/w2v_mlaad_english_a21_v1"
mkdir -p "$run_dir"
exec /home/jovyan/.pixi/bin/pixi run --as-is -e default python -u eval.py \
  evaluation=mlaad evaluation.device=auto \
  "steering.resume_dir=$run_dir" "hydra.run.dir=$run_dir" "$@" >> "$run_dir/launcher.log" 2>&1
