#!/usr/bin/env bash
# Complete development and WildSpoof scoring after the authorized CLEAR fit.
set -euo pipefail
cd /home/jovyan/iudin/w2v-steering
pixi=/home/jovyan/.pixi/bin/pixi
fit=logs/runs/2026-11-06/clear_projected_fit_v1
queue=logs/runs/2026-11-06/clear_followup_v1
mkdir -p "$queue"
exec 9>"$queue/.pipeline.lock"
flock -n 9
status() {
  "$pixi" run --as-is python -c 'import json,sys; from datetime import datetime,timezone; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps({"stage":sys.argv[2],"updated_at":datetime.now(timezone.utc).isoformat()},indent=2)+"\n")' "$queue/status.json" "$1"
}
trap 'status failed' ERR
status waiting_for_fit
while tmux has-session -t w2v-clear-fit 2>/dev/null; do
  sleep 10
done
"$pixi" run --as-is python -c 'import json,sys; from pathlib import Path; result=json.loads((Path(sys.argv[1])/"selection.json").read_text()); assert result["status"]=="complete" and result["mode"]=="projected"' "$fit"
status full_development
"$pixi" run --as-is python -u eval.py evaluation=asvspoof5_dev \
  steering.clear_run="$fit" evaluation.device=auto \
  hydra.run.dir=logs/runs/2026-11-06/w2v_full_dev_clear_v1
"$pixi" run --as-is python -m src.dev_eval.reporting
status wildspoof_best
"$pixi" run --as-is python -u eval.py evaluation=retention_a21 \
  evaluation.adaptation_run="$fit" evaluation.selection=best_a21 evaluation.cohort=external \
  evaluation.manifest_run=logs/runs/2026-10-18/w2v_wildspoof2026_transfer evaluation.device=auto \
  hydra.run.dir=logs/runs/2026-11-06/clear_wildspoof_best_v1
"$pixi" run --as-is python -m src.dev_eval.reporting
if "$pixi" run --as-is python -c 'import json,sys; from pathlib import Path; h=json.loads((Path(sys.argv[1])/"selection.json").read_text())["checkpoint_sha256"]; sys.exit(0 if h["guarded"] not in {h["baseline"],h["best_a21"]} else 1)' "$fit"; then
  status wildspoof_guarded
  "$pixi" run --as-is python -u eval.py evaluation=retention_a21 \
    evaluation.adaptation_run="$fit" evaluation.selection=guarded evaluation.cohort=external \
    evaluation.manifest_run=logs/runs/2026-10-18/w2v_wildspoof2026_transfer evaluation.device=auto \
    hydra.run.dir=logs/runs/2026-11-06/clear_wildspoof_guarded_v1
fi
"$pixi" run --as-is python -m src.dev_eval.reporting
status complete
