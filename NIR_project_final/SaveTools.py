import json
from pathlib import Path

from project_paths import TABLES_DIR, ensure_project_dirs


def save_experiment_info(path, parameters, data, metrics, uid):
    ensure_project_dirs()
    output_dir = Path(path) if path else TABLES_DIR
    if output_dir.name == "results":
        output_dir = TABLES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    file = output_dir / f"{uid}.json"
    payload = {
        "parameters": parameters,
        "data": data,
        "metrics": metrics,
    }

    with file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"File created: {file}")
