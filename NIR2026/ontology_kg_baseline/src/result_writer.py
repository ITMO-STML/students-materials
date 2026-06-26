import csv
import json
from pathlib import Path

from src.plotter import MetricsPlotter
class ResultWriter:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)

    def save_results(self, results: list[dict]):
        with open(self.output_dir / "results_full.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    def save_summary(self, summary: dict):
        with open(self.output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    def save_metrics_csv(self, summary: dict):
        with open(self.output_dir / "metrics.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)

            writer.writerow([
                "method",
                "precision",
                "recall",
                "f1",
                "fact_precision",
                "fact_recall",
                "fact_f1",
                "ontology_validity"
            ])

            for method_name, values in summary["methods"].items():
                writer.writerow([
                    method_name,
                    values["precision"],
                    values["recall"],
                    values["f1"],
                    values["fact_precision"],
                    values["fact_recall"],
                    values["fact_f1"],
                    values["ontology_validity"]
                ])

    def save_all(self, results: list[dict], summary: dict):
        self.save_results(results)
        self.save_summary(summary)
        self.save_metrics_csv(summary)

        plotter = MetricsPlotter(self.output_dir)
        plotter.plot_all(summary)