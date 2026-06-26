from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class MetricsPlotter:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)

    def plot_all(self, summary: dict):
        df = self._summary_to_dataframe(summary)

        self._plot_fact_metrics(df)
        self._plot_strict_metrics_ontology_only(df)
        self._plot_single_metric(df, "precision", "Precision by method", "precision", "precision_by_method.png")
        self._plot_single_metric(df, "recall", "Recall by method", "recall", "recall_by_method.png")
        self._plot_single_metric(df, "f1", "F1 score by method", "f1", "f1_by_method.png")
        self._plot_single_metric(df, "ontology_validity", "Ontology validity by method", "ontology_validity", "ontology_validity_by_method.png")
        self._plot_all_metrics(df)

    def _summary_to_dataframe(self, summary: dict) -> pd.DataFrame:
        rows = []

        for method_name, values in summary["methods"].items():
            rows.append({
                "method": method_name,
                "precision": values["precision"],
                "recall": values["recall"],
                "f1": values["f1"],
                "fact_precision": values["fact_precision"],
                "fact_recall": values["fact_recall"],
                "fact_f1": values["fact_f1"],
                "ontology_validity": values["ontology_validity"]
            })

        return pd.DataFrame(rows)

    def _add_bar_labels(self, ax):
        for container in ax.containers:
            ax.bar_label(
                container,
                fmt="%.3f",
                padding=3,
                fontsize=9
            )

    def _method_labels_ru(self, methods):
        mapping = {
            "A_no_ontology": "Без онтологии",
            "B_relation_names": "Отношения\nиз онтологии",
            "C_domain_range": "Отношения +\nDomain/Range",
            "D_domain_range_validator": "Отношения +\nDomain/Range +\nValidator",
            "E_domain_range_validator_repair": "Отношения +\nDomain/Range +\nRepair"
        }

        return [mapping.get(method, method) for method in methods]

    def _setup_axis(self, ax, title, ylabel="Значение"):
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

    def _plot_fact_metrics(self, df: pd.DataFrame):
        labels = self._method_labels_ru(df["method"])

        plot_df = pd.DataFrame({
            "Метод": labels,
            "Fact Precision": df["fact_precision"],
            "Fact Recall": df["fact_recall"],
            "Fact F1": df["fact_f1"]
        })

        ax = plot_df.plot(
            x="Метод",
            y=["Fact Precision", "Fact Recall", "Fact F1"],
            kind="bar",
            figsize=(10, 4.5)
        )

        self._setup_axis(
            ax,
            "Fact Precision, Fact Recall и Fact F1"
        )

        ax.legend(title="Метрика", loc="lower center")
        self._add_bar_labels(ax)

        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.output_dir / "chart_fact_metrics.png", dpi=300)
        plt.close()

    def _plot_strict_metrics_ontology_only(self, df: pd.DataFrame):
        ontology_df = df[
            df["method"].isin([
                "B_relation_names",
                "C_domain_range",
                "D_domain_range_validator"
            ])
        ].copy()

        labels = self._method_labels_ru(ontology_df["method"])

        plot_df = pd.DataFrame({
            "Метод": labels,
            "Precision": ontology_df["precision"],
            "Recall": ontology_df["recall"],
            "F1": ontology_df["f1"]
        })

        ax = plot_df.plot(
            x="Метод",
            y=["Precision", "Recall", "F1"],
            kind="bar",
            figsize=(10, 4.5)
        )

        self._setup_axis(
            ax,
            "Сравнение методов с использованием онтологии"
        )

        ax.legend(title="Метрика")
        self._add_bar_labels(ax)

        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.output_dir / "chart_strict_metrics_ontology_only.png", dpi=300)
        plt.close()

    def _plot_single_metric(
        self,
        df: pd.DataFrame,
        metric: str,
        title: str,
        ylabel: str,
        filename: str
    ):
        ax = df.plot(
            x="method",
            y=metric,
            kind="bar",
            figsize=(10, 5),
            legend=False
        )

        ax.set_title(title, fontsize=14)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Method")
        ax.set_ylim(0, 1.0)

        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300)
        plt.close()

    def _plot_all_metrics(self, df: pd.DataFrame):
        plot_df = df[[
            "method",
            "precision",
            "recall",
            "f1",
            "ontology_validity"
        ]]

        ax = plot_df.plot(
            x="method",
            y=[
                "precision",
                "recall",
                "f1",
                "ontology_validity"
            ],
            kind="bar",
            figsize=(11, 5)
        )

        ax.set_title("Comparison of KG generation methods", fontsize=14)
        ax.set_ylabel("Score")
        ax.set_xlabel("Method")
        ax.set_ylim(0, 1.0)
        ax.legend()

        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(self.output_dir / "all_metrics_comparison.png", dpi=300)
        plt.close()