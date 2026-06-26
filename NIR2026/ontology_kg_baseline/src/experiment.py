from src.metrics import MetricsCalculator


class Experiment:
    def __init__(
        self,
        examples: list[dict],
        ontology_relations: dict,
        extractors: list,
        validator,
        writer,
        dataset_name: str,
        model_name: str
    ):
        self.examples = examples
        self.ontology_relations = ontology_relations
        self.extractors = extractors
        self.validator = validator
        self.writer = writer
        self.dataset_name = dataset_name
        self.model_name = model_name

        self.method_names = [
            extractor.name
            for extractor in self.extractors
        ]

        self.method_names.extend([
            "D_domain_range_validator"
        ])

    def run(self):
        results = []

        metric_lists = {
            name: []
            for name in self.method_names
        }

        fact_metric_lists = {
            name: []
            for name in self.method_names
        }

        validity_lists = {
            name: []
            for name in self.method_names
        }

        print(f"Loaded {len(self.examples)} examples\n")

        for i, example in enumerate(self.examples, start=1):
            example_id = example["id"]
            text = example["sent"]
            gold = example["triples"]

            print("=" * 70)
            print(f"Example {i}: {example_id}")
            print(text)
            print()

            predictions = {}

            for extractor in self.extractors:
                print(
                    f"[{i}/{len(self.examples)}] "
                    f"{example_id} | Running {extractor.name}...")
                predictions[extractor.name] = extractor.extract(text)
                print(
                    f"[{i}/{len(self.examples)}] "
                    f"{example_id} | {extractor.name} done | "
                    f"Triples: {len(predictions[extractor.name])}")

                print()

            domain_range_prediction = predictions["C_domain_range"]

            print(
                f"[{i}/{len(self.examples)}] "
                f"{example_id} | Running validator...")

            validated_prediction, rejected = self.validator.validate(
                domain_range_prediction
            )
            print(
                f"[{i}/{len(self.examples)}] "
                f"{example_id} | Validator done | "
                f"Valid={len(validated_prediction)} "
                f"Rejected={len(rejected)}")

            predictions["D_domain_range_validator"] = validated_prediction

            metrics = {}
            fact_metrics = {}
            validity = {}

            for method_name, triples in predictions.items():
                metrics[method_name] = MetricsCalculator.compute_strict_metrics(
                    triples,
                    gold
                )

                fact_metrics[method_name] = MetricsCalculator.compute_fact_metrics(
                    triples,
                    gold
                )

                validity[method_name] = self.validator.ontology_validity(
                    triples
                )

                metric_lists[method_name].append(metrics[method_name])
                fact_metric_lists[method_name].append(fact_metrics[method_name])
                validity_lists[method_name].append(validity[method_name])

            for method_name in predictions:
                print(
                    method_name,
                    "| Strict F1:",
                    round(metrics[method_name]["f1"], 3),
                    "| Fact F1:",
                    round(fact_metrics[method_name]["fact_f1"], 3),
                    "| Ontology validity:",
                    round(validity[method_name], 3)
                )

            print()

            results.append({
                "id": example_id,
                "text": text,
                "gold": gold,
                "predictions": predictions,
                "rejected_by_validator": rejected,
                "strict_metrics": metrics,
                "fact_metrics": fact_metrics,
                "ontology_validity": validity
            })

        summary = self._build_summary(
            metric_lists,
            fact_metric_lists,
            validity_lists
        )

        self.writer.save_all(results, summary)

        print("=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)

        for method_name, values in summary["methods"].items():
            print(
                method_name,
                "| Strict F1:",
                round(values["f1"], 3),
                "| Fact F1:",
                round(values["fact_f1"], 3),
                "| Ontology validity:",
                round(values["ontology_validity"], 3)
            )

        return results, summary

    def _build_summary(
        self,
        metric_lists: dict,
        fact_metric_lists: dict,
        validity_lists: dict
    ):
        summary = {
            "dataset": self.dataset_name,
            "model": self.model_name,
            "methods": {}
        }

        for method_name in metric_lists:
            summary["methods"][method_name] = {
                **self._average_strict(metric_lists[method_name]),
                **self._average_fact(fact_metric_lists[method_name]),
                "ontology_validity": self._average(validity_lists[method_name])
            }

        return summary

    @staticmethod
    def _average(values):
        if not values:
            return 0

        return sum(values) / len(values)

    @staticmethod
    def _average_strict(metrics_list):
        if not metrics_list:
            return {
                "precision": 0,
                "recall": 0,
                "f1": 0
            }

        return {
            "precision": sum(m["precision"] for m in metrics_list) / len(metrics_list),
            "recall": sum(m["recall"] for m in metrics_list) / len(metrics_list),
            "f1": sum(m["f1"] for m in metrics_list) / len(metrics_list)
        }

    @staticmethod
    def _average_fact(metrics_list):
        if not metrics_list:
            return {
                "fact_precision": 0,
                "fact_recall": 0,
                "fact_f1": 0
            }

        return {
            "fact_precision": sum(m["fact_precision"] for m in metrics_list) / len(metrics_list),
            "fact_recall": sum(m["fact_recall"] for m in metrics_list) / len(metrics_list),
            "fact_f1": sum(m["fact_f1"] for m in metrics_list) / len(metrics_list)
        }