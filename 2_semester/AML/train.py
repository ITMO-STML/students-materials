import json
from argparse import ArgumentParser, Namespace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


class Bayes:
    """Bayesian heart-deasis classifier"""

    def __init__(self) -> None:
        self.state = None
        self.label2name = {1: "Heart risk", 0: "Health probs"}

    def fit(self, features: pd.DataFrame, labels: npt.ArrayLike) -> None:
        """Gen state of probabilities for Bayesian classifier
        Args:
            features (DataFrame): age, sex, cholesterin, max HR
            labels (ArrayLike): heart-deasiss
        """
        # our probs
        self.state = {}
        for label in labels.unique():
            label_features = features[labels == label]
            self.state[self.label2name[label]] = {}
            for feature in label_features.columns:
                values = label_features[feature]
                # if value discrete
                if features[feature].nunique() <= 2:
                    self.state[self.label2name[label]][feature] = values.value_counts(normalize=True).to_dict()
                # if value conitnues
                else:
                    self.state[self.label2name[label]][feature] = {"mean": values.mean(), "std": values.std()}
            self.state[self.label2name[label]]["prob"] = (labels == label).mean()

    def predict(self, features: npt.ArrayLike, plot: bool = False) -> tuple[npt.NDArray, Figure]:
        """Predicts heart deasis probability
        Args:
            features (DataFrame): age, sex, cholesterin, max HR
        """
        prob_yes = self.state[self.label2name[1]]["prob"]
        prob_no = self.state[self.label2name[0]]["prob"]
        if plot:
            fig, axs = plt.subplots(1, 4, figsize=(21, 4))

        for idx, (feature, value) in enumerate(features.items()):
            if "mean" in self.state[self.label2name[1]][feature]:
                mean_yes, std_yes = (
                    self.state[self.label2name[1]][feature]["mean"],
                    self.state[self.label2name[1]][feature]["std"],
                )
                mean_no, std_no = (
                    self.state[self.label2name[0]][feature]["mean"],
                    self.state[self.label2name[0]][feature]["std"],
                )
                self.plot_cont_features(mean_yes, std_yes, mean_no, std_no, value, axs[idx], feature)

                prob_yes *= (1 / (np.sqrt(2 * np.pi) * std_yes)) * np.exp(-0.5 * ((value - mean_yes) / std_yes) ** 2)
                prob_no *= (1 / (np.sqrt(2 * np.pi) * std_no)) * np.exp(-0.5 * ((value - mean_no) / std_no) ** 2)
            else:
                prob_yes *= self.state[self.label2name[1]][feature][int(value)]
                prob_no *= self.state[self.label2name[0]][feature][int(value)]
                self.plot_disc_features(
                    self.state[self.label2name[1]][feature][int(value)],
                    self.state[self.label2name[0]][feature][int(value)],
                    value,
                    axs[idx],
                    feature,
                )

        total = prob_yes + prob_no
        return round((prob_yes / total).item(), 4), fig

    def save(self, save_path: Path | str) -> None:
        if isinstance(save_path, Path):
            with save_path.open("w") as file:
                json.dump(self.state, file)
        else:
            with open(save_path, "w") as file:
                json.dump(self.state, file)

    def load(self, state_path: Path | str) -> None:
        if isinstance(state_path, Path):
            with state_path.open("r") as file:
                self.state = json.load(file)
        else:
            with open(state_path) as file:
                self.state = json.load(file)

    def print_state(self) -> None:
        """Printing nested state"""

        def print_nested(data, indent_level=0) -> None:
            indent = "  " * indent_level

            if isinstance(data, dict):
                for key, value in data.items():
                    print(f"{indent}{key}:")
                    print_nested(value, indent_level + 1)

            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, (dict, list)):
                        print_nested(item, indent_level)
                    else:
                        print(f"{indent}{item}")

            else:
                print(f"{indent}{round(data, 4)}")

        print_nested(self.state[self.label2name[1]])

    def plot_cont_features(
        self,
        mean_yes: float,
        std_yes: float,
        mean_no: float,
        std_no: float,
        value: int,
        ax: plt.Subplot,
        feature: str,
    ) -> None:
        """Plotting feature distribution"""
        x_min = min(mean_yes - 3 * std_yes, mean_no - 3 * std_no)
        x_max = max(mean_yes + 3 * std_yes, mean_no + 3 * std_no)
        x = np.linspace(x_min, x_max, 1000)

        y_yes = (1 / (np.sqrt(2 * np.pi) * std_yes)) * np.exp(-0.5 * ((x - mean_yes) / std_yes) ** 2)
        y_no = (1 / (np.sqrt(2 * np.pi) * std_no)) * np.exp(-0.5 * ((x - mean_no) / std_no) ** 2)

        ax.plot(x, y_yes, label="Heart risk", color="red")
        ax.plot(x, y_no, label="Health probs", color="green")
        ax.axvline(value, color="blue", linestyle="--", label=f"Value: {int(value)}")
        ax.set_title(f"{feature}")
        ax.legend(bbox_to_anchor=(0.5, 1.20), loc="upper center", ncol=3)
        ax.grid(True)

    def plot_disc_features(
        self,
        prob_yes,
        prob_no,
        value: int,
        ax: plt.Subplot,
        feature: str,
    ) -> None:
        """Plotting bar distribution"""
        ax.bar(["Healthy", "Ill"], [prob_no, prob_yes])
        ax.axvline(value, color="blue", linestyle="--", label=f"Value: {int(value)}")
        ax.set_title(f"{feature}")
        ax.legend(bbox_to_anchor=(0.5, 1.20), loc="upper center", ncol=3)
        ax.grid(True)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--df_path", type=str, help="Path to csv file with data", required=True)
    parser.add_argument("--state_save_path", type=str, default="weights/state.json", help="Path to save model state")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main(args: dict) -> None:
    data = pd.read_csv(args.df_path)
    X = data[["Sex", "Age", "Cholesterol", "Max HR"]]

    y = data["Heart Disease"] == "Presence"

    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)
    bayes = Bayes()
    bayes.fit(x_train, y_train)
    mode = y.mode()
    print("Mode classification report")
    print(classification_report(y_test, mode))

    print("\n\nTrained model classification report")
    preds = []
    for _, features in x_test.iterrows():
        pred, _ = bayes.predict(features) > 0.5
        preds.append(pred)
    print(classification_report(y_test, preds))

    print("\nSaving model state")
    bayes.save(args.state_save_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
