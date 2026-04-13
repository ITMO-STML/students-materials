import typing as tp
from argparse import ArgumentParser, Namespace
from pathlib import Path

import gradio as gr

from train import Bayes


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--state_path", type=str, hepl="Path to Bayes state.json", required=True)
    return parser.parse_args()


args = parse_args()
model = Bayes()
if not Path(args.state_path).exists():
    print(f"No model state found {args.state_path}")
    exit()
model.load(args.state_path)


def predict_heart_risk(age: int, sex: tp.Literal[0, 1], cholesterol: int, max_hr: int) -> dict:
    features = {"Age": age, "Sex": sex, "Cholesterol": cholesterol, "Max HR": max_hr}
    prob, fig = model.predict(features)

    prediction = {model.label2name[1]: prob, model.label2name[0]: round(1 - prob, 4)}

    return prediction, fig


with gr.Blocks() as demo:
    gr.Markdown("# Heart Disease Risk Predictor")
    gr.Markdown("Enter your parameters to get a risk assessment.")

    with gr.Row():
        age = gr.Number(label="Age", value=50, minimum=28, maximum=78, info="Age should be in range [28, 78]")
        sex = gr.Radio(label="Sex", choices=[0, 1], value=1, info="0: Female, 1: Male")

        cholesterol = gr.Number(
            label="Cholesterol",
            value=200,
            minimum=147,
            maximum=340,
            info="Cholesterol should be in range [147, 340]",
        )
        max_hr = gr.Number(
            label="Max Heart Rate",
            value=150,
            minimum=85,
            maximum=150,
            info="Max Heart Rate should be in range [85, 150]",
        )

    submit = gr.Button("Predict")

    # Вывод вероятностей
    output = gr.Label(label="Prediction", num_top_classes=2)
    # Вывод графиков распределений
    plot_output = gr.Plot(label="Feature Distributions")

    submit.click(fn=predict_heart_risk, inputs=[age, sex, cholesterol, max_hr], outputs=[output, plot_output])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)
