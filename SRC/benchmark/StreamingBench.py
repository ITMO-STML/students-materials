from tqdm import tqdm
import json
from typing import Iterable
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from .benchmark_abc import BenchmarkABC
from .utils import split_video, convert_time, convert_path
from model.model_abc import ModelABC
from configs import StreamingBenchConfig
from moviepy import VideoFileClip


class StreamingBench(BenchmarkABC):
    def __init__(self, cfg: StreamingBenchConfig = StreamingBenchConfig()) -> None:
        self.cfg = cfg

    def eval(self, data: Iterable, model: ModelABC, metric_path: str | Path):
        for idx, subset in enumerate(tqdm(data)):
            for question in subset["questions"]:
                if question.get(model.name(), False):
                    continue

                video_path = convert_path(subset["video_path"], self.cfg.dataset_dir)

                time_start, time_end = subset["time"].split(" - ")

                if time_start.strip().startswith("["):
                    time_start = time_start[1:]
                if time_end.strip().endswith("]"):
                    time_end = time_end[:-1]

                time_start = convert_time(time_start)
                time_end = convert_time(time_end)
                file = split_video(video_path, time_start, time_end)
                if file is None:
                    continue

                question_time = convert_time(question["time_stamp"])
                video = VideoFileClip(file)
                if time_start <= question_time < time_end:
                    video = video.subclipped(0, min(question_time - time_start + 2, video.duration))

                ques = question["question"]
                if "options" in question.keys():
                    options = question["options"]
                    if not options[0].startswith("A."):
                        options = [f"A. {options[0]}", f"B. {options[1]}", f"C. {options[2]}", f"D. {options[3]}"]
                    inp = self.cfg.PROMPT_TEMPLATE.format(ques, *options)
                    inp += "\n\nThe best option is:"
                else:
                    inp = self.cfg.PROMPT_TEMPLATE_WITHOUT_OPTIONS.format(ques)
                    inp += "\n\nAnswer:"

                response = model.run(video, inp)
                video.close()
                question[model.name()] = response

            if idx % 5 == 0:
                with open(metric_path, "w") as f:
                    json.dump(data, f, indent=4)

        self.compute_metrics(metric_path, model.name())

    def compute_metrics(self, metrics: str | Path | dict, model_name: str) -> None:
        if isinstance(metrics, (str, Path)):
            with open(metrics, "r") as file:
                metrics = json.load(file)
        preds = []
        true = []
        ans_dict = {"A": 1, "B": 2, "C": 3, "D": 4}

        for video in metrics:
            for question in video["questions"]:
                model_ans = question.get(model_name, False)
                if model_ans:
                    if "," in model_ans or len(model_ans) != 1:
                        preds.append(0)
                    else:
                        preds.append(ans_dict.get(model_ans, 0))
                    true.append(ans_dict[question.get("answer")])

        ac = accuracy_score(true, preds)
        pr, re, fs, _ = precision_recall_fscore_support(true, preds, average="macro", zero_division=0.)
        print(f"Accuracy: {ac:.4f}")
        print(f"Precision: {pr:.4f}")
        print(f"Recall: {re:.4f}")
        print(f"F-Score: {fs:.4f}")
