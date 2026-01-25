from tqdm import trange
import json
from typing import Iterable
from pathlib import Path

from .benchmark_abc import BenchmarkABC
from model.model_abc import ModelABC
from configs import VizWizConfig
from moviepy import ImageClip
from PIL import Image


class VizWiz(BenchmarkABC):
    def __init__(self, cfg: VizWizConfig = VizWizConfig()) -> None:
        self.cfg = cfg

    def eval(self, data: Iterable, model: ModelABC, metric_path: str | Path):
        with open(metric_path, "r", encoding="utf-8") as file:
                start = len(json.load(file))

        for idx in trange(start, len(data)):
            img = data[idx]
            question = self.cfg.PROMPT_UNCERTANTY.format(img["question"])
            image = Image.fromarray(ImageClip(self.cfg.dataset_dir + img["image"]).resized((448, 448)).get_frame(0))
            response = model.run(image, question)

            with open(metric_path, "r", encoding="utf-8") as file:
                existing_data = json.load(file)
            curr_data = [{
                "question": question,
                "model_ans": response,
                "user_ans": img["answers"]
            }]
            existing_data.extend(curr_data)
            with open(metric_path, "w", encoding="utf-8") as file:
                json.dump(existing_data, file, ensure_ascii=False, indent=4)
