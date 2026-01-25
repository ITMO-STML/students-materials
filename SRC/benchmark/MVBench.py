from tqdm import trange
import json

from typing import Iterable
from pathlib import Path
import numpy as np
from moviepy import VideoFileClip, ImageClip, concatenate_videoclips
from glob import glob
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import re

from .benchmark_abc import BenchmarkABC
from model.model_abc import ModelABC
from configs import MVBenchConfig

class MVBench(BenchmarkABC):
    def __init__(self, resolution: int, cfg: MVBenchConfig = MVBenchConfig()) -> None:
        self.cfg = cfg
        self.resolution = int(resolution)


    def eval(self, data: Iterable, model: ModelABC, metric_path: str | Path):
        with open(metric_path, "r", encoding="utf-8") as file:
            start = len(json.load(file))
        
        for idx in trange(start, data.shape[0]):
            row = data.iloc[idx]
            if row.get(model.name(), None) is not None:
                continue
            question = "You are an advanced video question-answering AI assistant. You have been provided with some frames from the video and a multiple-choice question related to the video. Your task is to carefully analyze the video and provide the best answer to question, choosing from the options provided. Respond with only the letter (A, B, C, or D) of the correct option.\n"
            question += f"Question: {row['question']}. \n"
            question += "Options:\n"
            list_candidates = eval(row['candidates'])
            for idx, c in enumerate(list_candidates):
                question += f"({chr(ord('A') + idx)}) {c}\n"

            question = question.rstrip()

            video_path = self.cfg.dataset_dir + "/" + row["prefix"] + row["video"]
            if Path(video_path).is_dir():
                clips = [ImageClip(clip).with_duration(0.3) for clip in glob(video_path+"/*.jpg")]
                video = concatenate_videoclips(clips, method="compose")
            else:
                video = VideoFileClip(video_path)
            if video.duration > (max(0, row['start'])):
                video = video.subclipped(max(0, row['start']), min(video.duration, row['end']))
            clip = video.resized((self.resolution, self.resolution))
            model_ans = model.run(clip, question)
            gt_ans = chr(ord("A") + list_candidates.index(row["answer"]))
            with open(metric_path, "r", encoding="utf-8") as file:
                existing_data = json.load(file)

            existing_data.extend([{"model": model_ans, "gt": gt_ans}])
            with open(metric_path, "w", encoding="utf-8") as file:
                json.dump(existing_data, file, ensure_ascii=False, indent=4)
        self.compute_metrics(metric_path)


    def get_index(self, bound, fps, max_frame, first_idx=0):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / self.num_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(self.num_segments)
        ])
        return frame_indices
    
    def compute_metrics(self, metrics: dict | str | Path) -> tuple[float,...]:
        if isinstance(metrics, (str, Path)):
            with open(metrics, "r") as file:
                metrics = json.load(file)
        preds = []
        true = []
        letter_A = ord("A")
        for question in metrics:
            if len(question["model"]) > 1:
                model_ans = re.search(r"\([A-Z]{1}\)", question["model"])
                if model_ans is not None:
                    model_ans = model_ans.group(0)[1:-1]
                else:
                    model_ans = chr(ord("A") - 1)
            else:
                model_ans = question["model"]
            preds.append(ord(model_ans) - letter_A + 1)
            true.append(ord(question["gt"]) - letter_A + 1)

        ac = accuracy_score(true, preds)
        pr, rec, fs, _ = precision_recall_fscore_support(true, preds, average="macro", zero_division=0.)
        print(f"Accuracy: {ac:.4f}" )
        print(f"Precision: {pr:.4f}", )
        print(f"Recall: {rec:.4f}", )
        print(f"F1: {fs:.4f}", )

    
