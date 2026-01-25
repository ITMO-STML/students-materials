import json
from typing import Iterable, Iterator
from pathlib import Path
from enum import Enum
import random

from tqdm import trange
import numpy as np
from moviepy import VideoFileClip
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from .benchmark_abc import BenchmarkABC
from model.model_abc import ModelABC
from configs import OmniMMIConfig


class OmniMMIType(Enum):
    PA = "proactive"
    MVQA = "multichoice vqa"
    YesNoVQA = "yes/no vqa"


class OmniMMIBench(BenchmarkABC):
    def __init__(self, cfg: OmniMMIConfig = OmniMMIConfig(), task_type: str | OmniMMIType = "proactive"):
        self.cfg = cfg
        self.task_type = OmniMMIType(task_type)
        match self.task_type:
            case self.task_type.PA:
                self.__get_prompt = self.get_pa_prompt
                self.__ts = False
            case self.task_type.MVQA:
                self.__get_prompt = self.get_mc_prompt
                self.__ts = "mutlichoice timestamps"
            case self.task_type.YesNoVQA:
                self.__get_prompt = self.get_yn_prompt
                self.__ts = "yes/no timestamps"


    def get_video(self, video_row: dict) -> Iterator:
        video_row['index'] = video_row['video'].split('.')[0]
        video_path = str(Path(self.cfg.dataset_dir, video_row['video']))
        video = VideoFileClip(video_path)
        clip = video.with_fps(self.cfg.output_fps)
        if self.__ts:
            clip = clip.subclipped(*video_row[self.__ts])

        return clip, video_row

    def get_yn_prompt(self, video_row: dict) -> str:
        self.gt_ans = video_row['yes/no answer']
        return self.cfg.YES_NO_PROACTIVE.format(video_row['yes/no question'])
    
    def get_mc_prompt(self, video_row: dict) -> str:
        avaialbe_options = video_row["multichoice options"][1:-1].split(", ")
        add = False
        if "none" not in avaialbe_options[-1].lower():
            add = True

        random.shuffle(avaialbe_options)
        letters = [chr(ord("A")+i) for i in range(len(avaialbe_options))]
        processed_options = "\n".join([letter + ": " + str(opt)
                        for letter, opt in zip(letters, avaialbe_options)])
        self.gt_ans = letters[avaialbe_options.index(video_row["multichoice answer"])]

        if add:
            processed_options += f"\n{chr(len(avaialbe_options)+65)}: None of these"

        return self.cfg.MULTICHOICE_PROACTIVE.format(video_row['multichoice question'], processed_options)

    def get_pa_prompt(self, video_row: dict) -> str:
        self.gt_ans = video_row["proactive answer"]
        return video_row['proactive question'] + self.cfg.ALERTING_PROACTIVE

    def get_proactive_item(self, clip: VideoFileClip, video_row: dict) -> Iterator:
        """
        Task:
        Proactive alerting

        Input:
        clip -- video data
        video_row -- question, answers, timestamps data
        """
        yield video_row

        prompt = self.__get_prompt(video_row)

        data = {'image': None, 'decision_question': prompt}
        yield data

        for j, frame in enumerate(clip.iter_frames()):
            frame_time = j * clip.fps
            image = Image.fromarray((frame).astype(np.uint8))
            data = {'image': image, 'answer': True, 'frame_time': frame_time}

            yield data


    def proactive_eval(self, data: dict, model: ModelABC) -> tuple[str, list[int]]:
        clip, video_row = self.get_video(data)
        hist = model.run(self.get_proactive_item(clip, video_row))
        return hist, self.gt_ans


    def eval(self, data: Iterable, model: ModelABC, metric_path: str | Path):
        """Evaluation of OmniMMI

        Args:
            data: ...
            model: ...
            metric_path: ...
        """
        random.seed(42)

        for i in trange(len(data)):
            d = data[i]
            if d["content_type"] != "video":
                continue
            model_ans, gt_ans = self.proactive_eval(d, model)
            with open(metric_path, "r", encoding="utf-8") as file:
                existing_data = json.load(file)

            existing_data.extend([{"model": model_ans, "gt": gt_ans}])
            with open(metric_path, "w", encoding="utf-8") as file:
                json.dump(existing_data, file, ensure_ascii=False, indent=4)
        self.compute_metrics(metric_path)
    
    def compute_metrics(self, metrics: str | Path | dict) -> None:
        if isinstance(metrics, (str, Path)):
            with open(metrics, "r") as file:
                metrics = json.load(file)

        match self.task_type:
            case self.task_type.PA:
                self.compute_proactive(metrics)
            case self.task_type.MVQA:
                ac, pr, re, f1 = self.compute_multichoice(metrics)
            case self.task_type.YesNoVQA:
                ac, pr, re, f1 = self.compute_yesno(metrics)

        if self.task_type == self.task_type.MVQA:
            print("OmniMMI:", self.task_type)
            print(f"Accuracy: {ac:.4f}" )
            print(f"Precision: {pr:.4f}", )
            print(f"Recall: {re:.4f}", )
            print(f"F1: {f1:.4f}", )

        elif self.task_type == self.task_type.YesNoVQA:
            print(f"Accuracy: {ac:.4f}" )
            for i in range(2):
                print(f"Precision_{i}: {pr[i]:.4f}", )
                print(f"Recall_{i}: {re[i]:.4f}", )
                print(f"F1_{i}: {f1[i]:.4f}", )


    def compute_yesno(self, metrics: dict) -> tuple[float,...]:
        pred = []
        gt = []
        for d in metrics:
            pred.append(d["model"].lower().strip() == "yes")
            gt.append(d["gt"].lower().strip() == "yes")
        
        ac = accuracy_score(gt, pred)
        pr, re, f1, _ = precision_recall_fscore_support(gt, pred)
        return ac, pr, re, f1


    def compute_multichoice(self, metrics: dict) -> tuple[float,...]:
        preds = []
        true = []
        ans_dict = {"A": 1, "B": 2, "C": 3, "D": 4}

        for question in metrics:
            preds.append(ans_dict.get(question["model"], 0))
            true.append(ans_dict[question["gt"]])

        ac = accuracy_score(true, preds)
        pr, re, fs, _ = precision_recall_fscore_support(true, preds, average="macro", zero_division=0.)
        return ac, pr, re, fs


    def compute_proactive(self, metrics: dict) -> tuple[float,...]:
        self.get_framewise_score(metrics)
        self.get_intervalwise_score(metrics)


    def get_framewise_score(self, metrics: dict, eps: float = 1e-8) -> None:
        """
        Calculates score per frame in all videos and then averaging it
        """
        precisions1, recalls1, precisions0, recalls0 = [], [], [], []
        accuracies = []

        for vid in metrics:
            if self.task_type == self.task_type.PA:
                start, end = vid["gt"]
            else:
                start_yn, end_yn = vid["model"][0][self.__ts]
                if start_yn == end_yn:
                    continue
                start_orig, end_orig = vid["model"][0]["proactive answer"]
                start = start_orig - start_yn
                end = end_orig - end_yn

            tp1 = fp1 = tp0 = fp0 = 0
            preds = vid["model"][0]["model_answers"]

            if self.task_type == self.task_type.MVQA:
                preds = np.array([1 if frame_pred == vid["gt"] else 0 for frame_pred in preds], dtype=np.uint8)
            else:
                preds = np.array([1 if frame_pred.lower() == "yes" else 0 for frame_pred in preds], dtype=np.uint8)
            if len(preds) == 0:
                print(start, end)
                continue

            gt_frames = np.zeros(len(preds))
            gt_frames[start:end + 1] = 1

            tp1 = np.sum(gt_frames * preds)
            fp1 = np.sum((1 - gt_frames) * preds)

            tp0 = np.sum((1 - gt_frames) * (1 - preds))
            fp0 = np.sum(gt_frames * (1 - preds))

            precision1 = tp1 / (tp1 + fp1 + eps)
            if np.isnan(precision1):
                precision1 = 0.0
            recall1 = tp1 / (sum(gt_frames) + eps)

            precision0 = tp0 / (tp0 + fp0 + eps)
            if np.isnan(precision0):
                precision0 = 0.0
            recall0 = tp0 / (sum(1 - gt_frames) + eps)

            accuracy = (tp1 + tp0) / len(preds)

            precisions1.append(precision1)
            recalls1.append(recall1)

            precisions0.append(precision0)
            recalls0.append(recall0)

            accuracies.append(accuracy)

        macro_precision_1 = np.mean(precisions1)
        macro_recall_1 = np.mean(recalls1)
        macro_f1_1 = 2 * macro_precision_1 * macro_recall_1 / (macro_precision_1 + macro_recall_1 + eps) 

        macro_precision_0 = np.mean(precisions0)
        macro_recall_0 = np.mean(recalls0)
        macro_f1_0 = 2 * macro_precision_0 * macro_recall_0 / (macro_precision_0 + macro_recall_0 + eps)

        accuracy = np.mean(accuracies)

        print('accuracy_framewise', np.round(accuracy, 4))
        print('precision_1_framewise', np.round(macro_precision_1, 4))
        print('recall_1_framewise', np.round(macro_recall_1, 4))
        print('f1_1_framewise', np.round(macro_f1_1, 4))
        print('precision_0_framewise', np.round(macro_precision_0, 4))
        print('recall_0_framewise', np.round(macro_recall_0, 4))
        print('f1_0_framewise', np.round(macro_f1_0, 4))


    def get_intervalwise_score(self, metrics: dict, eps: float = 1e-8) -> None:
        """
        Calculates score by ranges in video, collecting TP and FP 
        across all videos
        """
        tp = fp = tn = fn = 0
        for vid in metrics:
            if self.task_type == self.task_type.PA:
                start, end = vid["gt"]
            else:
                start_yn, end_yn = vid["model"][0][self.__ts]
                if start_yn == end_yn:
                    continue
                start_orig, end_orig = vid["model"][0]["proactive answer"]
                start = start_orig - start_yn
                end = end_orig - end_yn

            preds = vid["model"][0]["model_answers"]
            if self.task_type == self.task_type.MVQA:
                preds = np.array([1 if frame_pred == vid["gt"] else 0 for frame_pred in preds], dtype=np.uint8)
            else:
                preds = np.array([1 if frame_pred.lower() == "yes" else 0 for frame_pred in preds], dtype=np.uint8)
            if len(preds) == 0:
                print(start, end)

            pos_range = preds[start:end + 1]
            neg_range = np.concatenate([preds[:start], preds[(end + 1):]])

            if any(pos_range):
                tp += 1
            else:
                fn += 1

            if any(neg_range):
                fp += 1
            else:
                tn += 1
            
        precision_1 = tp / (tp + fp + eps)
        recall_1 = tp / (tp + fn + eps)
        f1_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1 + eps)
        precision_0 = tn / (tn + fn + eps)
        recall_0 = tn / (tn + fp + eps)
        f1_0 = 2 * precision_0 * recall_0 / (precision_0 + recall_0 + eps)
        accuracy = (tp + tn) / (len(preds) * 2)

        print('precision_1_intervalwise', np.round(precision_1, 4))
        print('recall_1_intervalwise', np.round(recall_1, 4))
        print('f1_1_intervalwise', np.round(f1_1, 4))
        print('precision_0_intervalwise', np.round(precision_0, 4))
        print('recall_0_intervalwise', np.round(recall_0, 4))
        print('f1_0_intervalwise', np.round(f1_0, 4))
        print('accuracy_intervalwise', np.round(accuracy, 4))
