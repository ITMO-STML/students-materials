from transformers import AutoModelForVision2Seq, AutoProcessor
from qwen_vl_utils import process_vision_info
from .model_abc import ModelABC
from typing import Any
from pathlib import Path
from configs import QwenConfig
from PIL import Image
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, field
from moviepy import VideoFileClip, VideoClip

import torch

@dataclass
class QwenState:
    query: str = ""
    frames: list = field(default_factory=list)
    llm_past_key_values: None = None
    sys_prompt_kv_cache: None = None


class Qwen3(ModelABC):
    def __init__(self, cfg: QwenConfig = QwenConfig(), *_, **__) -> None:
        self.cfg = cfg
        self.model = AutoModelForVision2Seq.from_pretrained(self.cfg.model_path, dtype="auto", device_map="auto", output_loading_info=False).eval().cuda()
        self.model = torch.compile(self.model, fullgraph=True, dynamic=True, mode="reduce-overhead")
        self.processor = AutoProcessor.from_pretrained(self.cfg.model_path)

        self.reset_state()
    
    def reset_state(self) -> None:
        self.state = QwenState()
        self.state.sys_prompt_kv_cache = self.state.llm_past_key_values


    def video_qa(self, file, question):
        if isinstance(file, (str, Path)):
            file = VideoFileClip(file)

        if isinstance(file, VideoClip):
            fps = max(file.duration / self.cfg.max_frames, 1)
            print(f"{file.duration=}, {fps=}")
            file = file.with_fps(fps)
            content = []
            for frame in file.iter_frames():
                content.append(Image.fromarray(frame.astype(np.uint8)))
            file.close()

        if isinstance(file, Image.Image):
            content = [file]

        if isinstance(file, list):
            content = file

        messages = [
            {"role": "user", "content": [
                    {"video": content,
                    "total_pixels": len(content) * self.cfg.img_size[0] * self.cfg.img_size[0],
                    "min_pixels": self.cfg.min_pixels, 
                    "max_frames": len(content),
                    'sample_fps': self.cfg.sample_fps},
                    {"type": "text", "text": question},
                ]
            },
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True, 
                                                                    image_patch_size= 16,
                                                                    return_video_metadata=True)
        if video_inputs is not None:
            video_inputs, video_metadatas = zip(*video_inputs)
            video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
        else:
            video_metadatas = None
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, video_metadata=video_metadatas, **video_kwargs, do_resize=False, return_tensors="pt")
        inputs = inputs.to(self.model.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.cfg.max_new_tokens,
            top_p=self.cfg.top_p,
            top_k=self.cfg.top_k,
            temperature=self.cfg.temperature,
            repetition_penalty=self.cfg.repetition_penalty
        )
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return output_text[0]

    def image_qa(self, file, question):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": file,
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.cfg.max_new_tokens,
            top_p=self.cfg.top_p,
            top_k=self.cfg.top_k,
            temperature=self.cfg.temperature,
            repetition_penalty=self.cfg.repetition_penalty
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]


    def pa(self, data, _):
        history = []
        video_meta = next(data)
        for clip in tqdm(data, leave=False):
            assert "image" in clip
            if clip.get("decision_question"):
                self.reset_state()
                self.state.query = clip["decision_question"]
                question_meta = {
                    k: v for k, v in clip.items() if k != "image" and k != "answer"
                }
                history.append(
                    question_meta
                    | video_meta
                    | {"frame_times": [], "model_answers": [], "response_times": []}
                )

            if clip["image"]:
                self.state.frames.append(clip["image"])

            if clip.get("answer"):
                model_output = self.video_qa(self.state.frames[-1], self.state.query)
                history[-1]["frame_times"].append(clip["frame_time"])
                history[-1]["model_answers"].append(model_output)

        return history


    @torch.inference_mode()
    def run(self, file: str | Path, question: dict | str | None = None) -> Any:
        # return self.image_qa(file, question)
        # return self.pa(file, question)
        return self.video_qa(file, question)


    @staticmethod
    def name() -> str:
        return "Qwen3-VL 8B"
