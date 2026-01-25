import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import numpy as np
from moviepy import VideoFileClip, VideoClip
from .model_abc import ModelABC
from .utils import TaskType
from pathlib import Path
from dataclasses import dataclass, field
from tqdm import tqdm

from configs import MiniCPMVConfig

@dataclass
class MiniCPMVState:
    query: str = ""
    system_prompt: str = ""
    system_interact_prompt: str = ""
    frames: list = field(default_factory=list)
    audios: list = field(default_factory=list)
    llm_past_key_values: None = None
    new_session: bool = True
    new_user_msg: bool = True
    llm_generated: bool = False
    llm_generate_completed: bool = False
    info_logits: None = None
    sys_prompt_kv_cache: None = None
    sys_prompt_kv_cache_interactive: None = None

class MiniCPMV(ModelABC):
    def __init__(self, cfg: MiniCPMVConfig = MiniCPMVConfig(), task_type: TaskType | str = "vqa") -> None:
        """
        Initialize the model by loading the pretrained MiniCPM-V model and tokenizer.
        """
        self.cfg = cfg
        self.model = AutoModel.from_pretrained(self.cfg.model_path, trust_remote_code=True, attn_implementation=self.cfg.attn_implementation, torch_dtype=self.cfg.torch_dtype)
        self.model = self.model.eval().cuda()
        self.model = torch.compile(self.model, fullgraph=True, dynamic=True, mode="reduce-overhead")
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_path, trust_remote_code=True)

        task_type = TaskType(task_type)
        match task_type:
            case TaskType.vqa:
                self.func = self.vqa
            case TaskType.pa:
                self.state = MiniCPMVState()
                self.reset_state()
                self.func = self.pa
                self.cfg.max_new_tokens = 512
            case _:
                raise ValueError(f"No such option {task_type}")

    def encode_video(self, video: VideoFileClip) -> list:
        """
        Encode the video frames from the video path.
        """
        duration = video.duration
        if duration > 0:
            fps = self.cfg.max_num_frames / duration
        else:
            fps = 1
        print(f"{duration=}, {fps=}")
        video = video.with_fps(fps)
        frames = [Image.fromarray(frame.astype(np.uint8)) for frame in video.iter_frames()]
        return frames

    def vqa(self,file: str | Path | Image.Image, question: str):
        if isinstance(file, Image.Image):
            frames = [file]
        elif isinstance(file, (str, Path)):
            frames = VideoFileClip(file)
        if isinstance(file, VideoClip):
            frames = self.encode_video(file)
            file.close()

        print(len(frames))
        msgs = [
            {'role': 'user', 'content': frames + [question]},
        ]

        answer = self.model.chat(
            image=None,
            msgs=msgs,
            tokenizer=self.tokenizer,
            max_slice_nums=self.cfg.max_slice_nums,
            use_image_id=self.cfg.use_image_id,
            max_new_tokens=self.cfg.max_new_tokens,
            repetition_penalty=self.cfg.repetition_penalty
        )
        return answer
    
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
                msgs = [{'role': 'user', 'content': [self.state.frames[-1], self.state.query]}]
                model_output = self.model.chat(
                    image=None,
                    msgs=msgs,
                    tokenizer=self.tokenizer,
                    max_slice_nums=self.cfg.max_slice_nums,
                    use_image_id=self.cfg.use_image_id,
                    max_new_tokens=self.cfg.max_new_tokens,
                    repetition_penalty=self.cfg.repetition_penalty
                )

                history[-1]["frame_times"].append(clip["frame_time"])
                history[-1]["model_answers"].append(model_output)

        return history


    def reset_state(self) -> None:
        self.state = MiniCPMVState()


    @torch.inference_mode()
    def run(self, file: str | Path | Image.Image, question: str | None = None):
        return self.func(file, question)


    def name(self) -> str:
        return "MiniCPM-V"
