import torch
from transformers import AutoModel, AutoTokenizer

import numpy as np
from PIL import Image
from .model_abc import ModelABC
from dataclasses import dataclass, field
from .utils import TaskType
from tqdm import tqdm
from configs import MiniCPMoConfig
from moviepy import VideoFileClip
from pathlib import Path
import soundfile as sf

@dataclass
class MiniCPMoState:
    query: str = ""
    session_id: str = "123"
    frames: list = field(default_factory=list)

class MiniCPMo(ModelABC):
    def __init__(self, cfg: MiniCPMoConfig = MiniCPMoConfig(), task_type: TaskType | str = "vqa"):
        self.cfg = cfg
        self.model = AutoModel.from_pretrained(
            self.cfg.model_path,
            trust_remote_code=self.cfg.trust_remote_code,
            attn_implementation=self.cfg.attn_implementation,
            torch_dtype=self.cfg.torch_dtype,
            local_files_only=self.cfg.local_files_only
        ).eval().cuda()
        self.model = torch.compile(self.model, fullgraph=True, dynamic=True, mode="reduce-overhead")
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_path, trust_remote_code=self.cfg.trust_remote_code)

        task_type = TaskType(task_type)
        match task_type:
            case TaskType.vqa:
                print("\n\n", "="*20, "RVQA", "="*20)
                self.func = self.rvqa
            case TaskType.pa:
                self.state = MiniCPMoState()
                self.func = self.pa
                self.cfg.max_new_tokens = 512
            case _:
                raise ValueError(f"No such option {task_type}")

    def vqa(self, file, inp):
        if isinstance(file, Image.Image):
            msgs = [{"role":"user", "content": ["<unit>", file] + [inp]}]
            res = self.model.chat(
                image=None, 
                msgs=msgs, 
                context=None,
                tokenizer=self.tokenizer,
                sampling=False,
                max_new_tokens=self.cfg.max_new_tokens,
                stream=False,
                stream_input=self.cfg.stream_input,
                omni_input=self.cfg.omni_input,
                use_tts=self.cfg.use_tts,
                max_slice_nums=1 if self.cfg.stream_input else self.cfg.max_slice_nums,
                use_image_id=False if self.cfg.stream_input else self.cfg.use_image_id,
            )
            return res

        if isinstance(file, (str, Path)):
            file = VideoFileClip(file)
        duration = file.duration
        print('video_duration:', duration)

        clip = file.with_fps(self.cfg.max_frames / duration)
        cnts= []

        for frame in clip.iter_frames():
            image = Image.fromarray(frame.astype(np.uint8))
            cnts += ["<unit>", image]

        file.close()
        msg = {"role":"user", "content": cnts + [inp]}
        msgs = [msg]

        res = self.model.chat(
            image=None, 
            msgs=msgs, 
            context=None,
            tokenizer=self.tokenizer,
            sampling=False,
            max_new_tokens=self.cfg.max_new_tokens,
            stream=False,
            stream_input=self.cfg.stream_input,
            omni_input=self.cfg.omni_input,
            use_tts=self.cfg.use_tts,
            max_slice_nums=1 if self.cfg.stream_input else self.cfg.max_slice_nums,
            use_image_id=False if self.cfg.stream_input else self.cfg.use_image_id,
        )
        return res

    def rvqa(self, file, inp):
        if isinstance(file, Image.Image):
            msgs = [{"role":"user", "content": ["<unit>", file] + [inp]}]
            res = self.model.chat(
                image=None, 
                msgs=msgs, 
                context=None,
                tokenizer=self.tokenizer,
                sampling=False,
                max_new_tokens=self.cfg.max_new_tokens,
                stream=False,
                stream_input=self.cfg.stream_input,
                omni_input=self.cfg.omni_input,
                use_tts=self.cfg.use_tts,
                max_slice_nums=1 if self.cfg.stream_input else self.cfg.max_slice_nums,
                use_image_id=False if self.cfg.stream_input else self.cfg.use_image_id,
            )
            return res

        if isinstance(file, (str, Path)):
            file = VideoFileClip(file)
        duration = file.duration
        print('video_duration:', duration)

        clip = file.with_fps(self.cfg.max_frames / duration)
        cnts= []

        for frame in clip.iter_frames():
            image = Image.fromarray(frame.astype(np.uint8))
            cnts += ["<unit>", image]

        file.close()
        self.model.reset_session()
        _ = self.model.streaming_prefill(
            session_id=self.cfg.session_id,
            msgs=[{"role":"user", "content": cnts + [inp]}],
            tokenizer=self.tokenizer
        )

        res = self.model.streaming_generate(
            session_id=self.cfg.session_id,
            tokenizer=self.tokenizer,
            temperature=self.cfg.temperature,
            generate_audio=self.cfg.generate_audio
        )

        audios = []
        text = ""

        if self.cfg.generate_audio:
            for r in res:
                audio_wav = r.audio_wav
                sampling_rate = r.sampling_rate
                txt = r.text.replace("<|tts_eos|>", "")

                audios.append(audio_wav)
                text += txt
                
            res = np.concatenate(audios)
            sf.write("output.wav", res, samplerate=sampling_rate)
        else:
            for r in res:
                text += r['text'].replace("<|tts_eos|>", "")

        return text


    def pa(self, data, _):
        history = []
        video_meta = next(data)
        for clip in tqdm(data, leave=False):
            assert "image" in clip
            if clip.get("decision_question"):
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
                contents = [self.state.query]
                for img in self.state.frames[-5:]:
                    contents.extend(["<unit>", img])
                _ = self.model.streaming_prefill(
                    session_id=self.cfg.session_id,
                    msgs=[{"role":"user", "content": contents}], 
                    tokenizer=self.tokenizer
                )
                res = self.model.streaming_generate(
                    session_id=self.cfg.session_id,
                    tokenizer=self.tokenizer,
                    temperature=self.cfg.temperature,
                    generate_audio=self.cfg.generate_audio
                )

                audios = []
                text = ""

                if self.cfg.generate_audio:
                    for r in res:
                        audio_wav = r.audio_wav
                        sampling_rate = r.sampling_rate
                        txt = r.text.replace("<|tts_eos|>", "")

                        audios.append(audio_wav)
                        text += txt
                        
                    res = np.concatenate(audios)
                    sf.write("output.wav", res, samplerate=sampling_rate)
                else:
                    for r in res:
                        text += r['text'].replace("<|tts_eos|>", "")

                history[-1]["frame_times"].append(clip["frame_time"])
                history[-1]["model_answers"].append(text)
                self.model.reset_session()

        return history


    @torch.inference_mode()
    def run(self, file, inp: str | None = None):
        return self.func(file, inp)

            
    def name(self):
        return "MiniCPMo"
