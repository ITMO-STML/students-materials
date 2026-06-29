from transformers import AutoModelForVision2Seq, AutoProcessor
from configs import QwenConfig, MVBenchConfig, StreamingBenchConfig
from benchmark.utils import convert_path, convert_time, split_video
from qwen_vl_utils import process_vision_info
import torch
import pandas as pd
from moviepy import VideoFileClip, ImageClip, concatenate_videoclips, VideoClip
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
from glob import glob
import argparse
from typing import Literal, Any
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True, help="Path to cleaned .tsv file")
    parser.add_argument("--type", type=str, required=True, choices=["MVBench", "StreamingBench"])
    args = parser.parse_args()
    return args

def preprocess_sb(video, cfg: StreamingBenchConfig = StreamingBenchConfig()) -> VideoClip:
    video_path = convert_path(video["video_path"], cfg.dataset_dir)
    time_start, time_end = video["time"].split(" - ")
    if time_start.strip().startswith("["):
        time_start = time_start[1:]
    if time_end.strip().endswith("]"):
        time_end = time_end[:-1]

    time_start = convert_time(time_start)
    time_end = convert_time(time_end)
    file = split_video(video_path, time_start, time_end)
    question_time = convert_time(video["time_stamp"])
    clip = VideoFileClip(file)
    if time_start <= question_time < time_end:
        clip = clip.subclipped(0, min(question_time - time_start + 2, clip.duration))
    return clip

def preprocess_mv(file, cfg: MVBenchConfig = MVBenchConfig(), resolution: int = 224) -> VideoClip:
    video_path = cfg.dataset_dir + "/" + file["video_path"]
    if Path(video_path).is_dir():
        clips = [ImageClip(clip).with_duration(0.3) for clip in glob(video_path+"/*.jpg")]
        video = concatenate_videoclips(clips, method="compose")
    else:
        video = VideoFileClip(video_path)
    if video.duration > (max(0, file['start'])):
        video = video.subclipped(max(0, file['start']), min(video.duration, file['end']))
    clip = video.resized((resolution, resolution))
    return clip

@torch.inference_mode()
def process_video(model: AutoModelForVision2Seq, processor: AutoProcessor, cfg: QwenConfig, row: dict[str | Path, Any], dataset_type = Literal["MVBench"] | Literal["StreamingBench"]) -> tuple[str, str]:
    """Generating image and video captions

    Args:
        model (AutoModelForVision2Seq): Qwen3 VL model
        processor (AutoProcessor): Qwen3 Image processor
        cfg (QwenConfig): Config for generation
        path (str, Path): path to media file

    Returns:
        tuple: Pair of image and video description
    """
    if dataset_type == "MVBench":
        clip = preprocess_mv(row)
    else:
        clip = preprocess_sb(row)
    fps = max(clip.duration / cfg.max_frames, 1)
    clip = clip.with_fps(fps)
    content = []
    for frame in clip.iter_frames():
        content.append(Image.fromarray(frame.astype(np.uint8)))
    clip.close()

    # image description
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": content[0],
                },
                {"type": "text", "text": "Describe this image in detail."},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=cfg.max_new_tokens,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        temperature=cfg.temperature,
        repetition_penalty=cfg.repetition_penalty)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    img_description = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    # video description
    messages = [
        {"role": "user", "content": [
                {"video": content,
                "total_pixels": len(content) * cfg.img_size[0] * cfg.img_size[0],
                "min_pixels": cfg.min_pixels, 
                "max_frames": len(content),
                'sample_fps': cfg.sample_fps},
                {"type": "text", "text": "Describe the video content in detail."},
            ]
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True, 
                                                                image_patch_size= 16,
                                                                return_video_metadata=True)
    if video_inputs is not None:
        video_inputs, video_metadatas = zip(*video_inputs)
        video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
    else:
        video_metadatas = None
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, video_metadata=video_metadatas, **video_kwargs, do_resize=False, return_tensors="pt")
    inputs = inputs.to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=cfg.max_new_tokens,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        temperature=cfg.temperature,
        repetition_penalty=cfg.repetition_penalty
    )
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    video_description = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

    return img_description, video_description


def main():
    args = parse_args()

    data = pd.read_csv(args.file_path, sep="\t")
    cfg = QwenConfig()
    cfg.max_new_tokens = 1024
    processor = AutoProcessor.from_pretrained(cfg.model_path)
    model = AutoModelForVision2Seq.from_pretrained(cfg.model_path, dtype="auto", device_map="auto", output_loading_info=False).eval()
    model = torch.compile(model, fullgraph=True, dynamic=True, mode="reduce-overhead")

    recaption = []

    for _, row in tqdm(data.iterrows(), total=len(data)):
        img_desc, vid_desc = process_video(model, processor, cfg, row, args.type)
        recaption.append({
            "orig_path": row["video_path"],
            "orig_question": row["question"],
            "img_desc": img_desc,
            "vid_desc": vid_desc,
            "category": row["task_type"]
        })
    
    with open(args.type+"_cleaned.json", "w", encoding="utf-8") as file:
        json.dump(recaption, file, indent=4)


if __name__=="__main__":
    main()
