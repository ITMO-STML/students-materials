import json
import argparse
from typing import Literal
from tqdm import tqdm
from pathlib import Path
import random

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mv_json", type=str, required=True, help="Path to input MVBench json")
    parser.add_argument("--sb_json", type=str, required=True, help="Path to input StreamingBench json")
    parser.add_argument("--video_folder", type=str, required=True, help="Path to video folder")
    parser.add_argument("--ft_type", type=str, required=True, choices=["sft", "dpo"], help="Type for generated dataset either SFT or DPO")
    return parser.parse_args()


def generate_sft_dataset(data: dict, data_type: Literal["sb", "mv"], video_folder: str) -> list:
    output_data = []
    video_folder = Path(video_folder)
    for i, row in enumerate(tqdm(data, desc=f"Processing {data_type}")):
        video_path = video_folder.joinpath(f"{data_type}_{i}.mp4")
        if not video_path.exists():
            continue
        query = {}
        real_objects = row["video_objects"]
        fake_objects = row["no_exist_video_objects"]
        total_objects = real_objects + fake_objects
        random.shuffle(total_objects)

        messages = []
        messages.append({
            "content": f"<video>What objects from the list are presented on the video? Objects: [{', '.join(total_objects)}]",
            "role": "user"
        })
        messages.append({
            "content": ", ".join(real_objects),
            "role": "assistant"
        })
        query["messages"] = messages
        query["videos"] = [str(video_path)]
        output_data.append(query)
    return output_data

def generate_dpo_dataset(data: dict, data_type: Literal["sb", "mv"], video_folder: str) -> list:
    output_data = []
    video_folder = Path(video_folder)
    id_prefix = "00" if data_type == "mv" else "01"

    for i, row in enumerate(tqdm(data, desc=f"Processing {data_type}")):
        video_path = video_folder.joinpath(f"{data_type}_{i}.mp4")
        if not video_path.exists():
            continue
        query = {}
        real_objects = row["video_objects"]
        fake_objects = row["no_exist_video_objects"]
        total_objects = real_objects + fake_objects
        random.shuffle(total_objects)

        conversations = [{
            "from": "human",
            "value": f"<video>What objects from the list are presented on the video? Objects: [{', '.join(total_objects)}]",
        }]
        chosen = [{
            "from": "gpt",
            "value": f"value: {', '.join(real_objects)}"
        }]
        rejected = [{
            "from": "gpt",
            "value": f"value: {', '.join(fake_objects)}"
        }]

        query["id"] = f"{id_prefix}-{i}"
        query["video"] = [str(video_path)]
        query["conversations"] = conversations
        query["chosen"] = chosen
        query["rejected"] = rejected
        output_data.append(query)
    return output_data

def main():
    random.seed(42)
    args = parse_args()
    output_json_path = f"{args.ft_type}.json"

    with open(args.mv_json, "r", encoding="utf-8") as file:
        mv_data = json.load(file)
    with open(args.sb_json, "r", encoding="utf-8") as file:
        sb_data = json.load(file)
    
    if args.ft_type == "sft":
        mv_output_json = generate_sft_dataset(mv_data, "mv", args.video_folder)
        sb_output_json = generate_sft_dataset(sb_data, "sb", args.video_folder)
        output_json = mv_output_json + sb_output_json
    else:
        mv_output_json = generate_dpo_dataset(mv_data, "mv", args.video_folder)
        sb_output_json = generate_dpo_dataset(sb_data, "sb", args.video_folder)
        output_json = mv_output_json + sb_output_json

    with open(output_json_path, "w", encoding="utf-8") as file:
        json.dump(output_json, file, indent=4)

if __name__=="__main__":
    main()