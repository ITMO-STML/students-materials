from lightx2v import LightX2VPipeline
import argparse
import json
from tqdm import tqdm
from pathlib import Path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to downloaded weights or to HuggingFace model")
    parser.add_argument("--model_cls", type=str, default="hunyuan_video_1.5" help="Model name")
    parser.add_argument("--transformer_model_name", type=str, default="720p_i2v", help="Model type")
    parser.add_argument("--task", type=str, default="i2v", choices=["i2v", "t2v"], help="Task type")

    parser.add_argument("--json_path", type=str, required=True, help="Path to .json file after rewrite")
    parser.add_argument("--dataset_type", type=str, choices=["mv", "sb"], required=True, help="Dataset type")
    parser.add_argument("--save_folder", type=str, required=True, help="Folder to save videos")
    parser.add_argument("--img_folder", type=str, required=True, help="Folder with generated images")
    return parser.parse_args()

def infer():
    args = parse_args()

    with open(args.json_path, encoding="utf-8") as file:
        data1 = json.load(file)
    print(f"\n\n{len(data1)=}\n\n")

    # Initialize pipeline for HunyuanVideo-1.5 I2V task
    pipe = LightX2VPipeline(
        model_path=args.model_path,
        model_cls=args.model_cls,
        transformer_model_name=args.transformer_model_name,
        task=args.task,
    )

    pipe.enable_offload(
        cpu_offload=True,
        offload_granularity="block",  # For HunyuanVideo-1.5, only "block" is supported
        text_encoder_offload=True,
        image_encoder_offload=False,
        vae_offload=False,
    )

    pipe.create_generator(
        attn_mode="flash_attn2",
        infer_steps=50,
        num_frames=121,
        guidance_scale=6.0,
        sample_shift=7.0,
        fps=24,
    )

    seed = 42
    for idx, row1 in tqdm(enumerate(data1)):
        img_path = Path(args.img_folder, f"{args.dataset_type}_{idx}.png")
        if not img_path.exists():
            print("failed to load path:", img_path)
            continue

        prompt = row1
        save_result_path = Path(args.save_folder, f"{args.dataset_type}_{idx}.mp4")
        if save_result_path.exists():
            print(f"Found {save_result_path}, skip")
            continue
        save_result_path = str(save_result_path)
        img_path = str(img_path)

        pipe.generate(
            seed=seed,
            prompt=prompt,
            image_path=img_path,
            negative_prompt="", # hunyan doesn't have it
            save_result_path=save_result_path,
        )

if __name__ == "__main__":
    infer()