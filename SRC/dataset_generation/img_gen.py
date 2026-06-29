from diffusers import ZImagePipeline
import torch
import json
from pathlib import Path
from tqdm.auto import tqdm
import argparse

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Tongyi-MAI/Z-Image", help="Path to downloaded weights or to HuggingFace model")
    parser.add_argument("--json_path", type=str, required=True, help="Path to .json file after rewrite")
    parser.add_argument("--save_folder", type=str, required=True, help="Folder to save images")
    return parser.parse_args()

def inference():
    args = parse_args()
    prefix = "streamingbench" if "streamingbench" in args.json_rewrite_path.lower() or "sb" in args.json_rewrite_path.lower() else "mvbench"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = ZImagePipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )
    pipe.to(device)
    pipe.set_progress_bar_config(leave=False)

    img_desc_key = "new_image_caption"
    neg_obj_key = "no_exist_image_objects"

    with open(args.json_rewrite_path, encoding="utf-8") as file:
        data = json.load(file)

    for i, f in tqdm(enumerate(data)):
        # Generate image
        path = Path(f"{args.save_folder}/{prefix}_{i}.png")
        if path.exists():
            continue
        prompt = f[img_desc_key]
        negative_prompt = ", ".join(f[neg_obj_key])
        if not prompt or not negative_prompt:
            continue

        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=1280,
            width=720,
            cfg_normalization=True,
            num_inference_steps=50,
            guidance_scale=4,
            generator=torch.Generator(device).manual_seed(42),
        ).images[0]

        image.save(path)

if __name__=="__main__":
    inference()
