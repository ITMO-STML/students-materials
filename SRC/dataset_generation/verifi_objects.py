from sentence_transformers import SentenceTransformer
import json
from tqdm import trange
import argparse
from torch import inference_mode
from scipy.spatial.distance import cosine

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeder_path", type=str, default="Qwen/Qwen3-VL-Embedding-8B", help="Path to Qwen3-VL-Embedding")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to MVBench or StreamingBench dataset")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save dataset")
    parser.add_argument("--lower", type=float, default=0.9, help="Cosine similarity lower bound")
    parser.add_argument("--upper", type=float, default=0.98, help="Cosine similarity upper bound")
    return parser.parse_args()

@inference_mode()
def main():
    args = parse_args()
    embeder = SentenceTransformer(
        args.embeder_path,
        model_kwargs={"attn_implementation": "sdpa", "torch_dtype": "auto"},
        processor_kwargs={"min_pixels": 28 * 28, "max_pixels": 600 * 600},
        revision="refs/pr/23",
    )

    with open(args.dataset_path) as file:
        data = json.load(file)
    cnt = 0
    for i in trange(len(data), desc="Iterating over dataset"):
        video_data = data[i]
        if not video_data.get("image_objects", False):
            data[i] = None
            continue
        pos_emb = embeder.encode(video_data["image_objects"]).mean(axis=0)
        new_neg_obj = []
        for neg_obj in video_data["no_exist_image_objects"]:
            cnt += 1
            neg_emb = embeder.encode(neg_obj)
            sim = 1 - cosine(pos_emb, neg_emb)
            if args.lower < sim < args.upper:
                new_neg_obj.append(neg_obj)
                cnt -= 1
        video_data["no_exist_image_objects"] = new_neg_obj
        if not new_neg_obj:
            data[i] = None
            continue
        if not video_data.get("video_objects", False):
            data[i] = None
            continue
        pos_emb = embeder.encode(video_data["video_objects"]).mean(axis=0)
        new_neg_obj = []
        for neg_obj in video_data["no_exist_video_objects"]:
            cnt += 1
            neg_emb = embeder.encode(neg_obj)
            sim = 1 - cosine(pos_emb, neg_emb)
            if args.lower < sim < args.upper:
                new_neg_obj.append(neg_obj)
                cnt -= 1
        video_data["no_exist_video_objects"] = new_neg_obj
        if not new_neg_obj:
            data[i] = None
    data_filtered = [d for d in data if d is not None]
    print(f"Removed {cnt} objects, {len(data) - len(data_filtered)} captions were removed")
    with open(args.save_path, "w", encoding="utf-8") as file:
        json.dump(data_filtered, file, indent=4)
        
    
if __name__=="__main__":
    main()