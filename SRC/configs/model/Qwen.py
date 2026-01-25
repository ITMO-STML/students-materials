from dataclasses import dataclass

@dataclass
class QwenConfig:
    model_path: str = "Qwen/Qwen3-VL-8B-Instruct"
    max_new_tokens: int = 512

    # img
    max_frames:int = 60
    min_frames: int = 5
    img_size: tuple[int, int] = (32, 32)
    total_pixels: int = max_frames * img_size[0] * img_size[1]
    min_pixels: int = min_frames * img_size[0] * img_size[1]
    sample_fps: int = 1

    # gen
    greedy: bool = False
    top_p: float = 0.8
    top_k: int = 20
    temperature: float = 0.7
    repetition_penalty: float = 1.0
    presence_penalty: float = 1.5