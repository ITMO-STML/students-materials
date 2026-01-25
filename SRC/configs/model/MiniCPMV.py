from dataclasses import dataclass
import torch


@dataclass
class MiniCPMVConfig:
    model_path: str = 'openbmb/MiniCPM-V-4_5'
    attn_implementation: str = "sdpa"
    torch_dtype: torch.dtype = torch.bfloat16

    max_num_frames: int = 64
    max_slice_nums: int = 1
    use_image_id: bool = False
    max_new_tokens: int = 512
    repetition_penalty: float = 1.05