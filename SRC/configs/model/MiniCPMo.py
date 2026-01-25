from dataclasses import dataclass
import torch

@dataclass
class MiniCPMoConfig:
    # model cfg
    model_path: str = "openbmb/MiniCPM-o-2_6"
    trust_remote_code: bool = True
    attn_implementation: str = "sdpa"
    torch_dtype: float = torch.bfloat16
    local_files_only: bool = True
    sample_rate: int = 16_000

    # gen cfg
    max_new_tokens: int = 128
    min_new_tokens: int = 0
    stream: bool = False
    stream_input: bool = True
    omni_input: bool = True
    use_tts: bool = False
    generate_audio: bool = False
    top_p: float = 0.8
    top_k: int = 100  
    temperature: float = 0.7
    repetition_penalty: float = 1.05
    do_sample: bool = True

    # will be overriden if stream_input
    max_slice_nums: int = 1
    use_image_id: bool = False

    max_frames: int = 30 # greater -> OOM

    session_id: str = '123'
