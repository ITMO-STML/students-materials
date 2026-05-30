from pathlib import Path

import torch
from safetensors.torch import load_model
from torch import nn

from .additive import AttentiveProbe
from .audio_preprocessor import AudioToMelSpectrogramPreprocessor
from .conformer import ConformerEncoder


class ConformerAttentiveProbe(nn.Module):
    def __init__(self, num_classes: int = 1, wegith_path: Path = Path("weights/"), num_layers: int = 4, num_heads: int = 4) -> None:
        super().__init__()
        self.enc = ConformerEncoder()
        self.proc = AudioToMelSpectrogramPreprocessor()
        conformer_out_dim = 176
        self.classifier = AttentiveProbe(conformer_out_dim, num_classes=num_classes, num_layers=num_layers, num_heads=num_heads)
        self.load_weights(wegith_path)

    def forward(self, wavs: torch.Tensor, unpaded_length: list[int]) -> torch.Tensor:
        assert len(wavs.shape) == 2, "Audio should be [batch, length] size"
        processed_features, length = self.proc.get_features(wavs, unpaded_length)
        emb, _ = self.enc(processed_features, length)
        emb = emb.transpose(1, 2)
        out = self.classifier(emb)
        return out

    def load_weights(self, path: Path) -> None:
        load_model(self.proc, path.joinpath("audio_preprocessor.safetensors"))
        load_model(self.enc, path.joinpath("conformer.safetensors"))
