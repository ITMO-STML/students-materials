from pathlib import Path

import torch
from safetensors.torch import load_model
from sentence_transformers import SentenceTransformer
from torch import nn

from .additive import AttentiveProbe
from .audio_preprocessor import AudioToMelSpectrogramPreprocessor
from .conformer import ConformerEncoder


class LateClassifier(nn.Module):
    def __init__(self, num_classes: int = 1, bert_weight_path: str = "weights/modernbert", conformer_weight_path: Path = Path("weights/"), num_layers: int = 8, num_heads: int = 8) -> None:
        super().__init__()
        self.text_enc = SentenceTransformer(bert_weight_path, local_files_only=True)
        bert_out_dim = 384

        self.audio_enc = ConformerEncoder()
        self.proc = AudioToMelSpectrogramPreprocessor()
        conformer_out_dim = 176

        hidden_size = 384

        self.text_classifier = AttentiveProbe(bert_out_dim, hidden_size, num_heads=num_heads, num_layers=num_layers)
        self.audio_classifier = AttentiveProbe(conformer_out_dim, hidden_size, num_heads=num_heads, num_layers=num_layers)

        self.classifier = nn.Linear(2 * hidden_size, num_classes)

        self.load_audio_weights(conformer_weight_path)

    def forward(self, wavs: torch.Tensor, unpaded_length: list[int], text: list[str]) -> torch.Tensor:
        assert len(wavs.shape) == 2, "Audio should be [batch, length] size"

        text_emb = self.text_enc.encode(text, convert_to_tensor=True)
        text_emb = text_emb.clone().unsqueeze(1)  # [batch, 1, hidden]

        processed_features, length = self.proc.get_features(wavs, unpaded_length)
        audio_emb, _ = self.audio_enc(processed_features, length)
        audio_emb = audio_emb.transpose(1, 2)

        text_emb = self.text_classifier(text_emb)
        audio_emb = self.audio_classifier(audio_emb)
        emb = torch.cat((text_emb, audio_emb), dim=-1)

        out = self.classifier(emb)
        return out

    def load_audio_weights(self, path: Path) -> None:
        load_model(self.proc, path.joinpath("audio_preprocessor.safetensors"))
        load_model(self.audio_enc, path.joinpath("conformer.safetensors"))
