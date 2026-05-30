import torch
from sentence_transformers import SentenceTransformer
from torch import nn

from .additive import AttentiveProbe


class BertAttentiveProbe(nn.Module):
    def __init__(self, num_classes: int = 1, weight_path: str = "weights/modernbert", num_layers: int = 4, num_heads: int = 4) -> None:
        super().__init__()
        self.enc = SentenceTransformer(weight_path, local_files_only=True)
        bert_out_dim = 384
        self.classifier = AttentiveProbe(bert_out_dim, num_classes=num_classes, num_layers=num_layers, num_heads=num_heads)

    def forward(self, text: list[str]) -> torch.Tensor:
        emb = self.enc.encode(text, convert_to_tensor=True)
        emb = emb.clone().unsqueeze(1)  # [batch, 1, hidden]
        out = self.classifier(emb)
        return out
