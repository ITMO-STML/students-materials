import torch
import torch.nn as nn


class AttentiveProbe(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, num_layers: int = 4, num_heads: int = 4) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            batch_first=True,
            activation=nn.functional.silu,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        assert len(features.shape) == 3, "Input should be [batch, seq_len, hidden] size"
        out = self.transformer(features)
        out = out.mean(dim=1)
        return self.classifier(out)
