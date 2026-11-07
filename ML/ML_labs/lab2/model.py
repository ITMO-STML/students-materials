import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from transformers import AutoModel
from typing import Tuple


class CLSCrossAttentionFusion(nn.Module):
    """
    Промежуточное фузирование:
    image CLS -> Query, текстовые токены -> Key/Value.
    """
    def __init__(self, num_classes: int = 10, num_genders: int = 3, freeze_encoders: bool = True) -> None:
        super().__init__()

        self.image_encoder: nn.Module = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.image_encoder.fc = nn.Identity()  # эмбеддинги картинок с ResNet

        self.text_encoder: nn.Module = AutoModel.from_pretrained("distilbert-base-uncased") # последовательность эмбеддингов для текста

        if freeze_encoders:
            for p in self.image_encoder.parameters():
                p.requires_grad = False
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        self.image_projection: nn.Linear = nn.Linear(512, 768) # выравнивание размерностей
        self.cross_attention: nn.MultiheadAttention = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True) # cross-attention
        self.norm: nn.LayerNorm = nn.LayerNorm(768)

        self.classifier_category: nn.Sequential = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

        self.classifier_gender: nn.Sequential = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_genders),
        )

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        img: torch.Tensor = self.image_encoder(images)                     # [B, 512]
        q: torch.Tensor = self.image_projection(img).unsqueeze(1)          # [B, 1, 768]

        txt: torch.Tensor = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state                                  # [B, L, 768]

        attended: torch.Tensor
        attended, _ = self.cross_attention(
            query=q, key=txt, value=txt,
            key_padding_mask=~attention_mask.bool()
        )
        fused: torch.Tensor = self.norm(attended.squeeze(1))               # [B, 768]

        logits_cat: torch.Tensor = self.classifier_category(fused)
        logits_gender: torch.Tensor = self.classifier_gender(fused)
        
        return logits_cat, logits_gender