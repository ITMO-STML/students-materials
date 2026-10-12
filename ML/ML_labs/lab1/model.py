import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from transformers import AutoModel


class CLSCrossAttentionFusion(nn.Module):
    """
    Промежуточное фузирование:
    image CLS -> Query, текстовые токены -> Key/Value.
    """
    def __init__(self, num_classes=10, freeze_encoders=True):
        super().__init__()

        self.image_encoder = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.image_encoder.fc = nn.Identity()  # эмбеддинги картинок с ResNet

        self.text_encoder = AutoModel.from_pretrained("distilbert-base-uncased") # последовательность эмбеддингов для текста

        if freeze_encoders:
            for p in self.image_encoder.parameters():
                p.requires_grad = False
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        self.image_projection = nn.Linear(512, 768) # выравнивание размерностей
        self.cross_attention = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True) # cross-attention
        self.norm = nn.LayerNorm(768)

        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, images, input_ids, attention_mask):
        img = self.image_encoder(images)                     # [B, 512]
        q = self.image_projection(img).unsqueeze(1)          # [B, 1, 768]

        txt = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state                                  # [B, L, 768]

        attended, _ = self.cross_attention(
            query=q, key=txt, value=txt,
            key_padding_mask=~attention_mask.bool()
        )
        fused = self.norm(attended.squeeze(1))               # [B, 768]
        return self.classifier(fused)