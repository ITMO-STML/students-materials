"""
Multitask Feed-Forward модель для baseline CMU-MOSEI.
Общий ствол -> две головы: sentiment (3 класса) и emotion (6 независимых бинарных меток).
"""

import torch
import torch.nn as nn


class MultitaskFFN(nn.Module):
    def __init__(self, input_dim, hidden_dims=(256, 128), dropout=0.3,
                 n_sentiment_classes=3, n_emotions=6):
        super().__init__()

        trunk_layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            trunk_layers += [
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev_dim = h
        self.trunk = nn.Sequential(*trunk_layers)

        self.sentiment_head = nn.Linear(prev_dim, n_sentiment_classes)
        self.emotion_head = nn.Linear(prev_dim, n_emotions)

    def forward(self, x):
        shared = self.trunk(x)
        sentiment_logits = self.sentiment_head(shared)   # (batch, 3) -> CrossEntropyLoss
        emotion_logits = self.emotion_head(shared)        # (batch, 6) -> BCEWithLogitsLoss
        return sentiment_logits, emotion_logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)