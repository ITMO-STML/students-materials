"""
Усложнённые модели для раздела 5 курсовой работы.
Три архитектуры энкодера последовательностей с единым multitask-интерфейсом:
вход (B, T, D) + маска (B, T) -> sentiment_logits (B, 3), emotion_logits (B, 6).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _MultitaskHead(nn.Module):
    """Общий ствол + две головы. Один и тот же для всех трёх архитектур."""

    def __init__(self, repr_dim, head_hidden=128, dropout=0.3,
                 n_sentiment=3, n_emotions=6):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(repr_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.sentiment_head = nn.Linear(head_hidden, n_sentiment)
        self.emotion_head = nn.Linear(head_hidden, n_emotions)

    def forward(self, repr_vec):
        h = self.shared(repr_vec)
        return self.sentiment_head(h), self.emotion_head(h)


def _masked_mean(seq, mask):
    """seq: (B, T, D), mask: (B, T). Возвращает (B, D) — среднее по реальным шагам."""
    mask_f = mask.float().unsqueeze(-1)
    summed = (seq * mask_f).sum(dim=1)
    counts = mask_f.sum(dim=1).clamp(min=1e-6)
    return summed / counts


class BiLSTMModel(nn.Module):
    """BiLSTM + конкатенация последних hidden states обоих направлений."""

    def __init__(self, input_dim, hidden_dim=128, num_layers=1,
                 dropout=0.3, head_hidden=128):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = _MultitaskHead(repr_dim=2 * hidden_dim,
                                    head_hidden=head_hidden, dropout=dropout)

    def forward(self, x, mask):
        lengths = mask.sum(dim=1).cpu().clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False,
        )
        _, (h_n, _) = self.lstm(packed)
        # h_n: (num_layers*2, B, hidden) -> берём последний слой, оба направления
        forward_last = h_n[-2]
        backward_last = h_n[-1]
        repr_vec = torch.cat([forward_last, backward_last], dim=-1)
        return self.head(repr_vec)


class CNN1DModel(nn.Module):
    """Параллельные Conv1d с разными размерами ядра + masked GlobalMaxPool."""

    def __init__(self, input_dim, num_filters=64, kernel_sizes=(3, 5, 7),
                 dropout=0.3, head_hidden=128):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=input_dim, out_channels=num_filters,
                       kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])
        self.head = _MultitaskHead(repr_dim=num_filters * len(kernel_sizes),
                                    head_hidden=head_hidden, dropout=dropout)

    def forward(self, x, mask):
        x = x.transpose(1, 2)  # (B, D, T) для Conv1d
        mask_f = mask.unsqueeze(1).float()  # (B, 1, T) для маскирования каналов
        feats = []
        for conv in self.convs:
            c = F.relu(conv(x))                                 # (B, F, T)
            c = c.masked_fill(mask_f == 0, float("-inf"))       # маскируем паддинг до max-pool
            pooled = c.max(dim=2).values                         # (B, F)
            feats.append(pooled)
        repr_vec = torch.cat(feats, dim=-1)
        return self.head(repr_vec)


class TransformerModel(nn.Module):
    """Линейная проекция -> Transformer encoder -> masked mean pooling."""

    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=2,
                 dim_feedforward=256, dropout=0.3, head_hidden=128):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True, activation="relu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = _MultitaskHead(repr_dim=d_model,
                                    head_hidden=head_hidden, dropout=dropout)

    def forward(self, x, mask):
        x = self.input_proj(x)
        # nn.TransformerEncoder: True означает "игнорировать", поэтому ~mask
        key_padding_mask = ~mask
        encoded = self.encoder(x, src_key_padding_mask=key_padding_mask)
        repr_vec = _masked_mean(encoded, mask)
        return self.head(repr_vec)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
class LateFusionModel(nn.Module):
    """Два независимых Transformer-энкодера (текст, аудио) -> конкатенация -> multitask head."""

    def __init__(self, text_dim, audio_dim, d_model=128, num_layers=2,
                 nhead=4, dim_feedforward=256, dropout=0.1, head_hidden=128):
        super().__init__()
        self.text_encoder = TransformerModel(
            input_dim=text_dim, d_model=d_model, num_layers=num_layers,
            nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,
        )
        self.audio_encoder = TransformerModel(
            input_dim=audio_dim, d_model=d_model, num_layers=num_layers,
            nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,
        )
        # «Отключаем» головы внутренних энкодеров — используем только их репрезентации
        # через _encode (см. ниже); проще переопределить forward, оставив input_proj
        # и encoder. Самое чистое решение — извлечь представление вручную:
        self.text_proj = self.text_encoder.input_proj
        self.text_enc = self.text_encoder.encoder
        self.audio_proj = self.audio_encoder.input_proj
        self.audio_enc = self.audio_encoder.encoder
        # Внутренние головы text/audio_encoder.head больше не нужны — голова одна общая
        del self.text_encoder.head
        del self.audio_encoder.head

        self.head = _MultitaskHead(repr_dim=2 * d_model, head_hidden=head_hidden, dropout=dropout)

    def _encode(self, x, mask, proj, encoder):
        x = proj(x)
        encoded = encoder(x, src_key_padding_mask=~mask)
        return _masked_mean(encoded, mask)

    def forward(self, xs, masks):
        text_x, audio_x = xs
        text_mask, audio_mask = masks
        text_repr = self._encode(text_x, text_mask, self.text_proj, self.text_enc)
        audio_repr = self._encode(audio_x, audio_mask, self.audio_proj, self.audio_enc)
        fused = torch.cat([text_repr, audio_repr], dim=-1)
        return self.head(fused)


class EarlyFusionModel(nn.Module):
    """Конкатенация признаков по последней размерности -> один Transformer-энкодер -> head."""

    def __init__(self, text_dim, audio_dim, d_model=128, num_layers=2,
                 nhead=4, dim_feedforward=256, dropout=0.1, head_hidden=128):
        super().__init__()
        self.input_proj = nn.Linear(text_dim + audio_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True, activation="relu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = _MultitaskHead(repr_dim=d_model, head_hidden=head_hidden, dropout=dropout)

    def forward(self, xs, masks):
        text_x, audio_x = xs
        text_mask, audio_mask = masks
        combined_mask = text_mask & audio_mask  # шаг реален только если он реален в ОБОИХ
        x = torch.cat([text_x, audio_x], dim=-1)
        x = self.input_proj(x)
        encoded = self.encoder(x, src_key_padding_mask=~combined_mask)
        repr_vec = _masked_mean(encoded, combined_mask)
        return self.head(repr_vec)


class CrossModalAttentionFusion(nn.Module):
    """Два энкодера + cross-attention (текст ↔ аудио) -> конкатенация -> head."""

    def __init__(self, text_dim, audio_dim, d_model=128, num_layers=2,
                 nhead=4, dim_feedforward=256, dropout=0.1, head_hidden=128):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, d_model)
        self.audio_proj = nn.Linear(audio_dim, d_model)

        enc_kwargs = dict(d_model=d_model, nhead=nhead,
                          dim_feedforward=dim_feedforward, dropout=dropout,
                          batch_first=True, activation="relu")
        self.text_self = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(**enc_kwargs), num_layers=num_layers,
        )
        self.audio_self = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(**enc_kwargs), num_layers=num_layers,
        )

        # Cross-attention: текст как query, аудио как key/value — и наоборот
        self.t2a_cross = nn.MultiheadAttention(d_model, num_heads=nhead,
                                                dropout=dropout, batch_first=True)
        self.a2t_cross = nn.MultiheadAttention(d_model, num_heads=nhead,
                                                dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm_t = nn.LayerNorm(d_model)
        self.norm_a = nn.LayerNorm(d_model)

        self.head = _MultitaskHead(repr_dim=2 * d_model, head_hidden=head_hidden, dropout=dropout)

    def forward(self, xs, masks):
        text_x, audio_x = xs
        text_mask, audio_mask = masks

        # 1) self-attention внутри каждой модальности
        t = self.text_self(self.text_proj(text_x), src_key_padding_mask=~text_mask)
        a = self.audio_self(self.audio_proj(audio_x), src_key_padding_mask=~audio_mask)

        # 2) cross-attention: текст смотрит на аудио, аудио смотрит на текст
        t_aware, _ = self.t2a_cross(query=t, key=a, value=a, key_padding_mask=~audio_mask)
        a_aware, _ = self.a2t_cross(query=a, key=t, value=t, key_padding_mask=~text_mask)
        t = self.norm_t(t + self.dropout(t_aware))
        a = self.norm_a(a + self.dropout(a_aware))

        # 3) агрегация и объединение
        t_repr = _masked_mean(t, text_mask)
        a_repr = _masked_mean(a, audio_mask)
        fused = torch.cat([t_repr, a_repr], dim=-1)
        return self.head(fused)