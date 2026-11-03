"""Компактный class-conditional U-Net для DDPM на MNIST."""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ConvBlock(nn.Module):
    """Conv -> GroupNorm -> SiLU + time/class эмбеддинг."""

    def __init__(self, in_ch: int, out_ch: int, emb_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()
        self.emb_proj = nn.Linear(emb_dim, out_ch)

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = h + self.emb_proj(emb)[:, :, None, None]
        h = self.act(self.norm2(self.conv2(h)))
        return h


class UNet(nn.Module):
    """U-Net 3 уровня. Вход: 1 + class_emb канал (class-embedding конкатенируется)."""

    def __init__(self, base_ch: int = 32, time_emb: int = 128, class_emb: int = 4, num_classes: int = 10) -> None:
        super().__init__()
        self.class_emb = class_emb
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb),
            nn.Linear(time_emb, time_emb),
            nn.SiLU(),
            nn.Linear(time_emb, time_emb),
        )
        self.label_emb = nn.Embedding(num_classes, class_emb)

        in_ch = 1 + class_emb
        self.enc1 = ConvBlock(in_ch, base_ch, time_emb)
        self.enc2 = ConvBlock(base_ch, base_ch * 2, time_emb)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4, time_emb)
        self.pool = nn.MaxPool2d(2)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_ch * 4, base_ch * 2, time_emb)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = ConvBlock(base_ch * 2, base_ch, time_emb)

        self.out = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x: Tensor, t: Tensor, labels: Tensor) -> Tensor:
        emb = self.time_mlp(t)
        label = self.label_emb(labels)[:, :, None, None]
        label = label.expand(-1, -1, x.shape[2], x.shape[3])
        x = torch.cat([x, label], dim=1)

        h1 = self.enc1(x, emb)              # 28x28
        h2 = self.enc2(self.pool(h1), emb)  # 14x14
        h3 = self.enc3(self.pool(h2), emb)  # 7x7

        u2 = self.up2(h3)
        u2 = self.dec2(torch.cat([u2, h2], dim=1), emb)
        u1 = self.up1(u2)
        u1 = self.dec1(torch.cat([u1, h1], dim=1), emb)
        return self.out(u1)