import math
from pathlib import Path

import torch
from safetensors.torch import load_model
from sentence_transformers import SentenceTransformer
from torch import nn

from .additive import AttentiveProbe
from .audio_preprocessor import AudioToMelSpectrogramPreprocessor
from .conformer import ConformerEncoder


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, qk_norm: bool) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        if qk_norm:
            self.q_norm = nn.LayerNorm(d_model)
            self.k_norm = nn.LayerNorm(d_model)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        Q = self.split_heads(self.q_norm(self.W_q(Q)))
        K = self.split_heads(self.k_norm(self.W_k(K)))
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, act: nn.Module) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.act = act

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class CrossAttnTransformer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float, act: nn.Module, qk_norm: bool) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, qk_norm)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, act)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        normed_q = self.norm1(q)
        normed_kv = self.norm2(kv)
        attn_output = self.self_attn(normed_q, normed_q, normed_q)
        q = q + self.dropout(attn_output)

        normed_q = self.norm3(q)
        attn_output = self.self_attn(normed_q, normed_kv, normed_kv)
        q = q + self.dropout(attn_output)

        normed_q = self.norm4(q)
        ff_output = self.feed_forward(normed_q)
        q = q + self.dropout(ff_output)
        return q


class CrossAttnClassifier(nn.Module):
    def __init__(self, num_classes: int = 1, bert_weight_path: str = "weights/modernbert", conformer_weight_path: Path = Path("weights/"), num_layers: int = 8, num_heads: int = 8) -> None:
        super().__init__()
        self.text_enc = SentenceTransformer(bert_weight_path, local_files_only=True)
        bert_out_dim = 384

        self.audio_enc = ConformerEncoder()
        self.proc = AudioToMelSpectrogramPreprocessor()
        conformer_out_dim = 176

        hidden_size = 384

        self.audio_proj = nn.Linear(conformer_out_dim, hidden_size)
        self.text_proj = nn.Linear(bert_out_dim, hidden_size)

        self.text_blocks = nn.ModuleList(CrossAttnTransformer(hidden_size, num_heads, hidden_size, 0.15, nn.functional.silu, True) for _ in range(num_layers))
        self.audio_blocks = nn.ModuleList(CrossAttnTransformer(hidden_size, num_heads, hidden_size, 0.15, nn.functional.silu, True) for _ in range(num_layers))

        self.classifier = AttentiveProbe(hidden_size, num_classes=num_classes, num_layers=num_layers, num_heads=num_heads)
        self.load_audio_weights(conformer_weight_path)

    def forward(self, wavs: torch.Tensor, unpaded_length: list[int], text: list[str]) -> torch.Tensor:
        assert len(wavs.shape) == 2, "Audio should be [batch, length] size"

        text_emb = self.text_enc.encode(text, convert_to_tensor=True)
        text_emb = text_emb.clone().unsqueeze(1)  # [batch, 1, hidden]

        processed_features, length = self.proc.get_features(wavs, unpaded_length)
        audio_emb, _ = self.audio_enc(processed_features, length)
        audio_emb = audio_emb.transpose(1, 2)

        text_emb = self.text_proj(text_emb)
        audio_emb = self.audio_proj(audio_emb)

        for text_block, audio_block in zip(self.text_blocks, self.audio_blocks):
            text_emb, audio_emb = text_block(text_emb, audio_emb), audio_block(audio_emb, text_emb)

        emb = torch.cat((text_emb, audio_emb), dim=1)

        out = self.classifier(emb)
        return out

    def load_audio_weights(self, path: Path) -> None:
        load_model(self.proc, path.joinpath("audio_preprocessor.safetensors"))
        load_model(self.audio_enc, path.joinpath("conformer.safetensors"))
