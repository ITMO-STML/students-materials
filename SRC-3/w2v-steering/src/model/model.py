"""Wav2Vec2 frontend and AASIST graph backend."""

from collections.abc import Mapping, Sequence
from typing import Any, NotRequired, TypedDict, override

import jaxtyping as jt
import torch
import torch.nn.functional as F
from torch import nn
from transformers import Wav2Vec2Model


def _xavier(*size: int) -> nn.Parameter:
    parameter = nn.Parameter(torch.empty(*size))
    nn.init.xavier_normal_(parameter)
    return parameter


def _pairwise_mul_nodes(
    nodes: jt.Float[torch.Tensor, "b n d"],
) -> jt.Float[torch.Tensor, "b n n d"]:
    count = nodes.size(1)
    expanded = nodes.unsqueeze(2).expand(-1, -1, count, -1)
    return expanded * expanded.transpose(1, 2)


def _top_k_graph(
    scores: jt.Float[torch.Tensor, "b n 1"],
    nodes: jt.Float[torch.Tensor, "b n d"],
    ratio: float,
) -> jt.Float[torch.Tensor, "b k d"]:
    _, count, width = nodes.size()
    kept = max(int(count * ratio), 1)
    _, index = torch.topk(scores, kept, dim=1)
    index = index.expand(-1, -1, width)
    return torch.gather(nodes * scores, 1, index)


class Wav2Vec2Frontend(nn.Module):
    """Wav2Vec2 encoder that replaces the fairseq XLS-R frontend.

    Args:
        model_name: Hugging Face checkpoint.
        encoder: Already constructed encoder. Skips the weight download.
        requires_grad: When false, encoder weights are frozen. Train and eval mode follow the parent module.

    """

    def __init__(
        self,
        model_name: str,
        encoder: Wav2Vec2Model | None = None,
        requires_grad: bool = False,
    ) -> None:
        super().__init__()
        self.model = encoder if encoder is not None else Wav2Vec2Model.from_pretrained(model_name)
        # fairseq was called with mask=False; do not SpecAugment the states.
        self.model.config.mask_time_prob = 0.0
        self.model.config.mask_feature_prob = 0.0
        self.out_dim = self.model.config.hidden_size
        if not requires_grad:
            for parameter in self.model.parameters():
                parameter.requires_grad = False

    @override
    def forward(
        self,
        input_values: jt.Float[torch.Tensor, "b 64600"],
        attention_mask: jt.Int[torch.Tensor, "b 64600"] | None = None,
    ) -> jt.Float[torch.Tensor, "b frames hidden"]:
        """Encode a batch of waveforms.

        Args:
            input_values: Audio of shape `(batch, time)`.
            attention_mask: Valid-sample mask of shape `(batch, time)`.

        Returns:
            Encoder states of shape `(batch, frames, hidden)`.

        """
        hidden = self.model(input_values, attention_mask=attention_mask)
        return hidden.last_hidden_state


class GraphAttentionLayer(nn.Module):
    """Graph attention over one node type.

    Args:
        in_dim: Input node width.
        out_dim: Output node width.
        temperature: Softmax temperature.

    """

    def __init__(self, in_dim: int, out_dim: int, temperature: float = 1.0) -> None:
        super().__init__()
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = _xavier(out_dim, 1)
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(p=0.2)
        self.act = nn.SELU(inplace=True)
        self.temp = temperature

    @override
    def forward(self, nodes: jt.Float[torch.Tensor, "b n d"]) -> jt.Float[torch.Tensor, "b n out"]:
        """Apply attention, projection, and normalization.

        Args:
            nodes: Nodes of shape `(batch, nodes, dim)`.

        Returns:
            Updated nodes of shape `(batch, nodes, out_dim)`.

        """
        hidden = self.input_drop(nodes)
        attention = self._attention_map(hidden)
        projected = self._project(hidden, attention)
        return self.act(self._batch_norm(projected))

    def _attention_map(self, nodes: jt.Float[torch.Tensor, "b n d"]) -> jt.Float[torch.Tensor, "b n n 1"]:
        attention = torch.tanh(self.att_proj(_pairwise_mul_nodes(nodes)))
        attention = torch.matmul(attention, self.att_weight) / self.temp
        return F.softmax(attention, dim=-2)

    def _project(
        self,
        nodes: jt.Float[torch.Tensor, "b n d"],
        attention: jt.Float[torch.Tensor, "b n n 1"],
    ) -> jt.Float[torch.Tensor, "b n out"]:
        attended = self.proj_with_att(torch.matmul(attention.squeeze(-1), nodes))
        return attended + self.proj_without_att(nodes)

    def _batch_norm(self, nodes: jt.Float[torch.Tensor, "b n d"]) -> jt.Float[torch.Tensor, "b n d"]:
        shape = nodes.size()
        normalized = self.bn(nodes.view(-1, shape[-1]))
        return normalized.view(shape)


class HtrgGraphAttentionLayer(nn.Module):
    """Heterogeneous graph attention over spectral and temporal nodes.

    Args:
        in_dim: Input node width.
        out_dim: Output node width.
        temperature: Softmax temperature.

    """

    def __init__(self, in_dim: int, out_dim: int, temperature: float = 1.0) -> None:
        super().__init__()
        self.proj_type1 = nn.Linear(in_dim, in_dim)
        self.proj_type2 = nn.Linear(in_dim, in_dim)
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_projM = nn.Linear(in_dim, out_dim)
        self.att_weight11 = _xavier(out_dim, 1)
        self.att_weight22 = _xavier(out_dim, 1)
        self.att_weight12 = _xavier(out_dim, 1)
        self.att_weightM = _xavier(out_dim, 1)
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.proj_with_attM = nn.Linear(in_dim, out_dim)
        self.proj_without_attM = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(p=0.2)
        self.act = nn.SELU(inplace=True)
        self.temp = temperature

    @override
    def forward(
        self,
        temporal: jt.Float[torch.Tensor, "b t d"],
        spectral: jt.Float[torch.Tensor, "b f d"],
        master: jt.Float[torch.Tensor, "#b 1 d"] | None = None,
    ) -> tuple[
        jt.Float[torch.Tensor, "b t out"],
        jt.Float[torch.Tensor, "b f out"],
        jt.Float[torch.Tensor, "b 1 out"],
    ]:
        """Update both node types and the master node.

        Args:
            temporal: Temporal nodes `(batch, time, dim)`.
            spectral: Spectral nodes `(batch, frequency, dim)`.
            master: Optional master node `(batch, 1, dim)`.

        Returns:
            Updated temporal nodes, spectral nodes, and master node.

        """
        num_temporal = temporal.size(1)
        num_spectral = spectral.size(1)
        hidden = torch.cat([self.proj_type1(temporal), self.proj_type2(spectral)], dim=1)
        if master is None:
            master = torch.mean(hidden, dim=1, keepdim=True)
        hidden = self.input_drop(hidden)
        attention = self._attention_map(hidden, num_temporal, num_spectral)
        master = self._update_master(hidden, master)
        projected = self.act(self._batch_norm(self._project(hidden, attention)))
        temporal_out = projected.narrow(1, 0, num_temporal)
        spectral_out = projected.narrow(1, num_temporal, num_spectral)
        return temporal_out, spectral_out, master

    def _update_master(
        self,
        nodes: jt.Float[torch.Tensor, "b n d"],
        master: jt.Float[torch.Tensor, "#b 1 d"],
    ) -> jt.Float[torch.Tensor, "b 1 out"]:
        attention = self._master_attention(nodes, master)
        attended = self.proj_with_attM(torch.matmul(attention.squeeze(-1).unsqueeze(1), nodes))
        return attended + self.proj_without_attM(master)

    def _master_attention(
        self,
        nodes: jt.Float[torch.Tensor, "b n d"],
        master: jt.Float[torch.Tensor, "#b 1 d"],
    ) -> jt.Float[torch.Tensor, "b n 1"]:
        attention = torch.tanh(self.att_projM(nodes * master))
        attention = torch.matmul(attention, self.att_weightM) / self.temp
        return F.softmax(attention, dim=-2)

    def _attention_map(
        self,
        nodes: jt.Float[torch.Tensor, "b n d"],
        num_temporal: int,
        num_spectral: int,
    ) -> jt.Float[torch.Tensor, "b n n 1"]:
        attention = torch.tanh(self.att_proj(_pairwise_mul_nodes(nodes)))
        board = torch.zeros_like(attention[:, :, :, 0]).unsqueeze(-1)
        temporal = slice(0, num_temporal)
        spectral = slice(num_temporal, num_temporal + num_spectral)
        board[:, temporal, temporal, :] = torch.matmul(attention[:, temporal, temporal, :], self.att_weight11)
        board[:, spectral, spectral, :] = torch.matmul(attention[:, spectral, spectral, :], self.att_weight22)
        board[:, temporal, spectral, :] = torch.matmul(attention[:, temporal, spectral, :], self.att_weight12)
        board[:, spectral, temporal, :] = torch.matmul(attention[:, spectral, temporal, :], self.att_weight12)
        return F.softmax(board / self.temp, dim=-2)

    def _project(
        self,
        nodes: jt.Float[torch.Tensor, "b n d"],
        attention: jt.Float[torch.Tensor, "b n n 1"],
    ) -> jt.Float[torch.Tensor, "b n out"]:
        attended = self.proj_with_att(torch.matmul(attention.squeeze(-1), nodes))
        return attended + self.proj_without_att(nodes)

    def _batch_norm(self, nodes: jt.Float[torch.Tensor, "b n d"]) -> jt.Float[torch.Tensor, "b n d"]:
        shape = nodes.size()
        normalized = self.bn(nodes.view(-1, shape[-1]))
        return normalized.view(shape)


class GraphPool(nn.Module):
    """Keep the highest-scoring fraction of graph nodes.

    Args:
        ratio: Fraction of nodes to keep.
        in_dim: Node width.
        dropout: Dropout probability applied before scoring.

    """

    def __init__(self, ratio: float, in_dim: int, dropout: float) -> None:
        super().__init__()
        self.ratio = ratio
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    @override
    def forward(self, nodes: jt.Float[torch.Tensor, "b n d"]) -> jt.Float[torch.Tensor, "b k d"]:
        """Pool nodes by learned scores.

        Args:
            nodes: Graph nodes `(batch, nodes, dim)`.

        Returns:
            Pooled nodes `(batch, kept, dim)`.

        """
        scores = self.sigmoid(self.proj(self.drop(nodes)))
        return _top_k_graph(scores, nodes, self.ratio)


class ResidualBlock(nn.Module):
    """RawNet2 residual block used by the AASIST encoder.

    Args:
        nb_filts: Input and output channel counts.
        first: Skip the leading normalization when this is the first block.

    """

    def __init__(self, nb_filts: Sequence[int], first: bool = False) -> None:
        super().__init__()
        self.first = first
        in_channels = nb_filts[0]
        out_channels = nb_filts[1]
        self.bn1 = None if first else nn.BatchNorm2d(num_features=in_channels)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(2, 3),
            padding=(1, 1),
            stride=1,
        )
        self.selu = nn.SELU(inplace=True)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(2, 3),
            padding=(0, 1),
            stride=1,
        )
        self.downsample = in_channels != out_channels
        self.conv_downsample = (
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                padding=(0, 1),
                kernel_size=(1, 3),
                stride=1,
            )
            if self.downsample
            else None
        )

    @override
    def forward(
        self,
        features: jt.Float[torch.Tensor, "b c freq time"],
    ) -> jt.Float[torch.Tensor, "b c_out freq time"]:
        """Convolve one feature map.

        Args:
            features: Map of shape `(batch, channels, frequency, time)`.

        Returns:
            Residual output with the block's output channel count.

        """
        identity = features
        if self.bn1 is not None:
            _ = self.selu(self.bn1(features))
        hidden = self.conv1(features)
        hidden = self.selu(self.bn2(hidden))
        hidden = self.conv2(hidden)
        if self.conv_downsample is not None:
            identity = self.conv_downsample(identity)
        return hidden + identity


class AntispoofingOutput(TypedDict):
    """Logits and optional cross-entropy from one forward pass."""

    logits: jt.Float[torch.Tensor, "b 2"]
    loss: NotRequired[jt.Float[torch.Tensor, ""]]


class Wav2Vec2AASIST(nn.Module):
    """AASIST anti-spoofing network with a transformers Wav2Vec2 frontend.

    Args:
        wav2vec2: Frontend settings, including the checkpoint name and optional `requires_grad`.
        aasist: Graph-backend sizes.
        criterion: Loss used when labels are provided. Cross-entropy for now.
        wav2vec: Optional encoder, used to skip the pretrained download.

    """

    def __init__(
        self,
        wav2vec2: Mapping[str, Any],
        aasist: Mapping[str, Any],
        criterion: nn.Module,
        wav2vec: Wav2Vec2Model | None = None,
    ) -> None:
        super().__init__()
        projection_dim = int(aasist["projection_dim"])
        spectral_bins = int(aasist["spectral_bins"])
        encoder_channels = int(aasist["encoder_channels"])
        num_classes = int(aasist["num_classes"])
        gat_dims = tuple(int(dim) for dim in aasist["gat_dims"])
        pool_ratios = tuple(float(ratio) for ratio in aasist["pool_ratios"])
        temperatures = tuple(float(temperature) for temperature in aasist["temperatures"])
        channel_schedule = tuple(tuple(int(channel) for channel in pair) for pair in aasist["channel_schedule"])

        self.criterion = criterion
        requires_grad = bool(wav2vec2["requires_grad"]) if "requires_grad" in wav2vec2 else False
        self.frontend = Wav2Vec2Frontend(
            model_name=str(wav2vec2["model_name"]),
            encoder=wav2vec,
            requires_grad=requires_grad,
        )
        self.LL = nn.Linear(self.frontend.out_dim, projection_dim)
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.first_bn1 = nn.BatchNorm2d(num_features=encoder_channels)
        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        self.selu = nn.SELU(inplace=True)
        self.encoder = nn.Sequential(
            *(
                nn.Sequential(ResidualBlock(nb_filts=channels, first=index == 0))
                for index, channels in enumerate(channel_schedule)
            ),
        )
        self.attention = nn.Sequential(
            nn.Conv2d(encoder_channels, projection_dim, kernel_size=(1, 1)),
            nn.SELU(inplace=True),
            nn.BatchNorm2d(projection_dim),
            nn.Conv2d(projection_dim, encoder_channels, kernel_size=(1, 1)),
        )
        self.pos_S = nn.Parameter(torch.randn(1, spectral_bins, encoder_channels))
        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.GAT_layer_S = GraphAttentionLayer(encoder_channels, gat_dims[0], temperature=temperatures[0])
        self.GAT_layer_T = GraphAttentionLayer(encoder_channels, gat_dims[0], temperature=temperatures[1])
        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(gat_dims[1], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(gat_dims[1], gat_dims[1], temperature=temperatures[2])
        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hS1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hS2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.out_layer = nn.Linear(5 * gat_dims[1], num_classes)

    @override
    def forward(
        self,
        input_values: jt.Float[torch.Tensor, "b 64600"],
        attention_mask: jt.Int[torch.Tensor, "b 64600"] | None = None,
        labels: jt.Int[torch.Tensor, " b"] | None = None,
    ) -> AntispoofingOutput:
        """Classify a batch of waveforms.

        Args:
            input_values: Audio of shape `(batch, time)` or `(batch, time, 1)`.
            attention_mask: Valid-sample mask of shape `(batch, time)`.
            labels: Class indices of shape `(batch,)`. Loss is omitted when this is absent.

        Returns:
            `logits` of shape `(batch, num_classes)`, and `loss` when labels are provided.

        """
        features = self._encode(input_values, attention_mask)
        spectral, temporal = self._spectrogram_nodes(features)
        logits = self._readout(spectral, temporal)
        if labels is None:
            return {"logits": logits}
        return {"logits": logits, "loss": self.criterion(logits, labels)}

    def _encode(
        self,
        input_values: jt.Float[torch.Tensor, "b 64600"],
        attention_mask: jt.Int[torch.Tensor, "b 64600"] | None,
    ) -> jt.Float[torch.Tensor, "b c freq time"]:
        waveform = input_values.squeeze(-1)
        if waveform.ndim == 3:
            waveform = waveform[:, :, 0]
        hidden = self.LL(self.frontend(waveform, attention_mask))
        hidden = hidden.transpose(1, 2).unsqueeze(1)
        hidden = self.selu(self.first_bn(F.max_pool2d(hidden, (3, 3))))
        hidden = self.selu(self.first_bn1(self.encoder(hidden)))
        return hidden

    def _spectrogram_nodes(
        self,
        features: jt.Float[torch.Tensor, "b c freq time"],
    ) -> tuple[jt.Float[torch.Tensor, "b freq c"], jt.Float[torch.Tensor, "b time c"]]:
        weights = self.attention(features)
        spectral = torch.sum(features * F.softmax(weights, dim=-1), dim=-1).transpose(1, 2)
        temporal = torch.sum(features * F.softmax(weights, dim=-2), dim=-2).transpose(1, 2)
        return spectral + self.pos_S, temporal

    def _readout(
        self,
        spectral: jt.Float[torch.Tensor, "b freq c"],
        temporal: jt.Float[torch.Tensor, "b time c"],
    ) -> jt.Float[torch.Tensor, "b 2"]:
        spectral_nodes = self.pool_S(self.GAT_layer_S(spectral))
        temporal_nodes = self.pool_T(self.GAT_layer_T(temporal))
        temporal_a, spectral_a, master_a = self._stack(
            temporal_nodes,
            spectral_nodes,
            self.master1,
            self.HtrgGAT_layer_ST11,
            self.HtrgGAT_layer_ST12,
            self.pool_hT1,
            self.pool_hS1,
        )
        temporal_b, spectral_b, master_b = self._stack(
            temporal_nodes,
            spectral_nodes,
            self.master2,
            self.HtrgGAT_layer_ST21,
            self.HtrgGAT_layer_ST22,
            self.pool_hT2,
            self.pool_hS2,
        )
        temporal_out = torch.max(self.drop_way(temporal_a), self.drop_way(temporal_b))
        spectral_out = torch.max(self.drop_way(spectral_a), self.drop_way(spectral_b))
        master = torch.max(self.drop_way(master_a), self.drop_way(master_b))
        temporal_max = torch.max(torch.abs(temporal_out), dim=1).values
        spectral_max = torch.max(torch.abs(spectral_out), dim=1).values
        pooled = torch.cat(
            [
                temporal_max,
                torch.mean(temporal_out, dim=1),
                spectral_max,
                torch.mean(spectral_out, dim=1),
                master.squeeze(1),
            ],
            dim=1,
        )
        return self.out_layer(self.drop(pooled))

    def _stack(
        self,
        temporal: jt.Float[torch.Tensor, "b t d"],
        spectral: jt.Float[torch.Tensor, "b f d"],
        master: jt.Float[torch.Tensor, "#b 1 d"],
        first_layer: HtrgGraphAttentionLayer,
        second_layer: HtrgGraphAttentionLayer,
        pool_temporal: GraphPool,
        pool_spectral: GraphPool,
    ) -> tuple[
        jt.Float[torch.Tensor, "b t_out d_out"],
        jt.Float[torch.Tensor, "b f_out d_out"],
        jt.Float[torch.Tensor, "b 1 d_out"],
    ]:
        temporal, spectral, master = first_layer(temporal, spectral, master=master)
        spectral = pool_spectral(spectral)
        temporal = pool_temporal(temporal)
        temporal_aug, spectral_aug, master_aug = second_layer(temporal, spectral, master=master)
        return temporal + temporal_aug, spectral + spectral_aug, master + master_aug
