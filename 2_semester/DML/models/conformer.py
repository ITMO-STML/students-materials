# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import random
from contextlib import nullcontext

import torch
import torch.distributed
from torch import nn
from torch.nn import LayerNorm
from torch.nn import functional as F

activation_registry = {
    "identity": nn.Identity,
    "hardtanh": nn.Hardtanh,
    "relu": nn.ReLU,
    "selu": nn.SELU,
    "swish": nn.SiLU,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
}

INF_VAL = 10000.0


def avoid_float16_autocast_context():
    """
    If the current autocast context is float16, cast it to bfloat16
    if available (unless we're in jit) or float32
    """

    if torch.is_autocast_enabled() and torch.get_autocast_gpu_dtype() == torch.float16:
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            return torch.amp.autocast("cuda", dtype=torch.float32)

        if torch.cuda.is_bf16_supported():
            return torch.amp.autocast("cuda", dtype=torch.bfloat16)
        return torch.amp.autocast("cuda", dtype=torch.float32)
    return nullcontext()


def compute_stochastic_depth_drop_probs(
    num_layers: int,
    stochastic_depth_drop_prob: float = 0.0,
    stochastic_depth_mode: str = "linear",
    stochastic_depth_start_layer: int = 1,
) -> list[float]:
    """Computes drop probabilities for stochastic depth regularization technique.
    The first layer is never dropped and the starting layer needs to be greater
    or equal to 1.

    Args:
        num_layers (int): number of layers in the network.
        stochastic_depth_drop_prob (float): if non-zero, will randomly drop
            layers during training. The higher this value, the more often layers
            are dropped. Defaults to 0.0.
        stochastic_depth_mode (str): can be either "linear" or "uniform". If
            set to "uniform", all layers have the same probability of drop. If
            set to "linear", the drop probability grows linearly from 0 for the
            first layer to the desired value for the final layer. Defaults to
            "linear".
        stochastic_depth_start_layer (int): starting layer for stochastic depth.
            All layers before this will never be dropped. Note that drop
            probability will be adjusted accordingly if mode is "linear" when
            start layer is > 1. Defaults to 1.
    Returns:
        List[float]: list of drop probabilities for all layers
    """
    if not (0 <= stochastic_depth_drop_prob < 1.0):
        raise ValueError("stochastic_depth_drop_prob has to be in [0, 1).")
    if not (1 <= stochastic_depth_start_layer <= num_layers):
        raise ValueError("stochastic_depth_start_layer has to be in [1, num layers].")

    # Layers before `stochastic_depth_start_layer` are never dropped
    layer_drop_probs = [0.0] * stochastic_depth_start_layer

    # Layers starting with `stochastic_depth_start_layer` may be dropped
    if (L := num_layers - stochastic_depth_start_layer) > 0:
        if stochastic_depth_mode == "linear":
            # we start with 1/L * drop_prob and and end with the desired drop probability.
            layer_drop_probs += [l / L * stochastic_depth_drop_prob for l in range(1, L + 1)]
        elif stochastic_depth_mode == "uniform":
            layer_drop_probs += [stochastic_depth_drop_prob] * L
        else:
            raise ValueError(f'stochastic_depth_mode has to be one of ["linear", "uniform"]. Current value: {stochastic_depth_mode}')
    return layer_drop_probs


class Swish(nn.SiLU):
    """
    Swish activation function introduced in 'https://arxiv.org/abs/1710.05941'
    Mathematically identical to SiLU. See note in nn.SiLU for references.
    """


class CausalConv1D(nn.Conv1d):
    """
    A causal version of nn.Conv1d where each step would have limited access to locations on its right or left
    All arguments are the same as nn.Conv1d except padding.

    If padding is set None, then paddings are set automatically to make it a causal convolution where each location would not see any steps on its right.

    If padding is set as a list (size of 2), then padding[0] would be used as left padding and padding[1] as right padding.
    It would make it possible to control the number of steps to be accessible on the right and left.
    This mode is not supported when stride > 1. padding[0]+padding[1] should be equal to (kernel_size - 1).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: str | int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        self.cache_drop_size = None
        if padding is None:
            self._left_padding = kernel_size - 1
            self._right_padding = stride - 1
        else:
            if stride != 1 and padding != kernel_size - 1:
                raise ValueError("No striding allowed for non-symmetric convolutions!")
            if isinstance(padding, int):
                self._left_padding = padding
                self._right_padding = padding
            elif isinstance(padding, list) and len(padding) == 2 and padding[0] + padding[1] == kernel_size - 1:
                self._left_padding = padding[0]
                self._right_padding = padding[1]
            else:
                raise ValueError(f"Invalid padding param: {padding}!")

        self._max_cache_len = self._left_padding

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

    def update_cache(self, x, cache=None):
        if cache is None:
            new_x = F.pad(x, pad=(self._left_padding, self._right_padding))
            next_cache = cache
        else:
            new_x = F.pad(x, pad=(0, self._right_padding))
            new_x = torch.cat([cache, new_x], dim=-1)
            if self.cache_drop_size > 0:
                next_cache = new_x[:, :, : -self.cache_drop_size]
            else:
                next_cache = new_x
            next_cache = next_cache[:, :, -cache.size(-1) :]
        return new_x, next_cache

    def forward(self, x, cache=None):
        x, cache = self.update_cache(x, cache=cache)
        x = super().forward(x)
        if cache is None:
            return x
        return x, cache


class ConformerFeedForward(nn.Module):
    """
    feed-forward module of Conformer model.
    use_bias (bool): Apply bias to all Linear and Conv1d layers improve activation flow and stabilize training of huge models.
    """

    def __init__(self, d_model, d_ff, dropout, activation=Swish(), use_bias=True):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.use_bias = use_bias
        self.linear1 = nn.Linear(d_model, d_ff, bias=self.use_bias)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(d_ff, d_model, bias=self.use_bias)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

    def reset_parameters_ff(self):
        ffn1_max = self.d_model**-0.5
        ffn2_max = self.d_ff**-0.5
        with torch.no_grad():
            nn.init.uniform_(self.linear1.weight, -ffn1_max, ffn1_max)
            nn.init.uniform_(self.linear2.weight, -ffn2_max, ffn2_max)
            if self.use_bias:
                nn.init.uniform_(self.linear1.bias, -ffn1_max, ffn1_max)
                nn.init.uniform_(self.linear2.bias, -ffn2_max, ffn2_max)


class ConformerConvolution(nn.Module):
    """The convolution module for the Conformer model.
    Args:
        d_model (int): hidden dimension
        kernel_size (int): kernel size for depthwise convolution
        pointwise_activation (str): name of the activation function to be used for the pointwise conv.
            Note that Conformer uses a special key `glu_` which is treated as the original default from
            the paper.
        use_bias (bool): Use bias in all Linear and Conv1d layers improve activation flow and stabilize training of huge models.
            Defaults to True
    """

    def __init__(
        self,
        d_model,
        kernel_size,
        norm_type="batch_norm",
        conv_context_size=None,
        pointwise_activation="glu_",
        use_bias=True,
    ):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.norm_type = norm_type
        self.use_bias = use_bias

        if conv_context_size is None:
            conv_context_size = (kernel_size - 1) // 2

        if pointwise_activation in activation_registry:
            self.pointwise_activation = activation_registry[pointwise_activation]()
            dw_conv_input_dim = d_model * 2

            if hasattr(self.pointwise_activation, "inplace"):
                self.pointwise_activation.inplace = True
        else:
            self.pointwise_activation = pointwise_activation
            dw_conv_input_dim = d_model

        self.pointwise_conv1 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=self.use_bias,
        )

        self.depthwise_conv = CausalConv1D(
            in_channels=dw_conv_input_dim,
            out_channels=dw_conv_input_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=conv_context_size,
            groups=dw_conv_input_dim,
            bias=self.use_bias,
        )

        self.batch_norm = nn.BatchNorm1d(dw_conv_input_dim)

        self.activation = Swish()
        self.pointwise_conv2 = nn.Conv1d(
            in_channels=dw_conv_input_dim,
            out_channels=d_model,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=self.use_bias,
        )

    def forward(self, x, pad_mask=None, cache=None):
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)

        # Compute the activation function or use GLU for original Conformer
        x = nn.functional.glu(x, dim=1)

        if pad_mask is not None:
            x = x.masked_fill(pad_mask.unsqueeze(1), 0.0)

        x = self.depthwise_conv(x, cache=cache)
        if cache is not None:
            x, cache = x

        if self.norm_type == "layer_norm":
            x = x.transpose(1, 2)
            x = self.batch_norm(x)
            x = x.transpose(1, 2)
        else:
            x = self.batch_norm(x)

        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = x.transpose(1, 2)
        if cache is None:
            return x
        return x, cache

    def reset_parameters_conv(self):
        pw1_max = pw2_max = self.d_model**-0.5
        dw_max = self.kernel_size**-0.5

        with torch.no_grad():
            nn.init.uniform_(self.pointwise_conv1.weight, -pw1_max, pw1_max)
            nn.init.uniform_(self.pointwise_conv2.weight, -pw2_max, pw2_max)
            nn.init.uniform_(self.depthwise_conv.weight, -dw_max, dw_max)
            if self.use_bias:
                nn.init.uniform_(self.pointwise_conv1.bias, -pw1_max, pw1_max)
                nn.init.uniform_(self.pointwise_conv2.bias, -pw2_max, pw2_max)
                nn.init.uniform_(self.depthwise_conv.bias, -dw_max, dw_max)


########
#
# attn
#
########
class MultiHeadAttention(nn.Module):
    """Multi-Head Attention layer of Transformer.
    Args:
        n_head (int): number of heads
        n_feat (int): size of the features
        dropout_rate (float): dropout rate
        use_bias (bool): whether to remove bias in linear and conv layers
        use_pytorch_sdpa (bool): use torch sdpa instead of manual attention
        use_pytorch_sdpa_backends list[str]: list of backend names to use in sdpa. None or empty list means all backends. e.g. ["MATH"]
    """

    def __init__(
        self,
        n_head,
        n_feat,
        dropout_rate,
        max_cache_len=0,
        use_bias=True,
        use_pytorch_sdpa=False,
        use_pytorch_sdpa_backends=None,
    ):
        """Construct an MultiHeadedAttention object."""
        super().__init__()
        self.use_pytorch_sdpa = use_pytorch_sdpa
        if self.use_pytorch_sdpa and use_pytorch_sdpa_backends:
            use_pytorch_sdpa_backends = [getattr(torch.nn.attention.SDPBackend, backend_name) for backend_name in use_pytorch_sdpa_backends]
        self.use_pytorch_sdpa_backends = use_pytorch_sdpa_backends

        self.cache_drop_size = None
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.s_d_k = math.sqrt(self.d_k)
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat, bias=use_bias)
        self.linear_k = nn.Linear(n_feat, n_feat, bias=use_bias)
        self.linear_v = nn.Linear(n_feat, n_feat, bias=use_bias)
        self.linear_out = nn.Linear(n_feat, n_feat, bias=use_bias)
        self.dropout = nn.Dropout(p=dropout_rate)

        self._max_cache_len = max_cache_len

    def forward_qkv(self, query, key, value):
        """Transforms query, key and value.
        Args:
            query (torch.Tensor): (batch, time1, size)
            key (torch.Tensor): (batch, time2, size)
            value (torch.Tensor): (batch, time2, size)
        returns:
            q (torch.Tensor): (batch, head, time1, size)
            k (torch.Tensor): (batch, head, time2, size)
            v (torch.Tensor): (batch, head, time2, size)
        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        return q, k, v

    def forward_attention(self, value, scores, mask):
        """Compute attention context vector.
        Args:
            value (torch.Tensor): (batch, time2, size)
            scores(torch.Tensor): (batch, time1, time2)
            mask(torch.Tensor): (batch, time1, time2)
        returns:
            value (torch.Tensor): transformed `value` (batch, time2, d_model) weighted by the attention scores
        """
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1)  # (batch, 1, time1, time2)
            scores = scores.masked_fill(mask, -INF_VAL)
            attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, head, time1, time2)
        else:
            attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).reshape(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, query, key, value, mask, pos_emb=None, cache=None):
        """Compute 'Scaled Dot Product Attention'.
        Args:
            query (torch.Tensor): (batch, time1, size)
            key (torch.Tensor): (batch, time2, size)
            value(torch.Tensor): (batch, time2, size)
            mask (torch.Tensor): (batch, time1, time2)
            cache (torch.Tensor) : (batch, time_cache, size)

        returns:
            output (torch.Tensor): transformed `value` (batch, time1, d_model) weighted by the query dot key attention
            cache (torch.Tensor) : (batch, time_cache_next, size)
        """
        key, value, query, cache = self.update_cache(key=key, value=value, query=query, cache=cache)

        if torch.is_autocast_enabled():
            query, key, value = query.to(torch.float32), key.to(torch.float32), value.to(torch.float32)

        # temporary until we solve this more gracefully
        with avoid_float16_autocast_context():
            q, k, v = self.forward_qkv(query, key, value)

            if self.use_pytorch_sdpa:
                n_batch = value.size(0)

                if mask is not None:
                    mask = ~mask.unsqueeze(1)

                dropout_rate = self.dropout_rate if self.training else 0
                if self.use_pytorch_sdpa_backends:
                    with torch.nn.attention.sdpa_kernel(self.use_pytorch_sdpa_backends):
                        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_rate)
                else:
                    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_rate)

                # this IF block can be deleted when https://github.com/pytorch/pytorch/pull/131863 is in the stable version
                if mask is not None:
                    all_masked_rows = torch.all(~mask, dim=-1)
                    all_masked_rows.unsqueeze_(-1)
                    out = out.masked_fill(all_masked_rows, 0.0)

                out = out.transpose(1, 2).reshape(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)
                out = self.linear_out(out)  # (batch, time1, d_model)
            else:
                scores = torch.matmul(q, k.transpose(-2, -1)) / self.s_d_k
                out = self.forward_attention(v, scores, mask)

        if cache is None:
            return out
        return out, cache

    def update_cache(self, key, value, query, cache):
        if cache is not None:
            key = value = torch.cat([cache, key], dim=1)
            q_keep_size = query.shape[1] - self.cache_drop_size
            cache = torch.cat([cache[:, q_keep_size:, :], query[:, :q_keep_size, :]], dim=1)
        return key, value, query, cache


class RelPositionMultiHeadAttention(MultiHeadAttention):
    """Multi-Head Attention layer of Transformer-XL with support of relative positional encoding.
    Paper: https://arxiv.org/abs/1901.02860
    Args:
        n_head (int): number of heads
        n_feat (int): size of the features
        dropout_rate (float): dropout rate
        use_bias (bool): whether to apply bias in linear and conv layers of MultiHeadAttention
    """

    def __init__(
        self,
        n_head,
        n_feat,
        dropout_rate,
        pos_bias_u,
        pos_bias_v,
        max_cache_len=0,
        use_bias=True,
        use_pytorch_sdpa=False,
        use_pytorch_sdpa_backends=None,
    ):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(
            n_head=n_head,
            n_feat=n_feat,
            dropout_rate=dropout_rate,
            max_cache_len=max_cache_len,
            use_bias=use_bias,
            use_pytorch_sdpa=use_pytorch_sdpa,
            use_pytorch_sdpa_backends=use_pytorch_sdpa_backends,
        )
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # these two learnable biases are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        if pos_bias_u is None or pos_bias_v is None:
            self.pos_bias_u = nn.Parameter(torch.FloatTensor(self.h, self.d_k))
            self.pos_bias_v = nn.Parameter(torch.FloatTensor(self.h, self.d_k))
            # nn.init.normal_(self.pos_bias_u, 0.0, 0.02)
            # nn.init.normal_(self.pos_bias_v, 0.0, 0.02)
            nn.init.zeros_(self.pos_bias_u)
            nn.init.zeros_(self.pos_bias_v)
        else:
            self.pos_bias_u = pos_bias_u
            self.pos_bias_v = pos_bias_v

    def rel_shift(self, x):
        """Compute relative positional encoding.
        Args:
            x (torch.Tensor): (batch, nheads, time, 2*time-1)
        """
        b, h, qlen, pos_len = x.size()  # (b, h, t1, t2)
        # need to add a column of zeros on the left side of last dimension to perform the relative shifting
        x = torch.nn.functional.pad(x, pad=(1, 0))  # (b, h, t1, t2+1)
        x = x.view(b, h, -1, qlen)  # (b, h, t2+1, t1)
        # need to drop the first row
        x = x[:, :, 1:].view(b, h, qlen, pos_len)  # (b, h, t1, t2)
        return x

    def forward(self, query, key, value, mask, pos_emb, cache=None):
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.
        Args:
            query (torch.Tensor): (batch, time1, size)
            key (torch.Tensor): (batch, time2, size)
            value(torch.Tensor): (batch, time2, size)
            mask (torch.Tensor): (batch, time1, time2)
            pos_emb (torch.Tensor) : (batch, time1, size)
            cache (torch.Tensor) : (batch, time_cache, size)

        Returns:
            output (torch.Tensor): transformed `value` (batch, time1, d_model) weighted by the query dot key attention
            cache (torch.Tensor) : (batch, time_cache_next, size)
        """
        key, value, query, cache = self.update_cache(key=key, value=value, query=query, cache=cache)

        if torch.is_autocast_enabled():
            query, key, value = query.to(torch.float32), key.to(torch.float32), value.to(torch.float32)

        # temporary until we solve this more gracefully
        with avoid_float16_autocast_context():
            q, k, v = self.forward_qkv(query, key, value)
            q = q.transpose(1, 2)  # (batch, time1, head, d_k)

            n_batch_pos = pos_emb.size(0)
            n_batch = value.size(0)
            p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
            p = p.transpose(1, 2)  # (batch, head, time1, d_k)

            # (batch, head, time1, d_k)
            q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
            # (batch, head, time1, d_k)
            q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

            # compute attention score
            # first compute matrix a and matrix c
            # as described in https://arxiv.org/abs/1901.02860 Section 3.3
            # (batch, head, time1, time2)

            # compute matrix b and matrix d
            # (batch, head, time1, time2)
            matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
            matrix_bd = self.rel_shift(matrix_bd)

            if self.use_pytorch_sdpa:
                scale_factor = 1 / math.sqrt(q_with_bias_u.size(-1))
                matrix_bd = matrix_bd[:, :, :, : k.size(-2)] * scale_factor

                if mask is not None:
                    mask = mask.unsqueeze(1)
                    matrix_bd.masked_fill_(mask, -INF_VAL)

                dropout_rate = self.dropout_rate if self.training else 0
                if self.use_pytorch_sdpa_backends:
                    with torch.nn.attention.sdpa_kernel(self.use_pytorch_sdpa_backends):
                        out = torch.nn.functional.scaled_dot_product_attention(q_with_bias_u, k, v, attn_mask=matrix_bd, dropout_p=dropout_rate)
                else:
                    out = torch.nn.functional.scaled_dot_product_attention(q_with_bias_u, k, v, attn_mask=matrix_bd, dropout_p=dropout_rate)

                # this IF block can be deleted when https://github.com/pytorch/pytorch/pull/131863 is in the stable version
                if mask is not None:
                    all_masked_rows = torch.all(mask, dim=-1)
                    all_masked_rows.unsqueeze_(-1)
                    all_masked_rows = all_masked_rows.expand(-1, out.size(1), -1, out.size(-1))
                    out = out.masked_fill(all_masked_rows, 0.0)

                out = out.transpose(1, 2).reshape(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)
                out = self.linear_out(out)  # (batch, time1, d_model)
            else:
                # drops extra elements in the matrix_bd to match the matrix_ac's size
                matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
                matrix_bd = matrix_bd[:, :, :, : matrix_ac.size(-1)]
                scores = (matrix_ac + matrix_bd) / self.s_d_k  # (batch, head, time1, time2)
                out = self.forward_attention(v, scores, mask)

        if cache is None:
            return out
        return out, cache


class ConformerLayer(nn.Module):
    """A single block of the Conformer encoder.

    Args:
        d_model (int): input dimension of MultiheadAttentionMechanism and PositionwiseFeedForward
        d_ff (int): hidden dimension of PositionwiseFeedForward
        self_attention_model (str): type of the attention layer and positional encoding
            'rel_pos': relative positional embedding and Transformer-XL
            'rel_pos_local_attn': relative positional embedding and Transformer-XL with local attention using
                overlapping chunks. Attention context is determined by att_context_size parameter.
            'abs_pos': absolute positional embedding and Transformer
            Default is rel_pos.
        global_tokens (int): number of tokens to be used for global attention.
            Only relevant if self_attention_model is 'rel_pos_local_attn'.
            Defaults to 0.
        global_tokens_spacing (int): how far apart the global tokens are
            Defaults to 1.
        global_attn_separate (bool): whether the q, k, v layers used for global tokens should be separate.
            Defaults to False.
        n_heads (int): number of heads for multi-head attention
        conv_kernel_size (int): kernel size for depthwise convolution in convolution module
        dropout (float): dropout probabilities for linear layers
        dropout_att (float): dropout probabilities for attention distributions
        use_bias (bool): Apply bias to all Linear and Conv1d layers from each ConformerLayer to improve activation flow and stabilize training of huge models.
            Defaults to True.
    """

    def __init__(
        self,
        d_model,
        d_ff,
        self_attention_model="rel_pos",
        global_tokens=0,
        global_tokens_spacing=1,
        global_attn_separate=False,
        n_heads=4,
        conv_kernel_size=31,
        conv_norm_type="batch_norm",
        conv_context_size=None,
        dropout=0.1,
        dropout_att=0.1,
        pos_bias_u=None,
        pos_bias_v=None,
        att_context_size=[-1, -1],
        use_bias=True,
        use_pytorch_sdpa=False,
        use_pytorch_sdpa_backends=None,
    ) -> None:
        super().__init__()

        self.use_pytorch_sdpa = use_pytorch_sdpa
        if use_pytorch_sdpa_backends is None:
            use_pytorch_sdpa_backends = []
        self.use_pytorch_sdpa_backends = use_pytorch_sdpa_backends
        self.self_attention_model = self_attention_model
        self.n_heads = n_heads
        self.fc_factor = 0.5

        # first feed forward module
        self.norm_feed_forward1 = LayerNorm(d_model)
        self.feed_forward1 = ConformerFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout, use_bias=use_bias)

        # convolution module
        self.norm_conv = LayerNorm(d_model)
        self.conv = ConformerConvolution(
            d_model=d_model,
            kernel_size=conv_kernel_size,
            norm_type=conv_norm_type,
            conv_context_size=conv_context_size,
            use_bias=use_bias,
        )

        # multi-headed self-attention module
        self.norm_self_att = LayerNorm(d_model)
        MHA_max_cache_len = att_context_size[0]

        self.self_attn = RelPositionMultiHeadAttention(
            n_head=n_heads,
            n_feat=d_model,
            dropout_rate=dropout_att,
            pos_bias_u=pos_bias_u,
            pos_bias_v=pos_bias_v,
            max_cache_len=MHA_max_cache_len,
            use_bias=use_bias,
            use_pytorch_sdpa=self.use_pytorch_sdpa,
            use_pytorch_sdpa_backends=self.use_pytorch_sdpa_backends,
        )

        # second feed forward module
        self.norm_feed_forward2 = LayerNorm(d_model)
        self.feed_forward2 = ConformerFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout, use_bias=use_bias)

        self.dropout = nn.Dropout(dropout)
        self.norm_out = LayerNorm(d_model)

    def forward(self, x, att_mask=None, pos_emb=None, pad_mask=None, cache_last_channel=None, cache_last_time=None):
        """
        Args:
            x (torch.Tensor): input signals (B, T, d_model)
            att_mask (torch.Tensor): attention masks(B, T, T)
            pos_emb (torch.Tensor): (L, 1, d_model)
            pad_mask (torch.tensor): padding mask
            cache_last_channel (torch.tensor) : cache for MHA layers (B, T_cache, d_model)
            cache_last_time (torch.tensor) : cache for convolutional layers (B, d_model, T_cache)
        Returns:
            x (torch.Tensor): (B, T, d_model)
            cache_last_channel (torch.tensor) : next cache for MHA layers (B, T_cache, d_model)
            cache_last_time (torch.tensor) : next cache for convolutional layers (B, d_model, T_cache)
        """
        residual = x
        x = self.norm_feed_forward1(x)
        x = self.feed_forward1(x)
        residual = residual + self.dropout(x) * self.fc_factor

        x = self.norm_self_att(residual)
        x = self.self_attn(query=x, key=x, value=x, mask=att_mask, pos_emb=pos_emb, cache=cache_last_channel)

        if x is not None and cache_last_channel is not None:
            (x, cache_last_channel) = x

        residual = residual + self.dropout(x)

        x = self.norm_conv(residual)
        x = self.conv(x, pad_mask=pad_mask, cache=cache_last_time)
        if cache_last_time is not None:
            (x, cache_last_time) = x
        residual = residual + self.dropout(x)

        x = self.norm_feed_forward2(residual)
        x = self.feed_forward2(x)
        residual = residual + self.dropout(x) * self.fc_factor

        x = self.norm_out(residual)

        if cache_last_channel is None:
            return x
        return x, cache_last_channel, cache_last_time


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding.
    Args:
        d_model (int): embedding dim
        dropout_rate (float): dropout rate
        max_len (int): maximum input length
        xscale (bool): whether to scale the input by sqrt(d_model)
        dropout_rate_emb (float): dropout rate for the positional embeddings
    """

    def __init__(self, d_model, dropout_rate, max_len=5000, xscale=None, dropout_rate_emb=0.0):
        """Construct an PositionalEncoding object."""
        super().__init__()
        self.d_model = d_model
        self.xscale = xscale
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.max_len = max_len
        if dropout_rate_emb > 0:
            self.dropout_emb = nn.Dropout(dropout_rate_emb)
        else:
            self.dropout_emb = None

    def create_pe(self, positions, dtype):
        pos_length = positions.size(0)
        pe = torch.zeros(pos_length, self.d_model, device=positions.device)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float32, device=positions.device) * -(math.log(INF_VAL) / self.d_model))
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        pe = pe.unsqueeze(0).to(dtype)
        if hasattr(self, "pe"):
            self.pe = pe
        else:
            self.register_buffer("pe", pe, persistent=False)

    def extend_pe(self, length, device, dtype):
        """Reset and extend the positional encodings if needed."""
        if hasattr(self, "pe") and self.pe.size(1) >= length:
            return
        positions = torch.arange(0, length, dtype=torch.float32, device=device).unsqueeze(1)
        self.create_pe(positions=positions, dtype=dtype)

    def forward(self, x: torch.Tensor, cache_len=0):
        """Adds positional encoding.
        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, feature_size)
            cache_len (int): the size of the cache which is used to shift positions
        Returns:
            x+pos_emb (torch.Tensor): Its shape is (batch, time, feature_size)
            pos_emb (torch.Tensor): Its shape is (1, time, feature_size)
        """
        input_len = x.size(1) + cache_len
        if self.xscale:
            x = x * self.xscale
        pos_emb = self.pe[:, :input_len]
        if self.dropout_emb:
            pos_emb = self.dropout_emb(pos_emb)
        x = x + pos_emb
        return self.dropout(x), pos_emb


class RelPositionalEncoding(PositionalEncoding):
    """Relative positional encoding for TransformerXL's layers
    See : Appendix B in https://arxiv.org/abs/1901.02860
    Args:
        d_model (int): embedding dim
        dropout_rate (float): dropout rate
        max_len (int): maximum input length
        xscale (bool): whether to scale the input by sqrt(d_model)
        dropout_rate_emb (float): dropout rate for the positional embeddings
    """

    def extend_pe(self, length, device, dtype):
        """Reset and extend the positional encodings if needed."""
        needed_size = 2 * length - 1
        if hasattr(self, "pe") and self.pe.size(1) >= needed_size:
            return
        # positions would be from negative numbers to positive
        # positive positions would be used for left positions and negative for right positions
        positions = torch.arange(length - 1, -length, -1, dtype=torch.float32, device=device).unsqueeze(1)
        self.create_pe(positions=positions, dtype=dtype)

    def forward(self, x, cache_len=0):
        """Compute positional encoding.
        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, feature_size)
            cache_len (int): the size of the cache which is used to shift positions
        Returns:
            x (torch.Tensor): Its shape is (batch, time, feature_size)
            pos_emb (torch.Tensor): Its shape is (1, time, feature_size)
        """

        if self.xscale:
            x = x * self.xscale

        # center_pos would be the index of position 0
        # negative positions would be used for right and positive for left tokens
        # for input of length L, 2*L-1 positions are needed, positions from (L-1) to -(L-1)
        input_len = x.size(1) + cache_len
        center_pos = self.pe.size(1) // 2 + 1
        start_pos = center_pos - input_len
        end_pos = center_pos + input_len - 1
        pos_emb = self.pe[:, start_pos:end_pos]
        if self.dropout_emb:
            pos_emb = self.dropout_emb(pos_emb)
        return self.dropout(x), pos_emb


########
#
# subsamp
#
########
def apply_channel_mask(tensor, mask):
    """Apply mask to tensor with channel dimension."""
    # tensor: (batch, channels, time, features)
    # mask: (batch, time, features)
    batch_size, channels, time, features = tensor.shape
    expanded_mask = mask.unsqueeze(1).expand(batch_size, channels, time, features)
    return tensor * expanded_mask


def calculate_conv_output_size(input_size: torch.Tensor, kernel_size: int, stride: int, padding: tuple[int, int]):
    """Calculate exact output size after convolution."""
    return (input_size + padding[0] + padding[1] - kernel_size) // stride + 1


class MaskedConvSequential(nn.Sequential):
    def forward(self, x, lengths):
        # Convert input (batch, time, features) to conv format
        x = x.unsqueeze(1)  # (batch, 1, time, features)
        current_lengths = lengths.clone().float()
        mask = self._create_mask(x, current_lengths.long())

        # Process through each layer with mask propagation
        for i, layer in enumerate(self):
            # Apply current mask before layer
            x = apply_channel_mask(x, mask)

            # Apply layer
            x = layer(x)

            # Update lengths for stride operations with proper padding
            if hasattr(layer, "stride") and layer.stride != (1, 1):
                if hasattr(layer, "_left_padding"):
                    padding = (layer._left_padding, layer._right_padding)  # CausalConv2D
                else:
                    padding = layer.padding
                current_lengths = calculate_conv_output_size(current_lengths, layer.kernel_size[0], layer.stride[0], padding)
                mask = self._create_mask(x, current_lengths.long())

        # Final masking
        x = apply_channel_mask(x, mask)
        return x, current_lengths.long()

    def _create_mask(self, tensor, lengths):
        """Create mask matching tensor dimensions."""
        batch_size, channels, time, features = tensor.shape
        time_mask = torch.arange(time, device=tensor.device).expand(batch_size, time) < lengths.unsqueeze(1)
        return time_mask.unsqueeze(-1).expand(batch_size, time, features).to(tensor.dtype)


class ConvSubsampling(nn.Module):
    """Convolutional subsampling which supports VGGNet and striding approach introduced in:
    VGGNet Subsampling: Transformer-transducer: end-to-end speech recognition with self-attention (https://arxiv.org/pdf/1910.12977.pdf)
    Striding Subsampling: "Speech-Transformer: A No-Recurrence Sequence-to-Sequence Model for Speech Recognition" by Linhao Dong et al. (https://ieeexplore.ieee.org/document/8462506)
    Args:
        subsampling (str): The subsampling technique from {"vggnet", "striding", "dw-striding"}
        subsampling_factor (int): The subsampling factor which should be a power of 2
        subsampling_conv_chunking_factor (int): Input chunking factor which can be -1 (no chunking)
        1 (auto) or a power of 2. Default is 1
        feat_in (int): size of the input features
        feat_out (int): size of the output features
        conv_channels (int): Number of channels for the convolution layers.
        activation (Module): activation function, default is nn.ReLU()
    """

    def __init__(
        self,
        subsampling,
        subsampling_factor,
        feat_in,
        feat_out,
        conv_channels,
        subsampling_conv_chunking_factor=1,
        activation=nn.ReLU(),
        is_causal=False,
    ):
        super().__init__()
        self._subsampling = subsampling
        self._conv_channels = conv_channels
        self._feat_in = feat_in
        self._feat_out = feat_out

        if subsampling_factor % 2 != 0:
            raise ValueError("Sampling factor should be a multiply of 2!")
        self._sampling_num = int(math.log(subsampling_factor, 2))
        self.subsampling_factor = subsampling_factor
        self.is_causal = is_causal

        if subsampling_conv_chunking_factor != -1 and subsampling_conv_chunking_factor != 1 and subsampling_conv_chunking_factor % 2 != 0:
            raise ValueError("subsampling_conv_chunking_factor should be -1, 1, or a power of 2")
        self.subsampling_conv_chunking_factor = subsampling_conv_chunking_factor

        in_channels = 1
        layers = []

        self._stride = 2
        self._kernel_size = 3
        self._ceil_mode = False

        if self.is_causal:
            self._left_padding = self._kernel_size - 1
            self._right_padding = self._stride - 1
            self._max_cache_len = subsampling_factor + 1
        else:
            self._left_padding = (self._kernel_size - 1) // 2
            self._right_padding = (self._kernel_size - 1) // 2
            self._max_cache_len = 0

        for i in range(self._sampling_num):
            layers.append(
                torch.nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=conv_channels,
                    kernel_size=self._kernel_size,
                    stride=self._stride,
                    padding=self._left_padding,
                )
            )
            layers.append(activation)
            in_channels = conv_channels

        in_length = torch.tensor(feat_in, dtype=torch.float)
        out_length = calc_length(
            lengths=in_length,
            all_paddings=self._left_padding + self._right_padding,
            kernel_size=self._kernel_size,
            stride=self._stride,
            ceil_mode=self._ceil_mode,
            repeat_num=self._sampling_num,
        )
        self.out = torch.nn.Linear(conv_channels * int(out_length), feat_out)
        self.conv2d_subsampling = True

        self.conv = MaskedConvSequential(*layers)

    def get_sampling_frames(self):
        return [1, self.subsampling_factor]

    def get_streaming_cache_size(self):
        return [0, self.subsampling_factor + 1]

    def forward(self, x, lengths):
        out_lengths = calc_length(
            lengths,
            all_paddings=self._left_padding + self._right_padding,
            kernel_size=self._kernel_size,
            stride=self._stride,
            ceil_mode=self._ceil_mode,
            repeat_num=self._sampling_num,
        )

        # Transpose to Channel First mode
        if not self.conv2d_subsampling:
            x = x.transpose(1, 2)

        # split inputs if chunking_factor is set
        if self.subsampling_conv_chunking_factor != -1 and self.conv2d_subsampling:
            if self.subsampling_conv_chunking_factor == 1:
                # if subsampling_conv_chunking_factor is 1, we split only if needed
                # avoiding a bug / feature limiting indexing of tensors to 2**31
                # see https://github.com/pytorch/pytorch/issues/80020
                x_ceil = 2**31 / self._conv_channels * self._stride * self._stride
                if torch.numel(x) > x_ceil:
                    need_to_split = True
                else:
                    need_to_split = False
            else:
                # if subsampling_conv_chunking_factor > 1 we always split
                need_to_split = True

            if need_to_split:
                x, lengths, success = self.conv_split_by_batch(x, lengths)
                if not success:  # if unable to split by batch, try by channel
                    if self._subsampling == "dw_striding":
                        # TODO: implement lengths inside conv_split_by_channel
                        x = self.conv_split_by_channel(x)
                        lengths = out_lengths
                    else:
                        x, lengths = self.conv(x, lengths)  # try anyway
            else:
                x, lengths = self.conv(x, lengths)
        else:
            x, lengths = self.conv(x)

        # Flatten Channel and Frequency Axes
        if self.conv2d_subsampling:
            b, c, t, f = x.size()
            x = self.out(x.transpose(1, 2).reshape(b, t, -1))
        # Transpose to Channel Last mode
        else:
            x = x.transpose(1, 2)

        return x, lengths

    def reset_parameters(self):
        # initialize weights
        if self._subsampling == "dw_striding":
            with torch.no_grad():
                # init conv
                scale = 1.0 / self._kernel_size
                dw_max = (self._kernel_size**2) ** -0.5
                pw_max = self._conv_channels**-0.5

                torch.nn.init.uniform_(self.conv[0].weight, -scale, scale)
                torch.nn.init.uniform_(self.conv[0].bias, -scale, scale)

                for idx in range(2, len(self.conv), 3):
                    torch.nn.init.uniform_(self.conv[idx].weight, -dw_max, dw_max)
                    torch.nn.init.uniform_(self.conv[idx].bias, -dw_max, dw_max)
                    torch.nn.init.uniform_(self.conv[idx + 1].weight, -pw_max, pw_max)
                    torch.nn.init.uniform_(self.conv[idx + 1].bias, -pw_max, pw_max)

                # init fc (80 * 64 = 5120 from https://github.com/kssteven418/Squeezeformer/blob/13c97d6cf92f2844d2cb3142b4c5bfa9ad1a8951/src/models/conformer_encoder.py#L487
                fc_scale = (self._feat_out * self._feat_in / self._sampling_num) ** -0.5
                torch.nn.init.uniform_(self.out.weight, -fc_scale, fc_scale)
                torch.nn.init.uniform_(self.out.bias, -fc_scale, fc_scale)

    def conv_split_by_batch(self, x, lengths):
        """Tries to split input by batch, run conv and concat results"""
        b, *_ = x.size()
        if b == 1:  # can't split if batch size is 1
            return x, lengths, False

        if self.subsampling_conv_chunking_factor > 1:
            cf = self.subsampling_conv_chunking_factor
        else:
            # avoiding a bug / feature limiting indexing of tensors to 2**31
            # see https://github.com/pytorch/pytorch/issues/80020
            x_ceil = 2**31 / self._conv_channels * self._stride * self._stride
            p = math.ceil(math.log(torch.numel(x) / x_ceil, 2))
            cf = 2**p

        new_batch_size = b // cf
        if new_batch_size == 0:  # input is too big
            return x, lengths, False

        ans = [
            self.conv(chunk, ln)
            for chunk, ln in zip(
                torch.split(x, new_batch_size, 0),
                torch.split(lengths, new_batch_size, 0),
            )
        ]
        return torch.cat([a[0] for a in ans]), torch.cat([a[1] for a in ans]), True

    def conv_split_by_channel(self, x):
        """For dw convs, tries to split input by time, run conv and concat results"""

        # Note: this method doesn't use the convolution masking implemented in MaskedConvolutionSequential
        x = x.unsqueeze(0)
        x = self.conv[0](x)  # full conv2D
        x = self.conv[1](x)  # activation

        for i in range(self._sampling_num - 1):
            _, c, t, _ = x.size()

            if self.subsampling_conv_chunking_factor > 1:
                cf = self.subsampling_conv_chunking_factor
            else:
                # avoiding a bug / feature limiting indexing of tensors to 2**31
                # see https://github.com/pytorch/pytorch/issues/80020
                p = math.ceil(math.log(torch.numel(x) / 2**31, 2))
                cf = 2**p

            new_c = int(c // cf)
            if new_c == 0:
                new_c = 1

            new_t = int(t // cf)
            if new_t == 0:
                new_t = 1

            x = self.channel_chunked_conv(self.conv[i * 3 + 2], new_c, x)  # conv2D, depthwise

            # splitting pointwise convs by time
            x = torch.cat([self.conv[i * 3 + 3](chunk) for chunk in torch.split(x, new_t, 2)], 2)  # conv2D, pointwise
            x = self.conv[i * 3 + 4](x)  # activation
        return x

    def channel_chunked_conv(self, conv, chunk_size, x):
        """Performs channel chunked convolution"""

        ind = 0
        out_chunks = []
        for chunk in torch.split(x, chunk_size, 1):
            step = chunk.size()[1]

            if self.is_causal:
                chunk = nn.functional.pad(chunk, pad=(self._kernel_size - 1, self._stride - 1, self._kernel_size - 1, self._stride - 1))
                ch_out = nn.functional.conv2d(
                    chunk,
                    conv.weight[ind : ind + step, :, :, :],
                    bias=conv.bias[ind : ind + step],
                    stride=self._stride,
                    padding=0,
                    groups=step,
                )
            else:
                ch_out = nn.functional.conv2d(
                    chunk,
                    conv.weight[ind : ind + step, :, :, :],
                    bias=conv.bias[ind : ind + step],
                    stride=self._stride,
                    padding=self._left_padding,
                    groups=step,
                )
            out_chunks.append(ch_out)
            ind += step

        return torch.cat(out_chunks, 1)

    def change_subsampling_conv_chunking_factor(self, subsampling_conv_chunking_factor: int):
        if subsampling_conv_chunking_factor != -1 and subsampling_conv_chunking_factor != 1 and subsampling_conv_chunking_factor % 2 != 0:
            raise ValueError("subsampling_conv_chunking_factor should be -1, 1, or a power of 2")
        self.subsampling_conv_chunking_factor = subsampling_conv_chunking_factor


def calc_length(lengths, all_paddings, kernel_size, stride, ceil_mode, repeat_num=1):
    """Calculates the output length of a Tensor passed through a convolution or max pooling layer"""
    add_pad: float = all_paddings - kernel_size
    one: float = 1.0
    for i in range(repeat_num):
        lengths = torch.div(lengths.to(dtype=torch.float) + add_pad, stride) + one
        if ceil_mode:
            lengths = torch.ceil(lengths)
        else:
            lengths = torch.floor(lengths)
    return lengths.to(dtype=torch.int)


class SubsamplingReductionModule(nn.Module):
    """Downsamples the audio signal in time dimension."""

    def __init__(self, reduction: str, d_model: int, reduction_factor: int = 2):
        super().__init__()

        assert reduction in ["pooling", "striding"]


class ConformerEncoder(nn.Module):
    """
    The encoder for ASR model of Conformer.
    Based on this paper:
    'Conformer: Convolution-augmented Transformer for Speech Recognition' by Anmol Gulati et al.
    https://arxiv.org/abs/2005.08100

    Args:
        feat_in (int): the size of feature channels
        n_layers (int): number of layers of ConformerBlock
        d_model (int): the hidden size of the model
        feat_out (int): the size of the output features
            Defaults to -1 (means feat_out is d_model)
        subsampling (str): the method of subsampling:
            choices = ['vggnet', 'striding', 'dw-striding', 'stacking', 'stacking_norm']
            Defaults to striding.
        subsampling_factor (int): the subsampling factor which should be power of 2
            Defaults to 4.
        subsampling_conv_chunking_factor(int): optionally, force chunk inputs (helpful for large inputs)
            Should be power of 2, 1 (auto-chunking, default), or -1 (no chunking)
        subsampling_conv_channels (int): the size of the convolutions in the subsampling module
            Defaults to -1 which would set it to d_model.
        reduction (str, Optional): the method of reduction, choices=['pooling', 'striding']. If no value
            is passed, then no reduction is performed and the models runs with the original 4x subsampling.
        reduction_position (int, Optional): the index of the layer to apply reduction. If -1, apply reduction
            at the end.
        reduction_factor (int): the reduction factor which should be either 1 or a power of 2
            Defaults to 1.
        ff_expansion_factor (int): the expansion factor in feed forward layers
            Defaults to 4.
        self_attention_model (str): the type of the attention layer and positional encoding.

            'rel_pos':
                relative positional embedding and Transformer-XL
            'rel_pos_local_attn':
                relative positional embedding and Transformer-XL with local attention using
                overlapping chunks. Attention context is determined by att_context_size parameter.
            'abs_pos':
                absolute positional embedding and Transformer

            Default is rel_pos.
        pos_emb_max_len (int): the maximum length of positional embeddings
            Defaults to 5000
        n_heads (int): number of heads in multi-headed attention layers
            Defaults to 4.
        att_context_size (List[Union[List[int],int]]): specifies the context sizes on each side.
            Each context size should be a list of two integers like `[100, 100]`.
            A list of context sizes like `[[100,100]`, `[100,50]]` can also be passed. -1 means unlimited context.
            Defaults to `[-1, -1]`
        att_context_probs (List[float]): a list of probabilities of each one of the att_context_size
            when a list of them is passed. If not specified, uniform distribution is being used.
            Defaults to None
        att_context_style (str): 'regular' or 'chunked_limited'.
            Defaults to 'regular'
        xscaling (bool): enables scaling the inputs to the multi-headed attention layers by `sqrt(d_model)`.
            Defaults to True.
        untie_biases (bool): whether to not share (untie) the bias weights between layers of Transformer-XL
            Defaults to True.
        conv_kernel_size (int): the size of the convolutions in the convolutional modules
            Defaults to 31.
        conv_norm_type (str): the type of the normalization in the convolutional modules
            Defaults to 'batch_norm'.
        conv_context_size (list): it can be"causal" or a list of two integers
            while `conv_context_size[0]+conv_context_size[1]+1==conv_kernel_size`.
            `None` means `[(conv_kernel_size-1)//2`, `(conv_kernel_size-1)//2]`, and 'causal' means
            `[(conv_kernel_size-1), 0]`.
            Defaults to None.
        conv_dual_mode (bool): specifies if convolution should be dual mode when dual_offline mode is being used.
            When enables, the left half of the convolution kernel would get masked in streaming cases.
            Defaults to False.
        use_bias (bool): Use bias in all Linear and Conv1d layers from each ConformerLayer to improve
            activation flow and stabilize training of huge models.
            Defaults to True.
        dropout (float): the dropout rate used in all layers except the attention layers
            Defaults to 0.1.
        dropout_pre_encoder (float): the dropout rate used before the encoder
            Defaults to 0.1.
        dropout_emb (float): the dropout rate used for the positional embeddings
            Defaults to 0.1.
        dropout_att (float): the dropout rate used for the attention layer
            Defaults to 0.0.
        stochastic_depth_drop_prob (float): if non-zero, will randomly drop
            layers during training. The higher this value, the more often layers
            are dropped. Defaults to 0.0.
        stochastic_depth_mode (str): can be either "linear" or "uniform". If
            set to "uniform", all layers have the same probability of drop. If
            set to "linear", the drop probability grows linearly from 0 for the
            first layer to the desired value for the final layer. Defaults to
            "linear".
        stochastic_depth_start_layer (int): starting layer for stochastic depth.
            All layers before this will never be dropped. Note that drop
            probability will be adjusted accordingly if mode is "linear" when
            start layer is > 1. Defaults to 1.
        global_tokens (int): number of tokens to be used for global attention.
            Only relevant if self_attention_model is 'rel_pos_local_attn'.
            Defaults to 0.
        global_tokens_spacing (int): how far apart the global tokens are
            Defaults to 1.
        global_attn_separate (bool): whether the q, k, v layers used for global tokens should be separate.
            Defaults to False.
        use_pytorch_sdpa (bool): use torch sdpa instead of manual attention.
            Defaults to False.
        use_pytorch_sdpa_backends (list[str]): list of backend names to use in sdpa.
            None or empty list means all backends. e.g. ["MATH"]
            Defaults to None.
        bypass_pre_encode: if True, skip the pre-encoder module and the `audio_signal` should be pre-encoded
            embeddings. The `audio_signal` input supports two formats depending on the `bypass_pre_encode`
            boolean flag. This determines the required format of the input variable `audio_signal`.
            Defaults to `bypass_pre_encode=False`. `bypass_pre_encode=True` is used for the cases
            where frame-level, context-independent embeddings are needed to be saved or reused.
            (e.g., speaker cache in streaming speaker diarization)
        sync_max_audio_length (bool): when true, performs NCCL all_reduce to allocate the same amount of memory for
            positional encoding buffers on all GPUs. Disabling this setting may help with deadlocks in certain
            scenarios such as model parallelism, or generally when this module is not being ran on some GPUs
            as a part of the training step.
    """

    def __init__(
        self,
        feat_in=80,
        n_layers=16,
        d_model=176,
        feat_out=-1,
        causal_downsampling=False,
        subsampling="striding",
        subsampling_factor=4,
        subsampling_conv_chunking_factor=1,
        subsampling_conv_channels=-1,
        reduction=None,
        reduction_position=None,
        reduction_factor=1,
        ff_expansion_factor=4,
        self_attention_model="rel_pos",
        n_heads=4,
        att_context_size=None,
        att_context_probs=None,
        att_context_style="regular",
        xscaling=True,
        untie_biases=True,
        pos_emb_max_len=5000,
        conv_kernel_size=31,
        conv_norm_type="batch_norm",
        conv_context_size=None,
        use_bias=True,
        dropout=0.1,
        dropout_pre_encoder=0.1,
        dropout_emb=0.1,
        dropout_att=0.0,
        stochastic_depth_drop_prob: float = 0.0,
        stochastic_depth_mode: str = "linear",
        stochastic_depth_start_layer: int = 1,
        global_tokens: int = 0,
        global_tokens_spacing: int = 1,
        global_attn_separate: bool = False,
        use_pytorch_sdpa: bool = False,
        use_pytorch_sdpa_backends=None,
        sync_max_audio_length: bool = True,
    ):
        super().__init__()
        d_ff = d_model * ff_expansion_factor
        self.d_model = d_model
        self.n_layers = n_layers
        self._feat_in = feat_in
        self.att_context_style = att_context_style
        self.subsampling_factor = subsampling_factor
        self.subsampling_conv_chunking_factor = subsampling_conv_chunking_factor

        self.self_attention_model = self_attention_model
        self.global_tokens = global_tokens
        self.global_attn_separate = global_attn_separate
        self.global_tokens_spacing = global_tokens_spacing
        self.use_pytorch_sdpa = use_pytorch_sdpa
        if use_pytorch_sdpa_backends is None:
            use_pytorch_sdpa_backends = []
        self.use_pytorch_sdpa_backends = use_pytorch_sdpa_backends
        self.sync_max_audio_length = sync_max_audio_length

        # Setting up the att_context_size
        (
            self.att_context_size_all,
            self.att_context_size,
            self.att_context_probs,
            self.conv_context_size,
        ) = self._calc_context_sizes(
            att_context_style=att_context_style,
            att_context_size=att_context_size,
            att_context_probs=att_context_probs,
            conv_context_size=conv_context_size,
            conv_kernel_size=conv_kernel_size,
        )

        if xscaling:
            self.xscale = math.sqrt(d_model)
        else:
            self.xscale = None

        # Subsampling
        if subsampling_conv_channels == -1:
            subsampling_conv_channels = d_model
        self.pre_encode = ConvSubsampling(
            subsampling=subsampling,
            subsampling_factor=subsampling_factor,
            feat_in=feat_in,
            feat_out=d_model,
            conv_channels=subsampling_conv_channels,
            subsampling_conv_chunking_factor=subsampling_conv_chunking_factor,
            activation=nn.ReLU(True),
            is_causal=causal_downsampling,
        )
        self.reduction_subsampling = None
        self.reduction_position = None

        self._feat_out = d_model

        # Biases for relative positional encoding
        if not untie_biases and self_attention_model == "rel_pos":
            d_head = d_model // n_heads
            pos_bias_u = nn.Parameter(torch.Tensor(n_heads, d_head))
            pos_bias_v = nn.Parameter(torch.Tensor(n_heads, d_head))
            nn.init.zeros_(pos_bias_u)
            nn.init.zeros_(pos_bias_v)
        else:
            pos_bias_u = None
            pos_bias_v = None

        # Positional encodings
        self.pos_emb_max_len = pos_emb_max_len
        self.pos_enc = RelPositionalEncoding(
            d_model=d_model,
            dropout_rate=dropout_pre_encoder,
            max_len=pos_emb_max_len,
            xscale=self.xscale,
            dropout_rate_emb=dropout_emb,
        )
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            layer = ConformerLayer(
                d_model=d_model,
                d_ff=d_ff,
                self_attention_model=self_attention_model,
                global_tokens=global_tokens,
                global_tokens_spacing=global_tokens_spacing,
                global_attn_separate=global_attn_separate,
                n_heads=n_heads,
                conv_kernel_size=conv_kernel_size,
                conv_norm_type=conv_norm_type,
                conv_context_size=self.conv_context_size,
                dropout=dropout,
                dropout_att=dropout_att,
                pos_bias_u=pos_bias_u,
                pos_bias_v=pos_bias_v,
                att_context_size=self.att_context_size,
                use_bias=use_bias,
                use_pytorch_sdpa=self.use_pytorch_sdpa,
                use_pytorch_sdpa_backends=self.use_pytorch_sdpa_backends,
            )
            self.layers.append(layer)

        if feat_out > 0 and feat_out != self._feat_out:
            self.out_proj = nn.Linear(self._feat_out, feat_out)
            self._feat_out = feat_out
        else:
            self.out_proj = None
            self._feat_out = d_model
        self.set_max_audio_length(self.pos_emb_max_len)
        self.use_pad_mask = True

        # self.setup_streaming_params()
        self.export_cache_support = False

        self.layer_drop_probs = compute_stochastic_depth_drop_probs(len(self.layers), stochastic_depth_drop_prob, stochastic_depth_mode, stochastic_depth_start_layer)
        # will be set in self.forward() if defined in AccessMixin config
        self.interctc_capture_at_layers = None

    def forward_for_export(self, audio_signal, length, cache_last_channel=None, cache_last_time=None, cache_last_channel_len=None):
        """
        Forward function for model export. Please see `forward()` for more details.
        """
        if cache_last_channel is not None:
            cache_last_channel = cache_last_channel.transpose(0, 1)
            cache_last_time = cache_last_time.transpose(0, 1)

        rets = self.forward_internal(
            audio_signal,
            length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
        )
        rets = self.streaming_post_process(rets, keep_all_outputs=False)
        if len(rets) == 2:
            return rets
        if rets[2] is None and rets[3] is None and rets[4] is None:
            return (rets[0], rets[1])
        return (
            rets[0],
            rets[1],
            rets[2].transpose(0, 1),
            rets[3].transpose(0, 1),
            rets[4],
        )

    # def streaming_post_process(self, rets, keep_all_outputs=True):
    #     """
    #     Post-process the output of the forward function for streaming.

    #     Args:
    #         rets: The output of the forward function.
    #         keep_all_outputs: Whether to keep all outputs.
    #     """
    #     if len(rets) == 2:
    #         return rets[0], rets[1], None, None, None

    #     (encoded, encoded_len, cache_last_channel_next, cache_last_time_next, cache_last_channel_next_len) = rets

    #     if cache_last_channel_next is not None and self.streaming_cfg.last_channel_cache_size >= 0:
    #         if self.streaming_cfg.last_channel_cache_size > 0:
    #             cache_last_channel_next = cache_last_channel_next[
    #                 :, :, -self.streaming_cfg.last_channel_cache_size :, :
    #             ]

    #     if self.streaming_cfg.valid_out_len > 0 and (not keep_all_outputs or self.att_context_style == "regular"):
    #         encoded = encoded[:, :, : self.streaming_cfg.valid_out_len]
    #         encoded_len = torch.clamp(encoded_len, max=self.streaming_cfg.valid_out_len)

    #     return (encoded, encoded_len, cache_last_channel_next, cache_last_time_next, cache_last_channel_next_len)

    def forward(
        self,
        audio_signal,
        length,
        cache_last_channel=None,
        cache_last_time=None,
        cache_last_channel_len=None,
        bypass_pre_encode=False,
    ):
        """
        Forward function for the ConformerEncoder accepting an audio signal and its corresponding length.
        The ``audio_signal`` input supports two formats depending on ``bypass_pre_encode``:

        - ``bypass_pre_encode=False`` (default): ``audio_signal`` must be a tensor
          containing audio features. Shape: ``(batch, feat_in, n_frames)``.
        - ``bypass_pre_encode=True``: ``audio_signal`` must be a tensor containing
          pre-encoded embeddings. Shape: ``(batch, n_frame, d_model)``.
        """
        if not bypass_pre_encode and audio_signal.shape[-2] != self._feat_in:
            raise ValueError(f"If bypass_pre_encode is False, audio_signal should have shape (batch, {self._feat_in}, n_frame) but got last dimension {audio_signal.shape[-2]}.")

        self.update_max_seq_length(seq_length=audio_signal.size(2), device=audio_signal.device)
        return self.forward_internal(
            audio_signal,
            length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
            bypass_pre_encode=bypass_pre_encode,
        )

    def forward_internal(
        self,
        audio_signal,
        length,
        cache_last_channel=None,
        cache_last_time=None,
        cache_last_channel_len=None,
        bypass_pre_encode=False,
    ):
        """
        The ``audio_signal`` input supports two formats depending on ``bypass_pre_encode``:

        - ``bypass_pre_encode=False`` (default): ``audio_signal`` must be a tensor
          containing audio features. Shape: ``(batch, feat_in, n_frames)``.
        - ``bypass_pre_encode=True``: ``audio_signal`` must be a tensor containing
          pre-encoded embeddings. Shape: ``(batch, n_frame, d_model)``.

        ``bypass_pre_encode=True`` is used in cases where frame-level, context-independent embeddings are
        needed to be saved or reused (e.g., speaker cache in streaming speaker diarization).
        """
        if length is None:
            length = audio_signal.new_full((audio_signal.size(0),), audio_signal.size(-1), dtype=torch.int64, device=audio_signal.device)

        # select a random att_context_size with the distribution specified by att_context_probs during training
        # for non-validation cases like test, validation or inference, it uses the first mode in self.att_context_size
        if self.training and len(self.att_context_size_all) > 1:
            cur_att_context_size = random.choices(self.att_context_size_all, weights=self.att_context_probs)[0]
        else:
            cur_att_context_size = self.att_context_size

        if not bypass_pre_encode:
            audio_signal = torch.transpose(audio_signal, 1, 2)

            if isinstance(self.pre_encode, nn.Linear):
                audio_signal = self.pre_encode(audio_signal)
            else:
                audio_signal, length = self.pre_encode(x=audio_signal, lengths=length)
                length = length.to(torch.int64)
                # `self.streaming_cfg` is set by setup_streaming_cfg(), called in the init
                # if self.streaming_cfg.drop_extra_pre_encoded > 0 and cache_last_channel is not None:
                #     audio_signal = audio_signal[:, self.streaming_cfg.drop_extra_pre_encoded :, :]
                #     length = (length - self.streaming_cfg.drop_extra_pre_encoded).clamp(min=0)

            if self.reduction_position is not None and cache_last_channel is not None:
                raise ValueError("Caching with reduction feature is not supported yet!")

        max_audio_length = audio_signal.size(1)
        if cache_last_channel is not None:
            print("pososal")
            # cache_len = self.streaming_cfg.last_channel_cache_size
            # cache_keep_size = max_audio_length - self.streaming_cfg.cache_drop_size
            # max_audio_length = max_audio_length + cache_len
            # padding_length = length + cache_len
            # offset = torch.neg(cache_last_channel_len) + cache_len
        else:
            padding_length = length
            cache_last_channel_next = None
            cache_len = 0
            offset = None

        audio_signal, pos_emb = self.pos_enc(x=audio_signal, cache_len=cache_len)

        # Create the self-attention and padding masks
        pad_mask, att_mask = self._create_masks(
            att_context_size=cur_att_context_size,
            padding_length=padding_length,
            max_audio_length=max_audio_length,
            offset=offset,
            device=audio_signal.device,
        )

        if cache_last_channel is not None:
            pad_mask = pad_mask[:, cache_len:]
            if att_mask is not None:
                att_mask = att_mask[:, cache_len:]
            # Convert caches from the tensor to list
            cache_last_time_next = []
            cache_last_channel_next = []

        for lth, (drop_prob, layer) in enumerate(zip(self.layer_drop_probs, self.layers)):
            original_signal = audio_signal
            if cache_last_channel is not None:
                cache_last_channel_cur = cache_last_channel[lth]
                cache_last_time_cur = cache_last_time[lth]
            else:
                cache_last_channel_cur = None
                cache_last_time_cur = None
            audio_signal = layer(
                x=audio_signal,
                att_mask=att_mask,
                pos_emb=pos_emb,
                pad_mask=pad_mask,
                cache_last_channel=cache_last_channel_cur,
                cache_last_time=cache_last_time_cur,
            )

            if cache_last_channel_cur is not None:
                (audio_signal, cache_last_channel_cur, cache_last_time_cur) = audio_signal
                cache_last_channel_next.append(cache_last_channel_cur)
                cache_last_time_next.append(cache_last_time_cur)

            # applying stochastic depth logic from https://arxiv.org/abs/2102.03216
            if self.training and drop_prob > 0.0:
                should_drop = torch.rand(1) < drop_prob
                # adjusting to match expectation
                if should_drop:
                    # that's not efficient, but it's hard to implement distributed
                    # version of dropping layers without deadlock or random seed meddling
                    # so multiplying the signal by 0 to ensure all weights get gradients
                    audio_signal = audio_signal * 0.0 + original_signal
                else:
                    # not doing this operation if drop prob is 0 as it's identity in that case
                    audio_signal = (audio_signal - original_signal) / (1.0 - drop_prob) + original_signal

            if self.reduction_position == lth:
                audio_signal, length = self.reduction_subsampling(x=audio_signal, lengths=length)
                max_audio_length = audio_signal.size(1)
                # Don't update the audio_signal here because then it will again scale the audio_signal
                # and cause an increase in the WER
                _, pos_emb = self.pos_enc(x=audio_signal, cache_len=cache_len)
                pad_mask, att_mask = self._create_masks(
                    att_context_size=cur_att_context_size,
                    padding_length=length,
                    max_audio_length=max_audio_length,
                    offset=offset,
                    device=audio_signal.device,
                )

        if self.out_proj is not None:
            audio_signal = self.out_proj(audio_signal)

        # Reduction
        if self.reduction_position == -1:
            audio_signal, length = self.reduction_subsampling(x=audio_signal, lengths=length)

        audio_signal = torch.transpose(audio_signal, 1, 2)
        length = length.to(dtype=torch.int64)

        if cache_last_channel is not None:
            print("dvazhdy")
            # cache_last_channel_next = torch.stack(cache_last_channel_next, dim=0)
            # cache_last_time_next = torch.stack(cache_last_time_next, dim=0)
            # return (
            #     audio_signal,
            #     length,
            #     cache_last_channel_next,
            #     cache_last_time_next,
            #     torch.clamp(cache_last_channel_len + cache_keep_size, max=cache_len),
            # )
        return audio_signal, length

    def update_max_seq_length(self, seq_length: int, device):
        """
        Updates the maximum sequence length for the model.

        Args:
            seq_length (int): New maximum sequence length.
            device (torch.device): Device to use for computations.
        """
        # Find global max audio length across all nodes
        if self.sync_max_audio_length and torch.distributed.is_initialized():
            global_max_len = torch.tensor([seq_length], dtype=torch.float32, device=device)

            # Update across all ranks in the distributed system
            torch.distributed.all_reduce(global_max_len, op=torch.distributed.ReduceOp.MAX)

            seq_length = global_max_len.int().item()

        if seq_length > self.max_audio_length:
            self.set_max_audio_length(seq_length)

    def set_max_audio_length(self, max_audio_length):
        """
        Sets maximum input length.
        Pre-calculates internal seq_range mask.

        Args:
            max_audio_length (int): New maximum sequence length.
        """
        self.max_audio_length = max_audio_length
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        self.pos_enc.extend_pe(max_audio_length, device, dtype)

    def _create_masks(self, att_context_size, padding_length, max_audio_length, offset, device):
        if self.self_attention_model != "rel_pos_local_attn":
            att_mask = torch.ones(1, max_audio_length, max_audio_length, dtype=torch.bool, device=device)

            if att_context_size[0] >= 0:
                att_mask = att_mask.triu(diagonal=-att_context_size[0])
            if att_context_size[1] >= 0:
                att_mask = att_mask.tril(diagonal=att_context_size[1])
        else:
            att_mask = None

        # pad_mask is the masking to be used to ignore paddings
        pad_mask = torch.arange(0, max_audio_length, device=device).expand(padding_length.size(0), -1) < padding_length.unsqueeze(-1)

        if offset is not None:
            pad_mask_off = torch.arange(0, max_audio_length, device=device).expand(padding_length.size(0), -1) >= offset.unsqueeze(-1)
            pad_mask = pad_mask_off.logical_and(pad_mask)

        if att_mask is not None:
            # pad_mask_for_att_mask is the mask which helps to ignore paddings
            pad_mask_for_att_mask = pad_mask.unsqueeze(1).repeat([1, max_audio_length, 1])
            pad_mask_for_att_mask = torch.logical_and(pad_mask_for_att_mask, pad_mask_for_att_mask.transpose(1, 2))
            # att_mask is the masking to be used by the MHA layers to ignore the tokens not supposed to be visible
            att_mask = att_mask[:, :max_audio_length, :max_audio_length]
            # paddings should also get ignored, so pad_mask_for_att_mask is used to ignore their corresponding scores
            att_mask = torch.logical_and(pad_mask_for_att_mask, att_mask.to(pad_mask_for_att_mask.device))
            att_mask = ~att_mask

        pad_mask = ~pad_mask
        return pad_mask, att_mask

    def enable_pad_mask(self, on=True):
        """
        Enables or disables the pad mask and assign the boolean state `on`.

        Returns:
            mask (bool): The current state of the pad mask.
        """
        # On inference, user may choose to disable pad mask
        mask = self.use_pad_mask
        self.use_pad_mask = on
        return mask

    def _calc_context_sizes(self, att_context_size, att_context_probs, att_context_style, conv_context_size, conv_kernel_size):
        # convert att_context_size to a standard list of lists
        att_context_size_all = [[-1, -1]]

        if att_context_probs:
            if len(att_context_probs) != len(att_context_size_all):
                raise ValueError("The size of the att_context_probs should be the same as att_context_size.")
            att_context_probs = list(att_context_probs)
            if sum(att_context_probs) != 1:
                raise ValueError("The sum of numbers in att_context_probs should be equal to one to be a distribution.")
        else:
            att_context_probs = [1.0 / len(att_context_size_all)] * len(att_context_size_all)

        conv_context_size = [(conv_kernel_size - 1) // 2, (conv_kernel_size - 1) // 2]
        return att_context_size_all, att_context_size_all[0], att_context_probs, conv_context_size

    def set_default_att_context_size(self, att_context_size):
        """
        Sets the default attention context size from `att_context_size` argument.

        Args:
            att_context_size (list): The attention context size to be set.
        """
        if att_context_size not in self.att_context_size_all:
            print("Logging")
        if att_context_size is not None:
            self.att_context_size = att_context_size

        # self.setup_streaming_params()

    # def setup_streaming_params(
    #     self,
    #     chunk_size: int = None,
    #     shift_size: int = None,
    #     left_chunks: int = None,
    #     att_context_size: list = None,
    #     max_context: int = 10000,
    # ):
    #     """
    #     This function sets the needed values and parameters to perform streaming.
    #     The configuration would be stored in self.streaming_cfg.
    #     The streaming configuration is needed to simulate streaming inference.

    #     Args:
    #         chunk_size (int): overrides the chunk size
    #         shift_size (int): overrides the shift size for chunks
    #         left_chunks (int): overrides the number of left chunks visible to each chunk
    #         max_context (int): the value used for the cache size of last_channel layers
    #                            if left context is set to infinity (-1)
    #                            Defaults to -1 (means feat_out is d_model)
    #     """
    #     streaming_cfg = CacheAwareStreamingConfig()

    #     # When att_context_size is not specified, it uses the default_att_context_size
    #     if att_context_size is None:
    #         att_context_size = self.att_context_size

    #     if chunk_size is not None:
    #         if chunk_size < 1:
    #             raise ValueError("chunk_size needs to be a number larger or equal to one.")
    #         lookahead_steps = chunk_size - 1
    #         streaming_cfg.cache_drop_size = chunk_size - shift_size
    #     elif self.att_context_style == "chunked_limited":
    #         lookahead_steps = att_context_size[1]
    #         streaming_cfg.cache_drop_size = 0
    #     elif self.att_context_style == "regular":
    #         lookahead_steps = att_context_size[1] * self.n_layers + self.conv_context_size[1] * self.n_layers
    #         streaming_cfg.cache_drop_size = lookahead_steps
    #     else:
    #         streaming_cfg.cache_drop_size = 0
    #         lookahead_steps = None

    #     if chunk_size is None:
    #         streaming_cfg.last_channel_cache_size = att_context_size[0] if att_context_size[0] >= 0 else max_context
    #     else:
    #         if left_chunks is None:
    #             streaming_cfg.last_channel_cache_size = att_context_size[0] if att_context_size[0] >= 0 else max_context
    #             logging.warning(
    #                 f"left_chunks is not set. Setting it to default: {streaming_cfg.last_channel_cache_size}."
    #             )
    #         else:
    #             streaming_cfg.last_channel_cache_size = left_chunks * chunk_size

    #     if hasattr(self.pre_encode, "get_sampling_frames"):
    #         sampling_frames = self.pre_encode.get_sampling_frames()
    #     else:
    #         sampling_frames = 0

    #     if isinstance(sampling_frames, list):
    #         streaming_cfg.chunk_size = [
    #             sampling_frames[0] + self.subsampling_factor * lookahead_steps,
    #             sampling_frames[1] + self.subsampling_factor * lookahead_steps,
    #         ]
    #     else:
    #         streaming_cfg.chunk_size = sampling_frames * (1 + lookahead_steps)

    #     if isinstance(sampling_frames, list):
    #         streaming_cfg.shift_size = [
    #             sampling_frames[0] + sampling_frames[1] * (lookahead_steps - streaming_cfg.cache_drop_size),
    #             sampling_frames[1] + sampling_frames[1] * (lookahead_steps - streaming_cfg.cache_drop_size),
    #         ]
    #     else:
    #         streaming_cfg.shift_size = sampling_frames * (1 + lookahead_steps - streaming_cfg.cache_drop_size)

    #     if isinstance(streaming_cfg.shift_size, list):
    #         streaming_cfg.valid_out_len = (
    #             streaming_cfg.shift_size[1] - sampling_frames[1]
    #         ) // self.subsampling_factor + 1
    #     else:
    #         streaming_cfg.valid_out_len = streaming_cfg.shift_size // self.subsampling_factor

    #     if hasattr(self.pre_encode, "get_streaming_cache_size"):
    #         streaming_cfg.pre_encode_cache_size = self.pre_encode.get_streaming_cache_size()
    #     else:
    #         streaming_cfg.pre_encode_cache_size = 0

    #     if isinstance(streaming_cfg.pre_encode_cache_size, list):
    #         if streaming_cfg.pre_encode_cache_size[1] >= 1:
    #             streaming_cfg.drop_extra_pre_encoded = (
    #                 1 + (streaming_cfg.pre_encode_cache_size[1] - 1) // self.subsampling_factor
    #             )
    #         else:
    #             streaming_cfg.drop_extra_pre_encoded = 0
    #     else:
    #         streaming_cfg.drop_extra_pre_encoded = streaming_cfg.pre_encode_cache_size // self.subsampling_factor

    #     for m in self.layers.modules():
    #         if hasattr(m, "_max_cache_len"):
    #             if isinstance(m, MultiHeadAttention):
    #                 m.cache_drop_size = streaming_cfg.cache_drop_size
    #             if isinstance(m, CausalConv1D):
    #                 m.cache_drop_size = streaming_cfg.cache_drop_size

    #     self.streaming_cfg = streaming_cfg

    # def get_initial_cache_state(self, batch_size=1, dtype=torch.float32, device=None, max_dim=0):
    #     if device is None:
    #         device = next(self.parameters()).device
    #     if max_dim > 0:
    #         create_tensor = torch.randn
    #     else:
    #         create_tensor = torch.zeros
    #     last_time_cache_size = self.conv_context_size[0]
    #     cache_last_channel = create_tensor(
    #         (
    #             len(self.layers),
    #             batch_size,
    #             self.streaming_cfg.last_channel_cache_size,
    #             self.d_model,
    #         ),
    #         device=device,
    #         dtype=dtype,
    #     )
    #     cache_last_time = create_tensor(
    #         (len(self.layers), batch_size, self.d_model, last_time_cache_size),
    #         device=device,
    #         dtype=dtype,
    #     )
    #     if max_dim > 0:
    #         cache_last_channel_len = torch.randint(
    #             0,
    #             min(max_dim, self.streaming_cfg.last_channel_cache_size),
    #             (batch_size,),
    #             device=device,
    #             dtype=torch.int64,
    #         )
    #         for i in range(batch_size):
    #             cache_last_channel[:, i, cache_last_channel_len[i] :, :] = 0
    #             # what is the right rule to zero out cache_last_time?
    #             if cache_last_channel_len[i] == 0:
    #                 cache_last_time[:, i, :, :] = 0
    #     else:
    #         cache_last_channel_len = torch.zeros(batch_size, device=device, dtype=torch.int64)
    #     return cache_last_channel, cache_last_time, cache_last_channel_len

    def change_attention_model(
        self,
        self_attention_model: str = None,
        att_context_size: list[int] = None,
        update_config: bool = True,
        device: torch.device = None,
    ):
        """
        Update the self_attention_model which changes the positional encoding and attention layers.

        Args:
            self_attention_model (str): type of the attention layer and positional encoding

                'rel_pos':
                    relative positional embedding and Transformer-XL

                'rel_pos_local_attn':
                    relative positional embedding and Transformer-XL with local attention using
                    overlapping windows. Attention context is determined by att_context_size parameter.

                'abs_pos':
                    absolute positional embedding and Transformer

                If None is provided, the self_attention_model isn't changed. Defaults to None.
            att_context_size (List[int]): List of 2 ints corresponding to left and right attention context sizes,
                or None to keep as it is. Defaults to None.
            update_config (bool): Whether to update the config or not with the new attention model.
                Defaults to True.
            device (torch.device): If provided, new layers will be moved to the device.
                Defaults to None.
        """

        if att_context_size:
            att_context_size = list(att_context_size)
        else:
            att_context_size = self.att_context_size

        if self_attention_model is None:
            self_attention_model = self.self_attention_model

        if self_attention_model == "rel_pos_local_attn" and max(att_context_size) <= 0:
            raise ValueError("When using local attention, context size must be set > 0")

        # new_pos_enc = RelPositionalEncoding(
        #     d_model=self._cfg.d_model,
        #     dropout_rate=self._cfg.dropout,
        #     max_len=self._cfg.pos_emb_max_len,
        #     xscale=self.xscale,
        #     dropout_rate_emb=self._cfg.dropout_emb,
        # )
        print("Trizhdy")
        # if device is not None:
        #     new_pos_enc = new_pos_enc.to(device=device)
        # del self.pos_enc
        # self.pos_enc = new_pos_enc
        # self.self_attention_model = self_attention_model
        # self.att_context_size = att_context_size
        # self.set_max_audio_length(self.pos_emb_max_len)

        # for _, m in self.named_modules():
        #     if isinstance(m, ConformerLayer):
        #         new_attn = RelPositionMultiHeadAttention(
        #             n_head=self._cfg.n_heads,
        #             n_feat=self._cfg.d_model,
        #             dropout_rate=self._cfg.dropout_att,
        #             max_cache_len=att_context_size[0],
        #             pos_bias_u=None,
        #             pos_bias_v=None,
        #             use_pytorch_sdpa=self.use_pytorch_sdpa,
        #             use_pytorch_sdpa_backends=self.use_pytorch_sdpa_backends,
        #         )
        #         if device is not None:
        #             new_attn = new_attn.to(device=device)
        #         new_attn.load_state_dict(m.self_attn.state_dict(), strict=False)
        #         del m.self_attn
        #         m.self_attn = new_attn
        #         m.self_attention_model = self_attention_model

        # if update_config:
        #     with open_dict(self._cfg):
        #         self._cfg.self_attention_model = self_attention_model
        #         self._cfg.att_context_size = att_context_size

    def change_subsampling_conv_chunking_factor(self, subsampling_conv_chunking_factor: int):
        """
        Update the conv_chunking_factor (int)
        Default is 1 (auto)
        Set it to -1 (disabled) or to a specific value (power of 2) if you OOM in the conv subsampling layers


        Args:
            subsampling_conv_chunking_factor (int)
        """

        if not hasattr(self.pre_encode, "change_subsampling_conv_chunking_factor"):
            return

        self.pre_encode.change_subsampling_conv_chunking_factor(subsampling_conv_chunking_factor=subsampling_conv_chunking_factor)
