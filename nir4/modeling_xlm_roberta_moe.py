from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch
from torch import nn

from transformers.activations import ACT2FN
from transformers.models.xlm_roberta.configuration_xlm_roberta import XLMRobertaConfig
from transformers.models.xlm_roberta.modeling_xlm_roberta import (
    XLMRobertaAttention,
    XLMRobertaEmbeddings,
    XLMRobertaLMHead,
    XLMRobertaPooler,
    XLMRobertaPreTrainedModel,
)
from transformers.utils import ModelOutput


class XLMRobertaMoEConfig(XLMRobertaConfig):
    """
    XLM-R config with per-layer MoE expert counts.

    New fields:
      - moe_num_experts: list[int] with length == num_hidden_layers
      - output_router_logits: whether router logits should be returned by default
    """

    model_type = "xlm-roberta-moe"

    def __init__(
        self,
        moe_num_experts: Optional[Union[int, Sequence[int]]] = None,
        output_router_logits: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.moe_num_experts = self._normalize_moe_num_experts(moe_num_experts)
        self.output_router_logits = output_router_logits

    def _normalize_moe_num_experts(
        self,
        moe_num_experts: Optional[Union[int, Sequence[int]]],
    ) -> list[int]:
        if moe_num_experts is None:
            return [1] * self.num_hidden_layers
        if isinstance(moe_num_experts, int):
            return [int(moe_num_experts)] * self.num_hidden_layers
        moe_num_experts = list(moe_num_experts)
        if len(moe_num_experts) != self.num_hidden_layers:
            raise ValueError(
                "`moe_num_experts` must be an int or a list with length == num_hidden_layers "
                f"({self.num_hidden_layers}), got {len(moe_num_experts)}"
            )
        if any(int(x) < 1 for x in moe_num_experts):
            raise ValueError("Each value in `moe_num_experts` must be >= 1")
        return [int(x) for x in moe_num_experts]


@dataclass
class XLMRobertaMoEModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor | None = None
    pooler_output: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    router_logits: tuple[torch.FloatTensor | None, ...] | None = None
    expert_indices: tuple[torch.LongTensor | None, ...] | None = None


@dataclass
class XLMRobertaMoEMaskedLMOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    router_logits: tuple[torch.FloatTensor | None, ...] | None = None
    expert_indices: tuple[torch.LongTensor | None, ...] | None = None


class XLMRobertaDenseFFN(nn.Module):
    """Single dense FFN block: up -> act -> down."""

    def __init__(self, config: XLMRobertaConfig):
        super().__init__()
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size)
        self.act_fn = ACT2FN[config.hidden_act] if isinstance(config.hidden_act, str) else config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.up_proj(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.down_proj(hidden_states)
        return hidden_states

    def copy_from_xlm_roberta_ffn(self, intermediate_dense: nn.Linear, output_dense: nn.Linear) -> None:
        with torch.no_grad():
            self.up_proj.weight.copy_(intermediate_dense.weight)
            self.up_proj.bias.copy_(intermediate_dense.bias)
            self.down_proj.weight.copy_(output_dense.weight)
            self.down_proj.bias.copy_(output_dense.bias)


class XLMRobertaFFNOutput(nn.Module):
    """Residual + dropout + layernorm part kept outside the expert FFN."""

    def __init__(self, config: XLMRobertaConfig):
        super().__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + residual)
        return hidden_states


class XLMRobertaMoEFFN(nn.Module):
    """
    FFN-MoE module.

    - Experts are full FFNs (up -> act -> down).
    - `expert_path` provides a hard route of shape [batch, seq] with expert ids.
    - If `expert_path` is None and num_experts > 1, the module uses a trainable router and returns logits.
    - Default routing without `expert_path` is dense softmax mixing to keep the router trainable.
    """

    def __init__(self, config: XLMRobertaMoEConfig, num_experts: int, layer_idx: int):
        super().__init__()
        self.config = config
        self.num_experts = num_experts
        self.layer_idx = layer_idx
        self.experts = nn.ModuleList([XLMRobertaDenseFFN(config) for _ in range(num_experts)])
        self.router = nn.Linear(config.hidden_size, num_experts, bias=True) if num_experts > 1 else None
        self.reset_router()

    def reset_router(self) -> None:
        if self.router is None:
            return
        with torch.no_grad():
            self.router.weight.zero_()
            self.router.bias.zero_()

    def copy_dense_ffn_to_all_experts(self, intermediate_dense: nn.Linear, output_dense: nn.Linear) -> None:
        for expert in self.experts:
            expert.copy_from_xlm_roberta_ffn(intermediate_dense, output_dense)
        self.reset_router()

    def _stack_expert_outputs(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [batch, seq, num_experts, hidden]
        return torch.stack([expert(hidden_states) for expert in self.experts], dim=-2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        expert_path: Optional[torch.LongTensor] = None,
        output_router_logits: bool = False,
    ) -> tuple[torch.Tensor, torch.FloatTensor | None, torch.LongTensor]:
        expert_outputs = self._stack_expert_outputs(hidden_states)
        batch_size, seq_len, _, hidden_size = expert_outputs.shape

        router_logits = None

        if self.num_experts == 1:
            selected_experts = torch.zeros(
                (batch_size, seq_len),
                dtype=torch.long,
                device=hidden_states.device,
            )
            hidden_states = expert_outputs[:, :, 0, :]
            return hidden_states, router_logits, selected_experts

        if expert_path is not None:
            if expert_path.dim() == 3 and expert_path.size(-1) == 1:
                expert_path = expert_path.squeeze(-1)
            if expert_path.shape != (batch_size, seq_len):
                raise ValueError(
                    "`expert_path` must have shape [batch, seq] for a single layer, got "
                    f"{tuple(expert_path.shape)}"
                )
            selected_experts = expert_path.clamp(min=0, max=self.num_experts - 1)
            gather_index = selected_experts.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, hidden_size)
            mixed_states = torch.gather(expert_outputs, dim=2, index=gather_index).squeeze(2)
            if output_router_logits:
                router_logits = self.router(hidden_states)
            return mixed_states, router_logits, selected_experts

        router_logits = self.router(hidden_states)
        router_probs = torch.softmax(router_logits, dim=-1)
        selected_experts = router_probs.argmax(dim=-1)
        mixed_states = torch.einsum("bseh,bse->bsh", expert_outputs, router_probs)
        if not output_router_logits:
            router_logits = None
        return mixed_states, router_logits, selected_experts


class XLMRobertaMoELayer(nn.Module):
    def __init__(self, config: XLMRobertaMoEConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention = XLMRobertaAttention(config, is_causal=config.is_decoder, layer_idx=layer_idx)
        self.moe = XLMRobertaMoEFFN(config, config.moe_num_experts[layer_idx], layer_idx=layer_idx)
        self.ffn_output = XLMRobertaFFNOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        expert_path: Optional[torch.LongTensor] = None,
        output_attentions: bool = False,
        output_router_logits: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.FloatTensor | None, torch.FloatTensor | None, torch.LongTensor]:
        attention_output, attn_weights = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            **kwargs,
        )
        ffn_states, router_logits, expert_indices = self.moe(
            attention_output,
            expert_path=expert_path,
            output_router_logits=output_router_logits,
        )
        layer_output = self.ffn_output(ffn_states, attention_output)
        if not output_attentions:
            attn_weights = None
        return layer_output, attn_weights, router_logits, expert_indices


class XLMRobertaMoEEncoder(nn.Module):
    def __init__(self, config: XLMRobertaMoEConfig):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([XLMRobertaMoELayer(config, layer_idx=i) for i in range(config.num_hidden_layers)])

    @staticmethod
    def _select_layer_path(
        expert_paths: Optional[torch.LongTensor],
        layer_idx: int,
        num_hidden_layers: int,
    ) -> Optional[torch.LongTensor]:
        if expert_paths is None:
            return None
        if expert_paths.dim() != 3:
            raise ValueError(
                "`expert_paths` must have shape [num_layers, batch, seq] or [batch, seq, num_layers]"
            )
        if expert_paths.size(0) == num_hidden_layers:
            return expert_paths[layer_idx]
        if expert_paths.size(-1) == num_hidden_layers:
            return expert_paths[..., layer_idx]
        raise ValueError(
            "Cannot infer layer dimension in `expert_paths`. Expected [num_layers, batch, seq] or [batch, seq, num_layers], "
            f"got {tuple(expert_paths.shape)}"
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        expert_paths: Optional[torch.LongTensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        output_router_logits: bool = False,
        **kwargs,
    ) -> tuple[
        torch.Tensor,
        tuple[torch.FloatTensor, ...] | None,
        tuple[torch.FloatTensor, ...] | None,
        tuple[torch.FloatTensor | None, ...] | None,
        tuple[torch.LongTensor | None, ...] | None,
    ]:
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        all_expert_indices = ()

        for layer_idx, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_path = self._select_layer_path(expert_paths, layer_idx, len(self.layer))
            hidden_states, attn_weights, router_logits, expert_indices = layer_module(
                hidden_states,
                attention_mask=attention_mask,
                expert_path=layer_path,
                output_attentions=output_attentions,
                output_router_logits=output_router_logits,
                **kwargs,
            )

            if output_attentions:
                all_attentions = all_attentions + (attn_weights,)
            if output_router_logits:
                all_router_logits = all_router_logits + (router_logits,)
            all_expert_indices = all_expert_indices + (expert_indices,)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return hidden_states, all_hidden_states, all_attentions, all_router_logits, all_expert_indices


class XLMRobertaMoEModel(XLMRobertaPreTrainedModel):
    config_class = XLMRobertaMoEConfig
    base_model_prefix = "roberta"
    _no_split_modules = ["XLMRobertaEmbeddings", "XLMRobertaMoELayer"]

    def __init__(self, config: XLMRobertaMoEConfig, add_pooling_layer: bool = True):
        super().__init__(config)
        self.config = config
        self.embeddings = XLMRobertaEmbeddings(config)
        self.encoder = XLMRobertaMoEEncoder(config)
        self.pooler = XLMRobertaPooler(config) if add_pooling_layer else None
        self.post_init()
        self._zero_all_routers()

    def _zero_all_routers(self) -> None:
        for layer in self.encoder.layer:
            layer.moe.reset_router()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embeddings.word_embeddings = value

    @staticmethod
    def _extend_attention_mask(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        else:
            raise ValueError(
                "`attention_mask` must have rank 2 or 3, got "
                f"rank={attention_mask.dim()}"
            )
        extended_attention_mask = extended_attention_mask.to(dtype=dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        return extended_attention_mask

    def init_moe_from_dense_model(self, dense_model: nn.Module) -> "XLMRobertaMoEModel":
        self.embeddings.load_state_dict(dense_model.embeddings.state_dict())

        if self.pooler is not None and getattr(dense_model, "pooler", None) is not None:
            self.pooler.load_state_dict(dense_model.pooler.state_dict())

        for moe_layer, dense_layer in zip(self.encoder.layer, dense_model.encoder.layer):
            moe_layer.attention.load_state_dict(dense_layer.attention.state_dict())
            moe_layer.ffn_output.LayerNorm.load_state_dict(dense_layer.output.LayerNorm.state_dict())
            moe_layer.moe.copy_dense_ffn_to_all_experts(
                dense_layer.intermediate.dense,
                dense_layer.output.dense,
            )
        return self

    @classmethod
    def from_dense_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        moe_num_experts: Union[int, Sequence[int]],
        add_pooling_layer: bool = False,
        **kwargs,
    ) -> "XLMRobertaMoEModel":
        from transformers import XLMRobertaModel

        dense_config = XLMRobertaConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        config_dict = dense_config.to_dict()
        config_dict["moe_num_experts"] = list(moe_num_experts) if not isinstance(moe_num_experts, int) else moe_num_experts
        config = XLMRobertaMoEConfig(**config_dict)
        moe_model = cls(config, add_pooling_layer=add_pooling_layer)
        dense_model = XLMRobertaModel.from_pretrained(
            pretrained_model_name_or_path,
            add_pooling_layer=add_pooling_layer,
            **kwargs,
        )
        moe_model.init_moe_from_dense_model(dense_model)
        return moe_model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        expert_paths: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[tuple, XLMRobertaMoEModelOutput]:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of `input_ids` or `inputs_embeds`.")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )

        if input_ids is not None:
            input_shape = input_ids.size()
            device = input_ids.device
        else:
            input_shape = inputs_embeds.size()[:-1]
            device = inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask = self._extend_attention_mask(attention_mask, self.dtype)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )

        sequence_output, all_hidden_states, all_attentions, all_router_logits, all_expert_indices = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            expert_paths=expert_paths,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            **kwargs,
        )

        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            outputs = [sequence_output, pooled_output]
            if output_hidden_states:
                outputs.append(all_hidden_states)
            if output_attentions:
                outputs.append(all_attentions)
            if output_router_logits:
                outputs.append(all_router_logits)
            outputs.append(all_expert_indices)
            return tuple(outputs)

        return XLMRobertaMoEModelOutput(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            router_logits=all_router_logits,
            expert_indices=all_expert_indices,
        )


class XLMRobertaMoEForMaskedLM(XLMRobertaPreTrainedModel):
    """
    Optional MLM wrapper so save/load can be done through a task head too.
    This is useful if you continue pretraining the MoE model with MLM.
    """

    config_class = XLMRobertaMoEConfig
    _tied_weights_keys = {
        "lm_head.decoder.weight": "roberta.embeddings.word_embeddings.weight",
        "lm_head.decoder.bias": "lm_head.bias",
    }

    def __init__(self, config: XLMRobertaMoEConfig):
        super().__init__(config)
        self.roberta = XLMRobertaMoEModel(config, add_pooling_layer=False)
        self.lm_head = XLMRobertaLMHead(config)
        self.post_init()

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.lm_head.decoder = new_embeddings

    @classmethod
    def from_dense_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        moe_num_experts: Union[int, Sequence[int]],
        **kwargs,
    ) -> "XLMRobertaMoEForMaskedLM":
        model = cls(
            XLMRobertaMoEConfig.from_pretrained(
                pretrained_model_name_or_path,
                moe_num_experts=moe_num_experts,
                **kwargs,
            )
        )
        model.roberta = XLMRobertaMoEModel.from_dense_pretrained(
            pretrained_model_name_or_path,
            moe_num_experts=moe_num_experts,
            add_pooling_layer=False,
            **kwargs,
        )
        return model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        expert_paths: Optional[torch.LongTensor] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[tuple, XLMRobertaMoEMaskedLMOutput]:
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            expert_paths=expert_paths,
            output_router_logits=output_router_logits,
            return_dict=True,
            **kwargs,
        )
        logits = self.lm_head(outputs.last_hidden_state)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not (return_dict if return_dict is not None else self.config.use_return_dict):
            return (loss, logits, outputs.hidden_states, outputs.attentions, outputs.router_logits, outputs.expert_indices)

        return XLMRobertaMoEMaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
            expert_indices=outputs.expert_indices,
        )
