from typing import Optional, Union, Tuple, Dict
import logging
from collections.abc import Sequence
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import (LayerNorm, CrossEntropyLoss, MSELoss, BCEWithLogitsLoss)
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import (BaseModelOutput, MaskedLMOutput, SequenceClassifierOutput,
                                        QuestionAnsweringModelOutput, MultipleChoiceModelOutput,
                                        TokenClassifierOutput)
from transformers.models.deberta_v2.modeling_deberta_v2 import (DisentangledSelfAttention,
                                            DebertaV2Attention,
                                            DebertaV2Layer,
                                            DebertaV2Embeddings,
                                            DebertaV2Encoder,
                                            DebertaV2Config,
                                            LegacyDebertaV2OnlyMLMHead,
                                            DebertaV2OnlyMLMHead,
                                            ContextPooler,
                                            LegacyDebertaV2LMPredictionHead,
                                            DebertaV2LMPredictionHead
                                            )
from transformers.utils import logging

from .padding import index_first_axis, pad_input
from .ops.flash_attention import flash_attention_with_disentangled
from .ops.flash_attention_varlen import flash_attention_with_disentangled_varlen

logger = logging.get_logger(__name__)

@torch.jit.script
def make_log_bucket_position(relative_pos, bucket_size: int, max_position: int):
    sign = torch.sign(relative_pos)
    mid = bucket_size // 2
    abs_pos = torch.where(
        (relative_pos < mid) & (relative_pos > -mid),
        torch.tensor(mid - 1).type_as(relative_pos),
        torch.abs(relative_pos),
    )
    log_pos = (
        torch.ceil(torch.log(abs_pos / mid) / torch.log(torch.tensor((max_position - 1) / mid)) * (mid - 1)) + mid
    )
    bucket_pos = torch.where(abs_pos <= mid, relative_pos.type_as(log_pos), log_pos * sign)
    return bucket_pos


def build_relative_position(query_layer, key_layer, bucket_size: int = -1, max_position: int = -1):
    """
    Build relative position according to the query and key

    We assume the absolute position of query \\(P_q\\) is range from (0, query_size) and the absolute position of key
    \\(P_k\\) is range from (0, key_size), The relative positions from query to key is \\(R_{q \\rightarrow k} = P_q -
    P_k\\)

    Args:
        query_size (int): the length of query
        key_size (int): the length of key
        bucket_size (int): the size of position bucket
        max_position (int): the maximum allowed absolute position
        device (`torch.device`): the device on which tensors will be created.

    Return:
        `torch.LongTensor`: A tensor with shape [1, query_size, key_size]
    """
    query_size = query_layer.size(2)
    key_size = key_layer.size(2)

    q_ids = torch.arange(query_size, dtype=torch.long, device=query_layer.device)
    k_ids = torch.arange(key_size, dtype=torch.long, device=key_layer.device)
    rel_pos_ids = q_ids[:, None] - k_ids[None, :]
    if bucket_size > 0 and max_position > 0:
        rel_pos_ids = make_log_bucket_position(rel_pos_ids, bucket_size, max_position)
    rel_pos_ids = rel_pos_ids.to(torch.long)
    rel_pos_ids = rel_pos_ids[:query_size, :]
    rel_pos_ids = rel_pos_ids.unsqueeze(0)
    return rel_pos_ids


@torch.jit.script
def scaled_size_sqrt(query_layer: torch.Tensor, scale_factor: int):
    return torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)


def _transform_for_scores(x: torch.Tensor, attention_heads: int) -> torch.Tensor:
    """Transform tensor from (B, L, H*D) to (B, H, L, D)"""
    new_x_shape = x.size()[:-1] + (attention_heads, -1)
    x = x.view(new_x_shape).permute(0, 2, 1, 3).contiguous()
    return x


def _get_heads(x: torch.Tensor, attention_heads: int) -> torch.Tensor:
    """Transform tensor from (B, L, H*D) to (B, L, H, D)"""
    new_x_shape = x.size()[:-1] + (attention_heads, -1)
    x = x.view(new_x_shape).contiguous()
    return x

@torch.jit.script
def build_rpos(query_layer, key_layer, relative_pos, position_buckets: int, max_relative_positions: int):
    if key_layer.size(-2) != query_layer.size(-2):
        return build_relative_position(
            key_layer,
            key_layer,
            bucket_size=position_buckets,
            max_position=max_relative_positions,
        )
    else:
        return relative_pos
    
class DebertaV2Config(PretrainedConfig):
    model_type = "deberta-v2"

    def __init__(
        self,
        vocab_size=128100,
        hidden_size=1536,
        num_hidden_layers=24,
        num_attention_heads=24,
        intermediate_size=6144,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=0,
        initializer_range=0.02,
        layer_norm_eps=1e-7,
        relative_attention=False,
        max_relative_positions=-1,
        pad_token_id=0,
        position_biased_input=True,
        pos_att_type=None,
        pooler_dropout=0,
        pooler_hidden_act="gelu",
        legacy=True,
        unpad_inputs=True,
        _attn_implementation_autoset = True,
        _attn_implementation='eager',
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.relative_attention = relative_attention
        self.max_relative_positions = max_relative_positions
        self.pad_token_id = pad_token_id
        self.position_biased_input = position_biased_input
        # Remove padding once in the encoder and run all layers on real tokens only (see FlashDebertaV2Encoder).
        self.unpad_inputs = unpad_inputs
        self._attn_implementation_autoset = _attn_implementation_autoset
        self._attn_implementation = _attn_implementation
        
        # Backwards compatibility
        if isinstance(pos_att_type, str):
            pos_att_type = [x.strip() for x in pos_att_type.lower().split("|")]

        self.pos_att_type = pos_att_type
        self.vocab_size = vocab_size
        self.layer_norm_eps = layer_norm_eps

        self.pooler_hidden_size = kwargs.get("pooler_hidden_size", hidden_size)
        self.pooler_dropout = pooler_dropout
        self.pooler_hidden_act = pooler_hidden_act
        self.legacy = legacy

class PackedBatch:
    """A padded batch with its padding removed: the real tokens of every sequence, end to end.

    The encoder builds one per forward pass and hands it to every layer in place of the
    attention mask. Hidden states are then ``(T, hidden)`` with ``T`` the number of real
    tokens, so the projections, LayerNorms and FFN skip padding too. Only attention needs
    the sequence boundaries, ``cu_seqlens``.
    """

    def __init__(self, mask_2d):
        self.batch_size, self.seq_len = mask_2d.shape
        # The one host sync of a packed forward: T sets the shape of every packed tensor.
        self.indices = torch.nonzero(mask_2d.flatten(), as_tuple=False).flatten()
        self.cu_seqlens = F.pad(mask_2d.sum(dim=-1, dtype=torch.int32).cumsum(0, dtype=torch.int32), (1, 0))
        self.num_tokens = self.indices.numel()
        # Only sizes the kernel's position lookup table, so the padded length is a valid bound
        # and saves reading the true maximum back from the GPU.
        self.max_seqlen = self.seq_len

    @property
    def has_padding(self):
        return self.num_tokens < self.batch_size * self.seq_len

    def pack(self, x):
        """(B, L, ...) -> (T, ...)"""
        return index_first_axis(x.reshape(self.batch_size * self.seq_len, *x.shape[2:]), self.indices)

    def unpack(self, x):
        """(T, ...) -> (B, L, ...), zeros at padding."""
        return pad_input(x, self.indices, self.batch_size, self.seq_len)


@torch.compiler.disable
def _sequence_lengths(attention_mask):
    """Per-row lengths of a right-padded batch, computed once and shared by all layers."""
    # Every layer gets the same mask tensor, so cache on it (and recompute if it was edited in place).
    cached = getattr(attention_mask, "_flashdeberta_lengths", None)
    if cached is not None and cached[0] == attention_mask._version:
        return cached[1]
    if attention_mask.dim() == 4:
        valid = attention_mask[:, 0].sum(dim=-1) > 0   # [B, 1, L, L] -> [B, L]
    elif attention_mask.dim() == 3:
        valid = attention_mask.sum(dim=-1) > 0         # [B, L, L] -> [B, L]
    else:
        valid = attention_mask > 0
    lengths = valid.sum(dim=-1, dtype=torch.int32)
    attention_mask._flashdeberta_lengths = (attention_mask._version, lengths)
    return lengths


class FlashDisentangledSelfAttention(DisentangledSelfAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _position_layers(self, rel_embeddings):
        """Projected relative embeddings (1, NH, 2 * span, head_dim) for c2p and p2c, or None."""
        if not self.relative_attention:
            return None, None
        rel_embeddings = self.pos_dropout(rel_embeddings).unsqueeze(0)
        project = lambda proj: _transform_for_scores(proj(rel_embeddings), self.num_attention_heads)
        pos_key_layer = pos_query_layer = None
        if "c2p" in self.pos_att_type:
            pos_key_layer = project(self.key_proj if self.share_att_key else self.pos_key_proj)
        if "p2c" in self.pos_att_type:
            pos_query_layer = project(self.query_proj if self.share_att_key else self.pos_query_proj)
        return pos_key_layer, pos_query_layer

    def forward(self, hidden_states,
                    attention_mask,
                    output_attentions=False,
                    query_states=None,
                    relative_pos=None,
                    rel_embeddings=None):
        """
        Flash attention with disentangled relative attention.

        Args:
            hidden_states (Tensor): (B, L, hidden_size) padded, or (T, hidden_size) when
                ``attention_mask`` is a ``PackedBatch``.
            attention_mask (Tensor | PackedBatch): a padding mask ((B, L), (B, L, L) or (B, 1, L, L);
                padding must be on the right), or the ``PackedBatch`` describing packed inputs.
            output_attentions (bool): Not supported; attention weights are never returned.
            query_states (Tensor, optional): If provided, used as Q.
            relative_pos (Tensor, optional): Ignored; relative positions are computed in the kernel.
            rel_embeddings (Tensor): Relative position embeddings.

        Returns:
            Tuple[Tensor, None]: output of the same shape as ``hidden_states``.
        """
        if query_states is None:
            query_states = hidden_states

        scale_factor = 1 + ("c2p" in self.pos_att_type) + ("p2c" in self.pos_att_type)
        sm_scale = 1 / math.sqrt(self.attention_head_size * scale_factor)
        pos_key_layer, pos_query_layer = self._position_layers(rel_embeddings)

        query_layer = self.query_proj(query_states)
        key_layer = self.key_proj(hidden_states)
        value_layer = self.value_proj(hidden_states)

        if isinstance(attention_mask, PackedBatch):
            batch = attention_mask
            T = hidden_states.shape[0]
            heads = lambda t: t.view(T, self.num_attention_heads, self.attention_head_size)
            out = flash_attention_with_disentangled_varlen(
                heads(query_layer), heads(key_layer), heads(value_layer),
                pos_key_layer, pos_query_layer,
                batch.cu_seqlens, batch.cu_seqlens,
                batch.max_seqlen, batch.max_seqlen,
                False, sm_scale,
                self.position_buckets,
                self.max_relative_positions,
            )
            return (out.reshape(T, self.all_head_size), None)

        B, L, _ = hidden_states.shape
        out = flash_attention_with_disentangled(
            _transform_for_scores(query_layer, self.num_attention_heads),  # (B, NH, L, head_dim)
            _transform_for_scores(key_layer, self.num_attention_heads),
            _transform_for_scores(value_layer, self.num_attention_heads),
            _sequence_lengths(attention_mask) if attention_mask is not None else None,
            pos_key_layer, pos_query_layer,
            False, sm_scale,
            self.position_buckets,
            self.max_relative_positions,
        )
        out = out.view(B, self.num_attention_heads, L, self.attention_head_size).transpose(1, 2).reshape(B, L, self.all_head_size)
        return (out, None)


DEBERTA_SELF_ATTENTION_CLASSES = {
    "eager": DisentangledSelfAttention,
    "flash_attention_2": FlashDisentangledSelfAttention,
}

class FlashDebertaV2Attention(DebertaV2Attention):
    def __init__(self, config):
        super().__init__(config)
        self.self = FlashDisentangledSelfAttention(config)


class FlashDebertaV2Layer(DebertaV2Layer):
    def __init__(self, config):
        super().__init__(config)
        self.attention = FlashDebertaV2Attention(config)


class FlashDebertaV2Encoder(DebertaV2Encoder):
    """DeBERTa-v2 encoder that removes padding once and runs every layer on packed tokens.

    ``config.unpad_inputs`` (default True) controls it and is read on every call, so
    ``model.config.unpad_inputs = False`` forces the padded path at runtime. Packing is used
    only for a 2D padding mask that contains padding; 3D/4D masks, ``query_states`` and
    fully-filled batches take the padded path.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.layer = nn.ModuleList([FlashDebertaV2Layer(config) for _ in range(config.num_hidden_layers)])

    @torch.compiler.disable
    def _packed_batch(self, hidden_states, attention_mask, query_states):
        if not getattr(self.config, "unpad_inputs", True):
            return None
        if query_states is not None or attention_mask.dim() > 2 or not torch.is_tensor(hidden_states):
            return None
        batch = PackedBatch(attention_mask > 0)
        return batch if batch.has_padding else None

    def forward(
        self,
        hidden_states,
        attention_mask,
        output_hidden_states=True,
        output_attentions=False,
        query_states=None,
        relative_pos=None,
        return_dict=True,
    ):
        if attention_mask.dim() <= 2:
            input_mask = attention_mask
        else:
            input_mask = attention_mask.sum(-2) > 0

        # Relative positions are computed inside the kernels, so the L x L `relative_pos`
        # tensor of the original encoder is never built.
        batch = self._packed_batch(hidden_states, attention_mask, query_states)
        if batch is None:
            layer_mask = self.get_attention_mask(attention_mask)
            next_kv = hidden_states
        else:
            layer_mask = batch
            next_kv = batch.pack(hidden_states)
        unpack = (lambda x: x) if batch is None else batch.unpack

        all_hidden_states = (hidden_states,) if output_hidden_states else None
        all_attentions = () if output_attentions else None
        rel_embeddings = self.get_rel_embedding()

        for i, layer_module in enumerate(self.layer):
            output_states, attn_weights = layer_module(
                next_kv,
                layer_mask,
                query_states=query_states,
                relative_pos=relative_pos,
                rel_embeddings=rel_embeddings,
                output_attentions=output_attentions,
            )
            if output_attentions:
                all_attentions = all_attentions + (attn_weights,)

            if i == 0 and self.conv is not None:
                # The convolution mixes neighbouring tokens, so it needs the padded layout.
                output_states = self.conv(hidden_states, unpack(output_states), input_mask)
                if batch is not None:
                    output_states = batch.pack(output_states)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (unpack(output_states),)

            if query_states is not None:
                query_states = output_states
                if isinstance(hidden_states, Sequence):
                    next_kv = hidden_states[i + 1] if i + 1 < len(self.layer) else None
            else:
                next_kv = output_states

        last_hidden_state = all_hidden_states[-1] if output_hidden_states else unpack(output_states)
        if not return_dict:
            return tuple(v for v in [last_hidden_state, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=last_hidden_state, hidden_states=all_hidden_states, attentions=all_attentions
        )

class FlashDebertaV2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = DebertaV2Config
    base_model_prefix = "deberta"
    _keys_to_ignore_on_load_unexpected = ["position_embeddings"]
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
        elif isinstance(module, (LegacyDebertaV2LMPredictionHead, DebertaV2LMPredictionHead)):
            module.bias.data.zero_()

    @classmethod
    def _autoset_attn_implementation(
        cls,
        config,
        use_flash_attention_2: bool = False,   
        torch_dtype: Optional[torch.dtype] = None,
        device_map: Optional[Union[str, Dict[str, int]]] = None,
        check_device_map: bool = True,
    ):
        """
        Decide which attention backend to use.
        Priority
        --------
        1. Respect an explicit value already sitting in `config._attn_implementation`
           (e.g. user passed `attn_implementation="sdpa"` to `from_pretrained`).
        2. If `use_flash_attention_2=True` **and** a compatible GPU, dtype, and
           flash-attn-2 kernels are available → choose `"flash_attention_2"`.
        3. If PyTorch’s scaled-dot-product attention is available → `"sdpa"`.
        4. Otherwise → `"eager"`.
        """
        if getattr(config, "_attn_implementation", None) and not getattr(
            config, "_attn_implementation_autoset", False
        ):
            return config

        torch_dtype = torch_dtype or getattr(config, "torch_dtype", None)

        on_cuda = (
            torch.cuda.is_available()
            and (device_map is None or (isinstance(device_map, str) and device_map != "cpu"))
        )

        if (
            use_flash_attention_2
            and cls._supports_flash_attn_2
            and on_cuda
            and torch_dtype in (None, torch.float16, torch.bfloat16)
        ):
            config._attn_implementation = "flash_attention_2"
            config._flash_attn_2_enabled = True
            config._attn_implementation_autoset = True
            return config
        
        if check_device_map and not on_cuda and torch.cuda.is_available():
            logger.warning_once(
                "FlashDeBERTa is being initialised on CPU. Move the model to GPU "
                "with `model.to('cuda')` to benefit from Flash-Attention/SDPA."
            )

        config._attn_implementation = "eager"
        config._attn_implementation_autoset = True
        return config

class FlashDebertaV2Model(FlashDebertaV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = DebertaV2Embeddings(config)
        self.encoder = FlashDebertaV2Encoder(config)
        self.z_steps = 0
        self.config = config
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings.word_embeddings = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError("The prune function is not implemented in DeBERTa model.")

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )

        # Hidden states of every layer are only needed when asked for or for z_steps; with packed
        # inputs each one costs a re-padding copy.
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask,
            output_hidden_states=output_hidden_states or self.z_steps > 1,
            output_attentions=output_attentions,
            return_dict=True,
        )
        sequence_output = encoder_outputs.last_hidden_state

        if self.z_steps > 1:
            encoded_layers = list(encoder_outputs.hidden_states)
            hidden_states = encoded_layers[-2]
            layers = [self.encoder.layer[-1] for _ in range(self.z_steps)]
            query_states = encoded_layers[-1]
            rel_embeddings = self.encoder.get_rel_embedding()
            attention_mask = self.encoder.get_attention_mask(attention_mask)
            rel_pos = self.encoder.get_rel_pos(embedding_output)
            for layer in layers[1:]:
                query_states = layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=False,
                    query_states=query_states,
                    relative_pos=rel_pos,
                    rel_embeddings=rel_embeddings,
                )
                encoded_layers.append(query_states)
            sequence_output = encoded_layers[-1]

        if not return_dict:
            extra = (encoder_outputs.hidden_states if output_hidden_states else None, encoder_outputs.attentions)
            return (sequence_output,) + tuple(v for v in extra if v is not None)

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states if output_hidden_states else None,
            attentions=encoder_outputs.attentions,
        )

class FlashDebertaV2ForMaskedLM(FlashDebertaV2PreTrainedModel):
    _tied_weights_keys = ["cls.predictions.decoder.weight", "cls.predictions.decoder.bias"]
    _keys_to_ignore_on_load_unexpected = r"mask_predictions.*"

    def __init__(self, config):
        super().__init__(config)
        self.legacy = config.legacy
        self.deberta = FlashDebertaV2Model(config)
        if self.legacy:
            self.cls = LegacyDebertaV2OnlyMLMHead(config)
        else:
            self._tied_weights_keys = ["lm_predictions.lm_head.weight", "deberta.embeddings.word_embeddings.weight"]
            self.lm_predictions = DebertaV2OnlyMLMHead(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        if self.legacy:
            return self.cls.predictions.decoder
        else:
            return self.lm_predictions.lm_head.dense

    def set_output_embeddings(self, new_embeddings):
        if self.legacy:
            self.cls.predictions.decoder = new_embeddings
            self.cls.predictions.bias = new_embeddings.bias
        else:
            self.lm_predictions.lm_head.dense = new_embeddings
            self.lm_predictions.lm_head.bias = new_embeddings.bias

    # Copied from transformers.models.deberta.modeling_deberta.DebertaForMaskedLM.forward with Deberta->DebertaV2
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        if self.legacy:
            prediction_scores = self.cls(sequence_output)
        else:
            prediction_scores = self.lm_predictions(sequence_output, self.deberta.embeddings.word_embeddings)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    

class FlashDebertaV2ForSequenceClassification(FlashDebertaV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        num_labels = getattr(config, "num_labels", 2)
        self.num_labels = num_labels

        self.deberta = FlashDebertaV2Model(config)
        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim

        self.classifier = nn.Linear(output_dim, num_labels)
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = nn.Dropout(drop_out)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.deberta.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.deberta.set_input_embeddings(new_embeddings)

    # Copied from transformers.models.deberta.modeling_deberta.DebertaForSequenceClassification.forward with Deberta->DebertaV2
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoder_layer = outputs[0]
        pooled_output = self.pooler(encoder_layer)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    # regression task
                    loss_fn = nn.MSELoss()
                    logits = logits.view(-1).to(labels.dtype)
                    loss = loss_fn(logits, labels.view(-1))
                elif labels.dim() == 1 or labels.size(-1) == 1:
                    label_index = (labels >= 0).nonzero()
                    labels = labels.long()
                    if label_index.size(0) > 0:
                        labeled_logits = torch.gather(
                            logits, 0, label_index.expand(label_index.size(0), logits.size(1))
                        )
                        labels = torch.gather(labels, 0, label_index.view(-1))
                        loss_fct = CrossEntropyLoss()
                        loss = loss_fct(labeled_logits.view(-1, self.num_labels).float(), labels.view(-1))
                    else:
                        loss = torch.tensor(0).to(logits)
                else:
                    log_softmax = nn.LogSoftmax(-1)
                    loss = -((log_softmax(logits) * labels).sum(-1)).mean()
            elif self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )
    

class FlashDebertaV2ForTokenClassification(FlashDebertaV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.deberta = FlashDebertaV2Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )

class FlashDebertaV2ForQuestionAnswering(FlashDebertaV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.deberta = FlashDebertaV2Model(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    # Copied from transformers.models.deberta.modeling_deberta.DebertaForQuestionAnswering.forward with Deberta->DebertaV2
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    

class FlashDebertaV2ForMultipleChoice(FlashDebertaV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        num_labels = getattr(config, "num_labels", 2)
        self.num_labels = num_labels

        self.deberta = FlashDebertaV2Model(config)
        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim

        self.classifier = nn.Linear(output_dim, 1)
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = nn.Dropout(drop_out)

        self.init_weights()

    def get_input_embeddings(self):
        return self.deberta.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.deberta.set_input_embeddings(new_embeddings)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.deberta(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            inputs_embeds=flat_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoder_layer = outputs[0]
        pooled_output = self.pooler(encoder_layer)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


####################################################################
# The following models are designed for custom training loops where 
# the trainer has more control over the loss computation.
####################################################################

class FlashDebertaV2ForMaskedLmTraining(FlashDebertaV2PreTrainedModel):
    """
    FlashDeBERTa model for manual Masked Language Modeling training.

    The LM head is available but NOT applied in forward().
    Loss is computed manually in the trainer, allowing for custom strategies:
    - joint training (MLM + other objectives)
    - custom masking strategies
    - custom loss functions
    """
    _tied_weights_keys = ["lm_head.weight", "deberta.embeddings.word_embeddings.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.legacy = config.legacy
        self.deberta = FlashDebertaV2Model(config)

        # LM head for manual use in trainer
        if self.legacy:
            self.lm_head = LegacyDebertaV2OnlyMLMHead(config)
        else:
            self.lm_head = DebertaV2OnlyMLMHead(config)

        self.post_init()

    def get_input_embeddings(self):
        return self.deberta.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.deberta.set_input_embeddings(new_embeddings)

    def get_output_embeddings(self):
        if self.legacy:
            return self.lm_head.predictions.decoder
        else:
            return self.lm_head.lm_head.dense

    def set_output_embeddings(self, new_embeddings):
        if self.legacy:
            self.lm_head.predictions.decoder = new_embeddings
            self.lm_head.predictions.bias = new_embeddings.bias
        else:
            self.lm_head.lm_head.dense = new_embeddings
            self.lm_head.lm_head.bias = new_embeddings.bias

    def tie_weights(self):
        """Tie embedding and output weights if configured."""
        if self.config.tie_word_embeddings:
            output_embeddings = self.get_output_embeddings()
            if output_embeddings is not None:
                self._tie_or_clone_weights(output_embeddings, self.get_input_embeddings())

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[torch.FloatTensor], BaseModelOutput]:
        """
        Forward pass through encoder only.

        LM head is NOT applied here - use model.lm_head() manually in compute_loss.
        This gives the trainer full control over logits computation and loss calculation.

        Returns:
            BaseModelOutput with last_hidden_state, hidden_states, and attentions.
            Trainer will apply lm_head and compute loss separately.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return outputs
