from typing import Optional, Union, Tuple
import math
import torch
from torch import nn
from torch.nn import (LayerNorm, CrossEntropyLoss, MSELoss, BCEWithLogitsLoss)
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import (BaseModelOutput, MaskedLMOutput, SequenceClassifierOutput,
                                        QuestionAnsweringModelOutput, MultipleChoiceModelOutput,
                                        TokenClassifierOutput)
from transformers.models.deberta_v2.modeling_deberta_v2 import (DisentangledSelfAttention,
                                            DebertaV2Attention,
                                            DebertaV2SelfOutput,
                                            DebertaV2Intermediate,
                                            DebertaV2Output,
                                            DebertaV2Layer,
                                            ConvLayer,
                                            DebertaV2Embeddings,
                                            DebertaV2Encoder,
                                            DebertaV2Config,
                                            LegacyDebertaV2OnlyMLMHead,
                                            DebertaV2OnlyMLMHead,
                                            ContextPooler,
                                            LegacyDebertaV2LMPredictionHead,
                                            DebertaV2LMPredictionHead
                                            )

from .padding import _upad_input, pad_input
from .ops.flash_attention import flash_attention_with_disentangled
from .ops.flash_attention_varlen import flash_attention_with_disentangled_varlen


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
        _attn_implementation_autoset = True,
        _attn_implementation='flash_attention_2',
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

class FlashDisentangledSelfAttention(DisentangledSelfAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, hidden_states,
                    attention_mask,
                    output_attentions=False,
                    query_states=None,
                    relative_pos=None,
                    rel_embeddings=None):
        """
        Performs the flash attention forward pass with disentangled relative attention.

        Args:
            hidden_states (Tensor): Input tensor of shape (B, L, hidden_size).
            attention_mask (Tensor): The attention mask.
            output_attentions (bool): Whether to return attention weights.
            query_states (Tensor, optional): If provided, used as Q.
            relative_pos (Tensor, optional): Relative position encoding. If None, will be built.
            causal (bool): Whether to apply causal masking.
            sm_scale (float, optional): Scaling factor for softmax.

        Returns:
            Tuple[Tensor, None]: A tuple where the first element is the output tensor of shape (B, L, hidden_size).
        """
        if attention_mask is not None:
            total_length = torch.sum(attention_mask)
            max_length = torch.prod(torch.tensor(attention_mask.shape).to(attention_mask.device))
            if max_length==total_length:
                varlen = False
            else:
                varlen = True
        else:
            varlen = False

        if query_states is None:
            query_states = hidden_states

        B, L, _ = hidden_states.shape

        def transform(x, attention_heads):
            new_x_shape = x.size()[:-1] + (attention_heads, -1)
            x = x.view(new_x_shape).permute(0, 2, 1, 3).contiguous()
            return x

        def get_heads(x, attention_heads):
            new_x_shape = x.size()[:-1] + (attention_heads, -1)
            x = x.view(new_x_shape).contiguous()
            return x

        query_layer = self.query_proj(query_states)
        key_layer = self.key_proj(hidden_states)
        value_layer = self.value_proj(hidden_states)

        scale_factor = 1
        if "c2p" in self.pos_att_type:
            scale_factor += 1
        if "p2c" in self.pos_att_type:
            scale_factor += 1

        sm_scale = 1/math.sqrt(self.attention_head_size*scale_factor)

        if self.relative_attention:
            rel_embeddings = self.pos_dropout(rel_embeddings)
            if self.share_att_key:
                pos_key_layer = transform(
                    self.key_proj(rel_embeddings.unsqueeze(0)), self.num_attention_heads
                ) # (1, NH, MD, head_dim)
                pos_query_layer = transform(
                    self.query_proj(rel_embeddings.unsqueeze(0)), self.num_attention_heads
                )
            else:
                if "c2p" in self.pos_att_type:
                    pos_key_layer = transform(
                        self.pos_key_proj(rel_embeddings.unsqueeze(0)), self.num_attention_heads
                    )

                if "p2c" in self.pos_att_type:
                    pos_query_layer = transform(
                        self.pos_query_proj(rel_embeddings.unsqueeze(0)), self.num_attention_heads
                    )
            pos_key = None
            pos_query = None
        else:
            pos_key, pos_query = None, None

        causal = False
        if not varlen:
            query_layer = transform(query_layer, self.num_attention_heads) # (B, NH, L, head_dim)
            key_layer = transform(key_layer, self.num_attention_heads)
            value_layer = transform(value_layer, self.num_attention_heads)

            if "c2p" in self.pos_att_type:
                pos_key = torch.matmul(query_layer, pos_key_layer.transpose(-1, -2))
            if "p2c" in self.pos_att_type:
                pos_query = torch.matmul(key_layer, pos_query_layer.transpose(-1, -2))

            out = flash_attention_with_disentangled(
                query_layer,
                key_layer,
                value_layer,
                pos_key,
                pos_query,
                causal,
                sm_scale,
                self.position_buckets,
                self.max_relative_positions,
            )
        else:
            query_layer = get_heads(query_layer, self.num_attention_heads) # (B, L, NH, head_dim)
            key_layer = get_heads(key_layer, self.num_attention_heads)
            value_layer = get_heads(value_layer, self.num_attention_heads)

            if "c2p" in self.pos_att_type:
                # query_layer = (1, NH, L, head_dim)
                pos_key = torch.einsum("bqhd,zhmd->bqhm", query_layer, pos_key_layer)
            if "p2c" in self.pos_att_type:
                pos_query = torch.einsum("bqhd,zhmd->bqhm", key_layer, pos_query_layer)

            (query_layer, 
             key_layer,
             value_layer,
             pos_key,
             pos_query,
             indices_q,
             cu_seq_lens,
             max_seq_lens) = _upad_input(
                query_layer,
                key_layer,
                value_layer,
                pos_key,
                pos_query,
                attention_mask,
                L,
                self.num_attention_heads)

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            out_unpad = flash_attention_with_disentangled_varlen(
                query_layer,
                key_layer,
                value_layer,
                pos_key,
                pos_query,
                cu_seqlens_q, cu_seqlens_k,
                max_seqlen_in_batch_q, max_seqlen_in_batch_k,
                causal,
                sm_scale,
                self.position_buckets,
                self.max_relative_positions,
            )
            out =  pad_input(out_unpad, indices_q, B, L).transpose(1, 2)

        out = out.view(B, self.num_attention_heads, L, self.attention_head_size).transpose(1, 2).reshape(B, L, self.all_head_size)
        return (out, None)

DEBERTA_SELF_ATTENTION_CLASSES = {
    "eager": DisentangledSelfAttention,
    "flash_attention_2": FlashDisentangledSelfAttention,
}

class FlashDebertaV2Attention(DebertaV2Attention):
    def __init__(self, config):
        super().__init__(config)
        self.self = DEBERTA_SELF_ATTENTION_CLASSES[config._attn_implementation](config)
        self.output = DebertaV2SelfOutput(config)
        self.config = config


class FlashDebertaV2Layer(DebertaV2Layer):
    def __init__(self, config):
        super().__init__(config)
        self.attention = FlashDebertaV2Attention(config)
        self.intermediate = DebertaV2Intermediate(config)
        self.output = DebertaV2Output(config)


class FlashDebertaV2Encoder(DebertaV2Encoder):
    def __init__(self, config):
        super().__init__(config)

        self.layer = nn.ModuleList([FlashDebertaV2Layer(config) for _ in range(config.num_hidden_layers)])
        self.relative_attention = getattr(config, "relative_attention", False)

        if self.relative_attention:
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings

            self.position_buckets = getattr(config, "position_buckets", -1)
            pos_ebd_size = self.max_relative_positions * 2

            if self.position_buckets > 0:
                pos_ebd_size = self.position_buckets * 2

            self.rel_embeddings = nn.Embedding(pos_ebd_size, config.hidden_size)

        self.norm_rel_ebd = [x.strip() for x in getattr(config, "norm_rel_ebd", "none").lower().split("|")]

        if "layer_norm" in self.norm_rel_ebd:
            self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=True)

        self.conv = ConvLayer(config) if getattr(config, "conv_kernel_size", 0) > 0 else None
        self.gradient_checkpointing = False

    def get_attention_mask(self, attention_mask):
        return attention_mask

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

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask,
            output_hidden_states=True,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )
        encoded_layers = encoder_outputs[1]

        if self.z_steps > 1:
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
            return (sequence_output,) + encoder_outputs[(1 if output_hidden_states else 2) :]

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
    