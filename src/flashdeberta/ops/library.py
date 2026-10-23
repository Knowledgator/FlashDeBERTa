"""The attention kernels as ``torch.library`` custom ops.

``torch.compile`` treats a custom op as one opaque node with a known output shape, so the
kernel launch code (shared-memory fallback, cached position plans) runs unchanged inside
compiled graphs and CUDA graphs. Each forward op has a registered backward, so compiled
training works too.

Ops:
    flashdeberta::attn_fwd / attn_bwd                 padded (B, H, L, D) inputs
    flashdeberta::attn_varlen_fwd / attn_varlen_bwd   packed (T, H, D) inputs

``precision`` is "native", "fp8_qk" or "fp8_qkv" (see ``ops.fp8``). A backward through
per-sequence position tables (``pos_*_index``) is not supported.

Inside a CUDA graph capture nothing here may sync with the host. The ops do not, with
two exceptions that only matter on first use: a new position plan is built on the host
and copied over, and a new kernel specialization is compiled. Both are cached, and
CUDA graph trees run every new shape once before recording it, so captures hit the cache.
Custom ``positions`` without ``position_span`` read the span from the GPU on every call.
"""
from typing import Optional

import torch
from torch import Tensor

from .flash_attention import padded_backward, padded_forward
from .flash_attention_varlen import varlen_backward, varlen_forward

_EMPTY = ()


def _or_empty(grad, like):
    # Custom ops cannot return None: a missing gradient travels as an empty tensor.
    return grad if grad is not None else like.new_empty(_EMPTY)


# ----------------------------------------------------------------------------- padded

@torch.library.custom_op("flashdeberta::attn_fwd", mutates_args=())
def attn_fwd(
    q: Tensor, k: Tensor, v: Tensor,
    seq_lengths: Optional[Tensor],
    pos_key_layer: Optional[Tensor], pos_query_layer: Optional[Tensor],
    pos_key_index: Optional[Tensor], pos_query_index: Optional[Tensor],
    positions: Optional[Tensor],
    causal: bool, sm_scale: float, position_buckets: int, max_relative_distance: int,
    position_span: Optional[int], precision: str,
) -> tuple[Tensor, Tensor]:
    return padded_forward(q, k, v, seq_lengths, pos_key_layer, pos_query_layer, pos_key_index, pos_query_index,
                          positions, causal, sm_scale, position_buckets, max_relative_distance, position_span,
                          precision)


@attn_fwd.register_fake
def _(q, k, v, seq_lengths, pos_key_layer, pos_query_layer, pos_key_index, pos_query_index, positions,
      causal, sm_scale, position_buckets, max_relative_distance, position_span, precision):
    B, H, M, _ = q.shape
    return torch.empty_like(q), q.new_empty((B, H, M), dtype=torch.float32)


@torch.library.custom_op("flashdeberta::attn_bwd", mutates_args=())
def attn_bwd(
    do: Tensor, q: Tensor, k: Tensor, v: Tensor, o: Tensor, lse: Tensor,
    seq_lengths: Optional[Tensor],
    pos_key_layer: Optional[Tensor], pos_query_layer: Optional[Tensor],
    positions: Optional[Tensor],
    causal: bool, sm_scale: float, position_buckets: int, max_relative_distance: int,
    position_span: Optional[int], precision: str,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    dq, dk, dv, dpk, dpq = padded_backward(
        do, q, k, v, o, lse, seq_lengths, pos_key_layer, pos_query_layer, positions, causal, sm_scale,
        position_buckets, max_relative_distance, position_span, precision)
    return dq, dk, dv, _or_empty(dpk, q), _or_empty(dpq, q)


@attn_bwd.register_fake
def _(do, q, k, v, o, lse, seq_lengths, pos_key_layer, pos_query_layer, positions,
      causal, sm_scale, position_buckets, max_relative_distance, position_span, precision):
    dpk = torch.empty_like(pos_key_layer) if pos_key_layer is not None else q.new_empty(_EMPTY)
    dpq = torch.empty_like(pos_query_layer) if pos_query_layer is not None else q.new_empty(_EMPTY)
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v), dpk, dpq


def _attn_setup(ctx, inputs, output):
    (q, k, v, seq_lengths, pos_key_layer, pos_query_layer, pos_key_index, pos_query_index, positions,
     causal, sm_scale, position_buckets, max_relative_distance, position_span, precision) = inputs
    o, lse = output
    ctx.save_for_backward(q, k, v, o, lse, seq_lengths, pos_key_layer, pos_query_layer, positions)
    ctx.per_sequence_tables = pos_key_index is not None or pos_query_index is not None
    ctx.has_pos = (pos_key_layer is not None, pos_query_layer is not None)
    ctx.args = (causal, sm_scale, position_buckets, max_relative_distance, position_span, precision)


def _attn_backward(ctx, do, dlse):
    if ctx.per_sequence_tables:
        raise NotImplementedError("backward through per-sequence position tables is not supported")
    q, k, v, o, lse, seq_lengths, pos_key_layer, pos_query_layer, positions = ctx.saved_tensors
    dq, dk, dv, dpk, dpq = attn_bwd(do, q, k, v, o, lse, seq_lengths, pos_key_layer, pos_query_layer,
                                    positions, *ctx.args)
    has_pk, has_pq = ctx.has_pos
    return (dq, dk, dv, None, dpk if has_pk else None, dpq if has_pq else None,
            None, None, None, None, None, None, None, None, None)


attn_fwd.register_autograd(_attn_backward, setup_context=_attn_setup)


# ----------------------------------------------------------------------------- packed

@torch.library.custom_op("flashdeberta::attn_varlen_fwd", mutates_args=())
def attn_varlen_fwd(
    q: Tensor, k: Tensor, v: Tensor,
    cu_seqlens_q: Tensor, cu_seqlens_k: Tensor, max_seqlen_q: int, max_seqlen_k: int,
    pos_key_layer: Optional[Tensor], pos_query_layer: Optional[Tensor],
    pos_key_index: Optional[Tensor], pos_query_index: Optional[Tensor],
    positions: Optional[Tensor],
    causal: bool, sm_scale: float, position_buckets: int, max_relative_distance: int,
    position_span: Optional[int], precision: str,
) -> tuple[Tensor, Tensor]:
    return varlen_forward(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                          pos_key_layer, pos_query_layer, pos_key_index, pos_query_index, positions,
                          causal, sm_scale, position_buckets, max_relative_distance, position_span, precision)


@attn_varlen_fwd.register_fake
def _(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, pos_key_layer, pos_query_layer,
      pos_key_index, pos_query_index, positions, causal, sm_scale, position_buckets, max_relative_distance,
      position_span, precision):
    return torch.empty_like(q), q.new_empty(q.shape[:2], dtype=torch.float32)


@torch.library.custom_op("flashdeberta::attn_varlen_bwd", mutates_args=())
def attn_varlen_bwd(
    do: Tensor, q: Tensor, k: Tensor, v: Tensor, o: Tensor, lse: Tensor,
    cu_seqlens_q: Tensor, cu_seqlens_k: Tensor, max_seqlen_q: int, max_seqlen_k: int,
    pos_key_layer: Optional[Tensor], pos_query_layer: Optional[Tensor],
    positions: Optional[Tensor],
    causal: bool, sm_scale: float, position_buckets: int, max_relative_distance: int,
    position_span: Optional[int], precision: str,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    dq, dk, dv, dpk, dpq = varlen_backward(
        do, q, k, v, o, lse, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
        pos_key_layer, pos_query_layer, positions, causal, sm_scale,
        position_buckets, max_relative_distance, position_span, precision)
    return dq, dk, dv, _or_empty(dpk, q), _or_empty(dpq, q)


@attn_varlen_bwd.register_fake
def _(do, q, k, v, o, lse, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, pos_key_layer,
      pos_query_layer, positions, causal, sm_scale, position_buckets, max_relative_distance, position_span,
      precision):
    dpk = torch.empty_like(pos_key_layer) if pos_key_layer is not None else q.new_empty(_EMPTY)
    dpq = torch.empty_like(pos_query_layer) if pos_query_layer is not None else q.new_empty(_EMPTY)
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v), dpk, dpq


def _varlen_setup(ctx, inputs, output):
    (q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, pos_key_layer, pos_query_layer,
     pos_key_index, pos_query_index, positions, causal, sm_scale, position_buckets, max_relative_distance,
     position_span, precision) = inputs
    o, lse = output
    ctx.save_for_backward(q, k, v, o, lse, cu_seqlens_q, cu_seqlens_k, pos_key_layer, pos_query_layer, positions)
    ctx.per_sequence_tables = pos_key_index is not None or pos_query_index is not None
    ctx.has_pos = (pos_key_layer is not None, pos_query_layer is not None)
    ctx.max_seqlens = (max_seqlen_q, max_seqlen_k)
    ctx.args = (causal, sm_scale, position_buckets, max_relative_distance, position_span, precision)


def _varlen_backward(ctx, do, dlse):
    if ctx.per_sequence_tables:
        raise NotImplementedError("backward through per-sequence position tables is not supported")
    q, k, v, o, lse, cu_seqlens_q, cu_seqlens_k, pos_key_layer, pos_query_layer, positions = ctx.saved_tensors
    dq, dk, dv, dpk, dpq = attn_varlen_bwd(do, q, k, v, o, lse, cu_seqlens_q, cu_seqlens_k, *ctx.max_seqlens,
                                           pos_key_layer, pos_query_layer, positions, *ctx.args)
    has_pk, has_pq = ctx.has_pos
    return (dq, dk, dv, None, None, None, None, dpk if has_pk else None, dpq if has_pq else None,
            None, None, None, None, None, None, None, None, None)


attn_varlen_fwd.register_autograd(_varlen_backward, setup_context=_varlen_setup)
