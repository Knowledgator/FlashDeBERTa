# Copyright 2023 BAAI
# Copyright 2026 Delyan Boychev
# Copyright 2024-2026 Knowledgator
# Kernel structure derived from FlagAttention: https://github.com/FlagOpen/FlagAttention
# Relative-position lookup planning and banded positional-gradient accumulation
# adapted from DisentangledFlash by Delyan Boychev:
#   https://github.com/delyan-boychev/disentangled-flash
#   (commits f185d9a6ae1cce42a54573208dba27fae828be98 and
#    892cdc27c98a3d21019c58e85b7e7ceae2318d5e)
# Modified by Knowledgator, 2026, for FlashDeBERTa's tensor layouts and kernels.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import math
import torch
import triton
import triton.language as tl
import functools
from typing import Tuple, Dict, Any

from .config import forward_config, backward_configs
from .launch import launch, clear_launch_cache
from .position import (position_plan, position_table, fold_matrix, fold_band_gradients,
                       clear_position_cache)

def cdiv(a, b):
    return (a + b - 1) // b

def get_mid(cu_seqlens_q, B, BLOCK_M):
    mid_batch = []
    mid_start = []
    MN = 0
    for batch in range(B):
        q_start = cu_seqlens_q[batch]
        q_end = cu_seqlens_q[batch+1]
        n_batch_blocks = (q_end-q_start+BLOCK_M-1).item()//BLOCK_M
        MN+=n_batch_blocks
        for block in range(n_batch_blocks):
            mid_start.append(q_start+(block)*BLOCK_M)
            mid_batch.append(batch)
    return (mid_batch, mid_start, MN)


@functools.lru_cache(maxsize=256)
def _get_mid_cached(cu_seqlens_tuple: Tuple[int, ...], B: int, BLOCK_M: int) -> Tuple[Tuple[int, ...], Tuple[int, ...], int]:
    """
    Cached version of get_mid that works with hashable tuple input.
    Returns tuples instead of lists for cacheability.
    """
    mid_batch = []
    mid_start = []
    MN = 0
    for batch in range(B):
        q_start = cu_seqlens_tuple[batch]
        q_end = cu_seqlens_tuple[batch + 1]
        n_batch_blocks = (q_end - q_start + BLOCK_M - 1) // BLOCK_M
        MN += n_batch_blocks
        for block in range(n_batch_blocks):
            mid_start.append(q_start + block * BLOCK_M)
            mid_batch.append(batch)
    return (tuple(mid_batch), tuple(mid_start), MN)


# Global tensor cache for mid tensors (avoid repeated CPU->GPU transfers)
_mid_tensor_cache: Dict[Tuple[Any, ...], Tuple[torch.Tensor, torch.Tensor, int]] = {}


@torch.compiler.disable
def get_mid_cached(cu_seqlens: torch.Tensor, B: int, BLOCK_M: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Get cached mid_batch and mid_start tensors.
    Caches both the computation and the GPU tensors to avoid repeated allocations.

    Note: Disabled for torch.compile as it involves CPU operations.
    """
    # Create cache key from cu_seqlens values
    cu_tuple = tuple(cu_seqlens.tolist())
    cache_key = (cu_tuple, B, BLOCK_M, device)

    if cache_key in _mid_tensor_cache:
        return _mid_tensor_cache[cache_key]

    # Compute using cached function
    mid_batch_tuple, mid_start_tuple, MN = _get_mid_cached(cu_tuple, B, BLOCK_M)

    # Create tensors on device
    mid_batch = torch.tensor(mid_batch_tuple, dtype=torch.long, device=device)
    mid_start = torch.tensor(mid_start_tuple, dtype=torch.long, device=device)

    # Cache the result (limit cache size)
    if len(_mid_tensor_cache) > 512:
        # Simple eviction: clear half the cache
        keys_to_remove = list(_mid_tensor_cache.keys())[:256]
        for k in keys_to_remove:
            del _mid_tensor_cache[k]

    _mid_tensor_cache[cache_key] = (mid_batch, mid_start, MN)
    return mid_batch, mid_start, MN


def clear_mid_cache():
    """
    Clear the mid tensor cache. Call this if memory is a concern.
    Note: This only clears the mid tensor cache, not the config caches.
    Use clear_config_cache_varlen() to clear config caches.
    """
    _mid_tensor_cache.clear()
    _get_mid_cached.cache_clear()

def clear_config_cache_varlen():
    """Clear cached launch configs and position plans."""
    clear_launch_cache()
    clear_position_cache()

def clear_all_varlen_caches():
    """Clear all caches for varlen kernels (mid tensors, launch configs and position plans)."""
    clear_mid_cache()
    clear_config_cache_varlen()


@triton.jit
def _fwd_kernel_deberta_disentangled_attention(
    Q, K, V,
    K_POS, Q_POS, POS_LUT,
    L, O,
    sm_scale,
    cu_seqlens_q, cu_seqlens_k,
    mid_batch, mid_start,
    stride_qz, stride_qh, stride_qk,
    stride_kz, stride_kh, stride_kk,
    stride_vz, stride_vh, stride_vk,
    stride_oz, stride_oh, stride_ok,
    stride_pk0, stride_pk1, stride_pk2,
    stride_pq0, stride_pq1, stride_pq2,
    H, MAX_N,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    HAS_C2P: tl.constexpr, HAS_P2C: tl.constexpr,
):
    input_dtype = Q.dtype.element_ty
    log2e: tl.constexpr = 1.4426950408889634
    qk_scale = sm_scale * log2e

    start_z = tl.program_id(0)
    off_h = tl.program_id(1)
    off_b = tl.load(mid_batch + start_z)
    off_m = tl.load(mid_start + start_z)

    q_start = tl.load(cu_seqlens_q + off_b)
    q_end = tl.load(cu_seqlens_q + off_b + 1)
    k_start = tl.load(cu_seqlens_k + off_b)
    k_end = tl.load(cu_seqlens_k + off_b + 1)

    lN = k_end - k_start
    P_SEQ = lN - (q_end - q_start)

    offs_m = off_m + tl.arange(0, BLOCK_M)
    offs_m_rel = offs_m - q_start
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_DMODEL)
    mask_m = offs_m < q_end

    q = tl.load(Q + offs_m[:, None] * stride_qz + off_h * stride_qh + offs_k[None, :] * stride_qk,
                mask=mask_m[:, None], other=0.0, cache_modifier=".cg")

    if IS_CAUSAL:
        hi = tl.maximum(tl.minimum(lN, P_SEQ + off_m - q_start + BLOCK_M), 0)
    else:
        hi = lN

    m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # Packed position scores span (tokens, H, 2 * ATT_SPAN): keep their offsets in int64.
    if HAS_C2P:
        k_pos_rows = K_POS + off_h.to(tl.int64) * stride_pk1 + offs_m.to(tl.int64) * stride_pk0
    if HAS_P2C:
        Q_POS += off_h.to(tl.int64) * stride_pq1

    k_ptrs = K + (offs_k[:, None] * stride_kk + (k_start + offs_n_base)[None, :] * stride_kz + off_h * stride_kh)
    v_ptrs = V + ((k_start + offs_n_base)[:, None] * stride_vz + off_h * stride_vh + offs_k[None, :] * stride_vk)

    for start_n in range(0, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + offs_n_base
        mask_n = offs_n < lN
        valid = mask_m[:, None] & mask_n[None, :]

        k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0, cache_modifier=".cg")
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0, cache_modifier=".cg")

        s = tl.dot(q, k)

        if HAS_C2P or HAS_P2C:
            # Slot of every (query, key) pair from the distance lookup table.
            slot = tl.load(POS_LUT + (offs_m_rel[:, None] - offs_n[None, :] + MAX_N - 1), mask=valid, other=0)
        if HAS_C2P:
            s += tl.load(k_pos_rows[:, None] + slot * stride_pk2, mask=valid, other=0.0).to(tl.float32)
        if HAS_P2C:
            s += tl.load(Q_POS + (k_start + offs_n).to(tl.int64)[None, :] * stride_pq0 + slot * stride_pq2,
                         mask=valid, other=0.0).to(tl.float32)

        s = s * qk_scale
        s = tl.where(mask_n[None, :], s, float("-inf"))

        if IS_CAUSAL:
            causal_mask = (P_SEQ + offs_m_rel[:, None]) >= offs_n[None, :]
            s = tl.where(causal_mask, s, float("-inf"))

        m_i_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(s - m_i_new[:, None])
        acc *= alpha[:, None]
        acc += tl.dot(p.to(input_dtype), v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

        k_ptrs += BLOCK_N * stride_kz
        v_ptrs += BLOCK_N * stride_vz

    # L is the natural-log LSE of the scaled scores; m_i and l_i are in base 2.
    if IS_CAUSAL:
        is_empty_line = (offs_m_rel + P_SEQ) < 0
        acc = tl.where(is_empty_line[:, None], 0.0, acc * (1.0 / l_i[:, None]))
        l_val = tl.where(is_empty_line, float("-inf"), (m_i + tl.math.log2(l_i)) / log2e)
    else:
        acc = acc * (1.0 / l_i[:, None])
        l_val = (m_i + tl.math.log2(l_i)) / log2e

    tl.store(L + offs_m * H + off_h, l_val, mask=mask_m, cache_modifier=".cg")
    tl.store(O + offs_m[:, None] * stride_oz + off_h * stride_oh + offs_k[None, :] * stride_ok,
             acc.to(input_dtype), mask=mask_m[:, None], cache_modifier=".cg")


def get_fwd_config(total_tokens, max_seqlen_q, max_seqlen_k, D, causal, disentangled=False, att_span=256):
    """
    Kernel configuration (BLOCK_M, BLOCK_N, num_stages, num_warps) for the forward pass.
    See ``ops.config`` for the defaults and the environment-variable overrides.
    """
    return forward_config(D)


def flash_attn_v2_fwd_dise(q, k, v, pos_key, pos_query, pos_lut, cu_seqlens_q, cu_seqlens_k,
                           max_seqlen_k, causal, sm_scale, config):
    """
    Forward pass of FlashAttention with DeBERTa-style disentangled relative attention
    over packed variable-length sequences.

    Args:
        q: (BM, H, D) packed queries; k, v: (BN, H, D) packed keys and values
        pos_key: C2P scores ``q @ pos_key_layer^T`` of shape (BM, H, 2 * ATT_SPAN), or None
        pos_query: P2C scores ``k @ pos_query_layer^T`` of shape (BN, H, 2 * ATT_SPAN), or None
        pos_lut: int32 slot of each distance ``i - j + max_seqlen_k - 1`` (see ``ops.position``)
        cu_seqlens_q, cu_seqlens_k: (B + 1,) cumulative sequence lengths
        causal: whether to apply causal masking
        sm_scale: softmax scale
        config: (BLOCK_M, BLOCK_N, num_stages, num_warps)

    Returns:
        o: (BM, H, D) attention output
        L: (BM, H) natural-log LSE of the scaled scores
    """
    B = len(cu_seqlens_q) - 1
    Z, H, D = q.shape

    has_c2p = pos_key is not None
    has_p2c = pos_query is not None

    o = torch.empty_like(q)
    L = torch.empty((q.shape[0], q.shape[1]), device=q.device, dtype=torch.float32)

    stride_pk = pos_key.stride() if has_c2p else (0, 0, 0)
    stride_pq = pos_query.stride() if has_p2c else (0, 0, 0)
    kwargs = dict(BLOCK_DMODEL=D, IS_CAUSAL=causal, HAS_C2P=has_c2p, HAS_P2C=has_p2c)

    def make_launch(block_m, block_n):
        # Programs are tiles of one sequence: the tile map depends on BLOCK_M.
        mid_batch, mid_start, MN = get_mid_cached(cu_seqlens_q, B, block_m, q.device)
        args = (
            q, k, v,
            pos_key, pos_query, pos_lut,
            L, o,
            sm_scale,
            cu_seqlens_q, cu_seqlens_k,
            mid_batch, mid_start,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            *stride_pk,
            *stride_pq,
            H, max_seqlen_k,
        )
        return (MN, H), args, kwargs

    with torch.cuda.device(q.device.index):
        launch(_fwd_kernel_deberta_disentangled_attention, ("varlen_fwd", D, q.dtype, causal, has_c2p, has_p2c),
               config, make_launch)

    return o, L


@triton.jit
def _bwd_preprocess_varlen(
    Out, DO, Delta,
    cu_seqlens_q, mid_batch, mid_start,
    stride_oz, stride_oh, stride_ok,
    stride_doz, stride_doh, stride_dok,
    B, H,
    BLOCK_M: tl.constexpr, D_HEAD: tl.constexpr,
):
    # grid: (MN, H), MN = total M-tiles across batch
    tile_m = tl.program_id(0)
    off_h  = tl.program_id(1)

    off_b = tl.load(mid_batch + tile_m)
    off_m = tl.load(mid_start + tile_m)  # absolute token start in Q (flattened)

    q_start = tl.load(cu_seqlens_q + off_b)
    q_end   = tl.load(cu_seqlens_q + off_b + 1)

    offs_m_base = tl.arange(0, BLOCK_M)
    offs_m_abs  = off_m + offs_m_base           # absolute in [0, BM)
    mask_m      = (offs_m_abs < q_end)

    offs_k = tl.arange(0, D_HEAD)

    o_ptrs  = Out + (offs_m_abs[:, None] * stride_oz + off_h * stride_oh + offs_k[None, :] * stride_ok)
    do_ptrs = DO  + (offs_m_abs[:, None] * stride_doz + off_h * stride_doh + offs_k[None, :] * stride_dok)

    o  = tl.load(o_ptrs,  mask=mask_m[:, None], other=0.0).to(tl.float32)
    do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
    delta = tl.sum(o * do, axis=1)  # (BLOCK_M,)

    # Delta layout matches L: (BM, H) — advance by H along rows
    delta_ptrs = Delta + (offs_m_abs * H + off_h)
    tl.store(delta_ptrs, delta, mask=mask_m)


def get_bwd_config_varlen(total_tokens_q, total_tokens_k, max_seqlen_q, max_seqlen_k, D, causal,
                          *, disentangled=True, att_span=256, dtype=torch.float16, max_shared_memory=None):
    """
    Kernel configurations for the backward pass: (dK/dV config, dQ config).
    See ``ops.config`` for the defaults and the environment-variable overrides.
    """
    return backward_configs(D)


@triton.jit
def _bwd_kv_dise_kernel_varlen(
    Q, K, V, K_POS, Q_POS, POS_LUT, sm_scale, DO,
    DK, DV, P2C_BAND,
    L, Delta,
    cu_seqlens_q, cu_seqlens_k, mid_batch_n, mid_start_n,
    stride_qz, stride_qh, stride_qk,
    stride_kz, stride_kh, stride_kk,
    stride_vz, stride_vh, stride_vk,
    stride_doz, stride_doh, stride_dok,
    stride_dkz, stride_dkh, stride_dkk,
    stride_dvz, stride_dvh, stride_dvk,
    stride_pk0, stride_pk1, stride_pk2,
    stride_pq0, stride_pq1, stride_pq2,
    stride_band_h,
    H, MAX_N, BAND_LOW, NUM_COLUMNS,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    HAS_C2P: tl.constexpr, HAS_P2C: tl.constexpr,
):
    input_dtype = Q.dtype.element_ty
    log2e: tl.constexpr = 1.4426950408889634
    qk_scale = sm_scale * log2e

    tile_n = tl.program_id(0)
    off_h  = tl.program_id(1)

    off_b   = tl.load(mid_batch_n + tile_n)
    n_start = tl.load(mid_start_n + tile_n)   # absolute start index in K/V (flattened)

    q_start = tl.load(cu_seqlens_q + off_b)
    q_end   = tl.load(cu_seqlens_q + off_b + 1)
    k_start = tl.load(cu_seqlens_k + off_b)
    k_end   = tl.load(cu_seqlens_k + off_b + 1)

    lM = q_end - q_start
    P_SEQ = (k_end - k_start) - lM

    offs_n_abs = n_start + tl.arange(0, BLOCK_N)
    offs_n_rel = offs_n_abs - k_start
    mask_n     = offs_n_abs < k_end
    offs_k     = tl.arange(0, BLOCK_DMODEL)
    offs_m_base = tl.arange(0, BLOCK_M)

    k = tl.load(K + offs_n_abs[:, None] * stride_kz + off_h * stride_kh + offs_k[None, :] * stride_kk,
                mask=mask_n[:, None], other=0.0)
    v = tl.load(V + offs_n_abs[:, None] * stride_vz + off_h * stride_vh + offs_k[None, :] * stride_vk,
                mask=mask_n[:, None], other=0.0)

    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    # P2C gradients of distances saturated into the first/last slot, per key row.
    low_sum = tl.zeros([BLOCK_N], dtype=tl.float32)
    high_sum = tl.zeros([BLOCK_N], dtype=tl.float32)
    # Packed position scores and bands span all tokens: keep their offsets in int64.
    if HAS_C2P:
        K_POS += off_h.to(tl.int64) * stride_pk1
    if HAS_P2C:
        q_pos_rows = Q_POS + off_h.to(tl.int64) * stride_pq1 + offs_n_abs.to(tl.int64) * stride_pq0
        band_rows = P2C_BAND + off_h.to(tl.int64) * stride_band_h + offs_n_abs.to(tl.int64) * NUM_COLUMNS

    if CAUSAL:
        lo = tl.maximum(n_start - k_start - P_SEQ, 0)
        lo = (lo // BLOCK_M) * BLOCK_M
    else:
        lo = 0

    for start_m in range(lo, lM, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m_rel = start_m + offs_m_base
        offs_m_abs = q_start + offs_m_rel
        mask_m = offs_m_rel < lM
        valid = mask_m[:, None] & mask_n[None, :]
        if CAUSAL:
            valid = valid & ((P_SEQ + offs_m_rel[:, None]) >= offs_n_rel[None, :])

        q  = tl.load(Q + offs_m_abs[:, None] * stride_qz + off_h * stride_qh + offs_k[None, :] * stride_qk,
                     mask=mask_m[:, None], other=0.0)
        do = tl.load(DO + offs_m_abs[:, None] * stride_doz + off_h * stride_doh + offs_k[None, :] * stride_dok,
                     mask=mask_m[:, None], other=0.0)
        l = tl.load(L + offs_m_abs * H + off_h, mask=mask_m, other=0.0)
        delta = tl.load(Delta + offs_m_abs * H + off_h, mask=mask_m, other=0.0)

        s = tl.dot(q, tl.trans(k))

        distance = offs_m_rel[:, None] - offs_n_rel[None, :] + MAX_N - 1
        if HAS_C2P or HAS_P2C:
            slot = tl.load(POS_LUT + distance, mask=valid, other=0)
        if HAS_C2P:
            s += tl.load(K_POS + offs_m_abs.to(tl.int64)[:, None] * stride_pk0 + slot * stride_pk2,
                         mask=valid, other=0.0).to(tl.float32)
        if HAS_P2C:
            s += tl.load(q_pos_rows[None, :] + slot * stride_pq2, mask=valid, other=0.0).to(tl.float32)

        p = tl.math.exp2(s * qk_scale - l[:, None] * log2e)
        p = tl.where(valid, p, 0.0)

        dv += tl.dot(tl.trans(p.to(input_dtype)), do)

        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None]) * sm_scale

        dk += tl.dot(tl.trans(ds.to(input_dtype)), q)

        if HAS_P2C:
            # Each (key row, distance) cell has one writer: plain stores, no atomics.
            column = distance - BAND_LOW
            inside = valid & (column > 0) & (column < NUM_COLUMNS - 1)
            tl.store(band_rows[None, :] + column, ds.to(input_dtype), mask=inside)
            low_sum += tl.sum(tl.where(column <= 0, ds, 0.0), axis=0)
            high_sum += tl.sum(tl.where(column >= NUM_COLUMNS - 1, ds, 0.0), axis=0)

    tl.store(DK + offs_n_abs[:, None] * stride_dkz + off_h * stride_dkh + offs_k[None, :] * stride_dkk,
             dk.to(input_dtype), mask=mask_n[:, None])
    tl.store(DV + offs_n_abs[:, None] * stride_dvz + off_h * stride_dvh + offs_k[None, :] * stride_dvk,
             dv.to(input_dtype), mask=mask_n[:, None])
    if HAS_P2C:
        tl.store(band_rows, low_sum.to(input_dtype), mask=mask_n)
        tl.store(band_rows + NUM_COLUMNS - 1, high_sum.to(input_dtype), mask=mask_n)


@triton.jit
def _bwd_q_dise_kernel_varlen(
    Q, K, V, K_POS, Q_POS, POS_LUT, sm_scale, DO,
    DQ, C2P_BAND,
    L, Delta,
    cu_seqlens_q, cu_seqlens_k, mid_batch_m, mid_start_m,
    stride_qz, stride_qh, stride_qk,
    stride_kz, stride_kh, stride_kk,
    stride_vz, stride_vh, stride_vk,
    stride_doz, stride_doh, stride_dok,
    stride_dqz, stride_dqh, stride_dqk,
    stride_pk0, stride_pk1, stride_pk2,
    stride_pq0, stride_pq1, stride_pq2,
    stride_band_h,
    H, MAX_N, BAND_LOW, NUM_COLUMNS,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr, HAS_C2P: tl.constexpr, HAS_P2C: tl.constexpr,
):
    input_dtype = Q.dtype.element_ty
    log2e: tl.constexpr = 1.4426950408889634
    qk_scale = sm_scale * log2e

    tile_m = tl.program_id(0)
    off_h  = tl.program_id(1)

    off_b = tl.load(mid_batch_m + tile_m)
    off_m = tl.load(mid_start_m + tile_m)   # absolute q start for this tile

    q_start = tl.load(cu_seqlens_q + off_b)
    q_end   = tl.load(cu_seqlens_q + off_b + 1)
    k_start = tl.load(cu_seqlens_k + off_b)
    k_end   = tl.load(cu_seqlens_k + off_b + 1)

    lN = k_end - k_start
    P_SEQ = lN - (q_end - q_start)

    offs_m_abs = off_m + tl.arange(0, BLOCK_M)
    offs_m_rel = offs_m_abs - q_start
    mask_m     = offs_m_abs < q_end
    offs_k     = tl.arange(0, BLOCK_DMODEL)
    offs_n_base = tl.arange(0, BLOCK_N)

    q  = tl.load(Q + offs_m_abs[:, None] * stride_qz + off_h * stride_qh + offs_k[None, :] * stride_qk,
                 mask=mask_m[:, None], other=0.0)
    do = tl.load(DO + offs_m_abs[:, None] * stride_doz + off_h * stride_doh + offs_k[None, :] * stride_dok,
                 mask=mask_m[:, None], other=0.0)
    l = tl.load(L + offs_m_abs * H + off_h, mask=mask_m, other=0.0)
    delta = tl.load(Delta + offs_m_abs * H + off_h, mask=mask_m, other=0.0)

    if CAUSAL:
        hi = tl.maximum(tl.minimum(lN, P_SEQ + off_m - q_start + BLOCK_M), 0)
    else:
        hi = lN

    dq = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # C2P gradients of distances saturated into the first/last slot, per query row.
    low_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
    high_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
    # Packed position scores and bands span all tokens: keep their offsets in int64.
    if HAS_C2P:
        k_pos_rows = K_POS + off_h.to(tl.int64) * stride_pk1 + offs_m_abs.to(tl.int64) * stride_pk0
        band_rows = C2P_BAND + off_h.to(tl.int64) * stride_band_h + offs_m_abs.to(tl.int64) * NUM_COLUMNS
    if HAS_P2C:
        Q_POS += off_h.to(tl.int64) * stride_pq1

    k_ptrs = K + ((k_start + offs_n_base)[:, None] * stride_kz + off_h * stride_kh + offs_k[None, :] * stride_kk)
    v_ptrs = V + ((k_start + offs_n_base)[:, None] * stride_vz + off_h * stride_vh + offs_k[None, :] * stride_vk)

    for start_n in range(0, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n_rel = start_n + offs_n_base
        mask_n = offs_n_rel < lN
        valid = mask_m[:, None] & mask_n[None, :]
        if CAUSAL:
            valid = valid & ((P_SEQ + offs_m_rel[:, None]) >= offs_n_rel[None, :])

        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        s = tl.dot(q, tl.trans(k))

        distance = offs_m_rel[:, None] - offs_n_rel[None, :] + MAX_N - 1
        if HAS_C2P or HAS_P2C:
            slot = tl.load(POS_LUT + distance, mask=valid, other=0)
        if HAS_C2P:
            s += tl.load(k_pos_rows[:, None] + slot * stride_pk2, mask=valid, other=0.0).to(tl.float32)
        if HAS_P2C:
            s += tl.load(Q_POS + (k_start + offs_n_rel).to(tl.int64)[None, :] * stride_pq0 + slot * stride_pq2,
                         mask=valid, other=0.0).to(tl.float32)

        p = tl.math.exp2(s * qk_scale - l[:, None] * log2e)
        p = tl.where(valid, p, 0.0)

        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None]) * sm_scale

        dq += tl.dot(ds.to(input_dtype), k)

        if HAS_C2P:
            # Each (query row, distance) cell has one writer: plain stores, no atomics.
            # Columns run in reverse distance order so stores ascend along the key axis.
            column = NUM_COLUMNS - 1 - (distance - BAND_LOW)
            inside = valid & (column > 0) & (column < NUM_COLUMNS - 1)
            tl.store(band_rows[:, None] + column, ds.to(input_dtype), mask=inside)
            high_sum += tl.sum(tl.where(column <= 0, ds, 0.0), axis=1)
            low_sum += tl.sum(tl.where(column >= NUM_COLUMNS - 1, ds, 0.0), axis=1)

        k_ptrs += BLOCK_N * stride_kz
        v_ptrs += BLOCK_N * stride_vz

    tl.store(DQ + offs_m_abs[:, None] * stride_dqz + off_h * stride_dqh + offs_k[None, :] * stride_dqk,
             dq.to(input_dtype), mask=mask_m[:, None])
    if HAS_C2P:
        tl.store(band_rows, high_sum.to(input_dtype), mask=mask_m)
        tl.store(band_rows + NUM_COLUMNS - 1, low_sum.to(input_dtype), mask=mask_m)


def flash_attn_v2_bwd_dise_varlen(
    o, do, q, k, v, k_pos, q_pos, L,
    cu_seqlens_q, cu_seqlens_k, max_seqlen_k,
    causal, sm_scale,
    pos_lut, band_low, num_columns, kv_config, q_config,
):
    """
    Backward pass over packed variable-length sequences.

    Returns dq, dk, dv and the C2P/P2C score gradients per distance column,
    c2p_band (H, BM, num_columns) and p2c_band (H, BN, num_columns), or None.
    """
    device = q.device
    BM, H, D = q.shape
    BN = k.shape[0]
    B = cu_seqlens_q.numel() - 1

    has_c2p = k_pos is not None
    has_p2c = q_pos is not None

    # Δ: (BM, H), L: (BM, H)  — matches kernel pointer math
    delta = torch.empty((BM, H), device=device, dtype=torch.float32)
    PRE_BLOCK_M = 64
    mid_m_batch, mid_m_start, MN = get_mid_cached(cu_seqlens_q, B, PRE_BLOCK_M, device)
    with torch.cuda.device(q.device.index):
        _bwd_preprocess_varlen[(MN, H)](
            o, do, delta,
            cu_seqlens_q, mid_m_batch, mid_m_start,
            o.stride(0), o.stride(1), o.stride(2),
            do.stride(0), do.stride(1), do.stride(2),
            B, H,
            BLOCK_M=PRE_BLOCK_M, D_HEAD=D,
        )

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    # Columns a row never reaches must read as zero.
    c2p_band = torch.zeros((H, BM, num_columns), device=device, dtype=q.dtype) if has_c2p else None
    p2c_band = torch.zeros((H, BN, num_columns), device=device, dtype=q.dtype) if has_p2c else None

    stride_pk = k_pos.stride() if has_c2p else (0, 0, 0)
    stride_pq = q_pos.stride() if has_p2c else (0, 0, 0)
    kwargs = dict(BLOCK_DMODEL=D, CAUSAL=causal, HAS_C2P=has_c2p, HAS_P2C=has_p2c)
    key = (D, q.dtype, causal, has_c2p, has_p2c)

    def make_kv_launch(block_m, block_n):
        mid_n_batch, mid_n_start, NK = get_mid_cached(cu_seqlens_k, B, block_n, device)
        args = (
            q, k, v, k_pos, q_pos, pos_lut, sm_scale, do,
            dk, dv, p2c_band,
            L, delta,
            cu_seqlens_q, cu_seqlens_k, mid_n_batch, mid_n_start,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            do.stride(0), do.stride(1), do.stride(2),
            dk.stride(0), dk.stride(1), dk.stride(2),
            dv.stride(0), dv.stride(1), dv.stride(2),
            *stride_pk,
            *stride_pq,
            p2c_band.stride(0) if has_p2c else 0,
            H, max_seqlen_k, band_low, num_columns,
        )
        return (NK, H), args, kwargs

    def make_q_launch(block_m, block_n):
        mid_batch, mid_start, MN = get_mid_cached(cu_seqlens_q, B, block_m, device)
        args = (
            q, k, v, k_pos, q_pos, pos_lut, sm_scale, do,
            dq, c2p_band,
            L, delta,
            cu_seqlens_q, cu_seqlens_k, mid_batch, mid_start,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            do.stride(0), do.stride(1), do.stride(2),
            dq.stride(0), dq.stride(1), dq.stride(2),
            *stride_pk,
            *stride_pq,
            c2p_band.stride(0) if has_c2p else 0,
            H, max_seqlen_k, band_low, num_columns,
        )
        return (MN, H), args, kwargs

    with torch.cuda.device(q.device.index):
        launch(_bwd_kv_dise_kernel_varlen, ("varlen_bwd_kv",) + key, kv_config, make_kv_launch)
        launch(_bwd_q_dise_kernel_varlen, ("varlen_bwd_q",) + key, q_config, make_q_launch)

    return dq, dk, dv, c2p_band, p2c_band


def _position_scores(rows, table):
    """(T, H, D) rows @ (H, R, D) table^T -> (T, H, R) view of head-major storage."""
    return torch.bmm(rows.transpose(0, 1), table.transpose(1, 2)).transpose(0, 1)


class FlashAttentionDisentangledVarlen(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, pos_key_layer, pos_query_layer,
                cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k,
                causal, sm_scale, position_buckets, max_relative_distance):

        BM, H, D = q.shape
        assert (k.shape[1], v.shape[1]) == (H, H) and (k.shape[2], v.shape[2]) == (D, D)

        ATT_SPAN = position_buckets if position_buckets > 0 else max_relative_distance
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(D)
        plan = position_plan(max_seqlen_q, max_seqlen_k, position_buckets, max_relative_distance, ATT_SPAN, q.device)

        # C2P/P2C scores are plain GEMMs; the kernel gathers them per (query, key) pair.
        pos_key = pos_query = None
        if pos_key_layer is not None:
            pos_key = _position_scores(q, position_table(pos_key_layer, H, ATT_SPAN))
        if pos_query_layer is not None:
            pos_query = _position_scores(k, position_table(pos_query_layer, H, ATT_SPAN))

        config = get_fwd_config(BM, max_seqlen_q, max_seqlen_k, D, causal, disentangled=True, att_span=ATT_SPAN)
        o, L = flash_attn_v2_fwd_dise(
            q, k, v, pos_key, pos_query, plan.lut, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_k, causal, sm_scale, config,
        )

        # Position scores are recomputed in backward instead of being kept alive.
        ctx.save_for_backward(q, k, v, pos_key_layer, pos_query_layer, o, L, cu_seqlens_q, cu_seqlens_k)
        ctx.sm_scale = sm_scale
        ctx.causal = causal
        ctx.position_buckets = position_buckets
        ctx.max_relative_distance = max_relative_distance
        ctx.ATT_SPAN = ATT_SPAN
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        return o

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, pos_key_layer, pos_query_layer, o, L, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
        ATT_SPAN = ctx.ATT_SPAN
        BM, H, D = q.shape
        BN = k.shape[0]
        plan = position_plan(ctx.max_seqlen_q, ctx.max_seqlen_k, ctx.position_buckets,
                             ctx.max_relative_distance, ATT_SPAN, q.device)

        pos_key = pos_query = None
        if pos_key_layer is not None:
            key_table = position_table(pos_key_layer, H, ATT_SPAN)
            pos_key = _position_scores(q, key_table)
        if pos_query_layer is not None:
            query_table = position_table(pos_query_layer, H, ATT_SPAN)
            pos_query = _position_scores(k, query_table)

        kv_config, q_config = get_bwd_config_varlen(
            BM, BN, ctx.max_seqlen_q, ctx.max_seqlen_k, D, ctx.causal,
            disentangled=True, att_span=ATT_SPAN, dtype=q.dtype
        )

        dq, dk, dv, c2p_band, p2c_band = flash_attn_v2_bwd_dise_varlen(
            o, grad_output, q, k, v, pos_key, pos_query, L,
            cu_seqlens_q, cu_seqlens_k, ctx.max_seqlen_k,
            ctx.causal, ctx.sm_scale,
            plan.lut, plan.band_low, plan.num_columns, kv_config, q_config,
        )

        d_pos_key_layer = d_pos_query_layer = None
        if c2p_band is not None or p2c_band is not None:
            fold = fold_matrix(ctx.max_seqlen_q, ctx.max_seqlen_k, ctx.position_buckets,
                               ctx.max_relative_distance, ATT_SPAN, q.device, q.dtype)
        if c2p_band is not None:
            grad_rows, grad_table = fold_band_gradients(c2p_band, q.transpose(0, 1), key_table, fold.flip(0))
            dq = dq + grad_rows.transpose(0, 1)
            d_pos_key_layer = grad_table.view(pos_key_layer.shape)
        if p2c_band is not None:
            grad_rows, grad_table = fold_band_gradients(p2c_band, k.transpose(0, 1), query_table, fold)
            dk = dk + grad_rows.transpose(0, 1)
            d_pos_query_layer = grad_table.view(pos_query_layer.shape)

        # Match forward signature: (q, k, v, pos_key_layer, pos_query_layer, cu_seqlens_q, cu_seqlens_k,
        # max_seqlen_q, max_seqlen_k, causal, sm_scale, position_buckets, max_relative_distance)
        return dq, dk, dv, d_pos_key_layer, d_pos_query_layer, None, None, None, None, None, None, None, None


def flash_attention_with_disentangled_varlen(
    q, k, v, pos_key_layer, pos_query_layer, cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k, causal=False, sm_scale=None,
    position_buckets=0, max_relative_distance=0,
):
    """
    Flash attention with DeBERTa-style disentangled attention for variable-length sequences.

    Args:
        q:  (BM, H, D)   flattened queries
        k:  (BN, H, D)
        v:  (BN, H, D)
        pos_key_layer: (1, H, 2*ATT_SPAN, D) projected relative embeddings for C2P, or None
        pos_query_layer: (1, H, 2*ATT_SPAN, D) projected relative embeddings for P2C, or None
        cu_seqlens_q / cu_seqlens_k: int32/64, shape (B+1)
        max_seqlen_q / max_seqlen_k: int
        causal: whether to apply causal masking
        sm_scale: softmax scale (default: 1/sqrt(D))
        position_buckets: number of relative position buckets
        max_relative_distance: maximum relative distance for bucketing

    Returns:
        Output tensor of shape (BM, H, D)
    """
    return FlashAttentionDisentangledVarlen.apply(
        q, k, v, pos_key_layer, pos_query_layer, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k, causal, sm_scale,
        position_buckets, max_relative_distance
    )
