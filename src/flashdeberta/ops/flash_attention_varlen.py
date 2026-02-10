import os
import math
import torch
import triton
import warnings
import triton.language as tl
import functools
from typing import Tuple, Dict, Any

def calculate_shared_memory_usage_varlen(BLOCK_M, BLOCK_N, BLOCK_DMODEL, num_stages, dtype, 
                                         has_c2p=False, has_p2c=False, ATT_SPAN=0):
    """
    Calculate the shared memory requirements for Flash Attention with disentangled attention
    for variable-length sequences.
    
    Args:
        BLOCK_M: Block size for query sequence dimension
        BLOCK_N: Block size for key sequence dimension
        BLOCK_DMODEL: Head dimension size
        num_stages: Number of pipeline stages
        dtype: Data type (torch.float16, torch.float32, etc.)
        has_c2p: Whether content-to-position bias is used
        has_p2c: Whether position-to-content bias is used
        ATT_SPAN: Attention span for relative position
    
    Returns:
        The estimated shared memory usage in bytes
    """
    # Determine byte size based on data type
    if dtype == torch.float16:
        dtype_size = 2
    elif dtype == torch.float32:
        dtype_size = 4
    else:
        dtype_size = 2  # Default to float16 size for other types

    # Core tensors that are always used
    q_size = BLOCK_M * BLOCK_DMODEL * dtype_size
    k_size = BLOCK_N * BLOCK_DMODEL * dtype_size
    v_size = BLOCK_N * BLOCK_DMODEL * dtype_size
    
    # Memory for attention scores and accumulator
    attn_matrix_size = BLOCK_M * BLOCK_N * dtype_size
    accumulator_size = BLOCK_M * BLOCK_DMODEL * dtype_size
    
    # Position embedding memory if needed
    pos_memory = 0
    if has_c2p:
        pos_memory += BLOCK_M * 2 * ATT_SPAN * dtype_size
    if has_p2c:
        pos_memory += BLOCK_N * 2 * ATT_SPAN * dtype_size
    
    # Additional buffers for intermediate calculations
    # This includes arrays for relative positions, bucket indices, etc.
    additional_buffers = BLOCK_M * BLOCK_N * 4  # For relative position indices and calculations
    
    # For variable length, we need additional bookkeeping arrays
    varlen_buffers = (BLOCK_M + BLOCK_N) * 4  # For sequence boundary tracking
    
    # Mid batch and mid start arrays (for batch mapping)
    mid_batch_memory = BLOCK_M * 4  # Int32 array
    
    # Total memory per stage including variable length overhead
    memory_per_stage = q_size + k_size + v_size + attn_matrix_size + pos_memory + additional_buffers + varlen_buffers
    
    # Total shared memory including all pipeline stages and bookkeeping
    total_shared_memory = num_stages * memory_per_stage + accumulator_size + mid_batch_memory
    
    return total_shared_memory // 2

def calculate_shared_memory_usage_varlen_bwd(
    BLOCK_M,
    BLOCK_N,
    BLOCK_DMODEL,
    num_stages,
    dtype=torch.float16,
    *,
    has_c2p=False,
    has_p2c=False,
    ATT_SPAN=0,
    store_lse=True,
    recompute_probs=True,
    accum_dq=True,
    accum_dkv=True,
):
    """
    Rough shared-memory estimator for FlashAttention v2 **backward** (varlen) with optional
    DeBERTa-style disentangled biases.

    We estimate the larger of the two backward passes:
      - KV pass (accumulates dK/dV, atomics to DKPOS/DQPOS)
      - Q  pass (accumulates dQ)

    Each per-stage tile may keep:
      - q, k, v tiles
      - do tile (and o only in a separate preprocess kernel; not counted here)
      - MxN probs/dp/ds scratch if recomputing probabilities
      - per-row softmax book-keeping (L/log-sum-exp and Delta)
      - partial accumulators for dq / dk / dv
      - positional tiles for c2p / p2c (heuristic, span-limited)
      - small varlen scratch (cu_seqlens/mid arrays) per tile
      - a small safety pad

    Returns:
        Estimated shared memory usage in **bytes** for the worst of the two passes.
    """
    # dtype size
    if dtype == torch.float16 or dtype == torch.bfloat16:
        t_sz = 2
    elif dtype == torch.float32:
        t_sz = 4
    else:
        t_sz = 2

    # Core tiles
    q_size  = BLOCK_M * BLOCK_DMODEL * t_sz
    k_size  = BLOCK_N * BLOCK_DMODEL * t_sz
    v_size  = BLOCK_N * BLOCK_DMODEL * t_sz
    do_size = BLOCK_M * BLOCK_DMODEL * t_sz  # needed in both passes

    # Recomputed probabilities / dp/ds tile
    probs_size = (BLOCK_M * BLOCK_N * t_sz) if recompute_probs else 0

    lse_size   = (BLOCK_M * 4) if store_lse else (BLOCK_M * 4)  # still brought for each row
    delta_size = (BLOCK_M * 4)

    # Partial accumulators
    dq_acc = BLOCK_M * BLOCK_DMODEL * t_sz if accum_dq  else 0
    dk_acc = BLOCK_N * BLOCK_DMODEL * t_sz if accum_dkv else 0
    dv_acc = BLOCK_N * BLOCK_DMODEL * t_sz if accum_dkv else 0

    pos_mem = 0
    if has_c2p:
        # C2P indexed by query-row: need per-M-row span
        pos_mem += BLOCK_M * (2 * ATT_SPAN) * t_sz
    if has_p2c:
        # P2C indexed by key-row: need per-N-row span
        pos_mem += BLOCK_N * (2 * ATT_SPAN) * t_sz

    varlen_buffers = (BLOCK_M + BLOCK_N) * 4          # indices/limits per tile
    mid_scratch_m  = BLOCK_M * 4                       # mid for m-tiles
    mid_scratch_n  = BLOCK_N * 4                       # mid for n-tiles

    # Misc safety pad for masks/scales/loop temps
    misc = 8 * 1024  # 8KB

    per_stage_kv = (
        q_size + k_size + v_size + do_size +
        probs_size + lse_size + delta_size +
        dk_acc + dv_acc +
        pos_mem +
        varlen_buffers + mid_scratch_n +
        misc
    )

    per_stage_q = (
        q_size + k_size + v_size + do_size +
        probs_size + lse_size + delta_size +
        dq_acc +
        pos_mem +
        varlen_buffers + mid_scratch_m +
        misc
    )

    total = num_stages * max(per_stage_kv, per_stage_q)

    return total // 2

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
    """Clear the configuration caches for varlen kernels."""
    _get_fwd_config_cached.cache_clear()
    _get_bwd_config_varlen_cached.cache_clear()

def clear_all_varlen_caches():
    """Clear all caches for varlen kernels (both mid tensors and configs)."""
    clear_mid_cache()
    clear_config_cache_varlen()


@triton.jit
def _fwd_kernel_deberta_disentangled_attention(
    Q, K, V,
    K_POS, Q_POS,
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
    B, H, M, N,    
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    HAS_C2P: tl.constexpr, HAS_P2C: tl.constexpr,
    ATT_SPAN: tl.constexpr,
    NUM_BUCKETS: tl.constexpr, MAX_DISTANCE: tl.constexpr
):
    input_dtype = Q.dtype.element_ty

    start_z = tl.program_id(0)
    off_h = tl.program_id(1)
    off_b = tl.load(mid_batch + start_z)
    off_m = tl.load(mid_start + start_z)

    q_start = tl.load(cu_seqlens_q + off_b)
    q_end = tl.load(cu_seqlens_q + off_b + 1)
    k_start = tl.load(cu_seqlens_k + off_b)
    k_end = tl.load(cu_seqlens_k + off_b + 1)

    lM = q_end - q_start
    lN = k_end - k_start
    P_SEQ = lM - lN

    log2e: tl.constexpr = 1.4426950408889634

    L += off_m * H + off_h

    offs_m_base = tl.arange(0, BLOCK_M)
    offs_m = offs_m_base + off_m
    offs_m_relative = offs_m - q_start
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_DMODEL)
    
    q_ptrs = Q + (offs_m[:, None] * stride_qz + off_h * stride_qh + offs_k[None, :] * stride_qk)
    o_ptrs = O + (offs_m[:, None] * stride_oz + off_h * stride_oh + offs_k[None, :] * stride_ok)
    l_ptrs = L + offs_m_base * H

    mask_m = offs_m < q_end
    q = tl.load(q_ptrs, mask=mask_m[:, None], cache_modifier=".cg")

    if IS_CAUSAL:
        hi = tl.minimum(lN, P_SEQ + (off_m + 1) * BLOCK_M)
        if lM > lN:
            hi = tl.maximum(0, hi)
    else:
        hi = lN

    m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    offs_n_init = k_start + offs_n_base
    k_ptrs = K + (offs_k[:, None] * stride_kk + offs_n_init[None, :] * stride_kz + off_h * stride_kh)
    v_ptrs = V + (offs_n_init[:, None] * stride_vz + off_h * stride_vh + offs_k[None, :] * stride_vk)


    if HAS_C2P:
        k_pos_ptrs = K_POS + (offs_m[:, None] * stride_pk0 + off_h * stride_pk1)

    for start_n in range(0, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + offs_n_base

        mask_n = offs_n < lN
        k = tl.load(k_ptrs, mask=mask_n[None, :], cache_modifier=".cg")
        v = tl.load(v_ptrs, mask=mask_n[:, None], cache_modifier=".cg")

        s = tl.zeros([BLOCK_M, BLOCK_N], dtype=input_dtype)
        s += tl.dot(q, k) * sm_scale

        relative_positions = offs_m_relative[:, None] - offs_n[None, :]
        sign = tl.where(relative_positions > 0.0, 1.0, tl.where(relative_positions < 0.0, -1.0, 0.0))
        mid_val = NUM_BUCKETS // 2
        abs_relative = tl.abs(relative_positions)
        condition = (relative_positions < mid_val) & (relative_positions > -mid_val)
        abs_pos = tl.where(condition, mid_val - 1.0, abs_relative)
        log_numer = tl.log(abs_pos / mid_val)
        log_denom = tl.log((MAX_DISTANCE - 1) / mid_val)
        log_scaled = log_numer / log_denom * (mid_val - 1.0)
        log_pos = tl.ceil(log_scaled) + mid_val
        bucket_pos = tl.where(abs_pos <= mid_val, relative_positions, log_pos * sign)

        if HAS_C2P:
            c2p_index = tl.minimum(tl.maximum(bucket_pos + ATT_SPAN, 0), 2 * ATT_SPAN - 1).to(tl.int32)
            k_pos_ptrs_ = k_pos_ptrs + c2p_index * stride_pk2
            c2p_bias = tl.load(k_pos_ptrs_, mask=mask_m[:, None] & (c2p_index < 2 * ATT_SPAN), other=0.0)
            s += c2p_bias * sm_scale

        if HAS_P2C:
            offs_n_abs = k_start + offs_n
            current_q_pos_ptrs = Q_POS + (offs_n_abs[:, None] * stride_pq0 + off_h * stride_pq1)
            p2c_index = tl.minimum(tl.maximum(bucket_pos + ATT_SPAN, 0), 2 * ATT_SPAN - 1).to(tl.int32).trans(1, 0)
            q_pos_ptrs_ = current_q_pos_ptrs + p2c_index * stride_pq2
            p2c_bias = tl.load(q_pos_ptrs_, mask=mask_n[:, None] & (p2c_index < 2 * ATT_SPAN), other=0.0).trans(1, 0)
            s += p2c_bias * sm_scale

        s = tl.where(mask_n[None, :], s, float("-inf"))

        if IS_CAUSAL:
            causal_mask = (P_SEQ + offs_m[:, None]) >= offs_n[None, :]
            s = tl.where(causal_mask, s, float("-inf"))

        m_i_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.math.exp2((m_i - m_i_new) * log2e)
        p = tl.math.exp2((s - m_i_new[:, None]) * log2e)
        acc *= alpha[:, None]
        acc += tl.dot(p.to(q.dtype), v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

        k_ptrs += BLOCK_N * stride_kz
        v_ptrs += BLOCK_N * stride_vz

    if IS_CAUSAL and lM > lN:
        is_empty_line = (offs_m_relative + P_SEQ) < 0
        acc = tl.where(is_empty_line[:, None], 0.0, acc * (1.0 / l_i[:, None]))
        l_val = tl.where(is_empty_line, float("-inf"), m_i + tl.log(l_i))
    else:
        acc = acc * (1.0 / l_i[:, None])
        l_val = m_i + tl.log(l_i)

    tl.store(l_ptrs, l_val, mask=mask_m, cache_modifier=".cg")
    tl.store(o_ptrs, acc.to(input_dtype), mask=mask_m[:, None], cache_modifier=".cg")


@functools.lru_cache(maxsize=128)
def _get_fwd_config_cached(total_tokens, max_seqlen_q, max_seqlen_k, D, causal, disentangled, att_span):
    """
    Cached version of configuration computation for variable-length forward pass.
    """
    if all(key in os.environ for key in ['FLASHDEBERTA_FWD_BLOCK_M', 'FLASHDEBERTA_FWD_BLOCK_N',
                                          'FLASHDEBERTA_FWD_NUM_STAGES', 'FLASHDEBERTA_FWD_NUM_WARPS']):
        return (
            int(os.environ['FLASHDEBERTA_FWD_BLOCK_M']),
            int(os.environ['FLASHDEBERTA_FWD_BLOCK_N']),
            int(os.environ['FLASHDEBERTA_FWD_NUM_STAGES']),
            int(os.environ['FLASHDEBERTA_FWD_NUM_WARPS'])
        )

    capability_map = {
         (7,0): 96000,
         (7,2): 96000,
         (7,5): 64000,
         (8,0): 163000,
         (8,6): 99000,
         (8,7): 163000,
         (8,9): 99000,
         (9,0): 227000,
         }

    capability = torch.cuda.get_device_capability()
    device_property = torch.cuda.get_device_properties()
    if hasattr(device_property,"shared_memory_per_block_optin"):
        shared_mem_per_block = device_property.shared_memory_per_block_optin
    elif capability in list(capability_map.keys()):
        shared_mem_per_block = capability_map[capability]
    elif capability[0] >= 8:
        shared_mem_per_block = 99000
    else:
        shared_mem_per_block = 48000

    max_shared_memory = shared_mem_per_block - 2000

    if capability[0] >= 8:
        if max_seqlen_q <= 64:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 16, 16, 2, 4
        elif max_seqlen_q <= 128:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 32, 32, 2, 4
        elif max_seqlen_q <= 256:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 32, 2, 4
        elif not causal:
            if D <= 64:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 3, 4
            else:
                if max_seqlen_q <= 1024:
                    BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 3, 4
                else:
                    BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 3, 8
        else:
            if D <= 64:
                if disentangled:
                    BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 3, 4
                else:
                    BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 4, 4
            else:
                if max_seqlen_q <= 1024:
                    BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 2, 4
                else:
                    BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 3, 8
    elif capability[0] == 7:
        if not causal:
            if D <= 64:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 3, 4
            else:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 2, 4
        else:
            if D <= 64:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 3, 4
            else:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 2, 4
    else:
        BLOCK_M, BLOCK_N, num_stages, num_warps = 16, 16, 1, 4

    avg_seq_len = total_tokens / max(1, torch.cuda.device_count())
    if avg_seq_len < 256:
        BLOCK_M = min(BLOCK_M, 64)
        BLOCK_N = min(BLOCK_N, 32)
        num_stages = max(1, num_stages - 1)

    has_pos = disentangled
    ATT_SPAN = att_span if has_pos else 0

    dtype = torch.float16

    shared_mem_usage = calculate_shared_memory_usage_varlen(
        BLOCK_M, BLOCK_N, D, num_stages, dtype,
        has_c2p=has_pos, has_p2c=has_pos, ATT_SPAN=ATT_SPAN
    )

    while shared_mem_usage > max_shared_memory and (BLOCK_M > 16 or BLOCK_N > 16 or num_stages > 1):
        if num_stages > 1:
            num_stages -= 1
        else:
            BLOCK_M //= 2
            BLOCK_N //= 2

        shared_mem_usage = calculate_shared_memory_usage_varlen(
            BLOCK_M, BLOCK_N, D, num_stages, dtype,
            has_c2p=has_pos, has_p2c=has_pos, ATT_SPAN=ATT_SPAN
        )

    return (BLOCK_M, BLOCK_N, num_stages, num_warps)

def get_fwd_config(total_tokens, max_seqlen_q, max_seqlen_k, D, causal, disentangled=False, att_span=256):
    """
    Determine optimal kernel configuration parameters for variable-length sequences.

    Args:
        total_tokens: Total number of tokens across all batches
        max_seqlen_q: Maximum query sequence length
        max_seqlen_k: Maximum key sequence length
        D: Per-head dimension
        causal: Whether causal masking is applied
        disentangled: Whether to use DeBERTa-style disentangled attention
        att_span: Size of the attention span for relative positions

    Returns:
        Tuple (BLOCK_M, BLOCK_N, num_stages, num_warps)
    """
    config = _get_fwd_config_cached(total_tokens, max_seqlen_q, max_seqlen_k, D, causal, disentangled, att_span)

    warnings.warn(f"INFO: Variable-length forward config is {config[0]}, {config[1]}, {config[2]}, {config[3]} for BLOCK_M, BLOCK_N stages and warps, respectively.\n"
                  "INFO: If you want to change it, feel free to check ops/flash_attention_varlen")

    return config



def flash_attn_v2_fwd_dise(q, k, v, pos_key, pos_query, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal, sm_scale, BLOCK_M, BLOCK_N,
                           position_buckets, max_relative_distance, num_warps, num_stages, ATT_SPAN):
    """
    Performs the forward pass of FlashAttention with DeBERTa-style disentangled relative attention.

    This function computes the attention output `o` and log-normalizer `L` for the input query (q),
    key (k), and value (v) tensors. It supports disentangled relative attention using optional
    positional projection matrices for content-to-position (C2P) and position-to-content (P2C) biases.

    Args:
        q (Tensor): Query tensor of shape (B, H, M, D) where
            B = batch size, H = number of heads, M = query sequence length, D = head dimension.
        k (Tensor): Key tensor of shape (B, H, N, D) where
            N = key sequence length.
        v (Tensor): Value tensor of shape (B, H, N, D).
        pos_key (Tensor or None): Relative position embedding tensor for C2P bias with shape (2 * max_distance, D),
            or None to disable content-to-position bias.
        pos_query (Tensor or None): Relative position embedding tensor for P2C bias with shape (2 * max_distance, D),
            or None to disable position-to-content bias.
        causal (bool): If True, applies causal (autoregressive) masking to the attention weights.
        sm_scale (float): Scaling factor applied to the dot-product attention scores.
        BLOCK_M (int): Block size for splitting the query sequence dimension.
        BLOCK_N (int): Block size for splitting the key sequence dimension.
        position_buckets (int): Number of relative position buckets. If > 0, bucketing is applied.
        max_relative_distance (int): Maximum relative distance used in bucketing or span window size.
        num_warps (int): Number of warps used in the Triton kernel (hardware-specific parallelism).
        num_stages (int): Number of pipeline stages in the Triton kernel.

    Returns:
        o (Tensor): Output attention tensor of shape (B, H, M, D), same shape as `q`.
        L (Tensor): Log-sum-exp normalizer tensor of shape (B, H, M), used for numerically stable softmax.

    Notes:
        - This function utilizes a custom Triton kernel to efficiently compute block-sparse FlashAttention
          with optional relative position biasing (both C2P and P2C).
        - The relative attention mechanism supports DeBERTa's disentangled attention formulation, where
          the attention bias is computed separately for position-query and key-position interactions.
        - The number of relative position buckets and max distance determines the size and behavior
          of the relative bias.
    """
    M = max_seqlen_q
    N = max_seqlen_k
    B = len(cu_seqlens_q)-1
    Z, H, D = q.shape

    # Use cached mid tensors to reduce CPU overhead
    mid_batch, mid_start, MN = get_mid_cached(cu_seqlens_q, B, BLOCK_M, q.device)

    # Determine if each bias term is present.
    has_c2p = pos_key is not None
    has_p2c = pos_query is not None

    grid = (MN, H)
    o = torch.empty_like(q)
    L = torch.empty((q.shape[0], q.shape[1]), device=q.device, dtype=torch.float32) 

    if has_c2p:
        stride_pk0, stride_pk1, stride_pk2 = pos_key.stride()
    else:
        stride_pk0 = stride_pk1 = stride_pk2 = 0
    if has_p2c:
        stride_pq0, stride_pq1, stride_pq2 = pos_query.stride()
    else:
        stride_pq0 = stride_pq1 = stride_pq2 = 0

    with torch.cuda.device(q.device.index):
        _fwd_kernel_deberta_disentangled_attention[grid](
            q, k, v,
            pos_key, pos_query,
            L, o,
            sm_scale,
            cu_seqlens_q, cu_seqlens_k,
            mid_batch, mid_start,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            stride_pk0, stride_pk1, stride_pk2,
            stride_pq0, stride_pq1, stride_pq2,
            B, H, M, N,
            BLOCK_M=BLOCK_M, BLOCK_DMODEL=D, BLOCK_N=BLOCK_N,
            IS_CAUSAL=causal,
            HAS_C2P=has_c2p, HAS_P2C=has_p2c,
            ATT_SPAN=ATT_SPAN,
            NUM_BUCKETS=position_buckets,
            MAX_DISTANCE=max_relative_distance,
            num_warps=num_warps, num_stages=num_stages,
        )

    return o, L


@functools.lru_cache(maxsize=128)
def _get_bwd_config_varlen_cached(total_tokens_q, total_tokens_k, max_seqlen_q, max_seqlen_k, D, causal,
                                   disentangled, att_span, dtype_size):
    """
    Cached version of configuration computation for variable-length backward pass.
    dtype_size: 2 for float16/bfloat16, 4 for float32
    """
    if all(key in os.environ for key in ['FLASHDEBERTA_BWD_BLOCK_M', 'FLASHDEBERTA_BWD_BLOCK_N',
                                          'FLASHDEBERTA_BWD_NUM_STAGES', 'FLASHDEBERTA_BWD_NUM_WARPS']):
        return (
            int(os.environ['FLASHDEBERTA_BWD_BLOCK_M']),
            int(os.environ['FLASHDEBERTA_BWD_BLOCK_N']),
            int(os.environ['FLASHDEBERTA_BWD_NUM_STAGES']),
            int(os.environ['FLASHDEBERTA_BWD_NUM_WARPS'])
        )

    capability_map = {
        (7,0):  96000, (7,2):  96000, (7,5):  64000,
        (8,0): 163000, (8,6):  99000, (8,7): 163000, (8,9):  99000,
        (9,0): 227000,
    }
    cap = torch.cuda.get_device_capability()
    prop = torch.cuda.get_device_properties(0)

    if hasattr(prop, "shared_memory_per_block_optin"):
        shared_mem_per_block = prop.shared_memory_per_block_optin
    elif cap in capability_map:
        shared_mem_per_block = capability_map[cap]
    elif cap[0] >= 8:
        shared_mem_per_block = 99000
    else:
        shared_mem_per_block = 48000
    max_shared_memory = max(0, shared_mem_per_block - 2048)

    if cap[0] >= 8:
        if max_seqlen_q <= 64:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 16, 16, 2, 4
        elif max_seqlen_q <= 128:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 32, 32, 2, 4
        elif max_seqlen_q <= 256:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 32, 2, 4
        elif cap[0] >= 9:
            BLOCK_M, BLOCK_N, num_stages, num_warps = (64, 64, 3, 4) if D <= 64 else (64, 64, 2, 8)
        else:
            BLOCK_M, BLOCK_N, num_stages, num_warps = (64, 64, 3, 4) if D <= 64 else (64, 64, 2, 8)
    else:
        BLOCK_M, BLOCK_N, num_stages, num_warps = 32, 32, 2, 4

    if max_seqlen_q > 256 and max_seqlen_k >= 2 * max_seqlen_q and BLOCK_N < 128 and cap[0] >= 8:
        BLOCK_N = 128
        num_warps = max(num_warps, 8)

    ATT_SPAN = att_span if disentangled else 0

    # Convert dtype_size back to dtype for calculation
    if dtype_size == 2:
        dtype = torch.float16
    elif dtype_size == 4:
        dtype = torch.float32
    else:
        dtype = torch.float16

    shm = calculate_shared_memory_usage_varlen_bwd(
        BLOCK_M, BLOCK_N, D, num_stages, dtype,
        has_c2p=disentangled, has_p2c=disentangled, ATT_SPAN=ATT_SPAN,
        store_lse=True, recompute_probs=True, accum_dq=True, accum_dkv=True,
    )

    def halve_pow2(x): return max(16, (x // 2)) if x > 16 else 16
    while shm > max_shared_memory and (num_stages > 1 or BLOCK_M > 16 or BLOCK_N > 16):
        if num_stages > 1:
            num_stages -= 1
        elif BLOCK_N >= BLOCK_M and BLOCK_N > 16:
            BLOCK_N = halve_pow2(BLOCK_N)
        elif BLOCK_M > 16:
            BLOCK_M = halve_pow2(BLOCK_M)
        else:
            break
        shm = calculate_shared_memory_usage_varlen_bwd(
            BLOCK_M, BLOCK_N, D, num_stages, dtype,
            has_c2p=disentangled, has_p2c=disentangled, ATT_SPAN=ATT_SPAN,
            store_lse=True, recompute_probs=True, accum_dq=True, accum_dkv=True,
        )

    if D <= 64 and BLOCK_M * BLOCK_N <= 64 * 64:
        num_warps = min(num_warps, 4)
    else:
        num_warps = max(num_warps, 8 if cap[0] >= 8 else 4)

    return (BLOCK_M, BLOCK_N, num_stages, num_warps)

def get_bwd_config_varlen(total_tokens_q, total_tokens_k, max_seqlen_q, max_seqlen_k, D, causal,
                          *, disentangled=True, att_span=256, dtype=torch.float16, max_shared_memory=None):
    """
    Determine optimal kernel configuration parameters for variable-length backward pass.
    Uses caching to avoid recomputing configurations for the same inputs.
    """
    # Convert dtype to size for caching (dtypes are not hashable)
    if dtype == torch.float16 or dtype == torch.bfloat16:
        dtype_size = 2
    elif dtype == torch.float32:
        dtype_size = 4
    else:
        dtype_size = 2

    config = _get_bwd_config_varlen_cached(total_tokens_q, total_tokens_k, max_seqlen_q, max_seqlen_k, D, causal,
                                            disentangled, att_span, dtype_size)

    warnings.warn(
        f"INFO: Varlen backward config -> "
        f"BLOCK_M={config[0]}, BLOCK_N={config[1]}, stages={config[2]}, warps={config[3]}."
    )
    return config

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

@triton.jit
def _bwd_kv_dise_kernel_varlen(
    Q, K, V, K_POS, Q_POS, sm_scale, DO,
    DK, DV, DKPOS, DQPOS,
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
    B, H,  # only for indexing/strides (BM/H known from strides)
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    HAS_C2P: tl.constexpr, HAS_P2C: tl.constexpr,
    ATT_SPAN: tl.constexpr, NUM_BUCKETS: tl.constexpr, MAX_DISTANCE: tl.constexpr,
):
    input_dtype = Q.dtype.element_ty
    log2e: tl.constexpr = 1.4426950408889634

    tile_n = tl.program_id(0)
    off_h  = tl.program_id(1)

    off_b   = tl.load(mid_batch_n + tile_n)
    n_start = tl.load(mid_start_n + tile_n)   # absolute start index in K/V (flattened)

    q_start = tl.load(cu_seqlens_q + off_b)
    q_end   = tl.load(cu_seqlens_q + off_b + 1)
    k_start = tl.load(cu_seqlens_k + off_b)
    k_end   = tl.load(cu_seqlens_k + off_b + 1)

    lM = q_end - q_start
    lN = k_end - k_start
    P_SEQ = lM - lN

    offs_n_base   = tl.arange(0, BLOCK_N)
    offs_n_abs    = n_start + offs_n_base
    offs_n_rel    = offs_n_abs - k_start
    mask_n        = (offs_n_abs < k_end)
    offs_k        = tl.arange(0, BLOCK_DMODEL)

    # Load K, V tiles for this N-block
    k_ptrs = K + (offs_n_abs[:, None] * stride_kz + off_h * stride_kh + offs_k[None, :] * stride_kk)
    v_ptrs = V + (offs_n_abs[:, None] * stride_vz + off_h * stride_vh + offs_k[None, :] * stride_vk)
    k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)

    offs_m_base = tl.arange(0, BLOCK_M)

    # causal lower-bound in m (relative indexing)
    if CAUSAL:
        start_n_rel = n_start - k_start
        lo_rel = tl.maximum(start_n_rel - P_SEQ, 0)
        lo_rel = (lo_rel // BLOCK_M) * BLOCK_M
    else:
        lo_rel = 0

    for start_m_rel in range(lo_rel, lM, BLOCK_M):
        start_m_rel = tl.multiple_of(start_m_rel, BLOCK_M)
        offs_m_rel = start_m_rel + offs_m_base
        offs_m_abs = q_start + offs_m_rel
        mask_m     = (offs_m_abs < q_end)

        offs_qk = tl.arange(0, BLOCK_DMODEL)

        q_ptrs  = Q  + (offs_m_abs[:, None] * stride_qz + off_h * stride_qh + offs_qk[None, :] * stride_qk)
        do_ptrs = DO + (offs_m_abs[:, None] * stride_doz + off_h * stride_doh + offs_qk[None, :] * stride_dok)

        q  = tl.load(q_ptrs,  mask=mask_m[:, None], other=0.0)
        do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)

        # ---- Recompute scores s (M x N) ----
        s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        s += tl.dot(q, tl.trans(k)) * sm_scale

        # Relative bucketization (match forward varlen)
        rel = (offs_m_rel[:, None] - offs_n_rel[None, :])  # q_pos - k_pos to match fixed-len kernel
        sign = tl.where(rel > 0.0, 1.0, tl.where(rel < 0.0, -1.0, 0.0))
        mid_val = NUM_BUCKETS // 2
        abs_rel = tl.abs(rel)
        cond = (rel < mid_val) & (rel > -mid_val)
        abs_pos = tl.where(cond, mid_val - 1.0, abs_rel)
        log_numer = tl.log(abs_pos / mid_val)
        log_denom = tl.log((MAX_DISTANCE - 1) / mid_val)
        log_scaled = log_numer / log_denom * (mid_val - 1.0)
        log_pos = tl.ceil(log_scaled) + mid_val
        bucket_pos = tl.where(abs_pos <= mid_val, rel, log_pos * sign)

        if HAS_C2P:
            c2p_index = tl.minimum(tl.maximum(bucket_pos + ATT_SPAN, 0), 2 * ATT_SPAN - 1).to(tl.int32)
            # K_POS layout: (BM, H, 2*ATT_SPAN), indexed by query row (offs_m_abs)
            kpos_base = K_POS + (offs_m_abs[:, None] * stride_pk0 + off_h * stride_pk1)
            kpos_ptrs = kpos_base + c2p_index * stride_pk2
            c2p_bias = tl.load(kpos_ptrs, mask=mask_m[:, None] & (c2p_index < 2*ATT_SPAN), other=0.0)
            s += c2p_bias * sm_scale

        if HAS_P2C:
            p2c_index = tl.minimum(tl.maximum(bucket_pos + ATT_SPAN, 0), 2 * ATT_SPAN - 1).to(tl.int32).trans(1, 0)
            # Q_POS layout: (BN, H, 2*ATT_SPAN), indexed by key row (offs_n_abs)
            qpos_base = Q_POS + (offs_n_abs[:, None] * stride_pq0 + off_h * stride_pq1)
            qpos_ptrs = qpos_base + p2c_index * stride_pq2
            p2c_bias = tl.load(qpos_ptrs, mask=mask_n[:, None] & (p2c_index < 2*ATT_SPAN), other=0.0).trans(1, 0)
            s += p2c_bias * sm_scale

        # Causal mask and p re-materialization
        if CAUSAL:
            # same mask as fwd (using P_SEQ and absolute/relative rows)
            causal_mask = (P_SEQ + offs_m_rel[:, None]) >= (offs_n_rel[None, :])
        l_ptrs = L + (offs_m_abs * H + off_h)
        l = tl.load(l_ptrs, mask=mask_m, other=0.0)
        p = tl.math.exp2((s - l[:, None]) * log2e)
        p = tl.where(mask_n[None, :], p, 0.0)
        p = tl.where(mask_m[:,  None], p, 0.0)
        if CAUSAL:
            p = tl.where(causal_mask, p, 0.0)

        # dv = p^T @ do
        dv += tl.dot(tl.trans(p.to(do.dtype)), do)

        # dp = do @ v^T
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dp += tl.dot(do.to(input_dtype), tl.trans(v))

        # ds = p * (dp - delta[:, None])
        delta_ptrs = Delta + (offs_m_abs * H + off_h)
        delta = tl.load(delta_ptrs, mask=mask_m, other=0.0)
        ds = p * (dp - delta[:, None])
        ds = tl.where(mask_n[None, :], ds, 0.0)
        if CAUSAL:
            ds = tl.where(causal_mask, ds, 0.0)

        ds_scaled = (ds * sm_scale).to(input_dtype)

        # dk += ds^T @ q
        dk += tl.dot(tl.trans(ds_scaled), q)

        # positional grads: atomic adds
        if HAS_C2P:
            # Cast to fp32 for atomic_add (bf16 not supported by Triton atomic operations)
            ds_scaled_fp32 = ds_scaled.to(tl.float32)
            kpos_base = DKPOS + (offs_m_abs[:, None] * stride_pk0 + off_h * stride_pk1)
            kpos_ptrs = kpos_base + c2p_index * stride_pk2
            tl.atomic_add(kpos_ptrs, ds_scaled_fp32, mask=mask_m[:, None] & mask_n[None, :] & (c2p_index < 2*ATT_SPAN),
                          sem="relaxed")

        if HAS_P2C:
            # Cast to fp32 for atomic_add (bf16 not supported by Triton atomic operations)
            ds_scaled_fp32 = ds_scaled.to(tl.float32)
            ds_scaled_t_fp32 = ds_scaled_fp32.trans(1, 0)
            qpos_base = DQPOS + (offs_n_abs[:, None] * stride_pq0 + off_h * stride_pq1)
            qpos_ptrs = qpos_base + p2c_index * stride_pq2
            tl.atomic_add(qpos_ptrs, ds_scaled_t_fp32,
                          mask=mask_n[:, None] & mask_m[None, :] & (p2c_index < 2*ATT_SPAN),
                          sem="relaxed")

    # store dk/dv
    dk_ptrs = DK + (offs_n_abs[:, None] * stride_dkz + offs_k[None, :] * stride_dkk + off_h * stride_dkh)
    dv_ptrs = DV + (offs_n_abs[:, None] * stride_dvz + offs_k[None, :] * stride_dvk + off_h * stride_dvh)
    tl.store(dk_ptrs, dk.to(input_dtype), mask=mask_n[:, None])
    tl.store(dv_ptrs, dv.to(input_dtype), mask=mask_n[:, None])


@triton.jit
def _bwd_q_dise_kernel_varlen(
    Q, K, V, K_POS, Q_POS, sm_scale, DO,
    DQ,
    L, Delta,
    cu_seqlens_q, cu_seqlens_k, mid_batch_m, mid_start_m,
    stride_qz, stride_qh, stride_qk,
    stride_kz, stride_kh, stride_kk,
    stride_vz, stride_vh, stride_vk,
    stride_doz, stride_doh, stride_dok,
    stride_dqz, stride_dqh, stride_dqk,
    stride_pk0, stride_pk1, stride_pk2,
    stride_pq0, stride_pq1, stride_pq2,
    B, H,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr, HAS_C2P: tl.constexpr, HAS_P2C: tl.constexpr,
    ATT_SPAN: tl.constexpr, NUM_BUCKETS: tl.constexpr, MAX_DISTANCE: tl.constexpr,
):
    input_dtype = Q.dtype.element_ty
    log2e: tl.constexpr = 1.4426950408889634

    tile_m = tl.program_id(0)
    off_h  = tl.program_id(1)

    off_b = tl.load(mid_batch_m + tile_m)
    off_m = tl.load(mid_start_m + tile_m)   # absolute q start for this tile

    q_start = tl.load(cu_seqlens_q + off_b)
    q_end   = tl.load(cu_seqlens_q + off_b + 1)
    k_start = tl.load(cu_seqlens_k + off_b)
    k_end   = tl.load(cu_seqlens_k + off_b + 1)

    lM = q_end - q_start
    lN = k_end - k_start
    P_SEQ = lM - lN

    offs_m_base = tl.arange(0, BLOCK_M)
    offs_m_abs  = off_m + offs_m_base
    offs_m_rel  = offs_m_abs - q_start
    mask_m      = (offs_m_abs < q_end)

    offs_k = tl.arange(0, BLOCK_DMODEL)

    q_ptrs  = Q  + (offs_m_abs[:, None] * stride_qz + off_h * stride_qh + offs_k[None, :] * stride_qk)
    dq_ptrs = DQ + (offs_m_abs[:, None] * stride_dqz + off_h * stride_dqh + offs_k[None, :] * stride_dqk)
    do_ptrs = DO + (offs_m_abs[:, None] * stride_doz + off_h * stride_doh + offs_k[None, :] * stride_dok)

    q  = tl.load(q_ptrs,  mask=mask_m[:, None])
    do = tl.load(do_ptrs, mask=mask_m[:, None])

    # p upper bound for this tile
    if CAUSAL:
        hi_rel = tl.minimum(lN, P_SEQ + (offs_m_rel[-1] + 1))  # conservative upper bound
        hi_rel = tl.maximum(0, hi_rel)
    else:
        hi_rel = lN

    dq = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    offs_n_base = tl.arange(0, BLOCK_N)
    k_ptrs = K + ((k_start + offs_n_base)[:, None] * stride_kz + off_h * stride_kh + offs_k[None, :] * stride_kk)
    v_ptrs = V + ((k_start + offs_n_base)[:, None] * stride_vz + off_h * stride_vh + offs_k[None, :] * stride_vk)

    for start_n_rel in range(0, hi_rel, BLOCK_N):
        start_n_rel = tl.multiple_of(start_n_rel, BLOCK_N)
        offs_n_rel  = start_n_rel + offs_n_base
        offs_n_abs  = k_start + offs_n_rel
        mask_n      = (offs_n_abs < k_end)

        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        # ---- Recompute scores s ----
        s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        s += tl.dot(q, tl.trans(k)) * sm_scale

        rel = (offs_m_rel[:, None] - offs_n_rel[None, :])
        sign = tl.where(rel > 0.0, 1.0, tl.where(rel < 0.0, -1.0, 0.0))
        mid_val = NUM_BUCKETS // 2
        abs_rel = tl.abs(rel)
        cond = (rel < mid_val) & (rel > -mid_val)
        abs_pos = tl.where(cond, mid_val - 1.0, abs_rel)
        log_numer = tl.log(abs_pos / mid_val)
        log_denom = tl.log((MAX_DISTANCE - 1) / mid_val)
        log_scaled = log_numer / log_denom * (mid_val - 1.0)
        log_pos = tl.ceil(log_scaled) + mid_val
        bucket_pos = tl.where(abs_pos <= mid_val, rel, log_pos * sign)

        if HAS_C2P:
            c2p_index = tl.minimum(tl.maximum(bucket_pos + ATT_SPAN, 0), 2*ATT_SPAN - 1).to(tl.int32)
            kpos_base = K_POS + (offs_m_abs[:, None] * stride_pk0 + off_h * stride_pk1)
            k_pos_ptrs = kpos_base + c2p_index * stride_pk2
            c2p_bias = tl.load(k_pos_ptrs, mask=mask_m[:, None] & (c2p_index < 2*ATT_SPAN), other=0.0)
            s += c2p_bias * sm_scale

        if HAS_P2C:
            p2c_index = tl.minimum(tl.maximum(bucket_pos + ATT_SPAN, 0), 2*ATT_SPAN - 1).to(tl.int32).trans(1, 0)
            qpos_base = Q_POS + (offs_n_abs[:, None] * stride_pq0 + off_h * stride_pq1)
            q_pos_ptrs = qpos_base + p2c_index * stride_pq2
            p2c_bias = tl.load(q_pos_ptrs, mask=mask_n[:, None] & (p2c_index < 2*ATT_SPAN), other=0.0).trans(1, 0)
            s += p2c_bias * sm_scale

        if CAUSAL:
            causal_mask = (P_SEQ + offs_m_rel[:, None]) >= (offs_n_rel[None, :])

        l_ptrs = L + (offs_m_abs * H + off_h)
        l = tl.load(l_ptrs, mask=mask_m)
        p = tl.math.exp2((s - l[:, None]) * log2e)
        p = tl.where(mask_n[None, :], p, 0.0)
        if CAUSAL:
            p = tl.where(causal_mask, p, 0.0)

        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dp += tl.dot(do.to(input_dtype), tl.trans(v))

        delta_ptrs = Delta + (offs_m_abs * H + off_h)
        delta = tl.load(delta_ptrs, mask=mask_m, other=0.0)
        ds = p * (dp - delta[:, None])
        ds = tl.where(mask_n[None, :], ds, 0.0)
        if CAUSAL:
            ds = tl.where(causal_mask, ds, 0.0)

        dq += tl.dot((ds * sm_scale).to(input_dtype), k)

        k_ptrs += BLOCK_N * stride_kz
        v_ptrs += BLOCK_N * stride_vz

    tl.store(dq_ptrs, dq.to(input_dtype), mask=mask_m[:, None])

def flash_attn_v2_bwd_dise_varlen(
    o, do, q, k, v, k_pos, q_pos, L,
    cu_seqlens_q, cu_seqlens_k,
    causal, sm_scale,
    BLOCK_M, BLOCK_N, position_buckets, max_relative_distance,
    num_warps, num_stages, ATT_SPAN,
):
    device = q.device
    BM, H, D = q.shape
    BN = k.shape[0]
    B = cu_seqlens_q.numel() - 1

    # Build M- and N- tile mappings using cached tensors
    mid_m_batch, mid_m_start, MN = get_mid_cached(cu_seqlens_q, B, BLOCK_M, device)
    mid_n_batch, mid_n_start, NK = get_mid_cached(cu_seqlens_k, B, BLOCK_N, device)

    # Δ: (BM, H), L: (BM, H)  — matches kernel pointer math
    delta = torch.empty((BM, H), device=device, dtype=torch.float32)

    grid_pre = (MN, H)
    with torch.cuda.device(q.device.index):
        _bwd_preprocess_varlen[grid_pre](
            o, do, delta,
            cu_seqlens_q, mid_m_batch, mid_m_start,
            o.stride(0), o.stride(1), o.stride(2),
            do.stride(0), do.stride(1), do.stride(2),
            B, H,
            BLOCK_M=BLOCK_M, D_HEAD=D, num_warps=num_warps, num_stages=num_stages
        )

    dk   = torch.empty_like(k)
    dv   = torch.empty_like(v)
    # bf16 is not supported in atomic_add, so we always use fp32 for positional gradients
    dk_pos = torch.zeros_like(k_pos, dtype=torch.float32) if k_pos is not None else None
    dq_pos = torch.zeros_like(q_pos, dtype=torch.float32) if q_pos is not None else None

    has_c2p = k_pos is not None
    has_p2c = q_pos is not None
    if has_c2p:
        stride_pk0, stride_pk1, stride_pk2 = k_pos.stride()
    else:
        stride_pk0 = stride_pk1 = stride_pk2 = 0
    if has_p2c:
        stride_pq0, stride_pq1, stride_pq2 = q_pos.stride()
    else:
        stride_pq0 = stride_pq1 = stride_pq2 = 0

    grid_kv = (NK, H)
    with torch.cuda.device(q.device.index):
        _bwd_kv_dise_kernel_varlen[grid_kv](
            q, k, v, k_pos if has_c2p else k, q_pos if has_p2c else v, sm_scale, do,
            dk, dv, dk_pos if has_c2p else k, dq_pos if has_p2c else v,
            L, delta,
            cu_seqlens_q, cu_seqlens_k, mid_n_batch, mid_n_start,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            do.stride(0), do.stride(1), do.stride(2),
            dk.stride(0), dk.stride(1), dk.stride(2),
            dv.stride(0), dv.stride(1), dv.stride(2),
            stride_pk0, stride_pk1, stride_pk2,
            stride_pq0, stride_pq1, stride_pq2,
            B, H,
            BLOCK_M=BLOCK_M, BLOCK_DMODEL=D, BLOCK_N=BLOCK_N,
            CAUSAL=causal,
            HAS_C2P=has_c2p, HAS_P2C=has_p2c,
            ATT_SPAN=ATT_SPAN, NUM_BUCKETS=position_buckets, MAX_DISTANCE=max_relative_distance,
            num_warps=num_warps, num_stages=num_stages,
        )

    dq = torch.empty_like(q)
    grid_q = (MN, H)
    with torch.cuda.device(q.device.index):
        _bwd_q_dise_kernel_varlen[grid_q](
            q, k, v, k_pos if has_c2p else q, q_pos if has_p2c else k, sm_scale, do,
            dq,
            L, delta,
            cu_seqlens_q, cu_seqlens_k, mid_m_batch, mid_m_start,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            do.stride(0), do.stride(1), do.stride(2),
            dq.stride(0), dq.stride(1), dq.stride(2),
            stride_pk0, stride_pk1, stride_pk2,
            stride_pq0, stride_pq1, stride_pq2,
            B, H,
            BLOCK_M=BLOCK_M, BLOCK_DMODEL=D, BLOCK_N=BLOCK_N,
            CAUSAL=causal, HAS_C2P=has_c2p, HAS_P2C=has_p2c,
            ATT_SPAN=ATT_SPAN, NUM_BUCKETS=position_buckets, MAX_DISTANCE=max_relative_distance,
            num_warps=num_warps, num_stages=num_stages,
        )

    return dq, dk, dv, dk_pos, dq_pos

class FlashAttentionDisentangledVarlen(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, k_pos, q_pos,
                cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k,
                causal, sm_scale, position_buckets, max_relative_distance):

        BM, H, D = q.shape
        BN = k.shape[0]
        assert (k.shape[1], v.shape[1]) == (H, H) and (k.shape[2], v.shape[2]) == (D, D)

        ATT_SPAN = position_buckets if position_buckets > 0 else max_relative_distance
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(D)

        # Forward config
        BLOCK_M, BLOCK_N, num_stages, num_warps = get_fwd_config(
            total_tokens=BM,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            D=D, causal=causal, disentangled=True, att_span=ATT_SPAN
        )

        # Run forward (NOTE: L should be (BM, H) to match kernels below)
        o, L = flash_attn_v2_fwd_dise(
            q, k, v, k_pos, q_pos, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k, causal, sm_scale,
            BLOCK_M, BLOCK_N, position_buckets, max_relative_distance,
            num_warps, num_stages, ATT_SPAN
        )

        # Save everything needed
        ctx.save_for_backward(q, k, v, k_pos, q_pos, o, L, cu_seqlens_q, cu_seqlens_k)
        ctx.sm_scale = sm_scale
        ctx.causal = causal
        ctx.position_buckets = position_buckets
        ctx.max_relative_distance = max_relative_distance
        ctx.ATT_SPAN = ATT_SPAN
        ctx.fwd_cfg = (BLOCK_M, BLOCK_N, num_stages, num_warps)
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        return o

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, k_pos, q_pos, o, L, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
        sm_scale = ctx.sm_scale
        causal = ctx.causal
        position_buckets = ctx.position_buckets
        max_relative_distance = ctx.max_relative_distance
        ATT_SPAN = ctx.ATT_SPAN
        max_seqlen_q = ctx.max_seqlen_q
        max_seqlen_k = ctx.max_seqlen_k
        BM, H, D = q.shape
        BN = k.shape[0]

        # Backward config (varlen heuristic)
        BLOCK_M_b, BLOCK_N_b, num_stages_b, num_warps_b = get_bwd_config_varlen(
            total_tokens_q=BM, total_tokens_k=BN,
            max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k, D=D,
            causal=causal, disentangled=True, att_span=ATT_SPAN, dtype=q.dtype
        )

        dq, dk, dv, dk_pos, dq_pos = flash_attn_v2_bwd_dise_varlen(
            o, grad_output, q, k, v, k_pos, q_pos, L,
            cu_seqlens_q, cu_seqlens_k,
            causal, sm_scale,
            BLOCK_M_b, BLOCK_N_b, position_buckets, max_relative_distance,
            num_warps_b, num_stages_b, ATT_SPAN
        )

        # Match forward signature: (q, k, v, q_pos, k_pos, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal, sm_scale, position_buckets, max_relative_distance)
        return dq, dk, dv, dk_pos, dq_pos, None, None, None, None, None, None, None, None

def flash_attention_with_disentangled_varlen(
    q, k, v, k_pos, q_pos, cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k, causal=False, sm_scale=None,
    position_buckets=0, max_relative_distance=0,
):
    """
    Flash attention with DeBERTa-style disentangled attention for variable-length sequences.

    Args:
        q:  (BM, H, D)   flattened queries
        k:  (BN, H, D)
        v:  (BN, H, D)
        k_pos: (BM, H, 2*ATT_SPAN) or None - content-to-position bias
        q_pos: (BN, H, 2*ATT_SPAN) or None - position-to-content bias
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
        q, k, v, k_pos, q_pos, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k, causal, sm_scale,
        position_buckets, max_relative_distance
    )
