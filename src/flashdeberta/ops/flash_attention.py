import math
import torch
import triton
import triton.language as tl

from .config import forward_config, backward_configs
from .launch import launch, clear_launch_cache
from .position import (position_plan, position_table, fold_matrix, fold_band_gradients,
                       clear_position_cache)

def cdiv(a, b):
    return (a + b - 1) // b

@triton.jit
def _fwd_kernel_deberta_disentangled_attention(
    Q, K, V,
    K_POS, Q_POS, POS_LUT,
    L, O,
    SEQ_LENGTHS,
    sm_scale,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_pk0, stride_pk1, stride_pk2, stride_pk3,
    stride_pq0, stride_pq1, stride_pq2, stride_pq3,
    Z, H, M, N, P_SEQ,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr, LARGER_M: tl.constexpr,
    HAS_C2P: tl.constexpr, HAS_P2C: tl.constexpr,
):
    input_dtype = Q.dtype.element_ty

    start_m = tl.program_id(0)
    off_h   = tl.program_id(1)
    off_z   = tl.program_id(2)

    log2e: tl.constexpr = 1.4426950408889634
    qk_scale = sm_scale * log2e

    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_h * stride_kh
    V += off_z * stride_vz + off_h * stride_vh
    O += off_z * stride_oz + off_h * stride_oh
    L += (off_z * H + off_h) * M

    if HAS_C2P:
        K_POS += off_z.to(tl.int64) * stride_pk0 + off_h * stride_pk1
    if HAS_P2C:
        Q_POS += off_z.to(tl.int64) * stride_pq0 + off_h * stride_pq1

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_DMODEL)

    q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)
    o_ptrs = O + (offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok)
    l_ptrs = L + offs_m

    seq_length = tl.load(SEQ_LENGTHS + off_z).to(tl.int32)
    mask_m = offs_m < seq_length

    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0, cache_modifier=".cg")

    m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    k_ptrs = K + (offs_k[:, None] * stride_kk + offs_n_base[None, :] * stride_kn)
    v_ptrs = V + (offs_n_base[:, None] * stride_vn + offs_k[None, :] * stride_vk)

    n_limit = ((seq_length + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
    if IS_CAUSAL:
        hi = tl.minimum(n_limit, P_SEQ + (start_m + 1) * BLOCK_M)
        hi = tl.minimum(hi, N)
    else:
        hi = n_limit

    for start_n in range(0, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + offs_n_base
        mask_n = offs_n < seq_length
        valid = mask_m[:, None] & mask_n[None, :]

        k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0, cache_modifier=".cg")
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0, cache_modifier=".cg")

        s = tl.dot(q, k)

        if HAS_C2P or HAS_P2C:
            # Slot of every (query, key) pair from the distance lookup table.
            slot = tl.load(POS_LUT + (offs_m[:, None] - offs_n[None, :] + N - 1), mask=valid, other=0)
        if HAS_C2P:
            s += tl.load(K_POS + offs_m[:, None] * stride_pk2 + slot * stride_pk3,
                         mask=valid, other=0.0).to(tl.float32)
        if HAS_P2C:
            s += tl.load(Q_POS + offs_n[None, :] * stride_pq2 + slot * stride_pq3,
                         mask=valid, other=0.0).to(tl.float32)

        s = s * qk_scale
        s = tl.where(mask_n[None, :], s, float("-inf"))

        if IS_CAUSAL:
            causal_mask = (P_SEQ + offs_m[:, None]) >= offs_n[None, :]
            s = tl.where(causal_mask, s, float("-inf"))

        m_i_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(s - m_i_new[:, None])
        acc *= alpha[:, None]
        acc += tl.dot(p.to(input_dtype), v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn

    # L is the natural-log LSE of the scaled scores; m_i and l_i are in base 2.
    if IS_CAUSAL and LARGER_M:
        is_empty_line = (offs_m + P_SEQ) < 0
        acc = tl.where(is_empty_line[:, None], 0.0, acc * (1.0 / l_i[:, None]))
        l = tl.where(is_empty_line, float("-inf"), (m_i + tl.math.log2(l_i)) / log2e)
    else:
        acc = acc * (1.0 / l_i[:, None])
        l = (m_i + tl.math.log2(l_i)) / log2e

    tl.store(l_ptrs, l, mask=mask_m, cache_modifier=".cg")
    tl.store(o_ptrs, acc.to(input_dtype), mask=mask_m[:, None], cache_modifier=".cg")

def get_fwd_config(B, H, M, N, D, causal, disentangled=False, max_shared_memory=None, att_span=256):
    """
    Kernel configuration (BLOCK_M, BLOCK_N, num_stages, num_warps) for the forward pass.
    See ``ops.config`` for the defaults and the environment-variable overrides.
    """
    return forward_config(D)


def flash_attn_v2_fwd_dise(q, k, v, seq_lengths, pos_key, pos_query, pos_lut, causal, sm_scale,
                           BLOCK_M, BLOCK_N, num_warps, num_stages):
    """
    Performs the forward pass of FlashAttention with DeBERTa-style disentangled relative attention.

    Args:
        q: Query tensor of shape (B, H, M, D)
        k: Key tensor of shape (B, H, N, D)
        v: Value tensor of shape (B, H, N, D)
        seq_lengths: Tensor of shape (B,) containing sequence lengths for each batch element.
                     If None, all sequences are assumed to have length M (full sequence).
        pos_key: C2P scores ``q @ pos_key_layer^T`` of shape (B, H, M, 2 * ATT_SPAN), or None
        pos_query: P2C scores ``k @ pos_query_layer^T`` of shape (B, H, N, 2 * ATT_SPAN), or None
        pos_lut: int32 slot of each distance ``i - j + N - 1`` (see ``ops.position``)
        causal: Whether to apply causal masking
        sm_scale: Softmax scale factor
        BLOCK_M, BLOCK_N: Block sizes for tiling
        num_warps, num_stages: Triton kernel parameters

    Returns:
        o: Output tensor of shape (B, H, M, D)
        L: Log-sum-exp tensor of shape (B, H, M)
    """
    B, H, M, D = q.shape
    N = k.shape[2]
    P_SEQ = N - M

    if sm_scale is None:
        sm_scale = 1. / math.sqrt(D)

    if seq_lengths is None:
        seq_lengths = torch.full((B,), M, dtype=torch.int32, device=q.device)

    has_c2p = pos_key is not None
    has_p2c = pos_query is not None

    o = torch.zeros_like(q)
    L = torch.zeros((B, H, M), device=q.device, dtype=torch.float32)

    stride_pk = pos_key.stride() if has_c2p else (0, 0, 0, 0)
    stride_pq = pos_query.stride() if has_p2c else (0, 0, 0, 0)

    args = (
        q, k, v,
        pos_key, pos_query, pos_lut,
        L, o,
        seq_lengths,
        sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        *stride_pk,
        *stride_pq,
        B, H, M, N, P_SEQ,
    )
    kwargs = dict(BLOCK_DMODEL=D, IS_CAUSAL=causal, LARGER_M=M > N, HAS_C2P=has_c2p, HAS_P2C=has_p2c)
    with torch.cuda.device(q.device.index):
        launch(
            _fwd_kernel_deberta_disentangled_attention,
            ("fwd", D, q.dtype, causal, has_c2p, has_p2c),
            (BLOCK_M, BLOCK_N, num_stages, num_warps),
            lambda block_m, block_n: ((cdiv(M, block_m), H, B), args, kwargs),
        )

    return o, L

def get_bwd_config(B, H, M, N, D, causal, *, disentangled=False, att_span=256, dtype=torch.float16,
                   max_shared_memory=None):
    """
    Kernel configurations for the backward pass: (dK/dV config, dQ config).
    See ``ops.config`` for the defaults and the environment-variable overrides.
    """
    return backward_configs(D)

@triton.jit
def _bwd_preprocess(
    Out, DO,
    Delta,
    SEQ_LENGTHS,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_doz, stride_doh, stride_dom, stride_dok,
    stride_dz, stride_dh, stride_dm,
    M,
    BLOCK_M: tl.constexpr, D_HEAD: tl.constexpr,
    DIVISIBLE_M: tl.constexpr,
):
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)
    Out += off_z * stride_oz + off_h * stride_oh
    DO += off_z * stride_doz + off_h * stride_doh
    Delta += off_z * stride_dz + off_h * stride_dh

    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_k = tl.arange(0, D_HEAD)

    o_ptrs = Out + off_m[:, None] * stride_om + off_k[None, :] * stride_ok
    do_ptrs = DO  + off_m[:, None] * stride_dom + off_k[None, :] * stride_dok

    seq_length = tl.load(SEQ_LENGTHS+off_z).to(tl.int32)

    mask_m = off_m < seq_length
    o  = tl.load(o_ptrs,  mask=mask_m[:, None], other=0.0).to(tl.float32)
    do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + off_m * stride_dm, delta, mask=mask_m)

@triton.jit
def _bwd_kv_dise_kernel(
    Q, K, V, SEQ_LENGTHS, K_POS, Q_POS, POS_LUT, sm_scale, DO,
    DK, DV, P2C_BAND,
    L, Delta,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_doz, stride_doh, stride_dom, stride_dok,
    stride_dkz, stride_dkh, stride_dkn, stride_dkk,
    stride_dvz, stride_dvh, stride_dvn, stride_dvk,
    stride_pk0, stride_pk1, stride_pk2, stride_pk3,
    stride_pq0, stride_pq1, stride_pq2, stride_pq3,
    stride_band_h, stride_band_z,
    Z, H, M, N, P_SEQ, BAND_LOW, NUM_COLUMNS,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    HAS_C2P: tl.constexpr, HAS_P2C: tl.constexpr,
):
    input_dtype = Q.dtype.element_ty
    log2e: tl.constexpr = 1.4426950408889634
    qk_scale = sm_scale * log2e

    start_n = tl.program_id(0)
    off_h   = tl.program_id(1)
    off_z   = tl.program_id(2)

    Q  += off_z*stride_qz  + off_h*stride_qh
    K  += off_z*stride_kz  + off_h*stride_kh
    V  += off_z*stride_vz  + off_h*stride_vh
    DO += off_z*stride_doz + off_h*stride_doh
    DK += off_z*stride_dkz + off_h*stride_dkh
    DV += off_z*stride_dvz + off_h*stride_dvh

    if HAS_C2P:
        K_POS += off_z.to(tl.int64)*stride_pk0 + off_h*stride_pk1
    if HAS_P2C:
        Q_POS += off_z.to(tl.int64)*stride_pq0 + off_h*stride_pq1
        P2C_BAND += off_h.to(tl.int64)*stride_band_h + off_z.to(tl.int64)*stride_band_z

    L     += (off_z*H + off_h) * M
    Delta += (off_z*H + off_h) * M

    seq_length = tl.load(SEQ_LENGTHS + off_z).to(tl.int32)

    m_limit = ((seq_length + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
    if CAUSAL:
        lo = tl.maximum(start_n * BLOCK_N - P_SEQ - (BLOCK_M - 1), 0)
        lo = ((lo + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
    else:
        lo = 0

    offs_m_base = tl.arange(0, BLOCK_M)
    offs_n      = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k      = tl.arange(0, BLOCK_DMODEL)

    mask_n = offs_n < seq_length
    k = tl.load(K + offs_n[:, None]*stride_kn + offs_k[None, :]*stride_kk, mask=mask_n[:, None], other=0.0)
    v = tl.load(V + offs_n[:, None]*stride_vn + offs_k[None, :]*stride_vk, mask=mask_n[:, None], other=0.0)

    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    # P2C gradients of distances saturated into the first/last slot, per key row.
    low_sum = tl.zeros([BLOCK_N], dtype=tl.float32)
    high_sum = tl.zeros([BLOCK_N], dtype=tl.float32)

    for start_m in range(lo, m_limit, BLOCK_M):
        offs_m = start_m + offs_m_base
        mask_m = offs_m < seq_length
        valid = mask_m[:, None] & mask_n[None, :]
        if CAUSAL:
            valid = valid & ((P_SEQ + offs_m[:, None]) >= offs_n[None, :])

        q  = tl.load(Q + offs_m[:, None]*stride_qm + offs_k[None, :]*stride_qk, mask=mask_m[:, None], other=0.0)
        do = tl.load(DO + offs_m[:, None]*stride_dom + offs_k[None, :]*stride_dok, mask=mask_m[:, None], other=0.0)
        l  = tl.load(L + offs_m, mask=mask_m, other=0.0)
        delta = tl.load(Delta + offs_m, mask=mask_m, other=0.0)

        s = tl.dot(q, tl.trans(k))

        distance = offs_m[:, None] - offs_n[None, :] + N - 1
        if HAS_C2P or HAS_P2C:
            slot = tl.load(POS_LUT + distance, mask=valid, other=0)
        if HAS_C2P:
            s += tl.load(K_POS + offs_m[:, None]*stride_pk2 + slot*stride_pk3,
                         mask=valid, other=0.0).to(tl.float32)
        if HAS_P2C:
            s += tl.load(Q_POS + offs_n[None, :]*stride_pq2 + slot*stride_pq3,
                         mask=valid, other=0.0).to(tl.float32)

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
            tl.store(P2C_BAND + offs_n[None, :]*NUM_COLUMNS + column, ds.to(input_dtype), mask=inside)
            low_sum += tl.sum(tl.where(column <= 0, ds, 0.0), axis=0)
            high_sum += tl.sum(tl.where(column >= NUM_COLUMNS - 1, ds, 0.0), axis=0)

    tl.store(DK + offs_n[:, None]*stride_dkn + offs_k[None, :]*stride_dkk, dk.to(input_dtype), mask=mask_n[:, None])
    tl.store(DV + offs_n[:, None]*stride_dvn + offs_k[None, :]*stride_dvk, dv.to(input_dtype), mask=mask_n[:, None])
    if HAS_P2C:
        band_rows = P2C_BAND + offs_n*NUM_COLUMNS
        tl.store(band_rows, low_sum.to(input_dtype), mask=mask_n)
        tl.store(band_rows + NUM_COLUMNS - 1, high_sum.to(input_dtype), mask=mask_n)


@triton.jit
def _bwd_q_dise_kernel(
    Q, K, V, SEQ_LENGTHS, K_POS, Q_POS, POS_LUT, sm_scale, DO,
    DQ, C2P_BAND,
    L, Delta,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_doz, stride_doh, stride_dom, stride_dok,
    stride_dqz, stride_dqh, stride_dqm, stride_dqk,
    stride_pk0, stride_pk1, stride_pk2, stride_pk3,
    stride_pq0, stride_pq1, stride_pq2, stride_pq3,
    stride_band_h, stride_band_z,
    Z, H, M, N, P_SEQ, BAND_LOW, NUM_COLUMNS,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr, HAS_C2P: tl.constexpr, HAS_P2C: tl.constexpr,
):
    input_dtype = Q.dtype.element_ty
    log2e: tl.constexpr = 1.4426950408889634
    qk_scale = sm_scale * log2e

    start_m = tl.program_id(0)
    off_h   = tl.program_id(1)
    off_z   = tl.program_id(2)

    Q  += off_z*stride_qz  + off_h*stride_qh
    K  += off_z*stride_kz  + off_h*stride_kh
    V  += off_z*stride_vz  + off_h*stride_vh
    DO += off_z*stride_doz + off_h*stride_doh
    DQ += off_z*stride_dqz + off_h*stride_dqh

    if HAS_C2P:
        K_POS += off_z.to(tl.int64)*stride_pk0 + off_h*stride_pk1
        C2P_BAND += off_h.to(tl.int64)*stride_band_h + off_z.to(tl.int64)*stride_band_z
    if HAS_P2C:
        Q_POS += off_z.to(tl.int64)*stride_pq0 + off_h*stride_pq1

    L     += (off_z*H + off_h) * M
    Delta += (off_z*H + off_h) * M

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_DMODEL)

    seq_length = tl.load(SEQ_LENGTHS + off_z).to(tl.int32)
    mask_m = offs_m < seq_length

    q  = tl.load(Q + offs_m[:, None]*stride_qm + offs_k[None, :]*stride_qk, mask=mask_m[:, None], other=0.0)
    do = tl.load(DO + offs_m[:, None]*stride_dom + offs_k[None, :]*stride_dok, mask=mask_m[:, None], other=0.0)
    delta = tl.load(Delta + offs_m, mask=mask_m, other=0.0)
    l = tl.load(L + offs_m, mask=mask_m, other=0.0)

    dq = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # C2P gradients of distances saturated into the first/last slot, per query row.
    low_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
    high_sum = tl.zeros([BLOCK_M], dtype=tl.float32)

    n_limit = ((seq_length + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
    if CAUSAL:
        hi = tl.minimum(n_limit, P_SEQ + (start_m + 1) * BLOCK_M)
        hi = tl.minimum(hi, N)
    else:
        hi = n_limit

    k_ptrs = K + (offs_n_base[:, None] * stride_kn + offs_k[None, :] * stride_kk)
    v_ptrs = V + (offs_n_base[:, None] * stride_vn + offs_k[None, :] * stride_vk)

    for start_n in range(0, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + offs_n_base
        mask_n = offs_n < seq_length
        valid = mask_m[:, None] & mask_n[None, :]
        if CAUSAL:
            valid = valid & ((P_SEQ + offs_m[:, None]) >= offs_n[None, :])

        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0, cache_modifier=".cg")
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0, cache_modifier=".cg")

        s = tl.dot(q, tl.trans(k))

        distance = offs_m[:, None] - offs_n[None, :] + N - 1
        if HAS_C2P or HAS_P2C:
            slot = tl.load(POS_LUT + distance, mask=valid, other=0)
        if HAS_C2P:
            s += tl.load(K_POS + offs_m[:, None]*stride_pk2 + slot*stride_pk3,
                         mask=valid, other=0.0).to(tl.float32)
        if HAS_P2C:
            s += tl.load(Q_POS + offs_n[None, :]*stride_pq2 + slot*stride_pq3,
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
            tl.store(C2P_BAND + offs_m[:, None]*NUM_COLUMNS + column, ds.to(input_dtype), mask=inside)
            high_sum += tl.sum(tl.where(column <= 0, ds, 0.0), axis=1)
            low_sum += tl.sum(tl.where(column >= NUM_COLUMNS - 1, ds, 0.0), axis=1)

        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn

    tl.store(DQ + offs_m[:, None]*stride_dqm + offs_k[None, :]*stride_dqk, dq.to(input_dtype), mask=mask_m[:, None])
    if HAS_C2P:
        band_rows = C2P_BAND + offs_m*NUM_COLUMNS
        tl.store(band_rows, high_sum.to(input_dtype), mask=mask_m)
        tl.store(band_rows + NUM_COLUMNS - 1, low_sum.to(input_dtype), mask=mask_m)


def flash_attn_v2_bwd_dise(o, do, q, k, v, seq_lengths, k_pos, q_pos, L, causal, sm_scale,
                           pos_lut, band_low, num_columns, kv_config, q_config):
    """
    Performs the backward pass of FlashAttention with DeBERTa-style disentangled relative attention.

    Args:
        o: Forward output tensor of shape (B, H, M, D)
        do: Gradient of output tensor of shape (B, H, M, D)
        q, k, v: Inputs of the forward pass
        seq_lengths: Tensor of shape (B,) with per-sequence lengths, or None for full length
        k_pos: C2P scores of shape (B, H, M, 2 * ATT_SPAN), or None
        q_pos: P2C scores of shape (B, H, N, 2 * ATT_SPAN), or None
        L: Log-sum-exp tensor from forward pass
        causal: Whether causal masking was applied
        sm_scale: Softmax scale factor
        pos_lut, band_low, num_columns: position plan (see ``ops.position``)
        kv_config, q_config: (BLOCK_M, BLOCK_N, num_stages, num_warps) of the dK/dV and dQ kernels

    Returns:
        dq, dk, dv: Gradients for q, k, v
        c2p_band: (H, B, M, num_columns) score gradients per distance column, or None
        p2c_band: (H, B, N, num_columns) score gradients per distance column, or None
    """
    B, H, M, D = q.shape
    N = k.shape[2]
    P_SEQ = N - M

    if seq_lengths is None:
        seq_lengths = torch.full((B,), M, dtype=torch.int32, device=q.device)

    has_c2p = k_pos is not None
    has_p2c = q_pos is not None

    delta = torch.zeros_like(L)
    BLOCK_M = 64
    grid = (cdiv(M, BLOCK_M), H, B)
    with torch.cuda.device(q.device.index):
        _bwd_preprocess[grid](
            o, do, delta,
            seq_lengths,
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            delta.stride(0), delta.stride(1), delta.stride(2),
            M,
            BLOCK_M=BLOCK_M, D_HEAD=D, DIVISIBLE_M=(M % BLOCK_M) == 0,
        )

    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    dq = torch.zeros_like(q)
    # Columns a row never reaches must read as zero.
    c2p_band = torch.zeros((H, B, M, num_columns), device=q.device, dtype=q.dtype) if has_c2p else None
    p2c_band = torch.zeros((H, B, N, num_columns), device=q.device, dtype=q.dtype) if has_p2c else None

    stride_pk = k_pos.stride() if has_c2p else (0, 0, 0, 0)
    stride_pq = q_pos.stride() if has_p2c else (0, 0, 0, 0)
    stride_c2p_band = c2p_band.stride()[:2] if has_c2p else (0, 0)
    stride_p2c_band = p2c_band.stride()[:2] if has_p2c else (0, 0)

    kv_args = (
        q, k, v, seq_lengths, k_pos, q_pos, pos_lut, sm_scale, do,
        dk, dv, p2c_band,
        L, delta,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
        dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
        *stride_pk,
        *stride_pq,
        *stride_p2c_band,
        B, H, M, N, P_SEQ, band_low, num_columns,
    )
    q_args = (
        q, k, v, seq_lengths, k_pos, q_pos, pos_lut, sm_scale, do,
        dq, c2p_band,
        L, delta,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
        *stride_pk,
        *stride_pq,
        *stride_c2p_band,
        B, H, M, N, P_SEQ, band_low, num_columns,
    )
    kwargs = dict(BLOCK_DMODEL=D, CAUSAL=causal, HAS_C2P=has_c2p, HAS_P2C=has_p2c)
    key = (D, q.dtype, causal, has_c2p, has_p2c)
    with torch.cuda.device(q.device.index):
        launch(_bwd_kv_dise_kernel, ("bwd_kv",) + key, kv_config,
               lambda block_m, block_n: ((cdiv(N, block_n), H, B), kv_args, kwargs))
        launch(_bwd_q_dise_kernel, ("bwd_q",) + key, q_config,
               lambda block_m, block_n: ((cdiv(M, block_m), H, B), q_args, kwargs))

    return dq, dk, dv, c2p_band, p2c_band


def clear_config_cache():
    """Clear cached launch configs and position plans."""
    clear_launch_cache()
    clear_position_cache()


def _head_rows(layer):
    """(B, H, L, D) -> (H, B * L, D)."""
    B, H, L, D = layer.shape
    return layer.permute(1, 0, 2, 3).reshape(H, B * L, D)


class FlashAttentionDisentangled(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, seq_lengths, pos_key_layer, pos_query_layer, causal,
                sm_scale, position_buckets, max_relative_distance):

        Dq, Dk, Dv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Dq == Dk == Dv, "Query, key, and value must have the same head dimension"

        B, H, M, D = q.shape
        N = k.shape[2]
        if sm_scale is None:
            sm_scale = 1. / math.sqrt(D)

        ATT_SPAN = position_buckets if position_buckets > 0 else max_relative_distance
        plan = position_plan(M, N, position_buckets, max_relative_distance, ATT_SPAN, q.device)

        # C2P/P2C scores are plain GEMMs; the kernel gathers them per (query, key) pair.
        pos_key = pos_query = None
        if pos_key_layer is not None:
            pos_key = torch.matmul(q, position_table(pos_key_layer, H, ATT_SPAN).transpose(-1, -2))
        if pos_query_layer is not None:
            pos_query = torch.matmul(k, position_table(pos_query_layer, H, ATT_SPAN).transpose(-1, -2))

        BLOCK_M, BLOCK_N, num_stages, num_warps = get_fwd_config(
            B, H, M, N, D, causal, disentangled=True, att_span=ATT_SPAN
        )

        o, L = flash_attn_v2_fwd_dise(
            q, k, v, seq_lengths, pos_key, pos_query, plan.lut, causal, sm_scale,
            BLOCK_M, BLOCK_N, num_warps, num_stages,
        )

        # Position scores are recomputed in backward instead of being kept alive.
        ctx.save_for_backward(q, k, v, pos_key_layer, pos_query_layer, o, L)
        ctx.seq_lengths = seq_lengths
        ctx.sm_scale = sm_scale
        ctx.causal = causal
        ctx.position_buckets = position_buckets
        ctx.max_relative_distance = max_relative_distance
        ctx.ATT_SPAN = ATT_SPAN
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, pos_key_layer, pos_query_layer, o, L = ctx.saved_tensors
        causal = ctx.causal
        ATT_SPAN = ctx.ATT_SPAN
        B, H, M, D = q.shape
        N = k.shape[2]
        plan = position_plan(M, N, ctx.position_buckets, ctx.max_relative_distance, ATT_SPAN, q.device)

        pos_key = pos_query = None
        if pos_key_layer is not None:
            key_table = position_table(pos_key_layer, H, ATT_SPAN)
            pos_key = torch.matmul(q, key_table.transpose(-1, -2))
        if pos_query_layer is not None:
            query_table = position_table(pos_query_layer, H, ATT_SPAN)
            pos_query = torch.matmul(k, query_table.transpose(-1, -2))

        kv_config, q_config = get_bwd_config(
            B, H, M, N, D, causal,
            disentangled=(pos_key is not None or pos_query is not None),
            att_span=ATT_SPAN,
            dtype=q.dtype
        )

        do = do.contiguous() if do.stride(-1) != 1 else do
        dq, dk, dv, c2p_band, p2c_band = flash_attn_v2_bwd_dise(
            o, do, q, k, v, ctx.seq_lengths, pos_key, pos_query, L, causal, ctx.sm_scale,
            plan.lut, plan.band_low, plan.num_columns, kv_config, q_config,
        )

        d_pos_key_layer = d_pos_query_layer = None
        if c2p_band is not None or p2c_band is not None:
            fold = fold_matrix(M, N, ctx.position_buckets, ctx.max_relative_distance, ATT_SPAN, q.device, q.dtype)
        if c2p_band is not None:
            grad_rows, grad_table = fold_band_gradients(
                c2p_band.view(H, B * M, -1), _head_rows(q), key_table, fold.flip(0))
            dq = dq + grad_rows.view(H, B, M, D).permute(1, 0, 2, 3)
            d_pos_key_layer = grad_table.view(pos_key_layer.shape)
        if p2c_band is not None:
            grad_rows, grad_table = fold_band_gradients(
                p2c_band.view(H, B * N, -1), _head_rows(k), query_table, fold)
            dk = dk + grad_rows.view(H, B, N, D).permute(1, 0, 2, 3)
            d_pos_query_layer = grad_table.view(pos_query_layer.shape)

        return dq, dk, dv, None, d_pos_key_layer, d_pos_query_layer, None, None, None, None


def flash_attention_with_disentangled(q, k, v, seq_lengths, pos_key_layer, pos_query_layer, causal=False,
                                      sm_scale=None, position_buckets=0, max_relative_distance=0):
    """
    Exact DeBERTa disentangled attention.

    Args:
        q, k, v: (B, H, L, D)
        seq_lengths: (B,) int32 valid lengths (right padding), or None
        pos_key_layer: (1, H, 2 * ATT_SPAN, D) projected relative embeddings for C2P, or None
        pos_query_layer: (1, H, 2 * ATT_SPAN, D) projected relative embeddings for P2C, or None
    """
    return FlashAttentionDisentangled.apply(q, k, v, seq_lengths, pos_key_layer, pos_query_layer, causal,
                                            sm_scale, position_buckets, max_relative_distance)
