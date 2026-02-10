import os
import math
import torch
import triton
import warnings
import functools
import triton.language as tl

def calculate_shared_memory_usage(BLOCK_M, BLOCK_N, BLOCK_DMODEL, num_stages, dtype, 
                                 has_c2p=False, has_p2c=False, ATT_SPAN=0):
    """
    Calculate the shared memory requirements for Flash Attention with disentangled attention.
    
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
    if dtype == torch.float16:
        dtype_size = 2
    elif dtype == torch.float32:
        dtype_size = 4
    else:
        dtype_size = 2

    q_size = BLOCK_M * BLOCK_DMODEL * dtype_size
    k_size = BLOCK_N * BLOCK_DMODEL * dtype_size
    v_size = BLOCK_N * BLOCK_DMODEL * dtype_size
    
    attn_matrix_size = BLOCK_M * BLOCK_N * dtype_size
    accumulator_size = BLOCK_M * BLOCK_DMODEL * dtype_size
    
    pos_memory = 0
    if has_c2p:
        pos_memory += BLOCK_M * 2 * ATT_SPAN * dtype_size
    if has_p2c:
        pos_memory += BLOCK_N * 2 * ATT_SPAN * dtype_size
    
    additional_buffers = BLOCK_M * BLOCK_N * 4
    
    memory_per_stage = q_size + k_size + v_size + attn_matrix_size + pos_memory + additional_buffers
    
    total_shared_memory = num_stages * memory_per_stage + accumulator_size
    
    return total_shared_memory // 2


def calculate_shared_memory_usage_bwd(
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
    Rough shared-memory estimator for FlashAttention v2 backward with optional
    DeBERTa-style disentangled biases.
    """
    if dtype == torch.float16:
        t_sz = 2
    elif dtype == torch.bfloat16:
        t_sz = 2
    elif dtype == torch.float32:
        t_sz = 4
    else:
        t_sz = 2

    q_size  = BLOCK_M * BLOCK_DMODEL * t_sz
    k_size  = BLOCK_N * BLOCK_DMODEL * t_sz
    v_size  = BLOCK_N * BLOCK_DMODEL * t_sz
    o_size  = BLOCK_M * BLOCK_DMODEL * t_sz
    do_size = BLOCK_M * BLOCK_DMODEL * t_sz

    probs_size = BLOCK_M * BLOCK_N * (t_sz if recompute_probs else 0)

    lse_size = BLOCK_M * 2 * 4 if store_lse else BLOCK_M * 2 * 4

    dq_acc = BLOCK_M * BLOCK_DMODEL * t_sz if accum_dq else 0
    dk_acc = BLOCK_N * BLOCK_DMODEL * t_sz if accum_dkv else 0
    dv_acc = BLOCK_N * BLOCK_DMODEL * t_sz if accum_dkv else 0

    pos_mem = 0
    if has_c2p:
        pos_mem += BLOCK_M * 2 * ATT_SPAN * t_sz
    if has_p2c:
        pos_mem += BLOCK_N * 2 * ATT_SPAN * t_sz

    misc = 8 * 1024

    per_stage = (q_size + k_size + v_size +
                 o_size + do_size +
                 probs_size + lse_size +
                 dq_acc + dk_acc + dv_acc +
                 pos_mem + misc)

    total = num_stages * per_stage

    return total//2

def cdiv(a, b):
    return (a + b - 1) // b

@triton.jit
def _fwd_kernel_deberta_disentangled_attention(
    Q, K, V,
    K_POS, Q_POS,
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
    DIVISIBLE_M: tl.constexpr, DIVISIBLE_N: tl.constexpr,
    HAS_C2P: tl.constexpr, HAS_P2C: tl.constexpr,
    ATT_SPAN: tl.constexpr,
    NUM_BUCKETS: tl.constexpr, MAX_DISTANCE: tl.constexpr
):
    input_dtype = Q.dtype.element_ty

    start_m = tl.program_id(0)
    off_h   = tl.program_id(1)
    off_z   = tl.program_id(2)

    log2e: tl.constexpr = 1.4426950408889634

    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_h * stride_kh
    V += off_z * stride_vz + off_h * stride_vh
    O += off_z * stride_oz + off_h * stride_oh
    L += (off_z * H + off_h) * M

    if HAS_C2P:
        K_POS += off_z*stride_pk0 + off_h*stride_pk1
    if HAS_P2C:
        Q_POS += off_z*stride_pq0 + off_h*stride_pq1

    offs_m_base = tl.arange(0, BLOCK_M)
    offs_m = start_m * BLOCK_M + offs_m_base
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_DMODEL)

    q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)
    o_ptrs = O + (offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok)
    l_ptrs = L + offs_m

    seq_length = tl.load(SEQ_LENGTHS+off_z).to(tl.int32)
    mask_m = offs_m < seq_length

    q = tl.load(q_ptrs, mask=mask_m[:, None], cache_modifier=".cg")

    m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    offs_n_init = offs_n_base
    k_ptrs = K + (offs_k[:, None] * stride_kk + offs_n_init[None, :] * stride_kn)
    v_ptrs = V + (offs_n_init[:, None] * stride_vn + offs_k[None, :] * stride_vk)

    n_limit = ((seq_length + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
    if IS_CAUSAL:
        hi = tl.minimum(n_limit, P_SEQ + (start_m + 1) * BLOCK_M)
        hi = tl.minimum(hi, N)
    else:
        hi = n_limit

    mid_val = NUM_BUCKETS // 2
    inv_log_denom = 1.0 / tl.log((MAX_DISTANCE - 1) / mid_val)

    for start_n in range(0, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + offs_n_base

        mask_n = offs_n < seq_length

        k = tl.load(k_ptrs, mask=mask_n[None, :], cache_modifier=".cg")
        v = tl.load(v_ptrs, mask=mask_n[:, None], cache_modifier=".cg")

        s = tl.zeros([BLOCK_M, BLOCK_N], dtype=input_dtype)
        s += tl.dot(q, k) * sm_scale

        relative_positions = offs_m[:, None]-offs_n[None, :]

        sign = tl.where(relative_positions > 0.0, 1.0, tl.where(relative_positions < 0.0, -1.0, 0.0))

        abs_relative = tl.abs(relative_positions)
        condition = (relative_positions < mid_val) & (relative_positions > -mid_val)
        abs_pos = tl.where(condition, mid_val - 1.0, abs_relative)

        log_scaled = (tl.log(abs_pos / mid_val)) * inv_log_denom * (mid_val - 1.0)
        log_pos = tl.ceil(log_scaled) + mid_val

        bucket_pos = tl.where(abs_pos <= mid_val, relative_positions, log_pos * sign)

        if HAS_C2P:
            c2p_index = tl.minimum(tl.maximum(bucket_pos + ATT_SPAN, 0), 2 * ATT_SPAN - 1).to(tl.int32)
            # K_POS indexed by query position: [B, H, M, 2*ATT_SPAN]
            # offs_m[:, None] shape [BLOCK_M, 1], c2p_index shape [BLOCK_M, BLOCK_N]
            k_pos_ptrs = K_POS + offs_m[:, None] * stride_pk2 + c2p_index * stride_pk3
            c2p_bias = tl.load(k_pos_ptrs, mask=mask_m[:, None] & (c2p_index < 2*ATT_SPAN), other=0.0, cache_modifier=".cg")
            s += c2p_bias * sm_scale

        if HAS_P2C:
            p2c_index = tl.minimum(tl.maximum(bucket_pos + ATT_SPAN, 0), 2 * ATT_SPAN - 1).to(tl.int32)
            # Q_POS indexed by key position: [B, H, N, 2*ATT_SPAN]
            # Need to transpose to access by offs_n
            p2c_index_t = p2c_index.trans(1, 0)  # [BLOCK_N, BLOCK_M]
            # offs_n[:, None] shape [BLOCK_N, 1], p2c_index_t shape [BLOCK_N, BLOCK_M]
            q_pos_ptrs = Q_POS + offs_n[:, None] * stride_pq2 + p2c_index_t * stride_pq3
            p2c_bias_t = tl.load(q_pos_ptrs, mask=mask_n[:, None] & (p2c_index_t < 2*ATT_SPAN), other=0.0, cache_modifier=".cg")
            # Transpose back to [BLOCK_M, BLOCK_N]
            p2c_bias = p2c_bias_t.trans(1, 0)
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

        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn

    if IS_CAUSAL and LARGER_M:
        is_empty_line = (offs_m + P_SEQ) < 0
        acc = tl.where(is_empty_line[:, None], 0.0, acc * (1.0 / l_i[:, None]))
        l = tl.where(is_empty_line, float("-inf"), m_i + tl.log(l_i))
    else:
        acc = acc * (1.0 / l_i[:, None])
        l = m_i + tl.log(l_i)

    tl.store(l_ptrs, l, mask=mask_m, cache_modifier=".cg")
    tl.store(o_ptrs, acc.to(q.dtype), mask=mask_m[:, None], cache_modifier=".cg")

@functools.lru_cache(maxsize=128)
def _get_fwd_config_cached(M, N, D, causal, disentangled, att_span):
    """
    Cached version of configuration computation for fixed-length forward pass.
    """
    # Check for environment variable overrides
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

    if capability[0] >= 8 :
        if not causal:
            if D <= 64:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 3, 4
            else:
                if M <= 1024:
                    BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 3, 4
                else:
                    BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 128, 3, 8
        else:
            if D <= 64:
                if disentangled:
                    BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 3, 4
                else:
                    BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 4, 4
            else:
                if M <= 1024:
                    BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 2, 4
                else:
                    BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 128, 3, 8
    elif capability[0] == 8:
        if not causal:
            if D <= 64:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 3, 4
            else:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 2, 4
        else:
            if D <= 64:
                if disentangled:
                    BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 3, 4
                else:
                    BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 3, 4
            else:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 2, 4
    else:
        BLOCK_M, BLOCK_N, num_stages, num_warps = 16, 16, 10, 4

    has_pos = disentangled
    ATT_SPAN = att_span if has_pos else 0

    dtype = torch.float16

    shared_mem_usage = calculate_shared_memory_usage(
        BLOCK_M, BLOCK_N, D, num_stages, dtype,
        has_c2p=has_pos, has_p2c=has_pos, ATT_SPAN=ATT_SPAN
    )

    while shared_mem_usage > max_shared_memory and (BLOCK_M > 16 or BLOCK_N > 16 or num_stages > 1):
        if num_stages > 1:
            num_stages -= 1
        if BLOCK_M > 32 and BLOCK_N > 32:
            BLOCK_M //= 2
            BLOCK_N //= 2
        elif BLOCK_M > 32:
            BLOCK_M //= 2
        elif BLOCK_N > 32:
            BLOCK_N //= 2
        elif BLOCK_M > 16:
            BLOCK_M //= 2
        elif BLOCK_N > 16:
            BLOCK_N //= 2

        shared_mem_usage = calculate_shared_memory_usage(
            BLOCK_M, BLOCK_N, D, num_stages, dtype,
            has_c2p=has_pos, has_p2c=has_pos, ATT_SPAN=ATT_SPAN
        )

    return (BLOCK_M, BLOCK_N, num_stages, num_warps)


def get_fwd_config(B, H, M, N, D, causal, disentangled=False, max_shared_memory=None, att_span=256):
    """
    Determine optimal kernel configuration parameters.
    Uses caching to avoid recomputing configurations for the same inputs.

    Args:
        B: Batch size (unused, kept for API compatibility)
        H: Number of heads (unused, kept for API compatibility)
        M: Query sequence length
        N: Key sequence length
        D: Per-head dimension
        causal: Whether causal masking is applied
        disentangled: Whether to use DeBERTa-style disentangled attention
        max_shared_memory: Maximum shared memory (unused, auto-detected)
        att_span: Size of the attention span for relative positions

    Returns:
        Tuple (BLOCK_M, BLOCK_N, num_stages, num_warps)
    """
    config = _get_fwd_config_cached(M, N, D, causal, disentangled, att_span)

    warnings.warn(f"INFO: Fixed-length forward config is {config[0]}, {config[1]}, {config[2]}, {config[3]} for BLOCK_M, BLOCK_N, stages and warps, respectively.\n"
                  "INFO: If you want to change it, set FLASHDEBERTA_FWD_BLOCK_M, FLASHDEBERTA_FWD_BLOCK_N, FLASHDEBERTA_FWD_NUM_STAGES, FLASHDEBERTA_FWD_NUM_WARPS environment variables.")
    return config


def flash_attn_v2_fwd_dise(q, k, v, seq_lengths, pos_key, pos_query, causal, sm_scale, BLOCK_M, BLOCK_N,
                           position_buckets, max_relative_distance, num_warps, num_stages, ATT_SPAN):
    """
    Performs the forward pass of FlashAttention with DeBERTa-style disentangled relative attention.

    Args:
        q: Query tensor of shape (B, H, M, D)
        k: Key tensor of shape (B, H, N, D)
        v: Value tensor of shape (B, H, N, D)
        seq_lengths: Tensor of shape (B,) containing sequence lengths for each batch element.
                     If None, all sequences are assumed to have length M (full sequence).
        pos_key: Positional key tensor for C2P bias, or None
        pos_query: Positional query tensor for P2C bias, or None
        causal: Whether to apply causal masking
        sm_scale: Softmax scale factor
        BLOCK_M, BLOCK_N: Block sizes for tiling
        position_buckets: Number of relative position buckets
        max_relative_distance: Maximum relative distance
        num_warps, num_stages: Triton kernel parameters
        ATT_SPAN: Attention span for relative positions

    Returns:
        o: Output tensor of shape (B, H, M, D)
        L: Log-sum-exp tensor of shape (B, H, M)
    """
    B, H, M, D = q.shape
    N = k.shape[2]
    P_SEQ = N - M

    if sm_scale is None:
        sm_scale = 1. / math.sqrt(D)

    # Handle seq_lengths=None: create default tensor with full sequence lengths
    if seq_lengths is None:
        seq_lengths = torch.full((B,), M, dtype=torch.int32, device=q.device)

    has_c2p = pos_key is not None
    has_p2c = pos_query is not None

    larger_m = M > N

    divisible_m = (M % BLOCK_M) == 0
    divisible_n = (N % BLOCK_N) == 0

    grid = (cdiv(M, BLOCK_M), H, B)
    o = torch.zeros_like(q)
    L = torch.zeros((B, H, M), device=q.device, dtype=torch.float32)

    if has_c2p:
        stride_pk0, stride_pk1, stride_pk2, stride_pk3 = pos_key.stride()
    else:
        stride_pk0 = stride_pk1 = stride_pk2 = stride_pk3 = 0
    if has_p2c:
        stride_pq0, stride_pq1, stride_pq2, stride_pq3 = pos_query.stride()
    else:
        stride_pq0 = stride_pq1 = stride_pq2 = stride_pq3 = 0

    with torch.cuda.device(q.device.index):
        _fwd_kernel_deberta_disentangled_attention[grid](
            q, k, v,
            pos_key, pos_query,
            L, o,
            seq_lengths,
            sm_scale,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            stride_pk0, stride_pk1, stride_pk2, stride_pk3,
            stride_pq0, stride_pq1, stride_pq2, stride_pq3,
            B, H, M, N, P_SEQ,
            BLOCK_M=BLOCK_M, BLOCK_DMODEL=D, BLOCK_N=BLOCK_N,
            IS_CAUSAL=causal, LARGER_M=larger_m,
            DIVISIBLE_M=divisible_m, DIVISIBLE_N=divisible_n,
            HAS_C2P=has_c2p, HAS_P2C=has_p2c,
            ATT_SPAN=ATT_SPAN,
            NUM_BUCKETS=position_buckets,
            MAX_DISTANCE=max_relative_distance,
            num_warps=num_warps, num_stages=num_stages,
        )

    return o, L

@functools.lru_cache(maxsize=128)
def _get_bwd_config_cached(M, N, D, causal, disentangled, att_span, dtype_size):
    """
    Cached version of configuration computation for fixed-length backward pass.
    dtype_size: 2 for float16/bfloat16, 4 for float32
    """
    # Check for environment variable overrides
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

    if cap[0] >= 9:
        if D <= 64:
            BLOCK_M, BLOCK_N, num_stages, num_warps = (64, 64, 3, 4) if not causal else (128, 64, 3, 4)
        else:
            BLOCK_M, BLOCK_N, num_stages, num_warps = (64, 64, 2, 8) if not causal else (128, 64, 2, 8)
    elif cap[0] >= 8:
        if D <= 64:
            if causal:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 3, 4
            else:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 3, 4
        else:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 2, 8
    else:
        BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 2, 4

    if N >= 2 * M and BLOCK_N < 128 and cap[0] >= 8:
        BLOCK_N = 128
        num_warps = max(num_warps, 8)

    has_pos = bool(disentangled)
    ATT_SPAN = att_span if has_pos else 0

    # Convert dtype_size back to dtype for calculation
    if dtype_size == 2:
        dtype = torch.float16
    elif dtype_size == 4:
        dtype = torch.float32
    else:
        dtype = torch.float16

    shm = calculate_shared_memory_usage_bwd(
        BLOCK_M, BLOCK_N, D, num_stages, dtype,
        has_c2p=has_pos, has_p2c=has_pos, ATT_SPAN=ATT_SPAN,
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

        shm = calculate_shared_memory_usage_bwd(
            BLOCK_M, BLOCK_N, D, num_stages, dtype,
            has_c2p=has_pos, has_p2c=has_pos, ATT_SPAN=ATT_SPAN,
            store_lse=True, recompute_probs=True, accum_dq=True, accum_dkv=True,
        )

    if D <= 64 and BLOCK_M * BLOCK_N <= 128 * 64:
        num_warps = min(num_warps, 4)
    else:
        num_warps = max(num_warps, 8 if cap[0] >= 8 else 4)

    return (BLOCK_M, BLOCK_N, num_stages, num_warps)


def get_bwd_config(
    B, H, M, N, D, causal,
    *,
    disentangled: bool = False,
    att_span: int = 256,
    dtype = torch.float16,
    max_shared_memory: int | None = None,
):
    """
    Heuristic selector for backward kernel tiling.
    Uses caching to avoid recomputing configurations for the same inputs.

    Args:
        B: Batch size (unused, kept for API compatibility)
        H: Number of heads (unused, kept for API compatibility)
        M: Query sequence length
        N: Key sequence length
        D: Per-head dimension
        causal: Whether causal masking is applied
        disentangled: Whether to use DeBERTa-style disentangled attention
        att_span: Size of the attention span for relative positions
        dtype: Data type of tensors
        max_shared_memory: Maximum shared memory (unused, auto-detected)

    Returns:
        Tuple (BLOCK_M, BLOCK_N, num_stages, num_warps)
    """
    # Convert dtype to size for caching (dtypes are not hashable)
    if dtype == torch.float16 or dtype == torch.bfloat16:
        dtype_size = 2
    elif dtype == torch.float32:
        dtype_size = 4
    else:
        dtype_size = 2

    config = _get_bwd_config_cached(M, N, D, causal, disentangled, att_span, dtype_size)

    warnings.warn(
        f"INFO: Fixed-length backward config -> "
        f"BLOCK_M={config[0]}, BLOCK_N={config[1]}, stages={config[2]}, warps={config[3]}."
        "\nINFO: Set FLASHDEBERTA_BWD_BLOCK_M, FLASHDEBERTA_BWD_BLOCK_N, FLASHDEBERTA_BWD_NUM_STAGES, FLASHDEBERTA_BWD_NUM_WARPS to override."
    )

    return config

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
    Q, K, V, SEQ_LENGTHS, K_POS, Q_POS, sm_scale, DO,
    DK, DV, DQPOS,
    L, Delta,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_doz, stride_doh, stride_dom, stride_dok,
    stride_dkz, stride_dkh, stride_dkn, stride_dkk,
    stride_dvz, stride_dvh, stride_dvn, stride_dvk,
    stride_pk0, stride_pk1, stride_pk2, stride_pk3,
    stride_pq0, stride_pq1, stride_pq2, stride_pq3,
    Z, H, M, N, P_SEQ,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    HAS_C2P: tl.constexpr, HAS_P2C: tl.constexpr,
    DIVISIBLE_M: tl.constexpr, DIVISIBLE_N: tl.constexpr,
    ATT_SPAN: tl.constexpr, NUM_BUCKETS: tl.constexpr, MAX_DISTANCE: tl.constexpr,
):
    input_dtype = Q.dtype.element_ty
    log2e: tl.constexpr = 1.4426950408889634

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
        K_POS  += off_z*stride_pk0 + off_h*stride_pk1
    if HAS_P2C:
        Q_POS  += off_z*stride_pq0 + off_h*stride_pq1
        DQPOS  += off_z*stride_pq0 + off_h*stride_pq1

    L     += (off_z*H + off_h) * M
    Delta += (off_z*H + off_h) * M

    seq_length = tl.load(SEQ_LENGTHS+off_z).to(tl.int32)

    m_limit = ((seq_length + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
    if CAUSAL:
        lo = tl.maximum(start_n * BLOCK_N - P_SEQ - (BLOCK_M - 1), 0)
        lo = ((lo + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
    else:
        lo = 0

    offs_m_base = tl.arange(0, BLOCK_M)
    offs_n      = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k      = tl.arange(0, BLOCK_DMODEL)

    k_ptrs  = K  + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
    v_ptrs  = V  + (offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk)

    dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_k[None, :] * stride_dkk)
    dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_k[None, :] * stride_dvk)

    mask_n = offs_n < seq_length
    k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0, cache_modifier=".cg")
    v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0, cache_modifier=".cg")

    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)

    mid_val = NUM_BUCKETS // 2
    inv_log_denom = 1.0 / tl.log((MAX_DISTANCE - 1) / mid_val)
    for start_m in range(lo, m_limit, BLOCK_M):
        offs_m  = start_m + offs_m_base
        mask_m  = offs_m < seq_length

        q  = tl.load(Q + offs_m[:, None]*stride_qm + offs_k[None,:]*stride_qk,  mask=mask_m[:, None], other=0.0)
        do = tl.load(DO + offs_m[:, None]*stride_dom + offs_k[None,:]*stride_dok, mask=mask_m[:, None], other=0.0)
        l  = tl.load(L + offs_m, mask=mask_m, other=float("inf"))
        delta = tl.load(Delta + offs_m, mask=mask_m, other=0.0)

        s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        s += tl.dot(q, tl.trans(k)) * sm_scale

        relative_positions = offs_m[:, None] - offs_n[None, :]
        sign = tl.where(relative_positions > 0, 1.0, tl.where(relative_positions < 0, -1.0, 0.0))
        
        abs_relative = tl.abs(relative_positions)
        condition = (relative_positions < mid_val) & (relative_positions > -mid_val)
        abs_pos = tl.where(condition, mid_val - 1.0, abs_relative)

        log_scaled = (tl.log(abs_pos / mid_val)) * inv_log_denom * (mid_val - 1.0)
        log_pos = tl.ceil(log_scaled) + mid_val
        bucket_pos = tl.where(abs_pos <= mid_val, relative_positions, log_pos * sign)

        if HAS_C2P:
            c2p_index = tl.minimum(tl.maximum(bucket_pos + ATT_SPAN, 0), 2*ATT_SPAN-1).to(tl.int32)
            k_pos_ptrs = K_POS + offs_m[:, None]*stride_pk2 + c2p_index*stride_pk3
            c2p_bias = tl.load(k_pos_ptrs, mask=mask_m[:, None]&(c2p_index<2*ATT_SPAN), other=0.0, cache_modifier=".cg")
            s += c2p_bias * sm_scale
        
        if HAS_P2C:
            p2c_index = tl.minimum(tl.maximum(bucket_pos + ATT_SPAN, 0), 2*ATT_SPAN-1).to(tl.int32)
            p2c_index_t = p2c_index.trans(1, 0)
            q_pos_ptrs = Q_POS + offs_n[:, None]*stride_pq2 + p2c_index_t*stride_pq3
            p2c_bias_t = tl.load(q_pos_ptrs, mask=mask_n[:, None]&(p2c_index_t<2*ATT_SPAN), other=0.0, cache_modifier=".cg")
            p2c_bias = p2c_bias_t.trans(1, 0)
            s += p2c_bias * sm_scale

        valid_mn = mask_m[:, None] & mask_n[None, :]
        if CAUSAL:
            causal_mask = (P_SEQ + offs_m[:, None]) >= offs_n[None, :]
            s = tl.where(causal_mask, s, float("-inf"))
        
        s = tl.where(valid_mn, s, float("-inf"))
        p = tl.exp2((s - l[:, None])*log2e)
        p = tl.where(valid_mn, p, 0.0)

        dv += tl.dot(tl.trans(p).to(tl.float32), do.to(tl.float32))

        dp = tl.dot(do.to(input_dtype), tl.trans(v))

        ds = p * (dp - delta[:, None])
        ds = tl.where(valid_mn, ds, 0.0)

        ds_scaled = (ds * sm_scale).to(input_dtype)

        dk += tl.dot(tl.trans(ds_scaled), q)

        if HAS_P2C:
            # Compute P2C gradients: ds_scaled is [BLOCK_M, BLOCK_N], need [BLOCK_N, BLOCK_M]
            # Cast to fp32 for atomic_add (bf16 not supported by Triton atomic operations)
            ds_scaled_fp32 = ds_scaled.to(tl.float32)
            ds_scaled_t_fp32 = ds_scaled_fp32.trans(1, 0)
            qpos_grad_ptrs = DQPOS + offs_n[:, None]*stride_pq2 + p2c_index_t*stride_pq3
            tl.atomic_add(qpos_grad_ptrs, ds_scaled_t_fp32,
                          mask=mask_n[:, None]&mask_m[None,:]&(p2c_index_t<2*ATT_SPAN),
                           sem="relaxed", scope="cta")

    tl.store(dk_ptrs, dk.to(input_dtype), mask=mask_n[:, None])
    tl.store(dv_ptrs, dv.to(input_dtype), mask=mask_n[:, None])

@triton.jit
def _bwd_q_dise_kernel(
    Q, K, V, SEQ_LENGTHS, K_POS, Q_POS, sm_scale, DO,
    DQ, DKPOS,
    L, Delta,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_doz, stride_doh, stride_dom, stride_dok,
    stride_dqz, stride_dqh, stride_dqm, stride_dqk,
    stride_pk0, stride_pk1, stride_pk2, stride_pk3,
    stride_pq0, stride_pq1, stride_pq2, stride_pq3,
    Z, H, M, N, P_SEQ,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr, HAS_C2P: tl.constexpr, HAS_P2C: tl.constexpr, LARGER_M: tl.constexpr,
    DIVISIBLE_M: tl.constexpr, DIVISIBLE_N: tl.constexpr,
    ATT_SPAN: tl.constexpr, NUM_BUCKETS: tl.constexpr, MAX_DISTANCE: tl.constexpr,
):
    input_dtype = Q.dtype.element_ty
    log2e: tl.constexpr = 1.4426950408889634

    start_m = tl.program_id(0)
    off_h   = tl.program_id(1)
    off_z   = tl.program_id(2)

    Q  += off_z*stride_qz  + off_h*stride_qh
    K  += off_z*stride_kz  + off_h*stride_kh
    V  += off_z*stride_vz  + off_h*stride_vh
    DO += off_z*stride_doz + off_h*stride_doh
    DQ += off_z*stride_dqz + off_h*stride_dqh

    if HAS_C2P:
        K_POS += off_z*stride_pk0 + off_h*stride_pk1
        DKPOS += off_z*stride_pk0 + off_h*stride_pk1
    if HAS_P2C:
        Q_POS += off_z*stride_pq0 + off_h*stride_pq1

    L     += (off_z*H + off_h) * M
    Delta += (off_z*H + off_h) * M

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_DMODEL)

    q_ptrs  = Q  + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)
    dq_ptrs = DQ + (offs_m[:, None] * stride_dqm + offs_k[None, :] * stride_dqk)
    do_ptrs = DO + (offs_m[:, None] * stride_dom + offs_k[None, :] * stride_dok)

    seq_length = tl.load(SEQ_LENGTHS+off_z).to(tl.int32)
    mask_m = offs_m < seq_length

    q  = tl.load(q_ptrs,  mask=mask_m[:, None])
    do = tl.load(do_ptrs, mask=mask_m[:, None])
    delta = tl.load(Delta + offs_m, mask=mask_m)
    l = tl.load(L + offs_m, mask=mask_m)

    dq = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    n_limit = ((seq_length + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
    if CAUSAL:
        hi = tl.minimum(n_limit, P_SEQ + (start_m + 1) * BLOCK_M)
        hi = tl.minimum(hi, N)
    else:
        hi = n_limit

    k_ptrs = K + (offs_n_base[:, None] * stride_kn + offs_k[None, :] * stride_kk)
    v_ptrs = V + (offs_n_base[:, None] * stride_vn + offs_k[None, :] * stride_vk)

    mid_val = NUM_BUCKETS // 2
    inv_log_denom = 1.0 / tl.log((MAX_DISTANCE - 1) / mid_val)

    for start_n in range(0, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + offs_n_base

        mask_n = offs_n < seq_length

        k = tl.load(k_ptrs, mask=mask_n[:, None], cache_modifier=".cg", other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], cache_modifier=".cg", other=0.0)

        s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        s += tl.dot(q, tl.trans(k)) * sm_scale

        relative_positions = offs_m[:, None] - offs_n[None, :]
        sign = tl.where(relative_positions > 0.0, 1.0, tl.where(relative_positions < 0.0, -1.0, 0.0))

        abs_relative = tl.abs(relative_positions)
        condition = (relative_positions < mid_val) & (relative_positions > -mid_val)
        abs_pos = tl.where(condition, mid_val - 1.0, abs_relative)

        log_scaled = (tl.log(abs_pos / mid_val)) * inv_log_denom * (mid_val - 1.0)
        log_pos = tl.ceil(log_scaled) + mid_val
        bucket_pos = tl.where(abs_pos <= mid_val, relative_positions, log_pos * sign)

        if HAS_C2P:
            c2p_index = tl.minimum(tl.maximum(bucket_pos + ATT_SPAN, 0), 2*ATT_SPAN - 1).to(tl.int32)
            k_pos_ptrs = K_POS + (offs_m[:, None] * stride_pk2 + c2p_index * stride_pk3)
            c2p_bias = tl.load(k_pos_ptrs, mask=mask_m[:, None] & (c2p_index < 2*ATT_SPAN), other=0.0, cache_modifier=".cg")
            s += c2p_bias * sm_scale

        if HAS_P2C:
            p2c_index = tl.minimum(tl.maximum(bucket_pos + ATT_SPAN, 0), 2*ATT_SPAN - 1).to(tl.int32)
            p2c_index_t = p2c_index.trans(1, 0)
            q_pos_ptrs = Q_POS + (offs_n[:, None] * stride_pq2 + p2c_index_t * stride_pq3)
            p2c_bias_t = tl.load(q_pos_ptrs, mask=mask_n[:, None] & (p2c_index_t < 2*ATT_SPAN), other=0.0, cache_modifier=".cg")
            p2c_bias = p2c_bias_t.trans(1, 0)
            s += p2c_bias * sm_scale

        if CAUSAL:
            causal_mask = (P_SEQ + offs_m[:, None]) >= (offs_n[None, :])

        s = tl.where(mask_m[:, None]&mask_n[None, :], s, float("-inf"))
        
        p = tl.math.exp2((s - l[:, None]) * log2e)
        if CAUSAL:
            p = tl.where(causal_mask, p, 0.0)

        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dp += tl.dot(do.to(input_dtype), tl.trans(v))

        ds = p * (dp - delta[:, None])
        if CAUSAL:
            ds = tl.where(causal_mask, ds, 0.0)

        ds_scaled = (ds * sm_scale).to(input_dtype)

        dq += tl.dot(ds_scaled, k)

        if HAS_C2P:
            # Compute C2P gradients: ds_scaled is [BLOCK_M, BLOCK_N], matches K_POS indexing
            # Cast to fp32 for atomic_add (bf16 not supported by Triton atomic operations)
            ds_scaled_fp32 = ds_scaled.to(tl.float32)
            kpos_grad_ptrs = DKPOS + offs_m[:, None]*stride_pk2 + c2p_index*stride_pk3
            tl.atomic_add(kpos_grad_ptrs, ds_scaled_fp32,
                          mask=mask_m[:, None]&mask_n[None,:]&(c2p_index<2*ATT_SPAN),
                          sem="relaxed", scope="cta")

        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn

    dq = dq.to(input_dtype)
    tl.store(dq_ptrs, dq, mask=mask_m[:, None])

def flash_attn_v2_bwd_dise(o, do, q, k, v, seq_lengths, k_pos, q_pos, L, causal, sm_scale,
                           BLOCK_M, BLOCK_N, position_buckets, max_relative_distance,
                           num_warps, num_stages, ATT_SPAN):
    """
    Performs the backward pass of FlashAttention with DeBERTa-style disentangled relative attention.

    Args:
        o: Forward output tensor of shape (B, H, M, D)
        do: Gradient of output tensor of shape (B, H, M, D)
        q: Query tensor of shape (B, H, M, D)
        k: Key tensor of shape (B, H, N, D)
        v: Value tensor of shape (B, H, N, D)
        seq_lengths: Tensor of shape (B,) containing sequence lengths for each batch element.
                     If None, all sequences are assumed to have length M (full sequence).
        k_pos: Positional key tensor for C2P bias, or None
        q_pos: Positional query tensor for P2C bias, or None
        L: Log-sum-exp tensor from forward pass
        causal: Whether causal masking was applied
        sm_scale: Softmax scale factor
        BLOCK_M, BLOCK_N: Block sizes for tiling
        position_buckets: Number of relative position buckets
        max_relative_distance: Maximum relative distance
        num_warps, num_stages: Triton kernel parameters
        ATT_SPAN: Attention span for relative positions

    Returns:
        dq, dk, dv: Gradients for q, k, v
        dk_pos, dq_pos: Gradients for positional embeddings (or None)
    """
    B, H, M, D = q.shape
    N = k.shape[2]
    P_SEQ = N - M
    larger_m = M > N
    divisible_m = (M % BLOCK_M) == 0
    divisible_n = (N % BLOCK_N) == 0

    # Handle seq_lengths=None: create default tensor with full sequence lengths
    if seq_lengths is None:
        seq_lengths = torch.full((B,), M, dtype=torch.int32, device=q.device)

    has_c2p = (k_pos is not None)
    has_p2c = (q_pos is not None)

    delta = torch.zeros_like(L)
    grid = (cdiv(M, BLOCK_M), H, B)
    with torch.cuda.device(q.device.index):
        _bwd_preprocess[grid](
            o, do, delta,
            seq_lengths,
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            delta.stride(0), delta.stride(1), delta.stride(2),
            M,
            BLOCK_M=BLOCK_M, D_HEAD=D, DIVISIBLE_M=divisible_m,
        )

    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    # bf16 is not supported in atomic_add, so we always use fp32 for positional gradients
    dk_pos = torch.zeros_like(k_pos, dtype=torch.float32) if has_c2p else None
    dq_pos = torch.zeros_like(q_pos, dtype=torch.float32) if has_p2c else None

    if has_c2p:
        stride_pk0, stride_pk1, stride_pk2, stride_pk3 = k_pos.stride()
    else:
        stride_pk0 = stride_pk1 = stride_pk2 = stride_pk3 = 0

    if has_p2c:
        stride_pq0, stride_pq1, stride_pq2, stride_pq3 = q_pos.stride()
    else:
        stride_pq0 = stride_pq1 = stride_pq2 = stride_pq3 = 0

    grid_kv = (cdiv(N, BLOCK_N), H, B)
    with torch.cuda.device(q.device.index):
        _bwd_kv_dise_kernel[grid_kv](
            q, k, v, seq_lengths, k_pos, q_pos, sm_scale, do,
            dk, dv, dq_pos,
            L, delta,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
            dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
            stride_pk0, stride_pk1, stride_pk2, stride_pk3,
            stride_pq0, stride_pq1, stride_pq2, stride_pq3,
            B, H, M, N, P_SEQ,
            BLOCK_M=BLOCK_M, BLOCK_DMODEL=D, BLOCK_N=BLOCK_N,
            CAUSAL=causal,
            HAS_C2P=has_c2p, HAS_P2C=has_p2c,
            DIVISIBLE_M=divisible_m, DIVISIBLE_N=divisible_n,
            ATT_SPAN=ATT_SPAN,
            NUM_BUCKETS=position_buckets,
            MAX_DISTANCE=max_relative_distance,
            num_warps=num_warps, num_stages=num_stages,
        )

    dq = torch.zeros_like(q)
    grid_q = (cdiv(M, BLOCK_M), H, B)
    with torch.cuda.device(q.device.index):
        _bwd_q_dise_kernel[grid_q](
            q, k, v, seq_lengths, k_pos, q_pos, sm_scale, do,
            dq, dk_pos,
            L, delta,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
            stride_pk0, stride_pk1, stride_pk2, stride_pk3,
            stride_pq0, stride_pq1, stride_pq2, stride_pq3,
            B, H, M, N, P_SEQ,
            BLOCK_M=BLOCK_M, BLOCK_DMODEL=D, BLOCK_N=BLOCK_N,
            CAUSAL=causal, HAS_C2P=has_c2p, HAS_P2C=has_p2c, LARGER_M=(M > N),
            DIVISIBLE_M=divisible_m, DIVISIBLE_N=divisible_n,
            ATT_SPAN=ATT_SPAN, NUM_BUCKETS=position_buckets, MAX_DISTANCE=max_relative_distance,
            num_warps=num_warps, num_stages=num_stages,
        )

    return dq, dk, dv, dk_pos, dq_pos

def clear_config_cache():
    """Clear the configuration caches for fixed-length kernels."""
    _get_fwd_config_cached.cache_clear()
    _get_bwd_config_cached.cache_clear()


class FlashAttentionDisentangled(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, seq_lengths, k_pos, q_pos, causal,
                sm_scale, position_buckets, max_relative_distance):

        Dq, Dk, Dv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Dq == Dk == Dv, "Query, key, and value must have the same head dimension"

        B, H, M, D = q.shape
        N = k.shape[2]
        if sm_scale is None:
            sm_scale = 1. / math.sqrt(D)

        ATT_SPAN = position_buckets if position_buckets > 0 else max_relative_distance
        BLOCK_M, BLOCK_N, num_stages, num_warps = get_fwd_config(
            B, H, M, N, D, causal, disentangled=True, att_span=ATT_SPAN
        )

        o, L = flash_attn_v2_fwd_dise(
            q, k, v, seq_lengths, k_pos, q_pos, causal, sm_scale,
            BLOCK_M, BLOCK_N, position_buckets,
            max_relative_distance, num_warps, num_stages, ATT_SPAN
        )

        ctx.save_for_backward(q, k, v, k_pos, q_pos, o, L)
        ctx.seq_lengths = seq_lengths
        ctx.sm_scale = sm_scale
        ctx.causal = causal
        ctx.position_buckets = position_buckets
        ctx.max_relative_distance = max_relative_distance
        ctx.ATT_SPAN = ATT_SPAN
        ctx.config = (BLOCK_M, BLOCK_N, num_stages, num_warps)
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, k_pos, q_pos, o, L = ctx.saved_tensors
        seq_lengths = ctx.seq_lengths
        sm_scale = ctx.sm_scale
        causal = ctx.causal
        position_buckets = ctx.position_buckets
        max_relative_distance = ctx.max_relative_distance
        ATT_SPAN = ctx.ATT_SPAN
        B, H, M, D = q.shape
        N = k.shape[2]

        BLOCK_M, BLOCK_N, num_stages, num_warps = get_bwd_config(
            B, H, M, N, D, causal,
            disentangled=(k_pos is not None or q_pos is not None),
            att_span=ATT_SPAN,
            dtype=q.dtype
        )

        dq, dk, dv, dk_pos, dq_pos = flash_attn_v2_bwd_dise(
            o, do, q, k, v, seq_lengths, k_pos, q_pos, L, causal, sm_scale,
            BLOCK_M, BLOCK_N, position_buckets, max_relative_distance,
            num_warps, num_stages, ATT_SPAN
        )

        return dq, dk, dv, None, dk_pos, dq_pos, None, None, None, None
    
def flash_attention_with_disentangled(q, k, v, seq_lengths, k_pos, q_pos, causal=False, sm_scale=None,
                                      position_buckets=0, max_relative_distance=0):
    return FlashAttentionDisentangled.apply(q, k, v, seq_lengths, k_pos, q_pos, causal, sm_scale,
                                            position_buckets, max_relative_distance)