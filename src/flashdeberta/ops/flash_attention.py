import math
import torch
import triton
import warnings
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
    additional_buffers = BLOCK_M * BLOCK_N * 4  # For relative position indices, floating-point calculations
    
    # Total memory per stage
    memory_per_stage = q_size + k_size + v_size + attn_matrix_size + pos_memory + additional_buffers
    
    # Total shared memory including all pipeline stages
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

    We count per-stage tiles:
      - q, k, v tiles
      - o, do tiles (needed to form dP)
      - probs/dP tile (M x N) if recompute_probs
      - small per-row LSE/m buffers
      - partial accumulators for dq, dk, dv
      - positional tiles for c2p / p2c (approx., span-limited)

    Notes:
      - This is an upper-bound-ish estimate; kernels keep some values in regs.
      - LSE often stored in fp32 on load; counted in shared for safety.
    """
    # dtype sizes
    if dtype == torch.float16:
        t_sz = 2
    elif dtype == torch.bfloat16:
        t_sz = 2
    elif dtype == torch.float32:
        t_sz = 4
    else:
        t_sz = 2

    # Core tiles
    q_size  = BLOCK_M * BLOCK_DMODEL * t_sz
    k_size  = BLOCK_N * BLOCK_DMODEL * t_sz
    v_size  = BLOCK_N * BLOCK_DMODEL * t_sz
    o_size  = BLOCK_M * BLOCK_DMODEL * t_sz
    do_size = BLOCK_M * BLOCK_DMODEL * t_sz

    # Recomputed probabilities / dP tile
    probs_size = BLOCK_M * BLOCK_N * (t_sz if recompute_probs else 0)

    # Per-row softmax bookkeeping (m, lse), count as fp32 for safety
    lse_size = BLOCK_M * 2 * 4 if store_lse else BLOCK_M * 2 * 4  # still bring rows

    # Partial accumulators
    dq_acc = BLOCK_M * BLOCK_DMODEL * t_sz if accum_dq else 0
    dk_acc = BLOCK_N * BLOCK_DMODEL * t_sz if accum_dkv else 0
    dv_acc = BLOCK_N * BLOCK_DMODEL * t_sz if accum_dkv else 0

    # Positional memory (heuristic upper bound)
    pos_mem = 0
    if has_c2p:
        pos_mem += BLOCK_M * 2 * ATT_SPAN * t_sz
    if has_p2c:
        pos_mem += BLOCK_N * 2 * ATT_SPAN * t_sz

    # Misc scratch (indices, masks, scales); keep small constant per tile
    misc = 8 * 1024  # 8KB safety pad

    per_stage = (q_size + k_size + v_size +
                 o_size + do_size +
                 probs_size + lse_size +
                 dq_acc + dk_acc + dv_acc +
                 pos_mem + misc)

    # Double buffering across stages (FA kernels pipeline tiles)
    total = num_stages * per_stage

    # Many kernels reuse a half-buffer scheme; keep headroom
    return total//2

def cdiv(a, b):
    return (a + b - 1) // b

@triton.jit
def _fwd_kernel_deberta_disentangled_attention(
    Q, K, V,
    K_POS, Q_POS,
    L, O,
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
    L += (off_z * H + off_h) * M  # L is of shape (B*H, M)

    if HAS_C2P:
        K_POS += off_z*stride_pk0 + off_h*stride_pk1
    if HAS_P2C:
        Q_POS += off_z*stride_pq0 + off_h*stride_pq1

    offs_m_base = tl.arange(0, BLOCK_M)
    offs_m = start_m * BLOCK_M + offs_m_base
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_DMODEL)

    q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)  # (BLOCK_M, BLOCK_DMODEL)
    o_ptrs = O + (offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok)  # (BLOCK_M, BLOCK_DMODEL)
    l_ptrs = L + offs_m

    mask_m = offs_m < M
    if DIVISIBLE_M:
        q = tl.load(q_ptrs, cache_modifier=".cg")
    else:
        q = tl.load(q_ptrs, mask=mask_m[:, None], cache_modifier=".cg")

    if BLOCK_DMODEL < 128:
        I = tl.where(offs_k[:, None] == offs_k,
                     tl.full((BLOCK_DMODEL, BLOCK_DMODEL), 1.0, dtype=q.dtype),
                     tl.full((BLOCK_DMODEL, BLOCK_DMODEL), 0.0, dtype=q.dtype))
        q = tl.dot(q, I).to(q.dtype)

    m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    offs_n_init = offs_n_base
    k_ptrs = K + (offs_k[:, None] * stride_kk + offs_n_init[None, :] * stride_kn)  # (BLOCK_DMODEL, BLOCK_N)
    v_ptrs = V + (offs_n_init[:, None] * stride_vn + offs_k[None, :] * stride_vk)  # (BLOCK_N, BLOCK_DMODEL)

    if IS_CAUSAL:
        hi = tl.minimum(N, P_SEQ + (start_m + 1) * BLOCK_M)
        if LARGER_M:
            hi = tl.maximum(0, hi)
    else:
        hi = N

    for start_n in range(0, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + offs_n_base

        mask_n = offs_n < N
        if DIVISIBLE_N:
            k = tl.load(k_ptrs, cache_modifier=".cg")
            v = tl.load(v_ptrs, cache_modifier=".cg")
        else:
            k = tl.load(k_ptrs, mask=mask_n[None, :], cache_modifier=".cg")
            v = tl.load(v_ptrs, mask=mask_n[:, None], cache_modifier=".cg")

        s = tl.zeros([BLOCK_M, BLOCK_N], dtype=input_dtype)
        s += tl.dot(q, k) * sm_scale

        relative_positions = offs_m[:, None]-offs_n[None, :]  # shape: (BLOCK_M, BLOCK_N)

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

            k_pos_ptrs = K_POS+offs_m[:, None]*stride_pk2 + c2p_index*stride_pk3

            c2p_bias = tl.load(k_pos_ptrs, mask=mask_m[:, None] & (c2p_index < 2*ATT_SPAN), other=0.0)

            s += c2p_bias * sm_scale

        if HAS_P2C:
            p2c_index = tl.minimum(tl.maximum(bucket_pos + ATT_SPAN, 0), 2 * ATT_SPAN - 1).to(tl.int32).trans(1, 0)

            q_pos_ptrs = Q_POS + (offs_n[:, None] * stride_pq2 + p2c_index * stride_pq3)

            p2c_bias = tl.load(q_pos_ptrs, mask=mask_n[:, None] & (p2c_index < 2*ATT_SPAN), other=0.0).trans(1, 0)
            s += p2c_bias * sm_scale

        if not DIVISIBLE_N:
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

    if DIVISIBLE_M:
        tl.store(l_ptrs, l, cache_modifier=".cg")
        tl.store(o_ptrs, acc.to(q.dtype), cache_modifier=".cg")
    else:
        tl.store(l_ptrs, l, mask=mask_m, cache_modifier=".cg")
        tl.store(o_ptrs, acc.to(q.dtype), mask=mask_m[:, None], cache_modifier=".cg")

def get_fwd_config(B, H, M, N, D, causal, disentangled=False, max_shared_memory=None, att_span=256):
    """
    Determine optimal kernel configuration parameters.

    Args:
        B, H, M, N, D: Batch, head, query length, key length, per-head dimension.
        causal (bool): Whether causal masking is applied.
        disentangled (bool): Whether to use the DeBERTa-style disentangled attention kernel.
        max_shared_memory (int, optional): Maximum available shared memory in bytes.
                                         If None, it will be queried from the device.

    Returns:
        Tuple (BLOCK_M, BLOCK_N, num_stages, num_warps)
    """
    # See more details on the mapping at: https://forums.developer.nvidia.com/t/dynamic-shared-memory-calculated-by-ncu-larger-than-max-shared-memory-per-block/265589

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

    max_shared_memory = shared_mem_per_block - 2000 # remove 2kb for ops overhead

    # Start with an aggressive configuration
    if capability[0] >= 8 :
        if not causal:
            if D <= 64:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 3, 4
            else:
                if M <= 1024:
                    BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 3, 4
                else:
                    BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 128, 3, 8
        else:  # causal
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
        else:  # causal
            if D <= 64:
                if disentangled:
                    BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 3, 4
                else:
                    BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 3, 4
            else:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 2, 4
    else:
        BLOCK_M, BLOCK_N, num_stages, num_warps = 16, 16, 10, 4
    
    # Calculate shared memory usage with current config
    has_pos = disentangled
    ATT_SPAN = att_span if has_pos else 0 
    
    dtype = torch.float16

    shared_mem_usage = calculate_shared_memory_usage(
        BLOCK_M, BLOCK_N, D, num_stages, dtype, 
        has_c2p=has_pos, has_p2c=has_pos, ATT_SPAN=ATT_SPAN
    )
    
    # If shared memory usage exceeds available, adjust parameters
    # We prioritize reducing num_stages first, then block sizes
    while shared_mem_usage > max_shared_memory and (BLOCK_M > 16 or BLOCK_N > 16 or num_stages > 1):
        # First try reducing num_stages
        if num_stages > 1:
            num_stages -= 1
        # Then try reducing block sizes
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
        
        # Recalculate with new parameters
        shared_mem_usage = calculate_shared_memory_usage(
            BLOCK_M, BLOCK_N, D, num_stages, dtype, 
            has_c2p=has_pos, has_p2c=has_pos, ATT_SPAN=ATT_SPAN
        )

    warnings.warn(f"INFO: Variable-length forward config is {BLOCK_M}, {BLOCK_N}, {num_stages}, {num_warps} for BLOCK_M, BLOCK_N stages and warps, respectively.\n"
                  "INFO: If you want to change it, feel free to check ops/flash_attention_varlen")
    return (BLOCK_M, BLOCK_N, num_stages, num_warps)


def flash_attn_v2_fwd_dise(q, k, v, pos_key, pos_query, causal, sm_scale, BLOCK_M, BLOCK_N,
                           position_buckets, max_relative_distance, num_warps, num_stages, ATT_SPAN):
    """
    Performs the forward pass of FlashAttention with DeBERTa-style disentangled relative attention.
    
    Args:
        ... (existing arguments)
        max_shared_memory (int, optional): Maximum available shared memory in bytes.
                                          If None, it will be queried from the device.
    """
    B, H, M, D = q.shape
    N = k.shape[2]
    P_SEQ = N - M
    
    if sm_scale is None:
        sm_scale = 1. / math.sqrt(D)
    
    # Determine if each bias term is present.
    has_c2p = pos_key is not None
    has_p2c = pos_query is not None

    larger_m = M > N

    divisible_m = (M % BLOCK_M) == 0
    divisible_n = (N % BLOCK_N) == 0

    # Determine if each bias term is present.
    has_c2p = pos_key is not None
    has_p2c = pos_query is not None

    # Setup grid: use a 3D grid (query blocks, heads, batch)
    grid = (cdiv(M, BLOCK_M), H, B)
    o = torch.empty_like(q)
    L = torch.empty((B, H, M), device=q.device, dtype=torch.float32)

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
    Returns (BLOCK_M, BLOCK_N, num_stages, num_warps).
    """
    capability_map = {
        (7,0):  96000, (7,2):  96000, (7,5):  64000,
        (8,0): 163000, (8,6):  99000, (8,7): 163000, (8,9):  99000,
        (9,0): 227000,
    }
    cap = torch.cuda.get_device_capability()
    prop = torch.cuda.get_device_properties(0)

    if max_shared_memory is None:
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
            BLOCK_M, BLOCK_N, num_stages, num_warps = (128, 64, 3, 4) if not causal else (128, 64, 3, 4)
        else:
            BLOCK_M, BLOCK_N, num_stages, num_warps = (128, 64, 2, 8) if not causal else (128, 64, 2, 8)
    elif cap[0] >= 8:
        if D <= 64:
            if causal:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 3, 4
            else:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 3, 4
        else:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 2, 8
    else:
        BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 2, 4

    if N >= 2 * M and BLOCK_N < 128 and cap[0] >= 8:
        BLOCK_N = 128
        num_warps = max(num_warps, 8)

    has_pos = bool(disentangled)
    ATT_SPAN = att_span if has_pos else 0

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
            break  # nothing else to shrink

        shm = calculate_shared_memory_usage_bwd(
            BLOCK_M, BLOCK_N, D, num_stages, dtype,
            has_c2p=has_pos, has_p2c=has_pos, ATT_SPAN=ATT_SPAN,
            store_lse=True, recompute_probs=True, accum_dq=True, accum_dkv=True,
        )

    if D <= 64 and BLOCK_M * BLOCK_N <= 128 * 64:
        num_warps = min(num_warps, 4)
    else:
        num_warps = max(num_warps, 8 if cap[0] >= 8 else 4)

    warnings.warn(
        f"INFO: Varlen backward config -> "
        f"BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}, stages={num_stages}, warps={num_warps}."
        "\nINFO: Adjust att_span/disentangled or set max_shared_memory to tune."
    )

    return (BLOCK_M, BLOCK_N, num_stages, num_warps)

@triton.jit
def _bwd_preprocess(
    Out, DO,
    Delta,
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

    if DIVISIBLE_M:
        o  = tl.load(o_ptrs).to(tl.float32)
        do = tl.load(do_ptrs).to(tl.float32)
        delta = tl.sum(o * do, axis=1)
        tl.store(Delta + off_m * stride_dm, delta)
    else:
        mask_m = off_m < M
        o  = tl.load(o_ptrs,  mask=mask_m[:, None]).to(tl.float32)
        do = tl.load(do_ptrs, mask=mask_m[:, None]).to(tl.float32)
        delta = tl.sum(o * do, axis=1)
        tl.store(Delta + off_m * stride_dm, delta, mask=mask_m)

@triton.jit
def _bwd_kv_dise_kernel(
    Q, K, V, K_POS, Q_POS, sm_scale, DO,
    DK, DV, DKPOS, DQPOS,
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

    # Offset tensors
    Q  += off_z*stride_qz  + off_h*stride_qh
    K  += off_z*stride_kz  + off_h*stride_kh
    V  += off_z*stride_vz  + off_h*stride_vh
    DO += off_z*stride_doz + off_h*stride_doh

    DK += off_z*stride_dkz + off_h*stride_dkh
    DV += off_z*stride_dvz + off_h*stride_dvh

    if HAS_C2P:
        K_POS  += off_z*stride_pk0 + off_h*stride_pk1
        DKPOS  += off_z*stride_pk0 + off_h*stride_pk1
    if HAS_P2C:
        Q_POS  += off_z*stride_pq0 + off_h*stride_pq1
        DQPOS  += off_z*stride_pq0 + off_h*stride_pq1

    L     += (off_z*H + off_h) * M
    Delta += (off_z*H + off_h) * M

    # Bounds in m for this block of n
    if CAUSAL:
        lo = tl.maximum(start_n * BLOCK_N - P_SEQ, 0)
        lo = (lo // BLOCK_M) * BLOCK_M
    else:
        lo = 0

    offs_m_init = lo + tl.arange(0, BLOCK_M)
    offs_n      = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m_base = tl.arange(0, BLOCK_M)
    offs_k      = tl.arange(0, BLOCK_DMODEL)

    # Pointers
    q_ptrs  = Q  + (offs_m_init[:, None] * stride_qm + offs_k[None, :] * stride_qk)     # (M, D)
    k_ptrs  = K  + (offs_n[:, None]      * stride_kn + offs_k[None, :] * stride_kk)     # (N, D)
    v_ptrs  = V  + (offs_n[:, None]      * stride_vn + offs_k[None, :] * stride_vk)     # (N, D)
    do_ptrs = DO + (offs_m_init[:, None] * stride_dom + offs_k[None, :] * stride_dok)   # (M, D)

    dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_k[None, :] * stride_dvk)
    dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_k[None, :] * stride_dkk)

    # Load K, V once per n-tile
    mask_n = offs_n < N
    if DIVISIBLE_N:
        v = tl.load(v_ptrs)
        k = tl.load(k_ptrs)
    else:
        v = tl.load(v_ptrs, mask=mask_n[:, None])
        k = tl.load(k_ptrs, mask=mask_n[:, None])

    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)

    for start_m in range(lo, M, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m  = start_m + offs_m_base
        mask_m  = offs_m < M

        if DIVISIBLE_M:
            q  = tl.load(q_ptrs)
            do = tl.load(do_ptrs)
            l  = tl.load(L + offs_m)
            delta = tl.load(Delta + offs_m)
        else:
            q  = tl.load(q_ptrs,  mask=mask_m[:, None])
            do = tl.load(do_ptrs, mask=mask_m[:, None])
            l  = tl.load(L + offs_m,     mask=mask_m)
            delta = tl.load(Delta + offs_m, mask=mask_m)

        # Recompute scores s = qk^T * sm_scale + biases
        s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        s += tl.dot(q, tl.trans(k)) * sm_scale

        relative_positions = offs_m[:, None] - offs_n[None, :]  # (M, N)
        sign = tl.where(relative_positions > 0.0, 1.0, tl.where(relative_positions < 0.0, -1.0, 0.0))
        mid_val = NUM_BUCKETS // 2
        abs_relative = tl.abs(relative_positions)
        condition = (relative_positions < mid_val) & (relative_positions > -mid_val)
        abs_pos = tl.where(condition, mid_val - 1.0, abs_relative)

        log_numer = tl.log(abs_pos / mid_val)
        log_denom = tl.log((MAX_DISTANCE - 1) / mid_val)
        log_scaled = log_numer / log_denom * (mid_val - 1.0)
        log_pos = tl.ceil(log_scaled) + mid_val
        bucket_pos = tl.where(abs_pos <= mid_val, relative_positions, log_pos * sign)  # signed

        if HAS_C2P:
            c2p_index = tl.minimum(tl.maximum(bucket_pos + ATT_SPAN, 0), 2 * ATT_SPAN - 1).to(tl.int32)
            k_pos_ptrs = K_POS + (offs_m[:, None] * stride_pk2 + c2p_index * stride_pk3)  # (M,N) logical
            c2p_bias = tl.load(k_pos_ptrs, mask=mask_m[:, None] & (c2p_index < 2*ATT_SPAN), other=0.0)
            s += c2p_bias * sm_scale

        if HAS_P2C:
            p2c_index = tl.minimum(tl.maximum(bucket_pos + ATT_SPAN, 0), 2 * ATT_SPAN - 1).to(tl.int32).trans(1, 0)
            q_pos_ptrs = Q_POS + (offs_n[:, None] * stride_pq2 + p2c_index * stride_pq3)  # (N,M)
            p2c_bias = tl.load(q_pos_ptrs, mask=mask_n[:, None] & (p2c_index < 2*ATT_SPAN), other=0.0).trans(1, 0)
            s += p2c_bias * sm_scale

        # Causal mask
        if CAUSAL:
            causal_mask = (P_SEQ + offs_m[:, None]) >= (offs_n[None, :])
        # Re-materialize p using saved log-normalizer l
        p = tl.math.exp2((s - l[:, None]) * log2e)
        if not DIVISIBLE_N:
            p = tl.where(mask_n[None, :], p, 0.0)
        if CAUSAL:
            p = tl.where(causal_mask, p, 0.0)

        # dv = p^T @ do
        dv += tl.dot(tl.trans(p.to(do.dtype)), do)

        # dp = do @ v^T
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dp += tl.dot(do.to(input_dtype), tl.trans(v))

        # ds = p * (dp - delta[:, None])
        ds = p * (dp - delta[:, None])
        if not DIVISIBLE_N:
            ds = tl.where(mask_n[None, :], ds, 0.0)
        if CAUSAL:
            ds = tl.where(causal_mask, ds, 0.0)

        ds_scaled = (ds * sm_scale).to(input_dtype)

        # dk += ds^T @ q
        dk += tl.dot(tl.trans(ds_scaled), q)

        # Positional grads via atomic adds
        if HAS_C2P:
            # DKPOS[m, bucket] += sum_n ds(m,n)*sm_scale where bucket = c2p_index(m,n)
            kpos_grad_ptrs = DKPOS + (offs_m[:, None] * stride_pk2 + c2p_index * stride_pk3)  # (M,N)
            tl.atomic_add(
                kpos_grad_ptrs,
                ds_scaled,
                mask=mask_m[:, None] & mask_n[None, :] & (c2p_index < 2*ATT_SPAN),
            )

        if HAS_P2C:
            # DQPOS[n, bucket] += sum_m ds(m,n)*sm_scale where bucket = p2c_index(n,m)
            qpos_grad_ptrs = DQPOS + (offs_n[:, None] * stride_pq2 + p2c_index * stride_pq3)  # (N,M)
            tl.atomic_add(
                qpos_grad_ptrs,
                ds_scaled.trans(1, 0),
                mask=mask_n[:, None] & mask_m[None, :] & (p2c_index < 2*ATT_SPAN),
            )

        # advance pointers
        q_ptrs  += BLOCK_M * stride_qm
        do_ptrs += BLOCK_M * stride_dom

    if DIVISIBLE_N:
        tl.store(dk_ptrs, dk.to(input_dtype))
        tl.store(dv_ptrs, dv.to(input_dtype))
    else:
        tl.store(dk_ptrs, dk.to(input_dtype), mask=mask_n[:, None])
        tl.store(dv_ptrs, dv.to(input_dtype), mask=mask_n[:, None])

@triton.jit
def _bwd_q_dise_kernel(
    Q, K, V, K_POS, Q_POS, sm_scale, DO,
    DQ,
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

    mask_m = offs_m < M
    if DIVISIBLE_M:
        q  = tl.load(q_ptrs)
        do = tl.load(do_ptrs)
        delta = tl.load(Delta + offs_m)
        l = tl.load(L + offs_m)
    else:
        q  = tl.load(q_ptrs,  mask=mask_m[:, None])
        do = tl.load(do_ptrs, mask=mask_m[:, None])
        delta = tl.load(Delta + offs_m, mask=mask_m)
        l = tl.load(L + offs_m, mask=mask_m)

    dq = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # Upper bound for N this row touches
    if CAUSAL:
        hi = tl.minimum(N, P_SEQ + (start_m + 1) * BLOCK_M)
        if LARGER_M:
            hi = tl.maximum(0, hi)
    else:
        hi = N

    k_ptrs = K + (offs_n_base[:, None] * stride_kn + offs_k[None, :] * stride_kk)
    v_ptrs = V + (offs_n_base[:, None] * stride_vn + offs_k[None, :] * stride_vk)

    for start_n in range(0, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + offs_n_base

        mask_n = offs_n < N
        if DIVISIBLE_N:
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)
        else:
            k = tl.load(k_ptrs, mask=mask_n[:, None])
            v = tl.load(v_ptrs, mask=mask_n[:, None])

        # Recompute s and p
        s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        s += tl.dot(q, tl.trans(k)) * sm_scale

        # (same bucketization as fwd)
        relative_positions = offs_m[:, None] - offs_n[None, :]
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
            c2p_index = tl.minimum(tl.maximum(bucket_pos + ATT_SPAN, 0), 2*ATT_SPAN - 1).to(tl.int32)
            k_pos_ptrs = K_POS + (offs_m[:, None] * stride_pk2 + c2p_index * stride_pk3)
            c2p_bias = tl.load(k_pos_ptrs, mask=mask_m[:, None] & (c2p_index < 2*ATT_SPAN), other=0.0)
            s += c2p_bias * sm_scale

        if HAS_P2C:
            p2c_index = tl.minimum(tl.maximum(bucket_pos + ATT_SPAN, 0), 2*ATT_SPAN - 1).to(tl.int32).trans(1, 0)
            q_pos_ptrs = Q_POS + (offs_n[:, None] * stride_pq2 + p2c_index * stride_pq3)
            p2c_bias = tl.load(q_pos_ptrs, mask=mask_n[:, None] & (p2c_index < 2*ATT_SPAN), other=0.0).trans(1, 0)
            s += p2c_bias * sm_scale

        if CAUSAL:
            causal_mask = (P_SEQ + offs_m[:, None]) >= (offs_n[None, :])

        p = tl.math.exp2((s - l[:, None]) * log2e)
        if not DIVISIBLE_N:
            p = tl.where(mask_n[None, :], p, 0.0)
        if CAUSAL:
            p = tl.where(causal_mask, p, 0.0)

        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dp += tl.dot(do.to(input_dtype), tl.trans(v))

        ds = p * (dp - delta[:, None])
        if not DIVISIBLE_N:
            ds = tl.where(mask_n[None, :], ds, 0.0)
        if CAUSAL:
            ds = tl.where(causal_mask, ds, 0.0)

        dq += tl.dot((ds * sm_scale).to(input_dtype), k)

        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn

    dq = dq.to(input_dtype)
    if DIVISIBLE_M:
        tl.store(dq_ptrs, dq)
    else:
        tl.store(dq_ptrs, dq, mask=mask_m[:, None])

def flash_attn_v2_bwd_dise(o, do, q, k, v, k_pos, q_pos, L, causal, sm_scale,
                           BLOCK_M, BLOCK_N, position_buckets, max_relative_distance,
                           num_warps, num_stages, ATT_SPAN):
    B, H, M, D = q.shape
    N = k.shape[2]
    P_SEQ = N - M
    larger_m = M > N
    divisible_m = (M % BLOCK_M) == 0
    divisible_n = (N % BLOCK_N) == 0

    has_c2p = (k_pos is not None)
    has_p2c = (q_pos is not None)

    # Preprocess: Delta = sum(o * do, dim=-1)
    delta = torch.empty_like(L)
    grid = (cdiv(M, BLOCK_M), H, B)
    with torch.cuda.device(q.device.index):
        _bwd_preprocess[grid](
            o, do, delta,
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            delta.stride(0), delta.stride(1), delta.stride(2),
            M,
            BLOCK_M=BLOCK_M, D_HEAD=D, DIVISIBLE_M=divisible_m,
        )

    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    dk_pos = torch.zeros_like(k_pos) if has_c2p else None
    dq_pos = torch.zeros_like(q_pos) if has_p2c else None

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
            q, k, v, k_pos, q_pos, sm_scale, do,
            dk, dv, dk_pos, dq_pos,
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

    dq = torch.empty_like(q)
    grid_q = (cdiv(M, BLOCK_M), H, B)
    with torch.cuda.device(q.device.index):
        _bwd_q_dise_kernel[grid_q](
            q, k, v, k_pos, q_pos, sm_scale, do,
            dq,
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

class FlashAttentionDisentangled(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, k_pos, q_pos, causal,
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
            q, k, v, k_pos, q_pos, causal, sm_scale,
            BLOCK_M, BLOCK_N, position_buckets,
            max_relative_distance, num_warps, num_stages, ATT_SPAN
        )

        # Save for backward
        ctx.save_for_backward(q, k, v, k_pos, q_pos, o, L)
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
        sm_scale = ctx.sm_scale
        causal = ctx.causal
        position_buckets = ctx.position_buckets
        max_relative_distance = ctx.max_relative_distance
        ATT_SPAN = ctx.ATT_SPAN
        B, H, M, D = q.shape
        N = k.shape[2]

        BLOCK_M, BLOCK_N, num_stages, num_warps = get_bwd_config(B, H, M, N, D, causal)

        dq, dk, dv, dk_pos, dq_pos = flash_attn_v2_bwd_dise(
            o, do, q, k, v, k_pos, q_pos, L, causal, sm_scale,
            BLOCK_M, BLOCK_N, position_buckets, max_relative_distance,
            num_warps, num_stages, ATT_SPAN
        )

        # match forward signature: (q, k, v, q_pos, k_pos, causal, sm_scale, position_buckets, max_relative_distance)
        return dq, dk, dv, dk_pos, dq_pos, None, None, None, None

def flash_attention_with_disentangled(q, k, v, k_pos, q_pos, causal=False, sm_scale=None,
                                      position_buckets=0, max_relative_distance=0):
    return FlashAttentionDisentangled.apply(q, k, v, k_pos, q_pos, causal, sm_scale,
                                            position_buckets, max_relative_distance)