import math
import torch
import triton
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
    
    return total_shared_memory // 5 # currently, overestimate by the factor of 5, so reduction is required

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
         (5,0): 48000,
         (5,2): 48000,
         (5,3): 48000,
         (6,0): 48000,
         (6,1): 48000,
         (6,2): 48000,
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
    
    if capability in list(capability_map.keys()):
        max_shared_memory = capability_map[capability] - 2000 # remove 2kb for ops overhead
    
    # if this is some unknown new arch -> default to minimal known for 8th 
    elif capability[0] >= 8:
        max_shared_memory = 99000 - 2000 # remove 2kb for ops overhead
    # if this is some older unknown arch -> default to minimal known for < 8th
    else:
        max_shared_memory = 48000 - 2000 # remove 2kb for ops overhead

    
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
        elif BLOCK_M > 32 and BLOCK_N > 32:
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
    print(f"INFO: Forward config is {BLOCK_M}, {BLOCK_N}, {num_stages}, {num_warps} for BLOCK_M, BLOCK_N stages and warps, respectfully.")
    print("INFO: If you want to change it, feel free to check ops/flash_attention")
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


class FlashAttentionDisentangled(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, q_pos, k_pos, causal,
                sm_scale, position_buckets, max_relative_distance):

        Dq, Dk, Dv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Dq == Dk == Dv, "Query, key, and value must have the same head dimension"
        
        B, H, M, D = q.shape
        N = k.shape[2]
        if sm_scale is None:
            sm_scale = 1. / math.sqrt(D)
        
        ATT_SPAN = position_buckets if position_buckets > 0 else max_relative_distance 
        
        config = get_fwd_config(B, H, M, N, D, causal, disentangled=True, att_span=ATT_SPAN)

        BLOCK_M, BLOCK_N, num_stages, num_warps = config
        
        o, L = flash_attn_v2_fwd_dise(q, k, v, q_pos, k_pos, causal, sm_scale,
                                      BLOCK_M, BLOCK_N, position_buckets,
                                      max_relative_distance, num_warps, num_stages, ATT_SPAN)
        return o

    @staticmethod
    def backward(ctx, grad_output):
        # Exclude backward capabilities by raising an error.
        raise RuntimeError("Backward pass is not implemented for FlashAttentionDisentangled")

def flash_attention_with_disentangled(q, k, v, q_pos, k_pos, causal=False, sm_scale=None,
                                      position_buckets=0, max_relative_distance=0):
    """
    An implementation of FlashAttention v2 with DeBERTa-style disentangled relative attention.
    This version does not support backward propagation.

    Args:
        q (Tensor): Queries of shape (B, H, M, D).
        k (Tensor): Keys of shape (B, H, N, D).
        v (Tensor): Values of shape (B, H, N, D).
        q_pos (Tensor): Relative projection tensor for content→position bias.
        k_pos (Tensor): Relative projection tensor for position→content bias.
        causal (bool): Whether to apply causal masking.
        sm_scale (float): Scaling factor for softmax (if None, uses 1/sqrt(D)).
        position_buckets (int): Number of position buckets.
        max_relative_distance (int): Maximum relative distance.

    Returns:
        out (Tensor): Output tensor of shape (B, H, M, D).

    Note:
        The backward pass is not implemented, so this function only supports forward propagation.
    """
    return FlashAttentionDisentangled.apply(q, k, v, q_pos, k_pos, causal, sm_scale,
                                            position_buckets, max_relative_distance)

