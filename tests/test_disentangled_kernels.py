"""Disentangled attention kernels vs a dense PyTorch reference: outputs and all gradients."""
import pytest
import torch
import torch.nn.functional as F

from flashdeberta.ops.flash_attention import flash_attention_with_disentangled
from flashdeberta.ops.flash_attention_varlen import flash_attention_with_disentangled_varlen
from flashdeberta.ops.position import (make_log_bucket_position, position_plan, positions_are_unique,
                                       prepare_positions)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

H, D, SPAN, MAX_REL = 4, 64, 256, 512
SM_SCALE = (3 * D) ** -0.5
# Relative L2 error bounds: Triton runs FP32 dots in TF32.
TOLERANCE = {torch.float32: 5e-3, torch.float16: 1e-2, torch.bfloat16: 2e-2}


def slot_index(M, N, device):
    rel = torch.arange(M, device=device)[:, None] - torch.arange(N, device=device)[None, :]
    return torch.clamp(make_log_bucket_position(rel, SPAN, MAX_REL).long() + SPAN, 0, 2 * SPAN - 1)


def reference(q, k, v, pos_key_layer, pos_query_layer, lengths, causal, token_positions=None):
    B, _, L, _ = q.shape
    if token_positions is None:
        idx = slot_index(L, L, q.device).expand(B, H, L, L)
    else:
        # Relative distance of a pair is the difference of its tokens' positions.
        rel = token_positions.long()[:, :, None] - token_positions.long()[:, None, :]
        idx = torch.clamp(make_log_bucket_position(rel, SPAN, MAX_REL).long() + SPAN, 0, 2 * SPAN - 1)
        idx = idx[:, None].expand(B, H, L, L)
    scores = q @ k.transpose(-1, -2)
    if pos_key_layer is not None:
        scores = scores + torch.gather(q @ pos_key_layer.transpose(-1, -2), -1, idx)
    if pos_query_layer is not None:
        p2c = torch.gather(k @ pos_query_layer.transpose(-1, -2), -1, idx.transpose(-1, -2))
        scores = scores + p2c.transpose(-1, -2)
    scores = scores * SM_SCALE
    positions = torch.arange(L, device=q.device)
    keep = positions[None, :] < lengths[:, None]
    scores = scores.masked_fill(~keep[:, None, None, :], float("-inf"))
    if causal:
        scores = scores.masked_fill(positions[None, :] > positions[:, None], float("-inf"))
    out = torch.softmax(scores, -1) @ v
    return out * keep[:, None, :, None]


def make_inputs(B, L, dtype, padded, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q, k, v = (torch.randn(B, H, L, D, device="cuda", dtype=dtype, generator=g) for _ in range(3))
    pk, pq = (torch.randn(1, H, 2 * SPAN, D, device="cuda", dtype=dtype, generator=g) * 0.5 for _ in range(2))
    lengths = torch.full((B,), L, device="cuda", dtype=torch.int32)
    if padded:
        lengths[1:] = torch.randint(L // 4, L, (B - 1,), device="cuda", dtype=torch.int32, generator=g)
    keep = torch.arange(L, device="cuda")[None, :] < lengths[:, None]
    grad_out = torch.randn(B, H, L, D, device="cuda", dtype=dtype, generator=g) * keep[:, None, :, None]
    return (q, k, v, pk, pq), lengths, grad_out


def run(fn, inputs, grad_out, dtype=None):
    leaves = [None if t is None else t.detach().to(dtype or t.dtype).requires_grad_() for t in inputs]
    out = fn(*leaves)
    out.backward(grad_out.to(out.dtype))
    return [out.detach()] + [None if t is None else t.grad for t in leaves]


def assert_close(got, expected, dtype):
    names = ["out", "dq", "dk", "dv", "d_pos_key", "d_pos_query"]
    for name, a, b in zip(names, got, expected):
        if b is None:
            assert a is None, name
            continue
        err = ((a.double() - b.double()).norm() / b.double().norm()).item()
        assert err < TOLERANCE[dtype], f"{name}: relative error {err:.2e}"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("B,L,padded", [(2, 64, False), (2, 100, True), (2, 512, False), (3, 700, True), (1, 2048, False)])
@pytest.mark.parametrize("terms", ["c2p+p2c", "c2p", "p2c"])
def test_fixed_length(dtype, B, L, padded, terms):
    inputs, lengths, grad_out = make_inputs(B, L, dtype, padded)
    q, k, v, pk, pq = inputs
    inputs = (q, k, v, pk if "c2p" in terms else None, pq if "p2c" in terms else None)

    def kernel(q, k, v, pk, pq):
        return flash_attention_with_disentangled(q, k, v, lengths if padded else None, pk, pq,
                                                 False, SM_SCALE, SPAN, MAX_REL)

    def dense(q, k, v, pk, pq):
        return reference(q, k, v, pk, pq, lengths, causal=False)

    assert_close(run(kernel, inputs, grad_out), run(dense, inputs, grad_out, torch.float64), dtype)


@pytest.mark.parametrize("L", [100, 600])
def test_fixed_length_causal(L):
    inputs, lengths, grad_out = make_inputs(2, L, torch.float32, padded=False)

    def kernel(q, k, v, pk, pq):
        return flash_attention_with_disentangled(q, k, v, None, pk, pq, True, SM_SCALE, SPAN, MAX_REL)

    def dense(q, k, v, pk, pq):
        return reference(q, k, v, pk, pq, lengths, causal=True)

    assert_close(run(kernel, inputs, grad_out), run(dense, inputs, grad_out, torch.float64), torch.float32)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("B,L", [(3, 100), (4, 1024), (2, 2048)])
def test_varlen(dtype, B, L):
    inputs, _, grad_out = make_inputs(B, L, dtype, padded=False)
    g = torch.Generator(device="cuda").manual_seed(1)
    lengths = torch.randint(L // 4, L + 1, (B,), device="cuda", dtype=torch.int32, generator=g)
    lengths[0] = L
    keep = torch.arange(L, device="cuda")[None, :] < lengths[:, None]
    grad_out = grad_out * keep[:, None, :, None]
    tokens = keep.flatten().nonzero().flatten()
    cu_seqlens = F.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0))

    def kernel(q, k, v, pk, pq):
        pack = lambda t: t.transpose(1, 2).reshape(B * L, H, D)[tokens]
        out = flash_attention_with_disentangled_varlen(pack(q), pack(k), pack(v), pk, pq, cu_seqlens, cu_seqlens,
                                                       L, L, False, SM_SCALE, SPAN, MAX_REL)
        full = torch.zeros(B * L, H, D, device="cuda", dtype=out.dtype).index_copy(0, tokens, out)
        return full.view(B, L, H, D).transpose(1, 2)

    def dense(q, k, v, pk, pq):
        return reference(q, k, v, pk, pq, lengths, causal=False)

    assert_close(run(kernel, inputs, grad_out), run(dense, inputs, grad_out, torch.float64), dtype)


def _varlen_case(lengths, L, dtype):
    B = len(lengths)
    inputs, _, grad_out = make_inputs(B, L, dtype, padded=False)
    lengths = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    keep = torch.arange(L, device="cuda")[None, :] < lengths[:, None]
    grad_out = grad_out * keep[:, None, :, None]
    tokens = keep.flatten().nonzero().flatten()
    cu_seqlens = F.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0))

    def kernel(q, k, v, pk, pq):
        pack = lambda t: t.transpose(1, 2).reshape(B * L, H, D)[tokens]
        out = flash_attention_with_disentangled_varlen(pack(q), pack(k), pack(v), pk, pq, cu_seqlens, cu_seqlens,
                                                       L, L, False, SM_SCALE, SPAN, MAX_REL)
        full = torch.zeros(B * L, H, D, device="cuda", dtype=out.dtype).index_copy(0, tokens, out)
        return full.view(B, L, H, D).transpose(1, 2)

    def dense(q, k, v, pk, pq):
        return reference(q, k, v, pk, pq, lengths, causal=False)

    return inputs, grad_out, kernel, dense


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_varlen_tile_boundaries(dtype):
    # Lengths on and around tile edges, including a single-token sequence.
    inputs, grad_out, kernel, dense = _varlen_case([300, 1, 63, 64, 65, 129, 17], 300, dtype)
    assert_close(run(kernel, inputs, grad_out), run(dense, inputs, grad_out, torch.float64), dtype)


def test_varlen_has_no_host_sync():
    L = 1100
    q, k, v, pk, pq = (t.detach().requires_grad_() for t in make_inputs(3, L, torch.float16, padded=False)[0])
    lengths = torch.tensor([L, 700, 333], device="cuda", dtype=torch.int32)
    cu_seqlens = F.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0))
    T = int(lengths.sum())
    pack = lambda t: t.detach()[0, :, :T].transpose(0, 1).contiguous().requires_grad_()
    q, k, v = pack(q), pack(k), pack(v)

    def step():
        out = flash_attention_with_disentangled_varlen(q, k, v, pk, pq, cu_seqlens, cu_seqlens,
                                                       L, L, False, SM_SCALE, SPAN, MAX_REL)
        out.backward(torch.ones_like(out))

    _assert_no_host_sync(step)


def _assert_no_host_sync(step):
    step()  # first call builds and uploads the position plan
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        step()
    finally:
        torch.cuda.set_sync_debug_mode("default")


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("with_positions", [False, True])
def test_fixed_length_has_no_host_sync(causal, with_positions):
    B, L = 3, 700
    (q, k, v, pk, pq), lengths, _ = make_inputs(B, L, torch.float16, padded=True)
    q, k, v, pk, pq = (t.requires_grad_() for t in (q, k, v, pk, pq))
    # Custom positions need an explicit span, otherwise their maximum is read back (documented sync).
    extra = (torch.arange(L, device="cuda", dtype=torch.int32).expand(B, L).contiguous(), L) if with_positions else ()

    def step():
        out = flash_attention_with_disentangled(q, k, v, None if causal else lengths, pk, pq,
                                                causal, SM_SCALE, SPAN, MAX_REL, *extra)
        out.backward(torch.ones_like(out))

    _assert_no_host_sync(step)


@pytest.mark.parametrize("L", [16, 300, 2048])
def test_position_plan_band_covers_every_distance(L):
    plan = position_plan(L, L, SPAN, MAX_REL, SPAN, "cuda")
    lut = plan.lut.long()
    # lut[i - j + L - 1] is the slot of pair (i, j).
    pairs = slot_index(L, L, "cuda")
    i, j = torch.meshgrid(torch.arange(L, device="cuda"), torch.arange(L, device="cuda"), indexing="ij")
    assert torch.equal(lut[i - j + L - 1], pairs)
    # Map every distance to its band column, as the kernels do, and back to a slot.
    columns = (torch.arange(lut.numel(), device="cuda") - plan.band_low).clamp(0, plan.num_columns - 1)
    assert torch.equal(plan.column_slots[columns], lut)
    # Distances of one row never share an interior column.
    interior = (columns > 0) & (columns < plan.num_columns - 1)
    assert columns[interior].unique().numel() == int(interior.sum())


def test_position_table_size_is_checked():
    inputs, _, _ = make_inputs(1, 64, torch.float32, padded=False)
    q, k, v, pk, pq = inputs
    with pytest.raises(ValueError, match="rows"):
        flash_attention_with_disentangled(q, k, v, None, pk[:, :, :SPAN], pq, False, SM_SCALE, SPAN, MAX_REL)


# ── custom token positions ────────────────────────────────────────────────────

def grid_positions(B, L, grid, stride):
    """[CLS] [IMG] grid x grid patches [/IMG] text...: patches at ``row * stride + column``."""
    rows, cols = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
    patches = 2 + (rows * stride + cols).flatten()
    close = 2 + (grid - 1) * stride + grid
    head = torch.cat([torch.tensor([0, 1]), patches, torch.tensor([close])])
    assert L >= head.numel(), "sequence too short for the grid"
    text = close + 1 + torch.arange(L - head.numel())
    return torch.cat([head, text]).to("cuda", torch.int32).expand(B, L).contiguous()


def make_positions(layout, B, L):
    if layout == "arange":
        return torch.arange(L, device="cuda", dtype=torch.int32).expand(B, L).contiguous()
    if layout == "grid":
        grid, stride = (16, 32) if L >= 300 else (4, 8)
        return grid_positions(B, L, grid, stride)
    if layout == "shuffled":  # unique positions in random order, spread over a span 3x the length
        g = torch.Generator().manual_seed(L)
        return torch.stack([torch.randperm(3 * L, generator=g)[:L] for _ in range(B)]).to("cuda", torch.int32)
    raise ValueError(layout)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("B,L,padded", [(2, 64, False), (2, 100, True), (2, 512, False), (3, 700, True), (1, 2048, False)])
@pytest.mark.parametrize("layout", ["arange", "grid", "shuffled"])
def test_fixed_length_positions(dtype, B, L, padded, layout):
    inputs, lengths, grad_out = make_inputs(B, L, dtype, padded)
    token_positions = make_positions(layout, B, L)
    assert positions_are_unique(token_positions, lengths)

    def kernel(q, k, v, pk, pq):
        return flash_attention_with_disentangled(q, k, v, lengths if padded else None, pk, pq,
                                                 False, SM_SCALE, SPAN, MAX_REL, positions=token_positions)

    def dense(q, k, v, pk, pq):
        return reference(q, k, v, pk, pq, lengths, causal=False, token_positions=token_positions)

    assert_close(run(kernel, inputs, grad_out), run(dense, inputs, grad_out, torch.float64), dtype)


@pytest.mark.parametrize("terms", ["c2p", "p2c"])
def test_fixed_length_positions_single_term(terms):
    inputs, lengths, grad_out = make_inputs(2, 600, torch.bfloat16, padded=True)
    q, k, v, pk, pq = inputs
    inputs = (q, k, v, pk if terms == "c2p" else None, pq if terms == "p2c" else None)
    token_positions = make_positions("grid", 2, 600)

    def kernel(q, k, v, pk, pq):
        return flash_attention_with_disentangled(q, k, v, lengths, pk, pq, False, SM_SCALE, SPAN, MAX_REL,
                                                 positions=token_positions, position_span=1024)

    def dense(q, k, v, pk, pq):
        return reference(q, k, v, pk, pq, lengths, causal=False, token_positions=token_positions)

    assert_close(run(kernel, inputs, grad_out), run(dense, inputs, grad_out, torch.float64), torch.bfloat16)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("B,L", [(3, 100), (4, 1024), (2, 2048)])
@pytest.mark.parametrize("layout", ["arange", "grid", "shuffled"])
def test_varlen_positions(dtype, B, L, layout):
    inputs, _, grad_out = make_inputs(B, L, dtype, padded=False)
    g = torch.Generator(device="cuda").manual_seed(1)
    lengths = torch.randint(L // 4, L + 1, (B,), device="cuda", dtype=torch.int32, generator=g)
    lengths[0] = L
    keep = torch.arange(L, device="cuda")[None, :] < lengths[:, None]
    grad_out = grad_out * keep[:, None, :, None]
    tokens = keep.flatten().nonzero().flatten()
    cu_seqlens = F.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0))
    token_positions = make_positions(layout, B, L)
    packed_positions = token_positions.flatten()[tokens]

    def kernel(q, k, v, pk, pq):
        pack = lambda t: t.transpose(1, 2).reshape(B * L, H, D)[tokens]
        out = flash_attention_with_disentangled_varlen(pack(q), pack(k), pack(v), pk, pq, cu_seqlens, cu_seqlens,
                                                       L, L, False, SM_SCALE, SPAN, MAX_REL,
                                                       positions=packed_positions)
        full = torch.zeros(B * L, H, D, device="cuda", dtype=out.dtype).index_copy(0, tokens, out)
        return full.view(B, L, H, D).transpose(1, 2)

    def dense(q, k, v, pk, pq):
        return reference(q, k, v, pk, pq, lengths, causal=False, token_positions=token_positions)

    assert_close(run(kernel, inputs, grad_out), run(dense, inputs, grad_out, torch.float64), dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("B,L,padded", [(2, 100, True), (2, 700, False)])
def test_sequential_positions_match_default_path(dtype, B, L, padded):
    """positions = 0..L-1 is the default behavior: same output and gradients."""
    inputs, lengths, grad_out = make_inputs(B, L, dtype, padded)
    sequential = make_positions("arange", B, L)

    def default(q, k, v, pk, pq):
        return flash_attention_with_disentangled(q, k, v, lengths if padded else None, pk, pq,
                                                 False, SM_SCALE, SPAN, MAX_REL)

    def custom(q, k, v, pk, pq):
        return flash_attention_with_disentangled(q, k, v, lengths if padded else None, pk, pq,
                                                 False, SM_SCALE, SPAN, MAX_REL, positions=sequential)

    for a, b in zip(run(custom, inputs, grad_out), run(default, inputs, grad_out)):
        assert torch.allclose(a.float(), b.float(), rtol=2e-2 if dtype == torch.bfloat16 else 1e-4, atol=1e-3)


def test_grid_positions_give_every_offset_its_own_slot():
    """With stride 32 on a 16-wide grid, (dy, dx) -> 32 * dy + dx is unique, and exact while below SPAN / 2."""
    dy, dx = torch.meshgrid(torch.arange(-15, 16), torch.arange(-15, 16), indexing="ij")
    distance = (32 * dy + dx).flatten()
    assert distance.unique().numel() == distance.numel()
    exact = distance.abs() < SPAN // 2
    slots = torch.clamp(make_log_bucket_position(distance, SPAN, MAX_REL).long() + SPAN, 0, 2 * SPAN - 1)
    assert slots[exact].unique().numel() == int(exact.sum())
    # Offsets of up to 3 rows in any direction are exact.
    assert bool(exact[(dy.flatten().abs() <= 3)].all())


def test_positions_are_validated():
    inputs, _, _ = make_inputs(2, 64, torch.float32, padded=False)
    q, k, v, pk, pq = inputs
    with pytest.raises(ValueError, match="shape"):
        flash_attention_with_disentangled(q, k, v, None, pk, pq, False, SM_SCALE, SPAN, MAX_REL,
                                          positions=make_positions("arange", 2, 32))
    with pytest.raises(ValueError, match="integer"):
        prepare_positions(torch.zeros(2, 64, device="cuda"), None, (2, 64), 64)
    duplicated = make_positions("arange", 2, 64).clone()
    duplicated[0, 5] = duplicated[0, 4]
    assert not positions_are_unique(duplicated)
    # A duplicate that sits in the padding does not count.
    assert positions_are_unique(duplicated, torch.tensor([4, 64], device="cuda"))
