"""Disentangled attention kernels vs a dense PyTorch reference: outputs and all gradients."""
import pytest
import torch
import torch.nn.functional as F

from flashdeberta.ops.flash_attention import flash_attention_with_disentangled
from flashdeberta.ops.flash_attention_varlen import flash_attention_with_disentangled_varlen
from flashdeberta.ops.position import make_log_bucket_position, position_plan

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

H, D, SPAN, MAX_REL = 4, 64, 256, 512
SM_SCALE = (3 * D) ** -0.5
# Relative L2 error bounds: Triton runs FP32 dots in TF32.
TOLERANCE = {torch.float32: 5e-3, torch.float16: 1e-2, torch.bfloat16: 2e-2}


def slot_index(M, N, device):
    rel = torch.arange(M, device=device)[:, None] - torch.arange(N, device=device)[None, :]
    return torch.clamp(make_log_bucket_position(rel, SPAN, MAX_REL).long() + SPAN, 0, 2 * SPAN - 1)


def reference(q, k, v, pos_key_layer, pos_query_layer, lengths, causal):
    B, _, L, _ = q.shape
    idx = slot_index(L, L, q.device).expand(B, H, L, L)
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
