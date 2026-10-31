"""FP8 attention kernels vs a fake-quantized PyTorch reference: outputs and all gradients.

In ``fp8_qk`` the kernels quantize every Q and K row (one head of one token) to e4m3 with scale
amax / 448 and compute Q K^T from the quantized rows; C2P/P2C scores and P V stay in the input
dtype. The backward recomputes the same scores and passes gradients straight through the
quantization: dQ = dS K and dK = dS^T Q use the original rows. The reference below does exactly
that, so the kernels must match it as closely as the native kernels match the exact reference.
"""
import pytest
import torch
import torch.nn.functional as F

from flashdeberta.ops.flash_attention import flash_attention_with_disentangled
from flashdeberta.ops.flash_attention_varlen import flash_attention_with_disentangled_varlen

from .test_disentangled_kernels import H, MAX_REL, SM_SCALE, SPAN, make_inputs, reference, run, slot_index

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
    reason="requires a GPU with FP8 tensor cores",
)

NAMES = ["out", "dq", "dk", "dv", "d_pos_key", "d_pos_query"]


def fake_fp8(x):
    """Per-row e4m3 quantize-dequantize, as the kernels do it."""
    scale = x.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 448
    return (x / scale).to(torch.float8_e4m3fn).to(x.dtype) * scale


def fp8_reference(q, k, v, pos_key_layer, pos_query_layer, lengths):
    """Scores from fake-quantized Q K^T; gradients flow as if Q K^T were exact (straight-through)."""
    B, _, L, _ = q.shape
    exact_qk = q @ k.transpose(-1, -2)
    quant_qk = fake_fp8(q.detach()) @ fake_fp8(k.detach()).transpose(-1, -2)
    scores = exact_qk + (quant_qk - exact_qk).detach()
    idx = slot_index(L, L, q.device).expand(B, H, L, L)
    if pos_key_layer is not None:
        scores = scores + torch.gather(q @ pos_key_layer.transpose(-1, -2), -1, idx)
    if pos_query_layer is not None:
        scores = scores + torch.gather(k @ pos_query_layer.transpose(-1, -2), -1, idx.transpose(-1, -2)).transpose(-1, -2)
    keep = torch.arange(L, device=q.device)[None, :] < lengths[:, None]
    scores = (scores * SM_SCALE).masked_fill(~keep[:, None, None, :], float("-inf"))
    return (torch.softmax(scores, -1) @ v) * keep[:, None, :, None]


def rel_err(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


def padded(precision, lengths):
    def fn(q, k, v, pk, pq):
        return flash_attention_with_disentangled(q, k, v, lengths, pk, pq, False, SM_SCALE, SPAN, MAX_REL,
                                                 precision=precision)
    return fn


def packed(precision, lengths):
    """The varlen kernel on the packed tokens, re-padded to (B, H, L, D) for comparison."""
    def fn(q, k, v, pk, pq):
        B, _, L, D = q.shape
        keep = torch.arange(L, device="cuda")[None, :] < lengths[:, None]
        tokens = keep.flatten().nonzero().flatten()
        cu_seqlens = F.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0))
        pack = lambda t: t.transpose(1, 2).reshape(B * L, H, D)[tokens]
        out = flash_attention_with_disentangled_varlen(pack(q), pack(k), pack(v), pk, pq, cu_seqlens, cu_seqlens,
                                                       L, L, False, SM_SCALE, SPAN, MAX_REL, precision=precision)
        full = torch.zeros(B * L, H, D, device="cuda", dtype=out.dtype).index_copy(0, tokens, out)
        return full.view(B, L, H, D).transpose(1, 2)
    return fn


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("B,L,is_padded", [(2, 100, True), (3, 700, True), (1, 2048, False)])
@pytest.mark.parametrize("kernel", [padded, packed])
def test_fp8_qk_matches_fake_quant_reference(dtype, B, L, is_padded, kernel):
    inputs, lengths, grad_out = make_inputs(B, L, dtype, is_padded)
    expected = run(lambda *t: fp8_reference(*[x.double() for x in t], lengths), inputs, grad_out, torch.float64)
    got = run(kernel("fp8_qk", lengths), inputs, grad_out)
    # Same bound as the native bf16 kernels against the exact reference.
    for name, a, b in zip(NAMES, got, expected):
        assert rel_err(a, b) < 1e-2, f"{name}: relative error {rel_err(a, b):.2e}"


@pytest.mark.parametrize("kernel", [padded, packed])
def test_fp8_qkv_is_close_to_exact(kernel):
    """P V in FP8 too: the output moves further from exact, gradients are those of fp8_qk."""
    inputs, lengths, grad_out = make_inputs(3, 700, torch.bfloat16, True)
    exact = run(lambda *t: reference(*[x.double() for x in t], lengths, False), inputs, grad_out, torch.float64)
    fake = run(lambda *t: fp8_reference(*[x.double() for x in t], lengths), inputs, grad_out, torch.float64)
    got = run(kernel("fp8_qkv", lengths), inputs, grad_out)
    assert rel_err(got[0], exact[0]) < 6e-2
    for name, a, b in zip(NAMES[1:], got[1:], fake[1:]):
        assert rel_err(a, b) < 2e-2, f"{name}: relative error {rel_err(a, b):.2e}"


@pytest.mark.parametrize("precision", ["fp8_qk", "fp8_qkv"])
def test_fp8_padded_and_packed_kernels_agree(precision):
    inputs, lengths, grad_out = make_inputs(3, 300, torch.bfloat16, True, seed=3)
    a = run(padded(precision, lengths), inputs, grad_out)
    b = run(packed(precision, lengths), inputs, grad_out)
    for name, x, y in zip(NAMES, a, b):
        assert rel_err(x, y) < 1e-2, name


def test_invalid_precision_is_rejected():
    inputs, lengths, _ = make_inputs(1, 64, torch.bfloat16, False)
    with pytest.raises(ValueError, match="precision must be one of"):
        flash_attention_with_disentangled(*inputs[:3], lengths, *inputs[3:], False, SM_SCALE, SPAN, MAX_REL,
                                          precision="fp4")


def test_fp8_attention_compiles_with_backward():
    inputs, lengths, grad_out = make_inputs(2, 200, torch.bfloat16, True, seed=5)
    fn = padded("fp8_qk", lengths)
    compiled = torch.compile(fn, fullgraph=True)
    eager = run(fn, inputs, grad_out)
    got = run(compiled, inputs, grad_out)
    for name, x, y in zip(NAMES, got, eager):
        assert torch.equal(x, y), name
