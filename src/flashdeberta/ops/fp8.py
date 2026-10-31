"""FP8 (e4m3) quantization for the attention kernels.

Q and K are quantized once per call, one scale per row (a head of a token), by
``quantize_rows``; the kernels load the fp8 rows and their scales. Quantizing in the
kernels instead would redo every K tile once per query block. Autograd only sees the
original inputs (the ops quantize internally). ``fp8_qkv`` also quantizes P with a
fixed scale (P <= 1) and every V tile in registers with one scale per head-dim column,
which factors out of the sum over keys (``fp8_quantize``).

The backward recomputes the scores from the same fp8 rows, so P matches the forward,
and treats quantization as the identity for gradients (straight-through).
"""
import torch
import triton
import triton.language as tl

FP8_MAX = 448.0

# precision name -> QUANT constexpr of the kernels
PRECISIONS = {"native": 0, "fp8_qk": 1, "fp8_qkv": 2}


def quant_mode(precision, device):
    """Validate ``precision`` and return the kernels' QUANT value."""
    try:
        mode = PRECISIONS[precision]
    except KeyError:
        raise ValueError(f"precision must be one of {sorted(PRECISIONS)}, got {precision!r}") from None
    if mode and torch.cuda.get_device_capability(device) < (8, 9):
        raise RuntimeError("fp8 attention needs a GPU with FP8 tensor cores (compute capability 8.9 or newer)")
    return mode


@triton.jit
def _quantize_rows_kernel(X, X8, SCALE, rows, D: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_R: tl.constexpr):
    offs_r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_d = tl.arange(0, BLOCK_D)
    mask = (offs_r[:, None] < rows) & (offs_d[None, :] < D)
    offs = offs_r[:, None].to(tl.int64) * D + offs_d[None, :]
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(x), 1), 1e-12) / 448.0
    tl.store(X8 + offs, (x / scale[:, None]).to(tl.float8e4nv), mask=mask)
    tl.store(SCALE + offs_r, scale, mask=offs_r < rows)


def quantize_rows(x):
    """``x`` (..., D) -> (fp8 rows (..., D) contiguous, fp32 scales (...,) contiguous), amax / 448 per row.

    One fused pass over ``x`` (read once, fp8 and scales written once).
    """
    x = x.contiguous()
    D = x.shape[-1]
    rows = x.numel() // D
    x8 = torch.empty(x.shape, device=x.device, dtype=torch.float8_e4m3fn)
    scale = torch.empty(x.shape[:-1], device=x.device, dtype=torch.float32)
    if rows:
        BLOCK_R = 64
        with torch.cuda.device(x.device.index):
            _quantize_rows_kernel[(triton.cdiv(rows, BLOCK_R),)](
                x, x8, scale, rows, D=D, BLOCK_D=triton.next_power_of_2(D), BLOCK_R=BLOCK_R)
    return x8, scale


@triton.jit
def fp8_quantize(x, AXIS: tl.constexpr):
    """Quantize ``x`` to e4m3 with one scale per slice along ``AXIS`` (amax / 448).

    Returns ``(x8, scale)``; ``x ~= x8 * scale`` broadcast along ``AXIS``. All-zero
    slices (masked rows) get a tiny scale and stay zero.
    """
    xf = x.to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(xf), AXIS), 1e-12) / 448.0
    if AXIS == 1:
        x8 = (xf / scale[:, None]).to(tl.float8e4nv)
    else:
        x8 = (xf / scale[None, :]).to(tl.float8e4nv)
    return x8, scale
