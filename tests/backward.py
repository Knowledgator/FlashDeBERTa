import math
import torch
from flashdeberta.model import FlashDisentangledSelfAttention
from transformers.models.deberta_v2.modeling_deberta_v2 import DisentangledSelfAttention

class DummyConfig:
    def __init__(self, hidden_size, num_attention_heads, position_buckets, max_relative_positions, pos_att_type=[], max_position_embeddings=512):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_probs_dropout_prob = 0.
        self.hidden_dropout_prob = 0.
        self.pos_att_type = pos_att_type
        self.relative_attention = True
        self.position_buckets = position_buckets
        self.max_relative_positions = max_relative_positions
        self.max_position_embeddings = max_position_embeddings
        self.share_att_key = False

def _make_varlen_mask(B, L, device, min_len=1):
    lengths = torch.randint(low=min_len, high=L + 1, size=(B,), device=device)
    mask = torch.zeros(B, L, device=device, dtype=torch.bool)
    for b, t in enumerate(lengths.tolist()):
        mask[b, :t] = True
    return mask

@torch.no_grad()
def _extended_mask(attention_mask):
    m = attention_mask.float()
    m1 = m.unsqueeze(1).unsqueeze(2)     # (B, 1, 1, L)
    m2 = m1.squeeze(-2).unsqueeze(-1)    # (B, 1, L, 1)
    return m1 * m2                       # (B, 1, L, L)

def _tensor_stats(x):
    if x is None:
        return float("nan"), float("nan"), float("nan")
    a = x.abs()
    max_abs = a.max().item()
    mean_abs = a.mean().item()
    l2 = x.pow(2).sum().sqrt().item()
    return max_abs, mean_abs, l2

def _grad_diff_record(name, ga, gb, atol, rtol, eps=1e-12):
    # ga/gb: gradients for parameter `name` (same shape)
    d = (ga - gb).detach()
    max_abs, mean_abs, l2 = _tensor_stats(d)
    ga_l2 = ga.detach().pow(2).sum().sqrt().item()
    gb_l2 = gb.detach().pow(2).sum().sqrt().item()
    rel_l2 = l2 / (max(ga_l2, gb_l2, eps))  # normalized by larger ref magnitude

    # Per-param pass criterion using absolute/relative on max element
    ref_max = max(ga.detach().abs().max().item(), gb.detach().abs().max().item(), eps)
    passed = (max_abs <= atol) or (max_abs <= rtol * ref_max)

    return {
        "name": name,
        "shape": tuple(ga.shape),
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "l2": l2,
        "ga_l2": ga_l2,
        "gb_l2": gb_l2,
        "rel_l2": rel_l2,
        "passed": passed,
    }

def compare_flash_and_deberta_backward(
    B, L, hidden_size,
    causal=False, sm_scale=None,
    position_buckets=32,
    max_relative_positions=64,
    pos_att_type=[],
    varlen=False,
    dtype=None,
    atol=None,
    rtol=None,
    seed=0,
    verbose=True,
    print_top_k=20,     # <── show top-K params by max_abs diff
):
    """
    Compare backward pass gradients parameter-by-parameter.
    Returns:
      {
        "passed": bool,
        "input_grad_max_abs_diff": float,
        "input_grad_mean_abs_diff": float,
        "rel_grad_max_abs_diff": float or nan,
        "rel_grad_mean_abs_diff": float or nan,
        "param_grad_max_abs_diff": float,
        "param_grad_mean_abs_diff": float,
        "per_param": {name: metrics_dict, ...},   # <── NEW
        "failed_params": [names...],              # <── NEW
      }
    """
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype is None:
        dtype = torch.float16 if device == "cuda" else torch.float32
    if atol is None:
        atol = 5e-3 if dtype == torch.float16 else 1e-5
    if rtol is None:
        rtol = 5e-3 if dtype == torch.float16 else 1e-5
    mode = "varlen" if varlen else "fixed-len"

    # Inputs
    hidden_states_a = torch.randn(B, L, hidden_size, device=device, dtype=dtype, requires_grad=True)
    rel_embeddings_a = torch.randn(max_relative_positions, hidden_size, device=device, dtype=dtype, requires_grad=True)

    attention_mask = _make_varlen_mask(B, L, device) if varlen else torch.ones(B, L, device=device, dtype=torch.bool)
    extended_attention_mask = _extended_mask(attention_mask)

    # Clone for flash path
    hidden_states_b = hidden_states_a.detach().clone().requires_grad_(True)
    rel_embeddings_b = rel_embeddings_a.detach().clone().requires_grad_(True)

    # Config + models
    num_attention_heads = 8
    assert hidden_size % num_attention_heads == 0
    config = DummyConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        position_buckets=position_buckets,
        max_relative_positions=max_relative_positions,
        pos_att_type=pos_att_type,
    )
    deberta_model = DisentangledSelfAttention(config).to(device).to(dtype)
    flash_model   = FlashDisentangledSelfAttention(config).to(device).to(dtype)
    flash_model.load_state_dict(deberta_model.state_dict())

    deberta_model.eval()
    flash_model.eval()

    # Forward
    out_a, _ = deberta_model(hidden_states_a, extended_attention_mask, rel_embeddings=rel_embeddings_a)
    out_b, _ = flash_model(hidden_states_b, attention_mask, rel_embeddings=rel_embeddings_b)

    diff = (out_a - out_b).abs().mean().item()
    if verbose:
        print(f"[Forward test — {mode}] dtype={dtype}, atol={atol}, rtol={rtol}")
        print("Mean absolute difference between outputs:", diff)
        passed = True if diff < rtol else False
        print(f"Forward passed: {passed}.")
    # Identical upstream grad
    grad_out = torch.randn_like(out_a)

    # Zero param grads
    for p in deberta_model.parameters():
        if p.grad is not None: p.grad = None
    for p in flash_model.parameters():
        if p.grad is not None: p.grad = None

    # Backward
    (out_a * grad_out).sum().backward()
    (out_b * grad_out.detach().clone()).sum().backward()

    # ---- Inputs ----
    in_diff = (hidden_states_a.grad - hidden_states_b.grad).detach()
    in_max, in_mean, _ = _tensor_stats(in_diff)

    # ---- Relative embeddings (if used) ----
    if any(t in pos_att_type for t in ("c2p", "p2c")):
        rel_diff = (rel_embeddings_a.grad - rel_embeddings_b.grad).detach()
        rel_max, rel_mean, _ = _tensor_stats(rel_diff)
    else:
        rel_max, rel_mean = float("nan"), float("nan")

    # ---- Per-param diffs ----
    per_param = {}
    diffs_max = []
    grads_a = {n: p.grad for n, p in deberta_model.named_parameters()}
    grads_b = {n: p.grad for n, p in flash_model.named_parameters()}
    failed_params = []

    for n, ga in grads_a.items():
        gb = grads_b.get(n, None)
        if ga is None or gb is None:
            continue
        rec = _grad_diff_record(n, ga, gb, atol, rtol)
        per_param[n] = rec
        diffs_max.append(rec["max_abs"])
        if not rec["passed"]:
            failed_params.append(n)

    param_max = max(diffs_max) if diffs_max else float("nan")
    param_mean = (sum(diffs_max) / len(diffs_max)) if diffs_max else float("nan")

    # Overall pass (all params + inputs + rel-emb if present)
    passed = True
    if not math.isnan(param_max):
        # overall: require all individual params passed instead of only global max
        passed &= (len(failed_params) == 0)
    if not math.isnan(in_max):
        passed &= (in_max <= atol) or (in_max <= rtol * max(1e-8, in_max))
    if not math.isnan(rel_max):
        passed &= (rel_max <= atol) or (rel_max <= rtol * max(1e-8, rel_max))

    if verbose:
        print(f"[Backward test — {mode}] dtype={dtype}, atol={atol}, rtol={rtol}")
        print(f" Input grad   diff: max={in_max:.3e}, mean={in_mean:.3e}")
        if not math.isnan(rel_max):
            print(f" Rel-emb grad diff: max={rel_max:.3e}, mean={rel_mean:.3e}")
        print(f" Param grad   diff: max={param_max:.3e}, mean(max_abs)={param_mean:.3e}")
        # Show top-K by max_abs diff
        if per_param:
            top = sorted(per_param.values(), key=lambda r: r["max_abs"], reverse=True)[:print_top_k]
            print(f" Top {len(top)} params by max_abs diff:")
            for r in top:
                flag = "" if r["passed"] else "  <-- FAIL"
                print(f"  {r['name']:55s} {str(r['shape']):>18s}  max={r['max_abs']:.3e}  "
                      f"mean={r['mean_abs']:.3e}  relL2={r['rel_l2']:.3e}{flag}")
        if failed_params:
            print(f" FAILED PARAMS ({len(failed_params)}): {failed_params}")
        print(" PASSED" if passed and not failed_params else " FAILED")

    return {
        "passed": passed and not failed_params,
        "input_grad_max_abs_diff": in_max,
        "input_grad_mean_abs_diff": in_mean,
        "rel_grad_max_abs_diff": rel_max,
        "rel_grad_mean_abs_diff": rel_mean,
        "param_grad_max_abs_diff": param_max,
        "param_grad_mean_abs_diff": param_mean,
        "per_param": per_param,           # name -> metrics dict
        "failed_params": failed_params,   # list of names
    }

# Example
if __name__ == "__main__":
    _ = compare_flash_and_deberta_backward(
        B=2, L=128, hidden_size=256,
        pos_att_type=["p2c", "c2p"],
        varlen=False, verbose=True, print_top_k=30
    )
