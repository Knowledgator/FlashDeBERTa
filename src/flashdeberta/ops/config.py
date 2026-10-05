"""Tile configs for the disentangled attention kernels.

Configs are (BLOCK_M, BLOCK_N, num_stages, num_warps). The kernels gather C2P/P2C
scores from global memory, so they are bound by gathers rather than by shared
memory: small tiles with one stage win. Defaults were measured on an RTX 5060 Ti
(SM120) in BF16 at head dim 64; a config that does not fit a GPU is shrunk at
launch (see ``ops.launch``).

Override with FLASHDEBERTA_FWD_{BLOCK_M,BLOCK_N,NUM_STAGES,NUM_WARPS} and
FLASHDEBERTA_BWD_* (both backward kernels), or FLASHDEBERTA_BWD_KV_* and
FLASHDEBERTA_BWD_Q_* for one backward kernel.
"""
import os


def _env_config(prefix):
    keys = [f"{prefix}_BLOCK_M", f"{prefix}_BLOCK_N", f"{prefix}_NUM_STAGES", f"{prefix}_NUM_WARPS"]
    if all(key in os.environ for key in keys):
        return tuple(int(os.environ[key]) for key in keys)
    return None


def forward_config(D):
    return _env_config("FLASHDEBERTA_FWD") or ((64, 64, 1, 4) if D <= 64 else (64, 32, 1, 4))


def backward_configs(D):
    """Return (dK/dV kernel config, dQ kernel config)."""
    shared = _env_config("FLASHDEBERTA_BWD")
    kv = _env_config("FLASHDEBERTA_BWD_KV") or shared or ((64, 32, 1, 4) if D <= 64 else (32, 32, 1, 4))
    q = _env_config("FLASHDEBERTA_BWD_Q") or shared or ((16, 64, 1, 4) if D <= 64 else (16, 32, 1, 4))
    return kv, q
