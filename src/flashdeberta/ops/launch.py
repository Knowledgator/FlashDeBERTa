"""Kernel launch with a shared-memory fallback.

Shared memory depends on the GPU, dtype and Triton version. If a config does not
fit, retry with fewer stages, then smaller tiles, and remember what worked.
"""
from triton.runtime.errors import OutOfResources

_RESOLVED = {}


def _shrink(config):
    block_m, block_n, num_stages, num_warps = config
    if num_stages > 1:
        return block_m, block_n, num_stages - 1, num_warps
    if block_n >= block_m and block_n > 16:
        return block_m, block_n // 2, num_stages, num_warps
    if block_m > 16:
        return block_m // 2, block_n, num_stages, num_warps
    return None


def launch(kernel, key, config, make_launch):
    """Launch ``kernel`` with ``config = (BLOCK_M, BLOCK_N, num_stages, num_warps)``.

    ``make_launch(BLOCK_M, BLOCK_N)`` returns ``(grid, args, kwargs)``. ``key``
    identifies the specialization so a fallback is resolved once.
    """
    requested = (key, config)
    config = _RESOLVED.get(requested, config)
    while True:
        block_m, block_n, num_stages, num_warps = config
        grid, args, kwargs = make_launch(block_m, block_n)
        try:
            kernel[grid](
                *args, BLOCK_M=block_m, BLOCK_N=block_n,
                num_stages=num_stages, num_warps=num_warps, **kwargs,
            )
        except OutOfResources:
            smaller = _shrink(config)
            if smaller is None:
                raise
            config = smaller
            continue
        _RESOLVED[requested] = config
        return config


def clear_launch_cache():
    _RESOLVED.clear()
