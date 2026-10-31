# Copyright 2026 Delyan Boychev
# Copyright 2026 Knowledgator
# Relative-position lookup planning and banded positional-gradient accumulation
# adapted from DisentangledFlash by Delyan Boychev:
#   https://github.com/delyan-boychev/disentangled-flash
#   (commits f185d9a6ae1cce42a54573208dba27fae828be98 and
#    892cdc27c98a3d21019c58e85b7e7ceae2318d5e)
# Modified by Knowledgator, 2026, for FlashDeBERTa's tensor layouts and kernels.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Relative-position plans shared by the disentangled attention kernels.

Kernels look up the C2P/P2C slot of a (query i, key j) pair through a table
indexed by the distance u = i - j + N - 1, instead of recomputing the log-bucket
for every score element.

The backward writes position-score gradients into "band" columns, one column
per distance. Every (row, distance) cell then has a single writer, so no
atomics are needed. Distances that saturate to the first/last slot share the
first/last column and are summed in registers. A one-hot [columns, slots]
matrix folds the band back into slots, so the gradients of the C2P/P2C score
GEMMs become a few batched GEMMs.
"""
import functools
from typing import NamedTuple

import torch


def make_log_bucket_position(relative_pos, bucket_size: int, max_position: int):
    """Same formula as transformers' DeBERTa-v2 ``make_log_bucket_position``."""
    sign = torch.sign(relative_pos)
    mid = bucket_size // 2
    abs_pos = torch.where(
        (relative_pos < mid) & (relative_pos > -mid),
        torch.tensor(mid - 1).type_as(relative_pos),
        torch.abs(relative_pos),
    )
    log_pos = (
        torch.ceil(torch.log(abs_pos / mid) / torch.log(torch.tensor((max_position - 1) / mid)) * (mid - 1)) + mid
    )
    return torch.where(abs_pos <= mid, relative_pos.type_as(log_pos), log_pos * sign)


_BUILDER = None


def _cached_on_device(build, device):
    """Run ``build()`` (which creates the device tensors of a cache entry) outside CUDA graph pools.

    Plans are cached across calls. CUDA graph trees (``torch.compile(mode="reduce-overhead")``) warm
    a graph up with every allocation of the calling thread routed to their memory pool, and refuse
    to record when a tensor allocated there outlives the run. Building on a helper thread keeps cache
    entries in the regular allocator; the device sync after it is paid once per new entry. Entries
    are built on first use, before capture, so a capture only hits the cache.
    """
    global _BUILDER
    device = torch.device(device)
    if device.type != "cuda":
        return build()
    if _BUILDER is None:
        from concurrent.futures import ThreadPoolExecutor
        _BUILDER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="flashdeberta-plan")

    def run():
        with torch.cuda.device(device):
            result = build()
            torch.cuda.current_stream(device).synchronize()
        return result

    return _BUILDER.submit(run).result()


class PositionPlan(NamedTuple):
    lut: torch.Tensor           # int32 [M + N - 1]: slot of distance u = i - j + N - 1
    band_low: int               # distances u <= band_low share column 0
    num_columns: int            # band columns; the last one holds distances u >= band_high
    column_slots: torch.Tensor  # int64 [num_columns]: slot of each band column


# Unbounded on purpose: a recorded CUDA graph keeps raw pointers to these tensors, so an entry must
# never be freed while a graph may replay. Entries are small, one per distinct (M, N, buckets) shape.
@functools.lru_cache(maxsize=None)
def _position_plan(M, N, position_buckets, max_relative_distance, att_span, device):
    deltas = torch.arange(-(N - 1), M, dtype=torch.long)
    if position_buckets > 0:
        deltas = make_log_bucket_position(deltas, position_buckets, max_relative_distance).to(torch.long)
    slots = torch.clamp(deltas + att_span, 0, 2 * att_span - 1)

    # Slots are monotone in distance: find where the saturated ends begin.
    values = slots.tolist()
    last = len(values) - 1
    low = 0
    while low < last and values[low + 1] == values[0]:
        low += 1
    high = last
    while high - 1 > low and values[high - 1] == values[last]:
        high -= 1
    high = max(high, low + 1)
    columns = [values[0], *values[low + 1:high], values[last]]

    lut, column_slots = _cached_on_device(
        lambda: (slots.to(device=device, dtype=torch.int32), torch.tensor(columns, dtype=torch.long, device=device)),
        device)
    return PositionPlan(lut=lut, band_low=low, num_columns=len(columns), column_slots=column_slots)


def position_plan(M, N, position_buckets, max_relative_distance, att_span, device):
    return _position_plan(int(M), int(N), int(position_buckets), int(max_relative_distance),
                          int(att_span), torch.device(device))


@functools.lru_cache(maxsize=None)
def _fold_matrix(M, N, position_buckets, max_relative_distance, att_span, device, dtype):
    plan = _position_plan(M, N, position_buckets, max_relative_distance, att_span, device)

    def build():
        slots = torch.arange(2 * att_span, device=device)
        return (plan.column_slots[:, None] == slots[None, :]).to(dtype)

    return _cached_on_device(build, device)


def fold_matrix(M, N, position_buckets, max_relative_distance, att_span, device, dtype):
    """One-hot [num_columns, 2 * att_span] map from band columns to position slots."""
    return _fold_matrix(int(M), int(N), int(position_buckets), int(max_relative_distance),
                        int(att_span), torch.device(device), dtype)


def position_table(table, num_heads, att_span):
    """(1, H, 2 * att_span, D) or (H, 2 * att_span, D) projected relative embeddings -> (H, 2 * att_span, D)."""
    if table.dim() == 4:
        if table.shape[0] != 1:
            raise ValueError("position tables must be shared across the batch: expected shape (1, H, R, D)")
        table = table[0]
    if table.shape[0] != num_heads:
        raise ValueError(f"position table has {table.shape[0]} heads, expected {num_heads}")
    # Slots are clamped to [0, 2 * att_span), so a shorter table would be read out of bounds.
    if table.shape[1] != 2 * att_span:
        raise ValueError(f"position table has {table.shape[1]} rows, expected 2 * att_span = {2 * att_span}")
    return table


def position_tables(table, index, num_heads, att_span, num_sequences):
    """Validate (S, H, 2 * att_span, D) tables and the (B,) table index of every sequence.

    Lets sequences of one batch use different position tables, e.g. relative embeddings projected
    by a different LoRA adapter per sequence. Inference only: the attention backward rejects them.
    """
    if table.dim() != 4:
        raise ValueError("per-sequence position tables must have shape (S, H, R, D)")
    position_table(table[:1], num_heads, att_span)  # validates heads and rows
    if index.dim() != 1 or index.shape[0] != num_sequences:
        raise ValueError(f"position table index must have shape ({num_sequences},), got {tuple(index.shape)}")
    return table


def per_sequence_tables(table, index, num_heads, att_span, num_sequences):
    """(S, H, 2 * att_span, D) tables and (B,) index -> (B, H, 2 * att_span, D), one table per sequence."""
    position_tables(table, index, num_heads, att_span, num_sequences)
    return table.index_select(0, index.to(device=table.device, dtype=torch.long))


def fold_band_gradients(band, rows, table, fold):
    """Gradients of ``scores = rows @ table^T`` from band-column score gradients.

    band:  [H, T, C] gradient per (row, distance column)
    rows:  [H, T, D] query (C2P) or key (P2C) rows
    table: [H, R, D] projected relative embeddings
    fold:  [C, R] one-hot column-to-slot map
    Returns (grad_rows [H, T, D], grad_table [H, R, D]).
    """
    grad_rows = torch.bmm(band, torch.matmul(fold, table))
    grad_table = torch.matmul(fold.t(), torch.bmm(band.transpose(1, 2), rows))
    return grad_rows, grad_table


def prepare_positions(positions, position_span, shape, num_keys):
    """Validate custom token positions and return ``(int32 positions, span)``, or ``(None, None)``.

    ``positions[i] - positions[j]`` replaces ``i - j`` as the relative distance of a pair, so a
    layout such as ``row * stride + column`` gives image patches 2-D relative positions. The
    backward writes one gradient cell per (row, distance): positions must be unique inside a
    sequence (see ``positions_are_unique``) and lie in ``[0, span)``.
    """
    if positions is None:
        return None, None
    if tuple(positions.shape) != tuple(shape):
        raise ValueError(f"positions must have shape {tuple(shape)}, got {tuple(positions.shape)}")
    if shape[-1] != num_keys:
        raise ValueError("positions require the same number of queries and keys (self-attention)")
    if positions.is_floating_point():
        raise ValueError("positions must be an integer tensor")
    span = int(positions.max()) + 1 if position_span is None else int(position_span)
    return positions.to(torch.int32).contiguous(), span


def positions_are_unique(positions, lengths=None):
    """True if no two valid tokens of a padded (B, L) batch share a position."""
    positions = positions.long()
    if lengths is not None:
        keep = torch.arange(positions.shape[1], device=positions.device)[None, :] < lengths[:, None]
        # Padding gets distinct negative values so it never collides.
        filler = -1 - torch.arange(positions.shape[1], device=positions.device)[None, :]
        positions = torch.where(keep, positions, filler)
    ordered = positions.sort(dim=1).values
    return bool((ordered[:, 1:] != ordered[:, :-1]).all())


def clear_position_cache():
    """Drop cached plans. Do not call while CUDA graphs recorded with them may still replay."""
    _position_plan.cache_clear()
    _fold_matrix.cache_clear()
