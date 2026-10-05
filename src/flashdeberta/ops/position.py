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


class PositionPlan(NamedTuple):
    lut: torch.Tensor           # int32 [M + N - 1]: slot of distance u = i - j + N - 1
    band_low: int               # distances u <= band_low share column 0
    num_columns: int            # band columns; the last one holds distances u >= band_high
    column_slots: torch.Tensor  # int64 [num_columns]: slot of each band column


@functools.lru_cache(maxsize=256)
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

    return PositionPlan(
        lut=slots.to(device=device, dtype=torch.int32),
        band_low=low,
        num_columns=len(columns),
        column_slots=torch.tensor(columns, dtype=torch.long, device=device),
    )


def position_plan(M, N, position_buckets, max_relative_distance, att_span, device):
    return _position_plan(int(M), int(N), int(position_buckets), int(max_relative_distance),
                          int(att_span), torch.device(device))


@functools.lru_cache(maxsize=256)
def _fold_matrix(M, N, position_buckets, max_relative_distance, att_span, device, dtype):
    plan = _position_plan(M, N, position_buckets, max_relative_distance, att_span, device)
    slots = torch.arange(2 * att_span, device=device)
    return (plan.column_slots[:, None] == slots[None, :]).to(dtype)


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


def clear_position_cache():
    _position_plan.cache_clear()
    _fold_matrix.cache_clear()
