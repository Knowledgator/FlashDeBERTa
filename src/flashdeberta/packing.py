"""How a batch is laid out for the encoder: padded, or packed (padding removed).

The encoder builds one descriptor per forward pass and hands it to every layer in place of the
attention mask.

* ``PaddedBatch``: hidden states stay (B, L, hidden); attention skips padding by sequence length.
* ``PackedBatch``: hidden states are (T, hidden), the real tokens of every sequence end to end, so
  projections, LayerNorms and the FFN skip padding too. Only attention needs the boundaries.

Packing needs the token count T on the host: one read from the GPU per forward pass. Under
``torch.compile`` that read happens in ``plan_batch`` outside the graph (one graph break), and T is
rounded up to a bucket (``packed_capacity``) so that compiled graphs and CUDA graphs see a fixed
shape. The extra rows are filler: they sit after ``cu_seqlens[-1]``, attention never reads them,
every other layer works row by row, and unpacking drops them.

A caller that packs on the host (``pack_inputs``) can feed packed inputs straight to
``FlashDebertaV2Model`` with no read and no graph break.
"""
from __future__ import annotations

from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F

Bucket = Union[None, int, str, Callable[[int], int]]

# Under torch.compile a packed batch always has a multiple of this many rows: FP8 GEMMs (torchao,
# torch._scaled_mm) need it on the token axis, which is the reduction axis of weight gradients.
COMPILED_ROW_ALIGNMENT = 16


def packed_capacity(num_tokens: int, bucket: Bucket = "auto") -> int:
    """Rows of a packed batch holding ``num_tokens`` real tokens.

    ``bucket``: None or 1 for exactly ``num_tokens``; an int rounds up to a multiple of it; a callable
    maps the count to a capacity; "auto" rounds up to a multiple of 64 up to 1024 tokens and then to
    an eighth of the power of two below (at most 12.5% filler, about 6% on average; eight sizes per
    doubling, so few distinct shapes for CUDA graphs).
    """
    if bucket is None or bucket == 1 or num_tokens == 0:
        return num_tokens
    if callable(bucket):
        capacity = int(bucket(num_tokens))
    elif bucket == "auto":
        step = 64 if num_tokens <= 1024 else (1 << (num_tokens.bit_length() - 1)) // 8
        capacity = -(-num_tokens // step) * step
    elif isinstance(bucket, int) and bucket > 0:
        capacity = -(-num_tokens // bucket) * bucket
    else:
        raise ValueError(f"invalid pack bucket {bucket!r}")
    if capacity < num_tokens:
        raise ValueError(f"pack bucket gave capacity {capacity} below the token count {num_tokens}")
    return capacity


class PaddedBatch:
    """A padded (B, L) batch: per-row lengths of right-padded sequences."""

    packed = False

    def __init__(self, lengths: torch.Tensor):
        self.lengths = lengths
        self.batch_size = lengths.shape[0]

    @classmethod
    def from_mask(cls, attention_mask: torch.Tensor) -> "PaddedBatch":
        """Lengths from a (B, L), (B, L, L) or (B, 1, L, L) padding mask (padding on the right)."""
        if attention_mask.dim() == 4:
            valid = attention_mask[:, 0].sum(dim=-1) > 0
        elif attention_mask.dim() == 3:
            valid = attention_mask.sum(dim=-1) > 0
        else:
            valid = attention_mask > 0
        return cls(valid.sum(dim=-1, dtype=torch.int32))

    def pack(self, x):
        return x

    def unpack(self, x):
        return x


class PackedBatch:
    """A batch with its padding removed: ``capacity`` rows, the real tokens of every sequence end to end.

    ``cu_seqlens`` (B + 1,) bounds the sequences; rows ``[cu_seqlens[-1], capacity)`` are filler.
    ``max_seqlen`` is an upper bound on the sequence lengths (the padded length): it only sizes the
    attention kernel's position lookup table, so no read of the true maximum is needed.
    """

    packed = True

    def __init__(self, cu_seqlens, max_seqlen, capacity, indices=None, batch_size=None, seq_len=None, mask=None):
        self.cu_seqlens = cu_seqlens
        self._max_seqlen = int(max_seqlen)
        self.capacity = int(capacity)
        # Flat (B * L) position of every packed row; filler rows hold B * L. None: inputs arrive packed.
        self.indices = indices
        self.batch_size = cu_seqlens.shape[0] - 1 if batch_size is None else batch_size
        self._seq_len = self._max_seqlen if seq_len is None else seq_len
        # The (B, L) mask a batch was packed from. Lengths are read from its shape so that under
        # torch.compile they stay symbolic (a Python int attribute would be a constant of the graph).
        self._mask = mask

    @property
    def seq_len(self):
        return self._mask.shape[1] if self._mask is not None else self._seq_len

    @property
    def max_seqlen(self):
        return self._mask.shape[1] if self._mask is not None else self._max_seqlen

    @classmethod
    def from_mask(cls, mask_2d: torch.Tensor, capacity: Optional[int] = None) -> "PackedBatch":
        """Pack a (B, L) boolean mask into ``capacity`` rows (default: the token count, one host read)."""
        B, L = mask_2d.shape
        lengths = mask_2d.sum(dim=-1, dtype=torch.int32)
        cu_seqlens = F.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0))
        if capacity is None:
            capacity = int(cu_seqlens[-1])
        # Static size: rows past the real tokens get the out-of-range index B * L.
        indices = torch.nonzero_static(mask_2d.flatten(), size=capacity, fill_value=B * L).flatten()
        return cls(cu_seqlens, L, capacity, indices, B, L, mask=mask_2d)

    def pack(self, x):
        """(B, L, ...) -> (capacity, ...); filler rows repeat the last padded position."""
        if self.indices is None:
            return x
        flat = x.reshape(self.batch_size * self.seq_len, *x.shape[2:])
        return flat.index_select(0, self.indices.clamp(max=flat.shape[0] - 1))

    def unpack(self, x):
        """(capacity, ...) -> (B, L, ...), zeros at padding; filler rows are dropped."""
        if self.indices is None:
            return x
        rows = self.batch_size * self.seq_len
        out = x.new_zeros(rows + 1, *x.shape[1:]).index_copy(0, self.indices, x)
        return out[:rows].view(self.batch_size, self.seq_len, *x.shape[1:])

    def token_positions(self):
        """(capacity,) position of every packed row inside its sequence (0 for filler)."""
        rows = torch.arange(self.capacity, device=self.cu_seqlens.device, dtype=torch.int32)
        cu = self.cu_seqlens.to(torch.int32)
        seq = torch.searchsorted(cu[1:].contiguous(), rows, right=True).clamp(max=self.batch_size - 1)
        return torch.where(rows < cu[-1], rows - cu[seq], torch.zeros_like(rows))

    def token_mask(self):
        """(capacity,) True for real tokens, False for filler."""
        rows = torch.arange(self.capacity, device=self.cu_seqlens.device)
        return rows < self.cu_seqlens[-1]


def _mark_dynamic(tensor, dim):
    if tensor is not None and tensor.dim() > dim:
        torch._dynamo.maybe_mark_dynamic(tensor, dim)


@torch.compiler.disable
def _plan_packed(attention_mask: torch.Tensor, bucket: Bucket, compiling: bool = False, inputs=()):
    mask_2d = attention_mask > 0
    B, L = mask_2d.shape
    lengths = mask_2d.sum(dim=-1, dtype=torch.int32)
    num_tokens = int(lengths.sum())  # the one host read of a packed forward
    capacity = packed_capacity(num_tokens, bucket)
    if compiling:
        capacity = -(-capacity // COMPILED_ROW_ALIGNMENT) * COMPILED_ROW_ALIGNMENT
    if capacity >= B * L:
        batch = PaddedBatch(lengths)  # nothing to gain from packing
    else:
        batch = PackedBatch.from_mask(mask_2d, capacity)
    if compiling:
        # Sequence length and packed size change from batch to batch: compile the graph after this
        # point for any value from the start, instead of once per value until dynamo notices.
        # (Marks must be set outside the traced code, which is why it happens here.)
        for tensor in (attention_mask, *inputs):
            _mark_dynamic(tensor, 1)
        if batch.packed:
            _mark_dynamic(batch.indices, 0)
            _mark_dynamic(batch._mask, 1)
    return batch


def plan_batch(attention_mask: torch.Tensor, config, query_states=None, compiling: bool = False,
               training: bool = False, inputs=()):
    """Choose the layout of a forward pass. See the module docstring.

    Packing is used for a 2D mask with padding when ``config.unpad_inputs`` is true (default).
    When ``compiling`` in inference, the token count is rounded up with ``config.pack_bucket``
    (default "auto") so CUDA graphs see few shapes. Training packs exactly: compiled training does not
    use CUDA graphs, so filler rows would only add work (set ``config.pack_bucket_training`` to bucket
    anyway, e.g. for ``mode="reduce-overhead"`` training). Compiled packed batches always get a
    multiple of ``COMPILED_ROW_ALIGNMENT`` rows (at most 15 filler rows) for FP8 GEMMs. ``inputs``: other (B, L, ...) model inputs,
    marked dynamic in their length together with the mask.
    """
    if (not getattr(config, "unpad_inputs", True) or query_states is not None
            or attention_mask is None or attention_mask.dim() > 2):
        return PaddedBatch.from_mask(attention_mask) if attention_mask is not None else None
    if not compiling:
        bucket = None
    elif training:
        bucket = getattr(config, "pack_bucket_training", None)
    else:
        bucket = getattr(config, "pack_bucket", "auto")
    return _plan_packed(attention_mask, bucket, compiling, inputs)


def pack_inputs(input_ids: torch.Tensor, attention_mask: torch.Tensor, bucket: Bucket = "auto",
                token_type_ids: Optional[torch.Tensor] = None, device=None) -> dict:
    """Pack a padded tokenizer batch on the host, for ``FlashDebertaV2Model(**packed)``.

    Returns ``input_ids`` (capacity,), ``cu_seqlens`` (B + 1,), ``max_seqlen`` and, if given,
    ``token_type_ids`` (capacity,). Filler rows are token 0. The model then needs no host read
    and no graph break, so it runs under ``torch.compile(fullgraph=True)`` and CUDA graphs; graphs
    depend only on (capacity, B, max_seqlen). The output is ``(capacity, hidden)``: sequence ``b`` is
    rows ``cu_seqlens[b]:cu_seqlens[b + 1]``.
    """
    mask = attention_mask.to("cpu") > 0
    B, L = mask.shape
    lengths = mask.sum(dim=-1, dtype=torch.int32)
    num_tokens = int(lengths.sum())
    capacity = packed_capacity(num_tokens, bucket)
    flat = mask.flatten().nonzero().flatten()

    def pack(t):
        out = torch.zeros(capacity, dtype=t.dtype)
        out[:num_tokens] = t.to("cpu").flatten()[flat]
        return out.to(device) if device is not None else out

    packed = {
        "input_ids": pack(input_ids),
        "cu_seqlens": F.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0)),
        "max_seqlen": L,
    }
    if device is not None:
        packed["cu_seqlens"] = packed["cu_seqlens"].to(device)
    if token_type_ids is not None:
        packed["token_type_ids"] = pack(token_type_ids)
    return packed
