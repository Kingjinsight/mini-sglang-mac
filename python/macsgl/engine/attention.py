from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import mlx.core as mx

from .kernel import paged_decode_attention


@dataclass
class ForwardMeta:
    is_decode: bool
    table_idx: mx.array  # [B]
    positions: mx.array  # [T]
    out_loc: mx.array  # [T], KV slot of each new token
    # page table entries written by this batch: page_table[pt_rows, pt_cols] = pt_slots
    pt_rows: mx.array
    pt_cols: mx.array
    pt_slots: mx.array
    input_ids: mx.array | None = None  # [T], prefill only (decode reads the last sampled tokens)
    last_idx: mx.array | None = None  # [B], prefill only
    lens: mx.array | None = None  # [B], decode only: context length including the new token
    # prefill only: (start, end, KV slots of the cached prefix or None) per request
    segments: List[Tuple[int, int, mx.array | None]] | None = None


class KVCache:
    def __init__(
        self,
        num_layers: int,
        num_slots: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: mx.Dtype,
        max_running_req: int,
        max_seq_len: int,
    ):
        shape = (num_slots, num_kv_heads, head_dim)
        self.k = [mx.zeros(shape, dtype) for _ in range(num_layers)]
        self.v = [mx.zeros(shape, dtype) for _ in range(num_layers)]
        # GPU copy of the scheduler's page table (token slot of every request position)
        self.page_table = mx.zeros((max_running_req, max_seq_len), dtype=mx.int32)
        mx.eval(self.k, self.v, self.page_table)

    def store(self, layer_id: int, k: mx.array, v: mx.array, loc: mx.array) -> None:
        # The scatter only runs in place if nothing else references the old pool array,
        # so every read of the pool must go through the updated array.
        self.k[layer_id][loc] = k
        self.v[layer_id][loc] = v

    def update_page_table(self, meta: ForwardMeta) -> None:
        self.page_table[meta.pt_rows, meta.pt_cols] = meta.pt_slots

    def arrays(self) -> List[mx.array]:
        return [*self.k, *self.v, self.page_table]


def attention(
    q: mx.array, k: mx.array, v: mx.array, layer_id: int, meta: ForwardMeta, cache: KVCache
) -> mx.array:
    """q: [T, Hq, D], k/v: [T, Hkv, D] -> [T, Hq * D]"""
    cache.store(layer_id, k, v, meta.out_loc)
    k_pool, v_pool = cache.k[layer_id], cache.v[layer_id]
    if meta.is_decode:
        assert meta.lens is not None
        o = paged_decode_attention(q, k_pool, v_pool, cache.page_table, meta.table_idx, meta.lens)
        return o.reshape(q.shape[0], -1)
    return _prefill(q, k, v, k_pool, v_pool, meta)


def _prefill(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    k_pool: mx.array,
    v_pool: mx.array,
    meta: ForwardMeta,
) -> mx.array:
    assert meta.segments is not None
    outs: List[mx.array] = []
    for start, end, prefix in meta.segments:
        ki, vi = k[start:end], v[start:end]
        if prefix is not None:
            ki = mx.concatenate([k_pool[prefix], ki])
            vi = mx.concatenate([v_pool[prefix], vi])
        o = mx.fast.scaled_dot_product_attention(
            q[start:end].transpose(1, 0, 2)[None],
            ki.transpose(1, 0, 2)[None],
            vi.transpose(1, 0, 2)[None],
            scale=q.shape[-1] ** -0.5,
            mask="causal",  # lower-right aligned: new tokens sit after the cached prefix
        )
        outs.append(o[0].transpose(1, 0, 2))
    o = outs[0] if len(outs) == 1 else mx.concatenate(outs)
    return o.reshape(q.shape[0], -1)
