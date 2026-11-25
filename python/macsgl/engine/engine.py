from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import mlx.core as mx
import numpy as np
import psutil
import torch

from macsgl.utils import init_logger

from .attention import ForwardMeta, KVCache
from .model import load_model
from .sample import Sampler, SamplingArgs

if TYPE_CHECKING:
    from macsgl.core import Batch

    from .config import EngineConfig

logger = init_logger(__name__)

_MX_DTYPE = {torch.bfloat16: mx.bfloat16, torch.float16: mx.float16, torch.float32: mx.float32}
# MLX keeps freed buffers for reuse without a bound; activations of varying sizes would pile up
_BUFFER_CACHE_LIMIT = 2 << 30


class TokenCopy:
    """After synchronize(), `out` holds the sampled tokens of the finished step."""

    def __init__(self, tokens: mx.array, out: torch.Tensor):
        self.tokens, self.out = tokens, out

    def synchronize(self) -> None:
        self.out.copy_(torch.from_numpy(np.array(self.tokens)))


class ForwardOutput(NamedTuple):
    next_tokens: mx.array
    next_tokens_cpu: torch.Tensor
    copy_done_event: TokenCopy


class Engine:
    def __init__(self, config: EngineConfig):
        self.dtype = _MX_DTYPE[config.dtype]
        mx.random.seed(42)
        mx.set_cache_limit(_BUFFER_CACHE_LIMIT)

        free_memory = min(
            psutil.virtual_memory().available,
            mx.device_info()["max_recommended_working_set_size"],
        )
        self.model = load_model(config, self.dtype)
        model_memory = mx.get_active_memory()

        model_config = config.model_config
        self.num_pages = self._determine_num_pages(free_memory, model_memory, config)
        num_tokens = self.num_pages * config.page_size
        self.max_seq_len = min(config.max_seq_len, num_tokens)
        self.kv_cache = KVCache(
            model_config.num_layers,
            num_tokens,
            model_config.num_kv_heads,
            model_config.head_dim,
            self.dtype,
            config.max_running_req,
            self.max_seq_len,
        )
        # CPU page table shared with the scheduler, mapping request positions to token slots
        self.page_table = torch.zeros((config.max_running_req, self.max_seq_len), dtype=torch.int32)
        self.last_tokens = mx.zeros((config.max_running_req,), dtype=mx.int32)
        self.sampler = Sampler(model_config.vocab_size)

    def _determine_num_pages(
        self, free_memory: int, model_memory: int, config: EngineConfig
    ) -> int:
        mc = config.model_config
        cache_per_page = (
            2 * mc.head_dim * mc.num_kv_heads * config.page_size * self.dtype.size * mc.num_layers
        )
        num_pages = config.num_page_override
        if num_pages is None:
            available = config.memory_ratio * free_memory - model_memory - _BUFFER_CACHE_LIMIT
            num_pages = int(available) // cache_per_page
        assert num_pages > 1, "Not enough memory for KV cache, try --num-pages or --memory-ratio"
        kv_gb = num_pages * cache_per_page / 2**30
        logger.info(
            f"Allocating {num_pages * config.page_size} tokens for KV cache ({kv_gb:.2f} GiB)"
        )
        return num_pages

    def prepare(self, batch: Batch) -> ForwardMeta:
        reqs = batch.reqs
        pt = self.page_table.numpy()
        table_idx = np.array([r.table_idx for r in reqs], dtype=np.int32)
        if batch.is_decode:
            lens = np.array([r.device_len for r in reqs], dtype=np.int32)
            table_idx_mx, positions = mx.array(table_idx), mx.array(lens - 1)
            out_loc = mx.array(pt[table_idx, lens - 1])
            return ForwardMeta(
                is_decode=True,
                table_idx=table_idx_mx,
                positions=positions,
                out_loc=out_loc,
                pt_rows=table_idx_mx,
                pt_cols=positions,
                pt_slots=out_loc,
                lens=mx.array(lens),
            )

        positions, out_loc, input_ids, segments = [], [], [], []
        pt_rows, pt_cols, pt_slots = [], [], []
        offset = 0
        for req, t in zip(reqs, table_idx):
            c, d = req.cached_len, req.device_len
            positions.append(np.arange(c, d, dtype=np.int32))
            out_loc.append(pt[t, c:d])
            input_ids.append(req.input_ids[c:d].numpy())
            segments.append((offset, offset + d - c, mx.array(pt[t, :c]) if c > 0 else None))
            # the whole row, so that prefixes matched in the radix cache reach the GPU table too
            pt_rows.append(np.full(d, t, dtype=np.int32))
            pt_cols.append(np.arange(d, dtype=np.int32))
            pt_slots.append(pt[t, :d])
            offset += d - c
        ends = np.array([end for _, end, _ in segments])
        return ForwardMeta(
            is_decode=False,
            table_idx=mx.array(table_idx),
            positions=mx.array(np.concatenate(positions)),
            out_loc=mx.array(np.concatenate(out_loc)),
            pt_rows=mx.array(np.concatenate(pt_rows)),
            pt_cols=mx.array(np.concatenate(pt_cols)),
            pt_slots=mx.array(np.concatenate(pt_slots)),
            input_ids=mx.array(np.concatenate(input_ids)),
            last_idx=mx.array(ends - 1),
            segments=segments,
        )

    def forward_batch(self, batch: Batch, args: SamplingArgs, meta: ForwardMeta) -> ForwardOutput:
        input_ids = self.last_tokens[meta.table_idx] if meta.is_decode else meta.input_ids
        self.kv_cache.update_page_table(meta)
        logits = self.model(input_ids, meta, self.kv_cache)
        for req in batch.reqs:
            req.complete_one()
        next_tokens = self.sampler.sample(logits, args)
        self.last_tokens[meta.table_idx] = next_tokens
        # The graph is built while the previous batch still runs on the GPU. Wait for it before
        # launching, otherwise the KV pool scatters copy the whole pool instead of writing in place.
        mx.synchronize()
        mx.async_eval(next_tokens, self.last_tokens, *self.kv_cache.arrays())
        next_tokens_cpu = torch.empty(batch.size, dtype=torch.int32)
        return ForwardOutput(next_tokens, next_tokens_cpu, TokenCopy(next_tokens, next_tokens_cpu))
