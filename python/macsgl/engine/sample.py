from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mlx.core as mx

if TYPE_CHECKING:
    from macsgl.core import Batch


@dataclass
class SamplingArgs:
    temperatures: mx.array | None
    top_k: mx.array | None = None
    top_p: mx.array | None = None


class Sampler:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def prepare(self, batch: Batch) -> SamplingArgs:
        params = [r.sampling_params for r in batch.reqs]
        if all(p.is_greedy for p in params):
            return SamplingArgs(temperatures=None)

        MIN_P = MIN_T = 1e-6
        ts = [max(0.0 if p.is_greedy else p.temperature, MIN_T) for p in params]
        top_ks = [p.top_k if p.top_k >= 1 else self.vocab_size for p in params]
        top_ps = [min(max(p.top_p, MIN_P), 1.0) for p in params]
        top_k, top_p = None, None
        if any(k != self.vocab_size for k in top_ks):
            top_k = mx.array(top_ks, dtype=mx.int32)
        if any(p < 1.0 for p in top_ps):
            top_p = mx.array(top_ps, dtype=mx.float32)
        return SamplingArgs(mx.array(ts, dtype=mx.float32), top_k=top_k, top_p=top_p)

    def sample(self, logits: mx.array, args: SamplingArgs) -> mx.array:
        if args.temperatures is None:
            return mx.argmax(logits, axis=-1).astype(mx.int32)

        logits = logits.astype(mx.float32) / args.temperatures[:, None]
        if args.top_k is None and args.top_p is None:
            return mx.random.categorical(logits).astype(mx.int32)

        order = mx.argsort(-logits, axis=-1)
        sorted_logits = mx.take_along_axis(logits, order, axis=-1)
        masks = []
        if args.top_k is not None:
            masks.append(mx.arange(logits.shape[-1])[None] >= args.top_k[:, None])
        if args.top_p is not None:
            probs = mx.softmax(sorted_logits, axis=-1)
            masks.append(mx.cumsum(probs, axis=-1) - probs > args.top_p[:, None])
        mask = masks[0] if len(masks) == 1 else masks[0] | masks[1]
        choice = mx.random.categorical(mx.where(mask, -mx.inf, sorted_logits))
        return mx.take_along_axis(order, choice[:, None], axis=-1)[:, 0].astype(mx.int32)
