from __future__ import annotations

import glob
import math
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict

import mlx.core as mx
import mlx.nn as nn
import torch

from macsgl.utils import download_hf_weight

from .attention import attention

if TYPE_CHECKING:
    from .attention import ForwardMeta, KVCache
    from .config import EngineConfig
    from .model_config import ModelConfig, RotaryConfig

# architecture -> (has_attn_bias, has_qk_norm)
_ARCH_FLAGS = {
    "LlamaForCausalLM": (False, False),
    "MistralForCausalLM": (False, False),
    "Mistral3ForConditionalGeneration": (False, False),
    "Qwen2ForCausalLM": (True, False),
    "Qwen3ForCausalLM": (False, True),
}
_MERGE_GROUPS = {
    "qkv_proj": ("q_proj", "k_proj", "v_proj"),
    "gate_up_proj": ("gate_proj", "up_proj"),
}
_MERGE_TARGET = {part: fused for fused, parts in _MERGE_GROUPS.items() for part in parts}


@partial(mx.compile, shapeless=True)
def _silu_and_mul(gate: mx.array, up: mx.array) -> mx.array:
    return nn.silu(gate) * up


def get_inv_freq_post_process(
    rotary_dim: int,
    base: float,
    rope_scaling: Dict[str, Any] | None,
) -> Callable[[torch.Tensor], torch.Tensor] | None:
    if rope_scaling is None:
        return None
    # need to test some cases:
    match rope_scaling["rope_type"]:
        case "default":
            return None

        case "llama3":
            scaling_factor: float = rope_scaling["factor"]
            low_freq_factor: float = rope_scaling["low_freq_factor"]
            high_freq_factor: float = rope_scaling["high_freq_factor"]
            original_max_position: int = rope_scaling["original_max_position_embeddings"]

            def post_process(inv_freq: torch.Tensor) -> torch.Tensor:
                # no smooth if low_freq_factor == high_freq_factor
                wave_len = 2 * math.pi / inv_freq
                if low_freq_factor == high_freq_factor:
                    return torch.where(
                        wave_len < original_max_position / high_freq_factor,
                        inv_freq,
                        inv_freq / scaling_factor,
                    )

                delta = high_freq_factor - low_freq_factor
                smooth = (original_max_position / wave_len - low_freq_factor) / delta
                smooth = torch.clamp(smooth, 0, 1)
                factor = (1 - smooth) / scaling_factor + smooth
                return factor * inv_freq

            return post_process

        case "yarn":
            factor: float = rope_scaling["factor"]
            beta_fast: float = rope_scaling.get("beta_fast", 32.0)
            beta_slow: float = rope_scaling.get("beta_slow", 1.0)
            orig_max_pos: int = rope_scaling["original_max_position_embeddings"]

            def _find_correction_dim(num_rotations: float) -> float:
                return (
                    rotary_dim
                    * math.log(orig_max_pos / (num_rotations * 2 * math.pi))
                    / (2 * math.log(base))
                )

            low = max(math.floor(_find_correction_dim(beta_fast)), 0)
            high = min(math.ceil(_find_correction_dim(beta_slow)), rotary_dim // 2 - 1)

            def post_process(inv_freq: torch.Tensor) -> torch.Tensor:
                ramp = torch.clamp(
                    (torch.arange(rotary_dim // 2, dtype=torch.float32) - low) / max(high - low, 1),
                    0,
                    1,
                )
                return (inv_freq / factor) * ramp + inv_freq * (1 - ramp)

            return post_process

    raise ValueError(f"Unsupported {rope_scaling = }")


class Rope:
    def __init__(self, config: RotaryConfig):
        self.dims = config.rotary_dim
        post_process = get_inv_freq_post_process(config.rotary_dim, config.base, config.scaling)
        self.base: float | None = config.base
        self.freqs: mx.array | None = None
        if post_process is not None:
            exponent = torch.arange(0, self.dims, 2, dtype=torch.float) / self.dims
            inv_freq = post_process(1.0 / (config.base**exponent))
            self.base, self.freqs = None, mx.array((1.0 / inv_freq).numpy())

    def __call__(self, x: mx.array, positions: mx.array) -> mx.array:
        # every token is its own "batch" so that each one gets its own position offset
        t, h, d = x.shape
        y = mx.fast.rope(
            x.reshape(t, h, 1, d),
            self.dims,
            traditional=False,
            base=self.base,
            scale=1.0,
            offset=positions,
            freqs=self.freqs,
        )
        return y.reshape(t, h, d)


class Attention(nn.Module):
    def __init__(self, config: ModelConfig, layer_id: int, rope: Rope, bias: bool, qk_norm: bool):
        super().__init__()
        self.layer_id = layer_id
        self.num_qo_heads = config.num_qo_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.rope = rope
        q_size = config.num_qo_heads * config.head_dim
        kv_size = config.num_kv_heads * config.head_dim
        self.qkv_proj = nn.Linear(config.hidden_size, q_size + 2 * kv_size, bias=bias)
        self.o_proj = nn.Linear(q_size, config.hidden_size, bias=False)
        if qk_norm:
            self.q_norm = nn.RMSNorm(config.head_dim, eps=config.rms_norm_eps)
            self.k_norm = nn.RMSNorm(config.head_dim, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array, meta: ForwardMeta, cache: KVCache) -> mx.array:
        t, d = x.shape[0], self.head_dim
        q_size, kv_size = self.num_qo_heads * d, self.num_kv_heads * d
        q, k, v = mx.split(self.qkv_proj(x), [q_size, q_size + kv_size], axis=-1)
        q = q.reshape(t, self.num_qo_heads, d)
        k = k.reshape(t, self.num_kv_heads, d)
        v = v.reshape(t, self.num_kv_heads, d)
        if "q_norm" in self:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = self.rope(q, meta.positions), self.rope(k, meta.positions)
        o = attention(q, k, v, self.layer_id, meta, cache)
        return self.o_proj(o)


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.hidden_act != "silu":
            raise ValueError(f"Unsupported activation function: {config.hidden_act}")
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gate, up = mx.split(self.gate_up_proj(x), 2, axis=-1)
        return self.down_proj(_silu_and_mul(gate, up))


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, layer_id: int, rope: Rope, bias: bool, qk_norm: bool):
        super().__init__()
        self.self_attn = Attention(config, layer_id, rope, bias, qk_norm)
        self.mlp = MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array, meta: ForwardMeta, cache: KVCache) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x), meta, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class DecoderModel(nn.Module):
    def __init__(self, config: ModelConfig, bias: bool, qk_norm: bool):
        super().__init__()
        rope = Rope(config.rotary_config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            DecoderLayer(config, i, rope, bias, qk_norm) for i in range(config.num_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, input_ids: mx.array, meta: ForwardMeta, cache: KVCache) -> mx.array:
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, meta, cache)
        if meta.last_idx is not None:
            x = x[meta.last_idx]
        return self.norm(x)


class CausalLM(nn.Module):
    def __init__(self, config: ModelConfig, bias: bool, qk_norm: bool):
        super().__init__()
        self.model = DecoderModel(config, bias, qk_norm)
        self.tie_word_embeddings = config.tie_word_embeddings
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(self, input_ids: mx.array, meta: ForwardMeta, cache: KVCache) -> mx.array:
        h = self.model(input_ids, meta, cache)
        if self.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(h)
        return self.lm_head(h)


def _merge_projections(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Fuse q/k/v and gate/up projections (also works for quantized weights/scales/biases)."""
    result: Dict[str, mx.array] = {}
    groups: Dict[str, Dict[str, mx.array]] = {}
    for name, value in weights.items():
        prefix, proj, suffix = name.rsplit(".", 2) if name.count(".") >= 2 else (name, "", "")
        if (fused := _MERGE_TARGET.get(proj)) is None:
            result[name] = value
        else:
            groups.setdefault(f"{prefix}.{fused}.{suffix}", {})[proj] = value
    for name, parts in groups.items():
        order = _MERGE_GROUPS[name.rsplit(".", 2)[1]]
        result[name] = mx.concatenate([parts[p] for p in order], axis=0)
    return result


def _load_checkpoint(model_path: str) -> Dict[str, mx.array]:
    folder = download_hf_weight(model_path)
    files = glob.glob(f"{folder}/*.safetensors")
    files = [f for f in files if not f.endswith("consolidated.safetensors")] or files
    weights: Dict[str, mx.array] = {}
    for file in files:
        for name, value in mx.load(file).items():
            if name.startswith(("vision_tower.", "multi_modal_projector.")):
                continue
            weights[name.removeprefix("language_model.")] = value
    return weights


def load_model(config: EngineConfig, dtype: mx.Dtype) -> CausalLM:
    model_config = config.model_config
    arch = model_config.architectures[0]
    if arch not in _ARCH_FLAGS:
        raise ValueError(f"Model architecture {arch} is not supported yet")
    model = CausalLM(model_config, *_ARCH_FLAGS[arch])

    if config.use_dummy_weight:
        model.set_dtype(dtype)
        mx.eval(model.parameters())
        return model

    weights = _load_checkpoint(config.model_path)
    if model_config.tie_word_embeddings:
        weights.pop("lm_head.weight", None)
    weights = _merge_projections(weights)

    # MLX-format quantized checkpoints (e.g. mlx-community/*-4bit)
    if (quant := getattr(config.hf_config, "quantization", None)) is not None:
        nn.quantize(
            model,
            group_size=quant["group_size"],
            bits=quant["bits"],
            mode=quant.get("mode", "affine"),
            class_predicate=lambda path, _: f"{path}.scales" in weights,
        )
    weights = {
        k: v.astype(dtype) if mx.issubdtype(v.dtype, mx.floating) else v for k, v in weights.items()
    }
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())
    return model
