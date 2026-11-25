from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import torch

from macsgl.utils import cached_load_hf_config

from .model_config import ModelConfig


@dataclass(frozen=True)
class EngineConfig:
    model_path: str
    dtype: torch.dtype
    max_running_req: int = 256
    page_size: int = 1
    # fraction of the currently available RAM, which the GPU shares with everything else
    memory_ratio: float = 0.5
    use_dummy_weight: bool = False
    max_seq_len_override: int | None = None
    num_page_override: int | None = None  # if not None, will override the number of pages

    @cached_property
    def hf_config(self):
        return cached_load_hf_config(self.model_path)

    @cached_property
    def model_config(self) -> ModelConfig:
        return ModelConfig.from_hf(self.hf_config)

    @property
    def max_seq_len(self) -> int:
        if self.max_seq_len_override is not None:
            return self.max_seq_len_override
        return self.model_config.rotary_config.max_position
