from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Set

from macsgl.core import Batch, Req


@dataclass
class DecodeManager:
    page_size: int
    running_reqs: Set[Req] = field(default_factory=set)

    def filter_reqs(self, reqs: Iterable[Req]) -> None:
        self.running_reqs = {req for req in self.running_reqs.union(reqs) if req.can_decode}

    def remove_req(self, req: Req) -> None:
        self.running_reqs.discard(req)

    def abort_req(self, uid: int) -> Req | None:
        for req in self.running_reqs:
            if req.uid == uid:
                self.running_reqs.remove(req)
                return req
        return None

    def inflight_tokens(self, new_token_ratio: float) -> int:
        tokens_reserved = (self.page_size - 1) * len(self.running_reqs)  # 1 page reserved
        expected = sum(math.ceil(req.remain_len * new_token_ratio) for req in self.running_reqs)
        return expected + tokens_reserved

    def schedule_next_batch(self) -> Batch | None:
        if not self.runnable:
            return None
        return Batch(reqs=sorted(self.running_reqs, key=lambda req: req.uid), phase="decode")

    @property
    def runnable(self) -> bool:
        return len(self.running_reqs) > 0
