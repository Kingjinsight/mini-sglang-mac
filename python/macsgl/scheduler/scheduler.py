from __future__ import annotations

from typing import TYPE_CHECKING, List, NamedTuple, NoReturn, Set, Tuple, TypeAlias

import mlx.core as mx

from macsgl.core import Batch, Req
from macsgl.engine import Engine
from macsgl.env import ENV
from macsgl.message import (
    AbortBackendMsg,
    BaseBackendMsg,
    BatchBackendMsg,
    DetokenizeMsg,
    ExitMsg,
    UserMsg,
)
from macsgl.utils import init_logger, load_tokenizer

from .cache import CacheManager
from .config import SchedulerConfig
from .decode import DecodeManager
from .io import SchedulerIOMixin
from .prefill import ChunkedReq, PrefillManager
from .table import TableManager

if TYPE_CHECKING:
    from macsgl.engine import ForwardMeta, ForwardOutput, SamplingArgs


logger = init_logger(__name__)

# Admission reserves this fraction of each request's remaining output tokens. It decays while
# decoding fits in memory and is re-estimated from the actual progress after a preemption.
INIT_NEW_TOKEN_RATIO = 0.7
MIN_NEW_TOKEN_RATIO = 0.1
NEW_TOKEN_RATIO_DECAY = 0.001
RETRACT_MARGIN_STEPS = 20


class ForwardInput(NamedTuple):
    batch: Batch
    sample_args: SamplingArgs
    meta: ForwardMeta


ForwardData: TypeAlias = "Tuple[ForwardInput, ForwardOutput]"


class Scheduler(SchedulerIOMixin):
    def __init__(self, config: SchedulerConfig):
        self.engine = Engine(config)
        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        self.cache_manager = CacheManager(
            self.engine.num_pages, config.page_size, self.engine.page_table, config.cache_type
        )
        self.decode_manager = DecodeManager(config.page_size)
        self.prefill_manager = PrefillManager(
            self.cache_manager, self.table_manager, self.decode_manager
        )

        self.finished_reqs: Set[Req] = set()
        self.tokenizer = load_tokenizer(config.model_path)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.prefill_budget = config.max_extend_tokens
        self.new_token_ratio = INIT_NEW_TOKEN_RATIO
        super().__init__(config)

    def run_when_idle(self) -> None:
        """Called when the scheduler is idle to perform background tasks."""
        logger.info("Scheduler is idle, waiting for new reqs...")
        self.cache_manager.check_integrity()

    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """
        The main loop of overlapping scheduling and execution.

        While the GPU runs the current batch, the CPU schedules the next one and processes
        the results of the last one, which hides the CPU latency.
        """
        blocking = not (
            last_data is not None  # don't block if we have a batch to be processed
            or self.prefill_manager.runnable
            or self.decode_manager.runnable
        )
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch(inflight=last_data is not None)
        ongoing_data = None
        if forward_input is not None:
            ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(last_data)
        return ongoing_data

    def normal_loop(self) -> None:
        blocking = not (self.prefill_manager.runnable or self.decode_manager.runnable)
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(ongoing_data)

    def run_forever(self) -> NoReturn:
        if ENV.DISABLE_OVERLAP_SCHEDULING:
            while True:
                self.normal_loop()
        data = None
        while True:
            data = self.overlap_loop(data)

    def shutdown(self) -> None:
        mx.synchronize()

    def _process_last_data(self, last_data: ForwardData | None) -> None:
        if last_data is None:
            return

        batch, (_, next_tokens_cpu, copy_done) = last_data[0].batch, last_data[1]
        copy_done.synchronize()
        reply: List[DetokenizeMsg] = []
        new_finished_reqs: Set[Req] = set()
        with self.cache_manager.lazy_free_region():
            for i, req in enumerate(batch.reqs):
                # NOTE: overlap scheduling runs a finished request for one extra step, drop it
                if isinstance(req, ChunkedReq) or req in self.finished_reqs:
                    continue
                next_token = next_tokens_cpu[i]
                req.append_host(next_token.unsqueeze(0))
                next_token = int(next_token.item())
                # count host tokens: under overlap, device_len already includes the in-flight step
                finished = len(req.input_ids) >= req.max_device_len
                if not req.sampling_params.ignore_eos:
                    finished |= next_token == self.eos_token_id
                reply.append(DetokenizeMsg(uid=req.uid, next_token=next_token, finished=finished))

                if finished:
                    self.decode_manager.remove_req(req)
                    self._free_req_resources(req)
                    new_finished_reqs.add(req)
                elif batch.is_prefill:  # for prefill, non-chunk req, cache the prefix
                    self.cache_manager.cache_req(req, finished=False)

        self.finished_reqs = new_finished_reqs
        self.send_result(reply)

    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        if isinstance(msg, BatchBackendMsg):
            for msg in msg.data:
                self._process_one_msg(msg)
        elif isinstance(msg, ExitMsg):
            raise KeyboardInterrupt
        elif isinstance(msg, UserMsg):
            logger.debug("Received user msg: %s", msg)
            input_len, max_seq_len = len(msg.input_ids), self.engine.max_seq_len
            max_output_len = max_seq_len - input_len
            if max_output_len <= 0:
                return logger.warning(
                    f"Input sequence length {input_len} exceeds {max_seq_len}, "
                    f"request {msg.uid} is dropped."
                )
            if msg.sampling_params.max_tokens > max_output_len:
                msg.sampling_params.max_tokens = max_output_len
                logger.warning(f"Adjust max_tokens to {max_output_len} for request {msg.uid}.")
            self.prefill_manager.add_one_req(msg)
        elif isinstance(msg, AbortBackendMsg):
            logger.debug("Aborting request %d", msg.uid)
            req_to_free = self.prefill_manager.abort_req(msg.uid)
            req_to_free = req_to_free or self.decode_manager.abort_req(msg.uid)
            if req_to_free is not None:
                self._free_req_resources(req_to_free)
        else:
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError

    def _free_req_resources(self, req: Req) -> None:
        self.table_manager.free(req.table_idx)
        self.cache_manager.cache_req(req, finished=True)

    def _prepare_batch(self, batch: Batch) -> ForwardInput:
        self.cache_manager.allocate_paged(batch.reqs)
        return ForwardInput(batch, self.engine.sampler.prepare(batch), self.engine.prepare(batch))

    def _schedule_next_batch(self, inflight: bool = False) -> ForwardInput | None:
        # TODO: support other policies: e.g. DECODE first
        batch = self.prefill_manager.schedule_next_batch(
            self.prefill_budget, self.new_token_ratio
        ) or self._schedule_decode_batch(inflight)
        return self._prepare_batch(batch) if batch else None

    def _schedule_decode_batch(self, inflight: bool) -> Batch | None:
        running = self.decode_manager.running_reqs
        if running and not self.cache_manager.can_allocate(running):
            if inflight:  # retract only after the in-flight batch has updated every request
                return None
            self._retract_decode_reqs()
        self.new_token_ratio = max(
            MIN_NEW_TOKEN_RATIO, self.new_token_ratio - NEW_TOKEN_RATIO_DECAY
        )
        return self.decode_manager.schedule_next_batch()

    def _retract_decode_reqs(self) -> None:
        """Preempt the newest requests until the next decode step fits in the KV cache."""
        running = sorted(self.decode_manager.running_reqs, key=lambda req: req.uid)
        num_retracted = 0
        while running and not self.cache_manager.can_allocate(running):
            req = running.pop()
            self.decode_manager.remove_req(req)
            self._free_req_resources(req)  # its KV stays in the prefix cache for re-admission
            self.prefill_manager.add_retracted_req(req)
            num_retracted += 1
        decoded = sum(req.output_len - req.remain_len for req in running)
        total = max(sum(req.output_len for req in running), 1)
        self.new_token_ratio = min(1.0, (decoded + RETRACT_MARGIN_STEPS * len(running)) / total)
        logger.warning(
            f"Retracted {num_retracted} requests, new_token_ratio={self.new_token_ratio:.3f}"
        )

    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        batch, sample_args, meta = forward_input
        output = self.engine.forward_batch(batch, sample_args, meta)
        self.decode_manager.filter_reqs(batch.reqs)
        return output
