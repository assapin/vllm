# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BatchedScheduler: a Scheduler subclass that gates prefill until a batch
threshold is met, enabling explicit batching via the scheduler_cls injection
point (SchedulerConfig.scheduler_cls).

Usage::

    from vllm.entrypoints.openai.batched_scheduler import BatchedScheduler

    # Set class-level config BEFORE creating AsyncLLM / LLMEngine
    BatchedScheduler.max_wait_ms = 100.0   # 0 = disabled (token gate only)
    BatchedScheduler.min_batch_tokens = 0  # 0 = disabled (time gate only)

    engine_args = AsyncEngineArgs(model=..., scheduler_cls=BatchedScheduler)
    async_llm = AsyncLLM.from_engine_args(engine_args)

Gate semantics
--------------
- ``max_wait_ms > 0``  : fire when the accumulation window expires.
- ``max_wait_ms = 0``  : time gate disabled; only token gate applies.
- ``min_batch_tokens > 0`` : fire when waiting-request token sum reaches
  this value.
- ``min_batch_tokens = 0`` : token gate disabled; only time gate applies.
- Both zero              : no gate — behaves identically to base Scheduler
  (fires every step).

Either active condition triggers the batch.  Running (decode) requests always
proceed without delay.
"""

import time

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler


class BatchedScheduler(Scheduler):
    """Scheduler that defers prefill until a batch threshold is met.

    See module docstring for gate semantics.
    """

    # Class-level config — set BEFORE creating the engine.
    max_wait_ms: float = 100.0
    min_batch_tokens: int = 0

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._batch_start: float | None = None
        # When the EngineCore subprocess is started via multiprocessing spawn,
        # it re-imports this module and loses any class-level mutations made in
        # the parent process.  Read instance-level overrides from env vars,
        # which ARE inherited by spawned children.  Env vars take precedence
        # over the class-level defaults.
        import os
        max_wait_env = os.environ.get("VLLM_BATCHED_MAX_WAIT_MS")
        if max_wait_env is not None:
            self.max_wait_ms = float(max_wait_env)
        min_tokens_env = os.environ.get("VLLM_BATCHED_MIN_BATCH_TOKENS")
        if min_tokens_env is not None:
            self.min_batch_tokens = int(min_tokens_env)

    # ------------------------------------------------------------------
    # has_requests(): keep the engine spinning (not blocking) while we
    # accumulate.  The engine's 1 ms sleep in _process_engine_step()
    # throttles the spin during the accumulation window.
    # ------------------------------------------------------------------
    def has_requests(self) -> bool:
        # Always let running (decode) steps and finished-req cleanup proceed.
        if self.running or self.has_finished_requests():
            return True
        # When only waiting requests exist, return True so the engine spins
        # rather than blocking on the input queue — the time gate needs to
        # fire even when no new requests arrive.
        return bool(self.waiting)

    # ------------------------------------------------------------------
    # schedule(): apply the accumulation gate for waiting-only state.
    # ------------------------------------------------------------------
    def schedule(self) -> SchedulerOutput:
        # Running requests: decode must never be gated.
        if self.running or not self.waiting:
            return super().schedule()

        # Both gates disabled → passthrough (behaves like base Scheduler).
        if self.max_wait_ms <= 0 and self.min_batch_tokens <= 0:
            return super().schedule()

        # Waiting-only: apply the accumulation gate.
        now = time.monotonic()
        if self._batch_start is None:
            self._batch_start = now

        elapsed_ms = (now - self._batch_start) * 1000
        waiting_tokens = sum(r.num_prompt_tokens for r in self.waiting)

        time_gate_open = (
            self.max_wait_ms > 0 and elapsed_ms >= self.max_wait_ms
        )
        token_gate_open = (
            self.min_batch_tokens > 0
            and waiting_tokens >= self.min_batch_tokens
        )

        if time_gate_open or token_gate_open:
            self._batch_start = None  # reset for the next batch window
            return super().schedule()

        # Gate not yet met — return an empty output.  The engine will sleep
        # 1 ms before calling schedule() again (existing code path).
        return SchedulerOutput.make_empty()
