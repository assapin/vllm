# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BatchedScheduler: a Scheduler subclass that gates prefill until a batch
threshold is met, enabling explicit batching via the scheduler_cls injection
point (SchedulerConfig.scheduler_cls).

Usage (standard vllm serve — no separate entrypoint needed)::

    vllm serve --model <model> \\
        --scheduler-cls vllm.entrypoints.openai.batched_scheduler.BatchedScheduler \\
        --additional-config '{"max_wait_ms": 100, "min_batch_tokens": 500}'

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

Configuration priority
----------------------
1. ``additional_config`` dict (passed via ``--additional-config`` CLI arg) —
   canonical path when using ``vllm serve``.
2. Env vars ``VLLM_BATCHED_MAX_WAIT_MS`` / ``VLLM_BATCHED_MIN_BATCH_TOKENS`` —
   inherited by the EngineCore subprocess on multiprocessing spawn, so they
   are set automatically from ``additional_config`` values before the engine
   is created.  Can also be set manually.
3. Class-level defaults (``max_wait_ms=100.0``, ``min_batch_tokens=0``).

Prometheus metrics
------------------
Three histograms emitted on each batch fire (lazily registered so buckets
can be derived from the actual config):

- ``vllm_batched_batch_delay_ms``      — ms from first waiting request seen
  by schedule() to gate fire. One observation per batch. Buckets derived
  from ``max_wait_ms``.
- ``vllm_batched_batch_size_requests`` — number of requests per fired batch.
- ``vllm_batched_batch_size_tokens``   — total prompt tokens per fired batch.
  Buckets derived from ``min_batch_tokens``.
"""

import os
import time
from typing import ClassVar

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler


class BatchedScheduler(Scheduler):
    """Scheduler that defers prefill until a batch threshold is met.

    See module docstring for gate semantics, configuration, and metrics.
    """

    # Class-level defaults — overridden per-instance from additional_config
    # or env vars.
    max_wait_ms: float = 100.0
    min_batch_tokens: int = 0

    # Lazily-initialised Prometheus histograms (one set per process).
    _hist_delay: ClassVar = None
    _hist_requests: ClassVar = None
    _hist_tokens: ClassVar = None

    def __init__(self, vllm_config, *args, **kwargs) -> None:
        super().__init__(vllm_config, *args, **kwargs)
        self._batch_start: float | None = None

        # 1. Read from additional_config (set via --additional-config CLI arg).
        #    This is the canonical path when using standard `vllm serve`.
        extra = (vllm_config.additional_config or {}) if vllm_config else {}
        self.max_wait_ms = float(
            extra.get("max_wait_ms", type(self).max_wait_ms))
        self.min_batch_tokens = int(
            extra.get("min_batch_tokens", type(self).min_batch_tokens))

        # 2. Env vars override additional_config — they are the subprocess
        #    inheritance mechanism when EngineCore is spawned.  batched_server
        #    (if used) sets them before spawning; users can also set manually.
        max_wait_env = os.environ.get("VLLM_BATCHED_MAX_WAIT_MS")
        if max_wait_env is not None:
            self.max_wait_ms = float(max_wait_env)
        min_tokens_env = os.environ.get("VLLM_BATCHED_MIN_BATCH_TOKENS")
        if min_tokens_env is not None:
            self.min_batch_tokens = int(min_tokens_env)

        self._init_metrics()

    # ------------------------------------------------------------------
    # Prometheus metric initialisation (lazy, config-aware buckets).
    # ------------------------------------------------------------------

    def _delay_buckets(self) -> list[float]:
        """Buckets for batch_delay_ms, centred around max_wait_ms."""
        w = self.max_wait_ms
        if w <= 0:
            return [1, 5, 10, 25, 50, 100, 200, 500, 1000]
        fractions = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
        return sorted({max(1.0, round(w * f, 1)) for f in fractions})

    def _token_buckets(self) -> list[float]:
        """Buckets for batch_size_tokens, centred around min_batch_tokens."""
        t = self.min_batch_tokens
        if t <= 0:
            return [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
        fractions = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        return sorted({max(1, round(t * f)) for f in fractions})

    def _init_metrics(self) -> None:
        """Lazily register Prometheus histograms once per process."""
        cls = type(self)
        if cls._hist_delay is not None:
            return  # already initialised in this process
        try:
            from prometheus_client import Histogram
        except ImportError:
            return  # prometheus_client not installed; metrics silently skipped
        cls._hist_delay = Histogram(
            "vllm_batched_batch_delay_ms",
            "Elapsed ms from first request entering waiting queue to batch"
            " fire (one observation per batch).",
            buckets=self._delay_buckets(),
        )
        cls._hist_requests = Histogram(
            "vllm_batched_batch_size_requests",
            "Number of requests in each fired prefill batch.",
            buckets=[1, 2, 4, 8, 16, 32, 64, 128, 256],
        )
        cls._hist_tokens = Histogram(
            "vllm_batched_batch_size_tokens",
            "Total prompt tokens in each fired prefill batch.",
            buckets=self._token_buckets(),
        )

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
            self._emit_metrics(elapsed_ms, waiting_tokens)
            return super().schedule()

        # Gate not yet met — return an empty output.  The engine will sleep
        # 1 ms before calling schedule() again (existing code path).
        return SchedulerOutput.make_empty()

    def _emit_metrics(self, elapsed_ms: float, waiting_tokens: int) -> None:
        """Emit per-batch Prometheus observations on gate fire."""
        if self._hist_delay is None:
            return
        self._hist_delay.observe(elapsed_ms)
        self._hist_requests.observe(len(self.waiting))
        self._hist_tokens.observe(waiting_tokens)
