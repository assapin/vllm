# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit and integration tests for BatchedScheduler.

Each test runs in a forked subprocess via @create_new_process_for_each_test()
(same pattern as tests/v1/engine/test_engine_core.py).  This gives full CUDA
memory isolation between tests without any manual cleanup.

Gate semantics (0 = disabled)
------------------------------
- max_wait_ms=0    : time gate disabled; only token gate applies.
- min_batch_tokens=0 : token gate disabled; only time gate applies.
- Both 0           : no gate — passthrough, fires every step.

Note on v1 engine output pipeline
----------------------------------
With async scheduling enabled (the default), the engine may pipeline the GPU
execution so that the RequestOutput objects are returned one step later than
the step that fired the batch.  Whether this happens depends on a race: if the
async output thread finishes the D2H copy before the done() check inside
step_with_batch_queue, the output is returned immediately; otherwise it is
deferred one step.

Tests use _drain() to collect output.  _drain() calls engine.step() in a loop
and returns on the first non-empty result.  When _drain() is used to both fire
the gate and collect the result, the first iteration fires the batch and
subsequent iterations collect it — regardless of whether pipelining defers the
output or not.

Timer ordering: _batch_start is set on the first schedule() call after a
request enters waiting.  Tests that sleep to trigger the time gate must call
engine.step() first (to start the timer and assert gate-is-closed), then
sleep, then _drain() to fire and collect.
"""

import os
import time

from vllm.engine.arg_utils import EngineArgs
from vllm.entrypoints.openai.batched_scheduler import BatchedScheduler
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.llm_engine import LLMEngine

from ...utils import create_new_process_for_each_test

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MODEL = "facebook/opt-125m"


def _make_engine() -> LLMEngine:
    """Create an in-process LLMEngine with BatchedScheduler injected."""
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    engine_args = EngineArgs(
        model=_MODEL,
        enforce_eager=True,
        enable_chunked_prefill=False,
        #async_scheduling=False,
        scheduler_cls=BatchedScheduler,
    )
    return LLMEngine.from_engine_args(engine_args)


def _add(engine: LLMEngine, req_id: str, prompt: str = "hello world",
         max_tokens: int = 1) -> None:
    engine.add_request(req_id, prompt, SamplingParams(max_tokens=max_tokens))


def _drain(engine: LLMEngine, max_steps: int = 5) -> list:
    """Step until RequestOutput objects arrive (accounts for pipelined output).

    The v1 engine fires the GPU on step N and delivers RequestOutput on step
    N+1.  Call this helper instead of a bare engine.step() whenever the test
    needs to *collect* output, not merely trigger a scheduling cycle.
    """
    for _ in range(max_steps):
        outputs = engine.step()
        if outputs:
            return outputs
    return []


# ---------------------------------------------------------------------------
# Test 1: token-only gate (time gate disabled via max_wait_ms=0)
# ---------------------------------------------------------------------------


@create_new_process_for_each_test()
def test_token_gate_holds_until_threshold():
    """Token gate holds the batch until min_batch_tokens is met.

    max_wait_ms=0 disables the time gate so only the token count controls
    when the batch fires.
    """
    BatchedScheduler.max_wait_ms = 0       # time gate disabled
    BatchedScheduler.min_batch_tokens = 500

    engine = _make_engine()
    try:
        # Add two small requests — combined tokens (≈4) stay below threshold.
        _add(engine, "0", "hi")
        _add(engine, "1", "hello")

        outputs = engine.step()
        assert outputs == [], "Gate should be closed: token threshold not met"

        # Add a long request that pushes the total over the threshold.
        # "word " * 500 ≈ 502 tokens; combined total ≈ 506 > 500.
        _add(engine, "2", "word " * 500)
        outputs = _drain(engine)
        assert len(outputs) > 0, "Gate should open: token threshold met"
    finally:
        engine.engine_core.shutdown()


# ---------------------------------------------------------------------------
# Test 2: time-only gate (token gate disabled via min_batch_tokens=0)
# ---------------------------------------------------------------------------


@create_new_process_for_each_test()
def test_time_gate_fires_after_wait_ms():
    """Time gate fires the batch after max_wait_ms has elapsed.

    min_batch_tokens=0 disables the token gate so only elapsed time controls
    when the batch fires.
    """
    BatchedScheduler.max_wait_ms = 50.0
    BatchedScheduler.min_batch_tokens = 0  # token gate disabled

    engine = _make_engine()
    try:
        _add(engine, "0", "hi")

        # First step starts the timer; gate is not open yet.
        outputs = engine.step()
        assert outputs == [], "Gate should be closed: time not elapsed"

        time.sleep(0.060)  # 60 ms > 50 ms threshold

        outputs = _drain(engine)
        assert len(outputs) > 0, "Gate should open: time threshold elapsed"
    finally:
        engine.engine_core.shutdown()


# ---------------------------------------------------------------------------
# Test 3: running (decode) requests bypass the gate
# ---------------------------------------------------------------------------


@create_new_process_for_each_test()
def test_running_requests_bypass_gate():
    """After prefill fires, decode steps proceed without the batching gate.

    Both gates set to 0 (passthrough) so prefill fires immediately, putting
    the request into running state.  BatchedScheduler then bypasses the gate
    entirely via `if self.running: return super().schedule()`.
    """
    BatchedScheduler.max_wait_ms = 0       # time gate disabled
    BatchedScheduler.min_batch_tokens = 0  # token gate disabled → passthrough

    engine = _make_engine()
    try:
        # max_tokens=3 ensures at least one decode step follows prefill.
        _add(engine, "0", "hello world", max_tokens=3)

        outputs = _drain(engine)
        assert len(outputs) > 0, "Prefill should proceed with passthrough config"

        # If the request is still generating, decode must not be gated.
        if not outputs[0].finished:
            outputs2 = _drain(engine)
            assert len(outputs2) > 0, "Decode step should not be gated"
    finally:
        engine.engine_core.shutdown()


# ---------------------------------------------------------------------------
# Test 4: gate timer resets after batch fires
# ---------------------------------------------------------------------------


@create_new_process_for_each_test()
def test_gate_resets_after_batch_fires():
    """_batch_start resets to None so the next batch starts a fresh timer.

    min_batch_tokens=0 disables the token gate so only the time gate is active.
    """
    BatchedScheduler.max_wait_ms = 50.0
    BatchedScheduler.min_batch_tokens = 0  # token gate disabled

    engine = _make_engine()
    try:
        _add(engine, "0", "hi")
        # Start the timer (first schedule() call sets _batch_start).
        engine.step()
        # Sleep so the gate opens on the next step.
        time.sleep(0.060)
        outputs = _drain(engine)
        assert len(outputs) > 0, "First batch should have fired"

        # Drain any remaining steps for the first request.
        for _ in range(50):
            remaining = engine.step()
            if not remaining or all(o.finished for o in remaining):
                break

        # Immediately add a new request — the timer just reset to None, so
        # elapsed_ms will be near 0, well below max_wait_ms=50.
        _add(engine, "1", "hello")
        outputs2 = engine.step()
        assert outputs2 == [], "Fresh timer should hold the new batch"
    finally:
        engine.engine_core.shutdown()


# ---------------------------------------------------------------------------
# Test 5: both gates disabled → passthrough
# ---------------------------------------------------------------------------


@create_new_process_for_each_test()
def test_passthrough_when_both_gates_disabled():
    """Both gates at 0 → no gating, behaves like base Scheduler."""
    BatchedScheduler.max_wait_ms = 0       # time gate disabled
    BatchedScheduler.min_batch_tokens = 0  # token gate disabled

    engine = _make_engine()
    try:
        _add(engine, "0", "hello")
        outputs = _drain(engine)
        assert len(outputs) > 0, "Should fire immediately with both gates disabled"
    finally:
        engine.engine_core.shutdown()


# ---------------------------------------------------------------------------
# Test 6: end-to-end via EngineArgs.scheduler_cls injection
# ---------------------------------------------------------------------------


@create_new_process_for_each_test()
def test_scheduler_cls_injection_end_to_end():
    """BatchedScheduler correctly defers prefill when injected via EngineArgs.

    max_wait_ms=0 disables the time gate so the batch fires purely on token
    count reaching min_batch_tokens.
    """
    BatchedScheduler.max_wait_ms = 0       # time gate disabled
    BatchedScheduler.min_batch_tokens = 100

    engine = _make_engine()
    try:
        # Short prompt — well below 100 tokens (≈2 tokens).
        engine.add_request("0", "hi", SamplingParams(max_tokens=1))

        outputs = engine.step()
        assert outputs == [], "Short prompt alone should not open the gate"

        # Add prompts that push the combined token count over 100:
        #   "tell me a very long story " * 10  ≈ 62 tokens
        #   "explain quantum entanglement in detail " * 5  ≈ 38 tokens
        #   total with "hi" ≈ 102 tokens > 100
        engine.add_request(
            "1",
            "tell me a very long story " * 10,
            SamplingParams(max_tokens=1),
        )
        engine.add_request(
            "2",
            "explain quantum entanglement in detail " * 5,
            SamplingParams(max_tokens=1),
        )
        outputs = _drain(engine)
        assert len(outputs) > 0, "Gate should open once token threshold is met"
    finally:
        engine.engine_core.shutdown()
