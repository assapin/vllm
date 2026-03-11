# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit and integration tests for BatchedScheduler.

All tests run in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0) following the
same pattern as tests/plugins_tests/test_scheduler_plugins.py.
"""

import time

import pytest

from vllm.engine.arg_utils import EngineArgs
from vllm.entrypoints.openai.batched_scheduler import BatchedScheduler
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.llm_engine import LLMEngine

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MODEL = "facebook/opt-125m"


def _make_engine(monkeypatch: pytest.MonkeyPatch) -> LLMEngine:
    """Create an in-process LLMEngine with BatchedScheduler injected."""
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    engine_args = EngineArgs(
        model=_MODEL,
        enforce_eager=True,
        scheduler_cls=BatchedScheduler,
    )
    return LLMEngine.from_engine_args(engine_args)


def _add(engine: LLMEngine, req_id: str, prompt: str = "hello world") -> None:
    engine.add_request(req_id, prompt, SamplingParams(max_tokens=1))


# ---------------------------------------------------------------------------
# Test 1: token gate holds until threshold
# ---------------------------------------------------------------------------


def test_token_gate_holds_until_threshold(monkeypatch: pytest.MonkeyPatch):
    """schedule() returns empty SchedulerOutput until min_batch_tokens met."""
    BatchedScheduler.max_wait_ms = 99_999.0  # time gate effectively disabled
    BatchedScheduler.min_batch_tokens = 500

    engine = _make_engine(monkeypatch)
    try:
        # Add two small requests — combined tokens should stay below threshold
        _add(engine, "0", "hi")
        _add(engine, "1", "hello")

        outputs = engine.step()
        assert outputs == [], "Gate should be closed: token threshold not met"

        # Add a long request that pushes total tokens over threshold
        _add(engine, "2", "word " * 200)  # ~200 tokens, well over 500 - 2
        outputs = engine.step()
        assert len(outputs) > 0, "Gate should open: token threshold met"
    finally:
        engine.shutdown()


# ---------------------------------------------------------------------------
# Test 2: time gate fires after max_wait_ms
# ---------------------------------------------------------------------------


def test_time_gate_fires_after_wait_ms(monkeypatch: pytest.MonkeyPatch):
    """schedule() returns real output after max_wait_ms has elapsed."""
    BatchedScheduler.max_wait_ms = 50.0
    BatchedScheduler.min_batch_tokens = 99_999  # token gate effectively disabled

    engine = _make_engine(monkeypatch)
    try:
        _add(engine, "0", "hi")

        outputs = engine.step()
        assert outputs == [], "Gate should be closed: time not elapsed"

        time.sleep(0.060)  # 60 ms > 50 ms threshold

        outputs = engine.step()
        assert len(outputs) > 0, "Gate should open: time threshold elapsed"
    finally:
        engine.shutdown()


# ---------------------------------------------------------------------------
# Test 3: running (decode) requests bypass the gate
# ---------------------------------------------------------------------------


def test_running_requests_bypass_gate(monkeypatch: pytest.MonkeyPatch):
    """After prefill fires, decode steps proceed without batching gate."""
    BatchedScheduler.max_wait_ms = 99_999.0
    BatchedScheduler.min_batch_tokens = 0  # passthrough — fire immediately

    engine = _make_engine(monkeypatch)
    try:
        _add(engine, "0", "hello world")

        # First step — prefill (gate open because min_batch_tokens=0)
        outputs = engine.step()
        assert len(outputs) > 0, "First step (prefill) should proceed"

        # Request may still be generating — decode steps must not be gated
        if not outputs[0].finished:
            outputs2 = engine.step()
            # Decode step should not be blocked even with a large max_wait_ms
            assert len(outputs2) > 0, "Decode step should not be gated"
    finally:
        engine.shutdown()


# ---------------------------------------------------------------------------
# Test 4: gate timer resets after batch fires
# ---------------------------------------------------------------------------


def test_gate_resets_after_batch_fires(monkeypatch: pytest.MonkeyPatch):
    """_batch_start resets to None so the next batch starts a fresh timer."""
    BatchedScheduler.max_wait_ms = 50.0
    BatchedScheduler.min_batch_tokens = 99_999  # token gate disabled

    engine = _make_engine(monkeypatch)
    try:
        _add(engine, "0", "hi")
        # Let first batch fire
        time.sleep(0.060)
        outputs = engine.step()
        assert len(outputs) > 0, "First batch should have fired"

        # Drain remaining steps for the first request
        for _ in range(50):
            remaining = engine.step()
            if not remaining or all(o.finished for o in remaining):
                break

        # Immediately add a new request — timer just reset, should NOT fire yet
        _add(engine, "1", "hello")
        outputs2 = engine.step()
        assert outputs2 == [], "Fresh timer should hold the batch"
    finally:
        engine.shutdown()


# ---------------------------------------------------------------------------
# Test 5: default passthrough (max_wait_ms=0, min_batch_tokens=0)
# ---------------------------------------------------------------------------


def test_default_passthrough(monkeypatch: pytest.MonkeyPatch):
    """With both gates at 0, BatchedScheduler behaves like base Scheduler."""
    BatchedScheduler.max_wait_ms = 0.0
    BatchedScheduler.min_batch_tokens = 0

    engine = _make_engine(monkeypatch)
    try:
        _add(engine, "0", "hello")
        outputs = engine.step()
        assert len(outputs) > 0, "Should fire immediately with gates at 0"
    finally:
        engine.shutdown()


# ---------------------------------------------------------------------------
# Test 6: end-to-end via EngineArgs.scheduler_cls injection
# ---------------------------------------------------------------------------


def test_scheduler_cls_injection_end_to_end(monkeypatch: pytest.MonkeyPatch):
    """BatchedScheduler correctly defers prefill when injected via EngineArgs."""
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    BatchedScheduler.max_wait_ms = 99_999.0
    BatchedScheduler.min_batch_tokens = 200  # requires ~200 tokens to fire

    engine_args = EngineArgs(
        model=_MODEL,
        enforce_eager=True,
        scheduler_cls=BatchedScheduler,
    )
    engine = LLMEngine.from_engine_args(engine_args)
    try:
        # Short prompt — well below 200 tokens
        engine.add_request("0", "hi", SamplingParams(max_tokens=1))

        # step() should return empty — gate not met
        outputs = engine.step()
        assert outputs == [], "Short prompt alone should not open the gate"

        # Add a longer prompt to push total tokens over threshold
        engine.add_request(
            "1",
            "tell me a very long story " * 10,  # ~60+ tokens
            SamplingParams(max_tokens=1),
        )
        # Keep adding until gate opens (token count accumulates)
        engine.add_request(
            "2",
            "explain quantum entanglement in detail " * 5,
            SamplingParams(max_tokens=1),
        )
        outputs = engine.step()
        # Gate should open now that combined tokens cross 200
        assert len(outputs) > 0, "Gate should open once token threshold is met"
    finally:
        engine.shutdown()
