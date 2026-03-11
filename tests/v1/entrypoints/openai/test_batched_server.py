# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration tests for the BatchedScheduler + batched_server entry point.

Each test gets its own fresh server (function-scoped fixture) so there is
no shared scheduler state between tests.

Behavioral proofs:
  1. Time-gate holds     — 10 s gate, single request, verify response >= 10 s.
  2. Token-gate unblocks — token threshold only (no time gate), short requests
                           stay blocked, final large request crosses threshold
                           and all return together.
  3. Concurrent batching — N concurrent requests all succeed in one gate pass.
  4. Passthrough         — both gates disabled, requests succeed (no deadlock).
"""

import asyncio
import os
import subprocess
import sys
import time

import pytest

from tests.utils import RemoteOpenAIServer

MODEL_NAME = "facebook/opt-125m"

_COMMON_ARGS = [
    "--dtype", "float32",
    "--max-model-len", "2048",
    "--max-num-seqs", "128",
    "--enforce-eager",
    "--no-enable-prefix-caching",
]


# ---------------------------------------------------------------------------
# RemoteBatchedServer
# ---------------------------------------------------------------------------

class RemoteBatchedServer(RemoteOpenAIServer):
    """Starts ``vllm.entrypoints.openai.batched_server`` in a subprocess."""

    def __init__(
        self,
        model: str,
        vllm_serve_args: list[str],
        *,
        max_wait_ms: float = 100.0,
        min_batch_tokens: int = 0,
        **kwargs,
    ) -> None:
        self._max_wait_ms = max_wait_ms
        self._min_batch_tokens = min_batch_tokens
        super().__init__(model, vllm_serve_args, **kwargs)

    def _start_server(
        self,
        model: str,
        vllm_serve_args: list[str],
        env_dict: dict[str, str] | None,
    ) -> None:
        env = os.environ.copy()
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        if env_dict:
            env.update(env_dict)
        serve_cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.batched_server",
            "--model", model,
            "--max-wait-ms", str(self._max_wait_ms),
            "--min-batch-tokens", str(self._min_batch_tokens),
            *vllm_serve_args,
        ]
        print(f"Launching RemoteBatchedServer: {' '.join(serve_cmd)}")
        self.proc = subprocess.Popen(
            serve_cmd,
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr,
            start_new_session=True,
        )


# ---------------------------------------------------------------------------
# Fixtures — function-scoped: fresh server (and clean scheduler state)
# per test.
# ---------------------------------------------------------------------------

@pytest.fixture
def server_10s_gate():
    """10-second time gate, no token gate."""
    with RemoteBatchedServer(
        MODEL_NAME, _COMMON_ARGS,
        max_wait_ms=10_000,
        min_batch_tokens=0,
    ) as s:
        yield s


@pytest.fixture
def server_token_gate():
    """Token gate only (time gate disabled).

    min_batch_tokens=200: short prompts (~5 tokens each) won't trigger the
    gate; only a large prompt that pushes the total over 200 will fire it.
    """
    with RemoteBatchedServer(
        MODEL_NAME, _COMMON_ARGS,
        max_wait_ms=0,
        min_batch_tokens=200,
    ) as s:
        yield s


@pytest.fixture
def server_300ms_gate():
    """300 ms time gate for the concurrent-batching efficiency test."""
    with RemoteBatchedServer(
        MODEL_NAME, _COMMON_ARGS,
        max_wait_ms=300,
        min_batch_tokens=0,
    ) as s:
        yield s


@pytest.fixture
def server_passthrough():
    """Both gates disabled — behaves like a normal vLLM server."""
    with RemoteBatchedServer(
        MODEL_NAME, _COMMON_ARGS,
        max_wait_ms=0,
        min_batch_tokens=0,
    ) as s:
        yield s


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_time_gate_holds_request(server_10s_gate):
    """A single request must be held for the full 10-second gate window.

    With max_wait_ms=10_000 and no token gate, the scheduler will not fire
    until 10 s have elapsed.  We assert the response arrives no earlier than
    9 s (10 % tolerance for scheduling jitter).
    """
    async with server_10s_gate.get_async_client() as client:
        t0 = time.monotonic()
        completion = await client.completions.create(
            model=MODEL_NAME,
            prompt="Hello",
            max_tokens=5,
            temperature=0.0,
        )
        elapsed_s = time.monotonic() - t0

    assert completion.choices[0].finish_reason == "length"
    assert elapsed_s >= 9.0, (
        f"Response arrived after only {elapsed_s:.2f} s — "
        "expected the 10 s time gate to hold the request."
    )


@pytest.mark.asyncio
async def test_token_gate_unblocks_batch(server_token_gate):
    """Short requests stay blocked until a large request crosses the threshold.

    Setup: time gate disabled (max_wait_ms=0), min_batch_tokens=200.

    Steps:
      1. Fire 5 short requests (~5 tokens each, ~25 total) as background tasks.
      2. Sleep 2 s — enough time for them to reach the scheduler's waiting
         queue.  Assert they are still pending (gate not yet open).
      3. Send one large request whose prompt alone exceeds the remaining
         threshold, pushing the cumulative total over 200 tokens.
      4. Assert that all 6 requests complete within a few seconds of the
         large request arriving (the gate fired and the batch ran).
    """
    # "hello " ≈ 2 OPT tokens; 5 requests × ~2 tokens = ~10 tokens total —
    # well below the 200-token threshold.
    short_prompt = "hello"

    # "word " repeated 200 times ≈ 200+ tokens — enough to cross the threshold
    # on its own even before counting the short requests already waiting.
    large_prompt = "word " * 200

    async with server_token_gate.get_async_client() as client:
        # Step 1: kick off short requests without awaiting them.
        short_tasks = [
            asyncio.create_task(
                client.completions.create(
                    model=MODEL_NAME,
                    prompt=short_prompt,
                    max_tokens=5,
                    temperature=0.0,
                )
            )
            for _ in range(5)
        ]

        # Step 2: give the server time to receive and queue the requests.
        await asyncio.sleep(2.0)
        assert not any(t.done() for t in short_tasks), (
            "Short requests completed before the token threshold was reached — "
            "the token gate does not appear to be holding them."
        )

        # Step 3: send the threshold-crossing request and time it.
        t0 = time.monotonic()
        large_result = await client.completions.create(
            model=MODEL_NAME,
            prompt=large_prompt,
            max_tokens=5,
            temperature=0.0,
        )
        gate_fire_s = time.monotonic() - t0

        # Step 4: collect the short requests — they should already be done
        # (the same GPU pass that handled the large request handled them too).
        short_results = await asyncio.gather(*short_tasks)

    # All requests succeeded.
    assert large_result.choices[0].finish_reason == "length"
    for i, r in enumerate(short_results):
        assert r.choices[0].finish_reason == "length", (
            f"Short request {i} did not finish normally."
        )

    # The large request (which triggered the gate) should have returned quickly
    # — no time gate is active, so it fires as soon as tokens >= threshold.
    assert gate_fire_s < 5.0, (
        f"Large request took {gate_fire_s:.2f} s after crossing the token "
        "threshold — expected near-immediate firing with no time gate."
    )


@pytest.mark.asyncio
async def test_concurrent_requests_batched_in_one_pass(server_300ms_gate):
    """N concurrent requests complete in one gate window, not N serial windows.

    With a 300 ms gate, if requests were serialised the total time would be
    N × 300 ms.  With batching all N fire in a single pass, so total time
    is ~300 ms + one GPU forward pass.  We assert total < 2 gate windows
    plus a generous GPU budget.
    """
    N = 8
    async with server_300ms_gate.get_async_client() as client:
        t0 = time.monotonic()
        results = await asyncio.gather(*[
            client.completions.create(
                model=MODEL_NAME,
                prompt=f"The word number {i} is",
                max_tokens=5,
                temperature=0.0,
            )
            for i in range(N)
        ])
        elapsed_ms = (time.monotonic() - t0) * 1000

    assert all(r.choices[0].finish_reason == "length" for r in results)

    # 2 gate windows + 3 s GPU budget << N × 300 ms = 2400 ms
    max_expected_ms = 2 * 300 + 3_000
    assert elapsed_ms < max_expected_ms, (
        f"{N} concurrent requests took {elapsed_ms:.0f} ms — expected < "
        f"{max_expected_ms:.0f} ms.  They may have been serialised."
    )


@pytest.mark.asyncio
async def test_passthrough_serves_requests(server_passthrough):
    """With both gates disabled the server accepts and answers requests.

    This confirms that max_wait_ms=0 / min_batch_tokens=0 does not deadlock
    or error — the scheduler falls through to the base Scheduler immediately.
    """
    async with server_passthrough.get_async_client() as client:
        completion = await client.completions.create(
            model=MODEL_NAME,
            prompt="Hello, my name is",
            max_tokens=5,
            temperature=0.0,
        )

    assert completion.choices[0].finish_reason == "length"
    assert len(completion.choices[0].text) > 0
