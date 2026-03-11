# Plan: vLLM Batched Inference Server

## Context

Add a **`BatchedScheduler`** that accumulates requests up to a time or token budget, then fires them all at once through the GPU in a single prefill batch. It plugs into the standard `vllm serve` entry point via `--scheduler-cls` and `--additional-config` — no separate server, no core files modified.

**CLI:**
```bash
vllm serve --model <model> \
    --scheduler-cls vllm.entrypoints.openai.batched_scheduler.BatchedScheduler \
    --additional-config '{"max_wait_ms": 100, "min_batch_tokens": 500}'
```

**YAML config (`--config batched.yaml`):**
```yaml
model: facebook/opt-125m
scheduler-cls: vllm.entrypoints.openai.batched_scheduler.BatchedScheduler
additional-config:
  max_wait_ms: 100
  min_batch_tokens: 500
port: 8001
enforce-eager: true
```

The YAML loader serializes nested dicts to JSON automatically, so `additional-config` as a YAML mapping is equivalent to the inline JSON form.

The implementation uses **`AsyncLLM`** (the standard v1 async engine) with `BatchedScheduler` injected via `SchedulerConfig.scheduler_cls` — a first-class vLLM injection point. No core files are modified.

---

## AsyncLLM Architecture

### Component Overview

```
┌─── Front-end Process ────────────────────────────────────────────┐
│                                                                    │
│  AsyncLLM (vllm/v1/engine/async_llm.py)                          │
│  ├── InputProcessor        tokenizes prompts → EngineCoreRequest  │
│  ├── OutputProcessor       EngineCoreOutput → RequestOutput        │
│  │     └── RequestOutputCollector  per-request asyncio.Queue      │
│  ├── EngineCoreClient      IPC bridge to background process        │
│  └── output_handler task   background loop: ZMQ recv → queues     │
│                                                                    │
│  generate(prompt, params, request_id) → AsyncGenerator            │
│    → add_request() → OutputProcessor.add_request()               │
│    → EngineCoreClient.add_request_async() (ZMQ send)             │
│    → yields from RequestOutputCollector queue                     │
└──────────────────────────────────────────┬───────────────────────┘
                                           │ ZMQ (IPC)
┌─── Background Process ────────────────────▼───────────────────────┐
│                                                                    │
│  EngineCoreProc (vllm/v1/engine/core.py)                         │
│  ├── input_thread       ZMQ ROUTER socket → input_queue           │
│  ├── output_thread      output_queue → ZMQ PULL socket            │
│  └── core_busy_loop()                                             │
│        ├── input_queue.get() → preprocess_add_request()           │
│        ├── scheduler.add_request()                                │
│        └── step():                                                │
│              scheduler.schedule() → SchedulerOutput              │
│              executor.execute_model() → ModelOutput (GPU)         │
│              scheduler.update_from_output() → EngineCoreOutputs  │
└────────────────────────────────────────────────────────────────────┘
```

### Component Roles

| Component | Location | Role |
|-----------|----------|------|
| `AsyncLLM` | `v1/engine/async_llm.py` | Top-level async engine; implements `EngineClient` protocol |
| `InputProcessor` | `v1/engine/input_processor.py` | Tokenizes + validates inputs → `EngineCoreRequest` |
| `OutputProcessor` | `v1/engine/output_processor.py` | Detokenizes + logprobs → `RequestOutput`; owns per-request queues |
| `RequestOutputCollector` | `v1/engine/output_processor.py` | Per-request `asyncio.Queue` for streaming outputs |
| `EngineCoreClient` | `v1/engine/core_client.py` | Abstract IPC layer; async variant uses ZMQ sockets |
| `AsyncMPClient` | `v1/engine/core_client.py` | Concrete IPC client for production (multiprocess + ZMQ) |
| `InprocClient` | `v1/engine/core_client.py` | In-process client for sync `LLMEngine` |
| `EngineCore` | `v1/engine/core.py` | Inner scheduling + GPU execution loop |
| `EngineCoreProc` | `v1/engine/core.py` | Wraps `EngineCore` in a background process with ZMQ |

### Request Lifecycle in AsyncLLM

```
caller: async for output in async_llm.generate(prompt, params, request_id):
  1. InputProcessor.process_inputs(prompt, params) → EngineCoreRequest
  2. OutputProcessor.add_request(request_id, ...)  → creates RequestOutputCollector
  3. AsyncMPClient.add_request_async(request)       → ZMQ send to background process
  4. [background] EngineCoreProc receives, schedules, runs GPU step
  5. [background] output_handler task: ZMQ recv → OutputProcessor.process_outputs()
       → detokenize, logprobs, create RequestOutput
       → RequestOutputCollector.put(output)
  6. generate() loop: yield from RequestOutputCollector queue
```

### Key Insight: Implicit Batching

When multiple `generate()` calls run concurrently (via `asyncio.gather`), they ALL call `add_request_async()` before the EngineCore's next `step()` iteration. The scheduler sees N queued requests and processes them together in a single prefill batch. This is how the standard async server batches naturally — and how our batched server achieves the same without `llm.chat()`.

### How AsyncLLM Compares to Sync LLM

| Aspect | `LLM` (sync) | `AsyncLLM` |
|--------|-------------|------------|
| Engine | `LLMEngine` (in-process) | `EngineCoreProc` (background process, ZMQ) |
| Client | `InprocClient` | `AsyncMPClient` |
| Batching | `_run_engine()` blocking loop | Scheduler sees concurrent requests |
| `batch_submit()` API | No — `_add_request()` called 1-by-1 | No — `generate()` called 1-by-1 concurrently |
| EngineClient protocol | No (shim needed) | **YES — use directly** |
| LoRA | `llm_engine.engine_core.add_lora()` | `async_llm.add_lora()` (async) |

---

## Architecture: BatchedScheduler + AsyncLLM

The HTTP queue from the original plan is unnecessary. The `BatchedScheduler` IS the queue — it accumulates requests in `scheduler.waiting` and gates the GPU step. Each HTTP client's connection is simply held open while the scheduler accumulates. When the gate opens, all accumulated requests fire together and all clients get their responses.

```
HTTP requests (N concurrent clients)
    ↓
Standard OpenAI endpoint: create_chat_completion(request, raw_request)
    → async_llm.generate() → ZMQ → scheduler.waiting.add(request)

EngineCore background process [BatchedScheduler]
    → has_requests() = True → loop spins at ~1ms (not blocking)
    → schedule() → empty SchedulerOutput until:
         elapsed >= max_wait_ms  OR  sum(waiting tokens) >= min_batch_tokens
    → When ready: super().schedule() → GPU fires on full batch
    → Outputs routed back to per-request generate() generators
    → Each create_chat_completion() returns → client gets response
```

### How `BatchedScheduler` Works

**Key files:**
- `vllm/v1/engine/core.py:131` — `Scheduler = vllm_config.scheduler_config.get_scheduler_cls()`
- `vllm/config/scheduler.py:125` — `scheduler_cls: str | type[object] | None = Field(default=None)`
- `vllm/config/scheduler.py:178` — handles a non-string class type directly

**The gate logic in `BatchedScheduler`** (`vllm/entrypoints/openai/batched_scheduler.py`):

```python
class BatchedScheduler(Scheduler):
    # Class-level config (set before creating AsyncLLM)
    max_wait_ms: float = 100.0
    min_batch_tokens: int = 0  # 0 = disabled

    def has_requests(self) -> bool:
        # Always process running (decode) requests and finished-req cleanup
        if self.running or self.has_finished_requests():
            return True
        # Spin (not block) during accumulation so the time gate can fire
        return bool(self.waiting)

    def schedule(self) -> SchedulerOutput:
        # Running requests: decode must never be gated
        if self.running or not self.waiting:
            return super().schedule()
        # Waiting-only: apply the accumulation gate
        if self._batch_start is None:
            self._batch_start = time.monotonic()
        elapsed_ms = (time.monotonic() - self._batch_start) * 1000
        waiting_tokens = sum(r.num_prompt_tokens for r in self.waiting)
        time_gate_open = elapsed_ms >= self.max_wait_ms
        token_gate_open = self.min_batch_tokens > 0 and waiting_tokens >= self.min_batch_tokens
        if time_gate_open or token_gate_open:
            self._batch_start = None  # reset for next window
            return super().schedule()
        return SchedulerOutput.make_empty()  # 1ms spin, no GPU work
```

**Why `has_requests() = True` (spin) instead of False (block):**
- When `has_work()` returns False, the engine calls `input_queue.get(block=True)` — blocks until a NEW request arrives via ZMQ
- During an accumulation window, requests are in `waiting` but the time gate needs to fire even with no new arrivals → must keep the loop spinning
- Cost: ~1ms CPU spin during accumulation (throttled by `time.sleep(0.001)` in `_process_engine_step`)

**Empty `SchedulerOutput` is explicitly safe:** `gpu_model_runner.py:3417` has `if not num_scheduled_tokens: return EMPTY_MODEL_RUNNER_OUTPUT` — no GPU work, no crash, no state corruption.

### Injecting the Scheduler

Pass via standard vLLM CLI — no extra entrypoint needed:

```bash
# Inline JSON
vllm serve --model <model> \
    --scheduler-cls vllm.entrypoints.openai.batched_scheduler.BatchedScheduler \
    --additional-config '{"max_wait_ms": 100, "min_batch_tokens": 500}'

# Or via YAML config file (--config batched.yaml)
# model: <model>
# scheduler-cls: vllm.entrypoints.openai.batched_scheduler.BatchedScheduler
# additional-config:
#   max_wait_ms: 100
#   min_batch_tokens: 500
```

`BatchedScheduler.__init__` reads `max_wait_ms` and `min_batch_tokens` from `vllm_config.additional_config`. It also checks env vars `VLLM_BATCHED_MAX_WAIT_MS` / `VLLM_BATCHED_MIN_BATCH_TOKENS` as a fallback (they are inherited by the EngineCore subprocess on spawn).

**Configuration priority:** `additional_config` → env vars → class-level defaults.

---

## Implementation

### New Files

| File | Lines | Purpose |
|------|-------|---------|
| `vllm/entrypoints/openai/batched_scheduler.py` | ~170 | `BatchedScheduler` class + Prometheus metrics |
| `tests/v1/core/test_batched_scheduler.py` | ~200 | Unit + integration tests |
| `tests/v1/entrypoints/openai/test_batched_server.py` | ~100 | End-to-end entrypoint tests |

No existing files modified.

### Why This vs the Original Sync LLM Plan

Original plan: sync `LLM` + `ThreadPoolExecutor` + `_BatchedEngineClient` shim (~450 lines, 1 file)

New plan: `BatchedScheduler` via `scheduler_cls` + `additional_config` injection:
- No shim, no thread pool, no custom HTTP queue logic, no separate entrypoint
- Standard `vllm serve` — prometheus, watchdog, graceful shutdown all work correctly
- `AsyncLLM` already implements `EngineClient` fully
- LoRA works out of the box via `AsyncLLM.add_lora()`
- Streaming also works (clients wait for batch gate, then stream tokens)

---

## Gotchas

1. **Empty `SchedulerOutput` is explicitly safe**: existing code path, no GPU work, no crash.
2. **`_batch_start` timer resets to `None` after firing**, NOT on each call.
3. **Config via `additional_config`**: read in `__init__` from `vllm_config.additional_config`. Env vars are the subprocess inheritance mechanism and are checked as fallback.
4. **`max_wait_ms=0, min_batch_tokens=0`** is a transparent passthrough — behaves identically to the base `Scheduler`.

---

## Testing

### Run Existing Tests

```bash
# Core scheduler unit tests
pytest tests/v1/core/test_scheduler.py -x

# Engine integration
pytest tests/v1/engine/test_async_llm.py -x
pytest tests/v1/engine/test_engine_core.py -x

# Existing scheduler plugin injection test (precedent)
pytest tests/plugins_tests/test_scheduler_plugins.py -x

# New batched scheduler tests
pytest tests/v1/core/test_batched_scheduler.py -x -v
```

### New Tests (`tests/v1/core/test_batched_scheduler.py`)

| Test | What it verifies |
|------|-----------------|
| `test_token_gate_holds_until_threshold` | Gate stays closed until `min_batch_tokens` met |
| `test_time_gate_fires_after_wait_ms` | Gate opens after `max_wait_ms` elapses |
| `test_running_requests_bypass_gate` | Decode steps are never gated |
| `test_gate_resets_after_batch_fires` | Timer resets so next batch starts fresh |
| `test_default_passthrough` | `max_wait_ms=0, min_batch_tokens=0` fires immediately |
| `test_scheduler_cls_injection_end_to_end` | Full `EngineArgs.scheduler_cls` injection path |

### Test Infrastructure Notes

**Process isolation:** Each test uses `@create_new_process_for_each_test()` (same pattern as `tests/v1/engine/test_engine_core.py`). This forks a subprocess per test, giving full CUDA memory isolation without manual cleanup — critical since each test loads the model and pre-allocates KV cache.

**`enable_chunked_prefill=False`:** The test engine disables chunked prefill. With chunked prefill enabled, the v1 engine also enables *asynchronous scheduling* (step N launches GPU work for step N-1's schedule), which adds extra pipeline stages and breaks the simple `step() → _drain()` pattern used in these tests.

**v1 async pipeline and the `_drain()` pattern:** With async scheduling enabled (the default for single-GPU eager), `step_with_batch_queue` tries to keep the pipeline full. After scheduling and submitting a real batch to the GPU, it checks `not future.done()` to decide whether to return early (`None, True`) and let the next step collect the result. This introduces an N+1 delay in the steady case. However, the check is a **race**: with a small model (e.g. opt-125m) on a fast GPU, the async output thread sometimes completes the D2H copy before the `done()` check runs, causing outputs to be returned in the *same* step that fired the batch rather than the next.

Tests that do `engine.step()` (fire) then `_drain()` (collect) are flaky against this race: when outputs come back from the fire step, they are discarded, and `_drain` returns empty. The fix is to use `_drain()` for both firing and collecting:

```python
# WRONG — flaky if GPU finishes before done() check:
engine.step()            # fires gate (sometimes also returns outputs — discarded!)
outputs = _drain(engine) # sometimes gets nothing

# CORRECT — _drain's first step fires the gate; subsequent steps collect:
outputs = _drain(engine) # works regardless of whether pipeline delays output
```

`_drain()` calls `engine.step()` in a loop and returns on the first non-empty result. If the gate fires and outputs arrive immediately (fast GPU), `_drain` returns from iteration 0. If outputs are pipelined to the next step, `_drain` catches them in iteration 1. Either way the test is correct.

**Timer ordering:** `BatchedScheduler._batch_start` is set on the *first* `schedule()` call, not when the request is added. Tests that sleep to trigger the time gate must call `step()` first (to start the timer and assert gate-is-closed), then sleep, then `_drain()` to fire and collect:
```python
engine.step()       # starts _batch_start; asserts gate is closed
time.sleep(0.060)
outputs = _drain(engine)  # fires (elapsed >= max_wait_ms) and collects
```

---

## Manual Smoke Test

```bash
# Start server with batched scheduler
vllm serve facebook/opt-125m \
    --scheduler-cls vllm.entrypoints.openai.batched_scheduler.BatchedScheduler \
    --additional-config '{"max_wait_ms": 200, "min_batch_tokens": 100}' \
    --port 8001 --enforce-eager

# Fire 10 concurrent requests — all should batch together
for i in $(seq 10); do
  curl -s -X POST http://localhost:8001/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"Say hello"}],"model":"facebook/opt-125m"}' &
done; wait

# Check batch metrics
curl -s http://localhost:8001/metrics | grep vllm_batched
```

---

## Prometheus Metrics

Three histograms are emitted from `BatchedScheduler` when the gate fires, using the standard `prometheus_client` library already present in vLLM.

| Metric | Type | Per | Description |
|--------|------|-----|-------------|
| `vllm_batched_batch_delay_ms` | Histogram | batch | ms from first request seen by `schedule()` to gate fire. Buckets derived from `max_wait_ms`. |
| `vllm_batched_batch_size_requests` | Histogram | batch | Number of requests in each fired batch. Fixed power-of-two buckets. |
| `vllm_batched_batch_size_tokens` | Histogram | batch | Total prompt tokens in each fired batch. Buckets derived from `min_batch_tokens`. |

### Design Rationale

**Per-request queue wait** is already captured by the built-in `vllm:request_queue_time_seconds` (`queued_ts → scheduled_ts`, monotonic timestamps inside the base `Scheduler`). Since `BatchedScheduler` doesn't override those event-recording paths, the accumulation window delay is included automatically. No duplicate needed.

**`vllm_batched_batch_delay_ms`** is the max-wait across a batch (one point per batch fire). It tells you whether the time gate or token gate fired early, and by how much.

**Bucket ranges reflect the actual config**: for `max_wait_ms=50`, delay buckets span 5 ms–100 ms; for `min_batch_tokens=1000`, token buckets span 100–2000. Resolution stays meaningful relative to the operator's chosen thresholds.

### Implementation

Histograms are lazily registered as class-level attributes on first
`__init__`, so bucket ranges can be derived from the actual config.
No changes to any existing vLLM file.

```python
# _delay_buckets(): fractions of max_wait_ms
# e.g. max_wait_ms=50 → [5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 75.0, 100.0]
fractions = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]

# _token_buckets(): fractions of min_batch_tokens (or fixed defaults when 0)
# e.g. min_batch_tokens=1000 → [100, 250, 500, 750, 1000, 1250, 1500, 2000]
fractions = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

# batch_size_requests: fixed power-of-two buckets
buckets = [1, 2, 4, 8, 16, 32, 64, 128, 256]
```

Metrics are emitted from `_emit_metrics()`, called in `schedule()` on the
gate-fire branch.  `elapsed_ms` and `waiting_tokens` are already computed
at that point — no extra work required.

### Non-invasiveness

- No existing vLLM files are modified
- `prometheus_client` is already a vLLM dependency; import failure is
  silently tolerated (metrics skipped, scheduler still works)
- Histograms are only registered when `BatchedScheduler` is imported
- The `/metrics` endpoint is served automatically by `serve_http()` via
  vLLM's existing Prometheus middleware

---

## Future Improvement

**Sub-millisecond time gate precision**: The current spin approach adds up to 1ms latency to batch release. For higher precision, inject `input_queue` into `BatchedScheduler` from `EngineCore.__init__` (~3 lines in `core.py`) and use `threading.Timer` to poke the queue when `max_wait_ms` expires — eliminating the spin entirely. Acceptable future PR.
