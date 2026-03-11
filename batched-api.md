# Plan: vLLM Batched Inference Server

## Context

Add a standalone OpenAI-compatible FastAPI server (`batched_server.py`) that accumulates requests up to a time or token budget, then fires them all at once through the GPU in a single prefill batch. The user starts the server in this mode *instead of* the normal async server.

The implementation uses **`AsyncLLM`** (the standard v1 async engine) with a custom **`BatchedScheduler`** injected via `SchedulerConfig.scheduler_cls` — a first-class vLLM injection point. No core files are modified.

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

```python
# In run_server(), BEFORE creating AsyncLLM:
BatchedScheduler.max_wait_ms = args.max_wait_ms
BatchedScheduler.min_batch_tokens = args.min_batch_tokens
args.scheduler_cls = BatchedScheduler   # flows into VllmConfig.scheduler_config
```

The background process imports `BatchedScheduler` from the module, picking up the class-level values set before the process is spawned.

---

## Implementation

### New Files

| File | Lines | Purpose |
|------|-------|---------|
| `vllm/entrypoints/openai/batched_scheduler.py` | ~60 | `BatchedScheduler` class |
| `vllm/entrypoints/openai/batched_server.py` | ~80 | Entry point with `--max-wait-ms`, `--min-batch-tokens` |
| `tests/v1/core/test_batched_scheduler.py` | ~200 | Unit + integration tests |

No existing files modified.

### Why This vs the Original Sync LLM Plan

Original plan: sync `LLM` + `ThreadPoolExecutor` + `_BatchedEngineClient` shim (~450 lines, 1 file)

New plan: `BatchedScheduler` via `scheduler_cls` injection + thin `batched_server.py` entry point:
- No shim, no thread pool, no custom HTTP queue logic
- `AsyncLLM` already implements `EngineClient` fully
- `serve_http()` gives watchdog + graceful shutdown for free
- LoRA works out of the box via `AsyncLLM.add_lora()`
- Streaming also works (clients wait for batch gate, then stream tokens)

---

## Gotchas

1. **Empty `SchedulerOutput` is explicitly safe**: existing code path, no GPU work, no crash.
2. **`_batch_start` timer resets to `None` after firing**, NOT on each call.
3. **Class-level config on `BatchedScheduler`**: set BEFORE `AsyncLLM` is created. The background process imports the class from the module, picking up the set values.
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

---

## Manual Smoke Test

```bash
# Start batched server
python -m vllm.entrypoints.openai.batched_server \
    --model facebook/opt-125m \
    --max-wait-ms 200 --min-batch-tokens 100 --port 8001 --enforce-eager

# Fire 10 concurrent requests — all should batch together
for i in $(seq 10); do
  curl -s -X POST http://localhost:8001/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"Say hello"}],"model":"facebook/opt-125m"}' &
done; wait

# Verify health and models endpoints
curl http://localhost:8001/health
curl http://localhost:8001/v1/models
```

---

## Future Improvement

**Sub-millisecond time gate precision**: The current spin approach adds up to 1ms latency to batch release. For higher precision, inject `input_queue` into `BatchedScheduler` from `EngineCore.__init__` (~3 lines in `core.py`) and use `threading.Timer` to poke the queue when `max_wait_ms` expires — eliminating the spin entirely. Acceptable future PR.
