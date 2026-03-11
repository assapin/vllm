# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batched-inference OpenAI-compatible server.

Nearly identical to api_server.py, with two extra CLI arguments:

  --max-wait-ms         Max milliseconds to accumulate waiting requests before
                        firing the next prefill batch (default: 100).
  --min-batch-tokens    Minimum total prompt tokens to trigger the batch early.
                        0 (default) disables this threshold.

The batching is implemented entirely inside BatchedScheduler, which is
injected via SchedulerConfig.scheduler_cls before the engine is created.
No other vLLM code is modified.

Start::

    python -m vllm.entrypoints.openai.batched_server \\
        --model facebook/opt-125m \\
        --max-wait-ms 200 --min-batch-tokens 100 --port 8001
"""

import uvloop

from vllm.entrypoints.openai.api_server import (
    build_app,
    init_app_state,
    run_server_worker,
    setup_server,
)
from vllm.entrypoints.openai.batched_scheduler import BatchedScheduler
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.entrypoints.utils import cli_env_setup
from vllm.utils.argparse_utils import FlexibleArgumentParser


def make_batched_arg_parser() -> FlexibleArgumentParser:
    parser = FlexibleArgumentParser(
        description="vLLM Batched Inference Server — OpenAI-compatible API "
        "server with explicit prefill-batch accumulation."
    )
    # Batching-specific args (prepended so they appear first in --help)
    parser.add_argument(
        "--max-wait-ms",
        type=float,
        default=100.0,
        help="Max milliseconds to accumulate waiting requests before firing "
        "the prefill batch (default: 100.0).",
    )
    parser.add_argument(
        "--min-batch-tokens",
        type=int,
        default=0,
        help="Fire the batch early when the total prompt-token count of "
        "waiting requests reaches this value. 0 disables this threshold "
        "(default: 0).",
    )
    # All standard vLLM server + engine args (--model, --host, --port,
    # --tensor-parallel-size, --chat-template, --api-key, etc.)
    parser = make_arg_parser(parser)
    return parser


async def run_server(args, **uvicorn_kwargs) -> None:
    """Run the batched inference server."""
    import os

    # Set env vars BEFORE the EngineCore subprocess is spawned so they are
    # inherited by the child process even when multiprocessing spawn is used.
    # BatchedScheduler.__init__ reads these env vars to set instance-level
    # attributes, overriding the class-level defaults.
    os.environ["VLLM_BATCHED_MAX_WAIT_MS"] = str(args.max_wait_ms)
    os.environ["VLLM_BATCHED_MIN_BATCH_TOKENS"] = str(args.min_batch_tokens)

    # Also set class-level for any in-process / fork-based instantiation.
    BatchedScheduler.max_wait_ms = args.max_wait_ms
    BatchedScheduler.min_batch_tokens = args.min_batch_tokens
    args.scheduler_cls = BatchedScheduler

    listen_address, sock = setup_server(args)
    await run_server_worker(listen_address, sock, args, **uvicorn_kwargs)


if __name__ == "__main__":
    cli_env_setup()
    parser = make_batched_arg_parser()
    args = parser.parse_args()
    validate_parsed_serve_args(args)

    uvloop.run(run_server(args))
