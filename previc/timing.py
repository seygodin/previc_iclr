"""Optional stage-timing registry for the efficiency profiler.

`reset_timing()` clears the registry; `record(stage, ms)` appends a sample;
`summary()` returns `{stage: [durations_ms, ...]}`.

Fine intra-PreViC sub-stage timings (R1 novelty / k-NN retrieval / ridge solve)
are populated only when running the instrumented core. When they are absent,
`summary()` is empty and callers fall back to the coarse pipeline stages (vision
encode, PreViC compression total, LLM prefill, decode, end-to-end) that they
time with their own CUDA events — these coarse stages reproduce the paper's main
efficiency numbers (compression overhead as a fraction of E2E, prefill savings).
"""
from __future__ import annotations

from contextlib import contextmanager

_STAGE_TIMES: dict[str, list[float]] = {}


def reset_timing() -> None:
    _STAGE_TIMES.clear()


def record(stage: str, ms: float) -> None:
    _STAGE_TIMES.setdefault(stage, []).append(float(ms))


def summary() -> dict[str, list[float]]:
    return {k: list(v) for k, v in _STAGE_TIMES.items()}


@contextmanager
def timed(stage: str):
    """CPU wall-clock timer context manager (use CUDA events for GPU-accurate timing)."""
    import time
    t0 = time.perf_counter()
    try:
        yield
    finally:
        record(stage, (time.perf_counter() - t0) * 1e3)
