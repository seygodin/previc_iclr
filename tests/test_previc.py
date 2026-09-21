"""Self-contained checks for the PreViC selector.

These need only torch and numpy. No VLM, no checkpoint, no dataset. They exercise
the selector on synthetic token tensors and assert the invariants the paper states.

    python -m pytest tests/ -q        # or:  python tests/test_previc.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from previc import PreViCConfig, previc  # noqa: E402

CFG = PreViCConfig()               # paper defaults: a=2, beta=0.3, k_nn=4, full-D
F, T, D = 8, 16, 32                # frames, tokens per frame, hidden size


def _video(seed: int = 0, moving: bool = True) -> torch.Tensor:
    """A [F*T, D] token sequence with a near-static background and, optionally, drift.

    Every token carries a unique tag in its last channel so that a selection can be
    recovered from the returned tokens without the selector exposing its mask.
    """
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(T, D, generator=g)
    frames = []
    for t in range(F):
        f = base.clone()
        if moving:
            f[: T // 4] += 0.5 * t          # a quarter of the tokens change over time
        f[:, -1] = torch.arange(T, dtype=f.dtype) + t * T   # unique tag
        frames.append(f)
    return torch.stack(frames).reshape(F * T, D)


def _kept_mask(V: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Recover a [F, T] boolean mask from the returned tokens using the unique tag."""
    tags = out[:, -1].round().long()
    mask = torch.zeros(F * T, dtype=torch.bool)
    mask[tags] = True
    return mask.reshape(F, T)


def test_budget_target_and_floor():
    """The discrete target is m0 = min(FT, max(4F, round(a*F*T))).

    The realized count can exceed m0 because the per-frame floor is applied after the
    budget is spent. With a small T the floor dominates, so we only require that the
    realized count is at least the target and never exceeds the input.
    """
    V = _video()
    for alpha in (0.1, 0.3, 0.5, 0.7, 0.9):
        out = previc(V, alpha, T, CFG)
        m0 = min(F * T, max(CFG.min_keep_per_frame * F, round(F * T * alpha)))
        assert m0 <= out.shape[0] <= F * T, (alpha, out.shape[0], m0)
        assert out.shape[1] == D


def test_realized_ratio_tracks_request_at_realistic_T():
    """At the token count a real VLM produces, the floor is negligible.

    The paper reports realized retention within a fraction of a point of the request.
    With T = 196 tokens per frame the 4-token floor cannot move the ratio much, so the
    realized ratio should sit close to the requested one.
    """
    T_real, F_real = 196, 8
    g = torch.Generator().manual_seed(1)
    V = torch.randn(F_real * T_real, D, generator=g)
    for alpha in (0.1, 0.3, 0.5, 0.7):
        realized = previc(V, alpha, T_real, CFG).shape[0] / (F_real * T_real)
        assert abs(realized - alpha) < 0.01, (alpha, realized)


def test_per_frame_floor():
    """No frame is emptied, even at the sparsest budget."""
    V = _video()
    keep = _kept_mask(V, previc(V, 0.05, T, CFG))
    per_frame = keep.sum(dim=1)
    assert int(per_frame.min()) >= CFG.min_keep_per_frame, per_frame.tolist()


def test_temporal_order_preserved():
    """Selected tokens come back in frame order, and are a subset of the input."""
    V = _video()
    out = previc(V, 0.3, T, CFG)
    idx = out[:, -1].round().long()
    assert torch.equal(idx, torch.sort(idx).values), "tokens must come back in frame order"
    assert torch.equal(out, V[idx]), "output must be a subset of the input, unmodified"


def test_deterministic():
    """Same input, same output. No hidden RNG in the default path."""
    V = _video()
    a = previc(V, 0.3, T, CFG)
    b = previc(V, 0.3, T, CFG)
    assert torch.equal(a, b)


def test_prefers_unpredictable_tokens():
    """R2 keeps the drifting tokens over the static background it can predict."""
    V = _video(moving=True)
    keep = _kept_mask(V, previc(V, 0.3, T, CFG))
    moving, static = keep[:, : T // 4].float().mean(), keep[:, T // 4 :].float().mean()
    assert moving > static, (float(moving), float(static))


def test_passthrough_at_full_ratio():
    V = _video()
    assert torch.equal(previc(V, 1.0, T, CFG), V)


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except AssertionError as exc:
                fails += 1
                print(f"  FAIL  {name}: {exc}")
    raise SystemExit(fails)
