"""Install PreViC into a video VLM by importing this module.

`import previc.preload` monkey-patches the base VLM's projector-output
token-compression hook so that visual tokens are compressed by PreViC before the
LLM. PreViC is pre-LLM and prompt-agnostic: it changes only the LLM *input*, so
the transformer forward pass is untouched.

The base VLM must expose a hook `fn(image_feature[F,T,D], num_tokens_per_frame,
merging_ratio) -> compressed[m,D]` at the projector output. For the paper's
Set `HOOK_MODULE` / `HOOK_ATTR` (or the `PREVIC_HOOK_MODULE` / `PREVIC_HOOK_ATTR`
environment variables) to the function your host calls with the projector output.

Configuration is read from environment variables at import time.

  keep ratio α : PREVIC_KEEP_RATIO    (if unset, the host hook's ratio is used)
  anchors a    : PREVIC_ANCHORS        (default 2)
  share β      : PREVIC_ANCHOR_SHARE   (default 0.3)
  k-NN         : PREVIC_KNN            (default 4)
  ridge λ      : PREVIC_LAMBDA         (default "adaptive")
  full-D / JL  : PREVIC_USE_JL         (JL off, i.e. full-D, by default)
                 PREVIC_JL_RATIO       (<1.0 enables the JL ablation)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from previc.core import PreViCConfig, previc  # noqa: E402

# Edit these to match your codebase's projector-output hook.
# The host function that receives the post-projector visual tokens. Override via
# PREVIC_HOOK_MODULE / PREVIC_HOOK_ATTR without editing this file.
HOOK_MODULE = os.environ.get("PREVIC_HOOK_MODULE", "")
HOOK_ATTR = os.environ.get("PREVIC_HOOK_ATTR", "")


def _env(*names, default=None):
    for n in names:
        if n in os.environ:
            return os.environ[n]
    return default


def config_from_env() -> PreViCConfig:
    jl_ratio = _env("PREVIC_JL_RATIO")
    use_jl_env = _env("PREVIC_USE_JL")
    if use_jl_env is not None:
        use_jl = use_jl_env == "1"
    elif jl_ratio is not None:
        use_jl = float(jl_ratio) < 1.0            # ratio 1.0 means full-D, i.e. JL off
    else:
        use_jl = False
    return PreViCConfig(
        a=int(_env("PREVIC_ANCHORS", default="2")),
        beta=float(_env("PREVIC_ANCHOR_SHARE", default="0.3")),
        k_nn=int(_env("PREVIC_KNN", default="4")),
        lam=_env("PREVIC_LAMBDA", default="adaptive"),
        use_jl=use_jl,
        jl_ratio=float(jl_ratio) if (use_jl and jl_ratio is not None) else 0.25,
        # ablation toggles (default = full method); used to reproduce the appendix studies
        r1=_env("PREVIC_R1_RANDOM", default="0") != "1",
        r2=_env("PREVIC_R2_RANDOM", default="0") != "1",
        r3=_env("PREVIC_R3_EQUAL", default="0") != "1",
        r2_signal=_env("PREVIC_R2_SIGNAL", default="knn_residual"),
        anchor_overflow=_env("PREVIC_ANCHOR_OVERFLOW", default="mean_dist"),
        norm_variant=_env("PREVIC_NORM_VARIANT", default="raw"),
    )


CONFIG = config_from_env()
_KEEP = _env("PREVIC_KEEP_RATIO")
_KEEP = float(_KEEP) if _KEEP is not None else None


def previc_hook(image_feature: torch.Tensor,
                num_tokens_per_frame: int = 196,
                merging_ratio: float = 0.5) -> torch.Tensor:
    keep = _KEEP if _KEEP is not None else float(merging_ratio)
    return previc(image_feature, keep, num_tokens_per_frame, CONFIG)


def install() -> None:
    import importlib
    if not HOOK_MODULE or not HOOK_ATTR:
        raise RuntimeError(
            "PreViC hook target is not set. Point PREVIC_HOOK_MODULE and PREVIC_HOOK_ATTR "
            "at the host function that receives the post-projector visual tokens, or set "
            "HOOK_MODULE / HOOK_ATTR in this file. See integration/ for an example.")
    mod = importlib.import_module(HOOK_MODULE)
    setattr(mod, HOOK_ATTR, previc_hook)
    print(f"[PreViC] installed at {HOOK_MODULE}.{HOOK_ATTR} "
          f"(a={CONFIG.a}, beta={CONFIG.beta}, k_nn={CONFIG.k_nn}, "
          f"lambda={CONFIG.lam}, full_D={not CONFIG.use_jl})",
          file=sys.stderr)


install()
