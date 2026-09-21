# PreViC — Predictive Visual Compression for Video VLMs

Anonymous code release accompanying the paper submission
*"PreViC: Predictive Visual Compression for Video Vision-Language Models."*

PreViC is a **training-free, prompt-agnostic, pre-LLM** visual-token compressor
for video VLMs, inspired by predictive coding in video codecs. It runs on the
post-projector visual tokens (before the LLM) and keeps only the tokens that a
compact **anchor memory cannot predict**, so the compressed representation is
query-independent and can be reused across questions about the same video.

Because PreViC changes only the **LLM input** (fewer visual tokens into an
otherwise unmodified transformer), it needs no change to the model forward pass
and drops into standard inference stacks.

## Method (paper Sec. 3)

| Component | What it does | Codec analog |
|---|---|---|
| **R1 — Anchor-Frame Pool** | Select `a` frames with highest temporal novelty `ν_t = mean_p ‖v_{t,p} − v_{t−1,p}‖²` (frame 0 always included); their tokens form the reference memory. | I-frame / reference buffer |
| **R2 — Predictive Residual Scoring** | For each non-anchor token, retrieve its `k` nearest anchor tokens (cosine) and reconstruct it by ridge least squares; keep tokens with large residual `ρ(x)=‖x−x̂‖₂`. | Motion-compensated P-frame |
| **R3 — Hierarchical Rate Allocation** | Split the budget `m=min{FT, max[4F, round(αFT)]}` into an anchor share (`β·m`, default 0.3, capped at the anchor-pool size) and a residual share; each frame is then topped up to at least 4 tokens. | GOP rate allocation |

The default operates in the **full post-projector feature space** (no channel
selection or projection), matching the paper's headline setting.

## Repository layout

```
previc/
  core.py                     # the PreViC algorithm (R1/R2/R3) — entry point previc(...)
  preload.py                  # install PreViC by patching the VLM projector-output hook
  timing.py                   # optional stage-timing registry for the profiler
  __init__.py
integration/
  llava_onevision_patch.py    # thin example wrapper around previc.preload
tests/
  test_previc.py              # self-contained checks: budget, per-frame floor, ordering,
                              # determinism, and that R2 keeps the unpredictable tokens
requirements.txt
LICENSE
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # torch, numpy
```

## Quick start (standalone)

```python
import torch
from previc import previc, PreViCConfig

# post-projector visual tokens: F frames × T tokens/frame × D dims
V = torch.randn(32, 196, 896)          # e.g. LLaVA-OneVision-0.5B
kept = previc(V, keep_ratio=0.3)       # keep 30% of tokens
print(kept.shape)                      # torch.Size([1882, 896]), in frame order

# override hyperparameters / run an ablation
cfg = PreViCConfig(a=2, beta=0.3, k_nn=4, lam="adaptive")   # paper defaults
kept = previc(V, keep_ratio=0.5, cfg=cfg)
```

## Attach to a video VLM

`import previc.preload` monkey-patches the base VLM's projector-output token hook
so visual tokens are compressed before the LLM. Set `HOOK_MODULE`/`HOOK_ATTR` in
`previc/preload.py` to your codebase's insertion point (the only requirement is
access to the post-projector `[F, T, D]` token tensor). Configuration is read
from environment variables (see below). Importing the module is what installs
the hook, so point it at your host's function with two environment variables and
import it:

```bash
export PREVIC_HOOK_MODULE=your_vlm.model.arch
export PREVIC_HOOK_ATTR=the_projector_output_hook
PREVIC_KEEP_RATIO=0.3 python -c "import previc.preload"
```

`integration/llava_onevision_patch.py` is a thin example wrapper.

## Hyperparameters

| Name | `PreViCConfig` field | Env var | Default |
|---|---|---|---|
| Retain ratio α | `keep_ratio` (arg) | `PREVIC_KEEP_RATIO` | — |
| Anchor frames `a` | `a` | `PREVIC_ANCHORS` | 2 |
| Anchor budget share β | `beta` | `PREVIC_ANCHOR_SHARE` | 0.3 |
| k-NN size | `k_nn` | `PREVIC_KNN` | 4 |
| Ridge λ | `lam` | `PREVIC_LAMBDA` | `adaptive` |
| Min tokens / frame | `min_keep_per_frame` | — | 4 |
| JL projection (ablation) | `use_jl` | `PREVIC_USE_JL` | off (full-D) |

All configuration is read from `PREVIC_*` environment variables by
`previc/preload.py` at import time.

Ablation switches in `PreViCConfig` (`r1`/`r2`/`r3`, `r2_signal`,
`anchor_overflow`, `norm_variant`) reproduce the paper's component and
sensitivity studies.

## Reproduce the paper

- **Algorithm:** `previc/core.py` — `previc(...)` is the single entry point.
- **Checks:** `python tests/test_previc.py` (or `python -m pytest tests/ -q`). These need
  only torch and numpy, no checkpoint and no dataset.

## Notes

- Evaluation uses **greedy decoding** on a **fixed** frame sampling (32 frames)
  and fixed benchmark subsets, so the reported numbers are exactly reproducible.
- This release is fully anonymized for review. It contains the algorithm, the
  plug-in and the self-contained checks. It does not ship evaluation harnesses,
  model checkpoints or benchmark videos, which come from their own repositories.
