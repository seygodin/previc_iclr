"""PreViC — Predictive Visual Compression for video VLMs.

Training-free, prompt-agnostic, pre-LLM visual-token compression, inspired by
predictive coding in video codecs. Operates on the post-projector visual tokens
(before the LLM) and returns a compressed token sequence in frame order.

Three components (see paper Sec. 3):

  R1 — Anchor-Frame Pool (I-frame / reference-frame analog):
       Select `a` frames with highest temporal novelty
           nu_t = (1/T) * sum_p || v_{t,p} - v_{t-1,p} ||^2
       (frame 0 is always included as the initial reference). All tokens of the
       anchor frames form the reference memory (pool).

  R2 — Predictive Residual Scoring (motion-compensated P-frame analog):
       For each non-anchor token x, retrieve its k nearest anchor tokens (cosine
       similarity) and reconstruct x by ridge-regularized least squares over that
       local basis. The residual rho(x) = || x - x_hat ||_2 is the importance
       score: a large residual means x carries information the memory cannot
       predict (keep it); a small residual means x is redundant (drop it).

  R3 — Hierarchical Rate Allocation (GOP rate allocation analog):
       Split the global keep budget m = min(F*T, max(4*F, round(alpha*F*T))) into an anchor share
       (beta * m, default 0.3) and a residual share. Anchor tokens are scored by
       distance to the anchor-pool mean; non-anchor tokens by R2 residual.

The method operates in the full post-projector feature space (no channel
selection or projection) by default, matching the paper's headline setting.
An optional Johnson-Lindenstrauss projection is provided as an ablation only.

The single entry point is `previc(...)`, a drop-in for a mm-projector-output
token-compression hook. See `integration/llava_onevision_patch.py` for how to
attach it to a video VLM.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
@dataclass
class PreViCConfig:
    """PreViC hyperparameters. Defaults reproduce the paper's headline setting."""

    a: int = 2                       # R1: number of anchor frames (frame 0 always included)
    beta: float = 0.3               # R3: anchor budget share
    k_nn: int = 4                   # R2: neighbours for local reconstruction
    lam: float | str = "adaptive"   # R2: ridge lambda; "adaptive" -> max(eps, eps*tr(BB^T)/k)
    eps: float = 1e-3               # R2: base ridge scale for the adaptive rule
    min_keep_per_frame: int = 4     # floor on kept tokens per frame

    # --- Ablation switches (all default to the full method) ---------------- #
    r1: bool = True                 # R1 on; if False, anchor frames chosen randomly
    r2: bool = True                 # R2 on; if False, non-anchor tokens scored randomly
    r3: bool = True                 # R3 on; if False, equal split + random anchor overflow
    r2_signal: str = "knn_residual" # {knn_residual, pool_mean, nearest_anchor, cosine_anchor, frame_pair_cos}
    anchor_overflow: str = "mean_dist"  # {mean_dist, k_center, farthest_point}
    norm_variant: str = "raw"       # {raw, mean_center_video, channel_standardize, l2_normalize}

    # --- Optional JL projection (ablation only; paper default is full-D) ---- #
    use_jl: bool = False
    jl_ratio: float = 0.25
    jl_min: int = 256
    seed: int = 42                  # RNG seed (JL / random ablations) for reproducibility


# --------------------------------------------------------------------------- #
# R1 — anchor-frame selection                                                 #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def select_anchor_frames(V: torch.Tensor, cfg: PreViCConfig) -> torch.Tensor:
    """Return anchor-frame indices (sorted) by temporal novelty. Frame 0 always in.

    V: [F, T, D] visual tokens.
    """
    F_, T, D = V.shape
    if F_ == 1:
        return torch.tensor([0], device=V.device)

    a = max(1, min(cfg.a, F_))

    if not cfg.r1:  # ablation: random anchor frames (frame 0 still forced in)
        gen = torch.Generator(device=V.device).manual_seed(cfg.seed + 7)
        cand = torch.randperm(F_ - 1, generator=gen, device=V.device) + 1
        idx = torch.cat([torch.tensor([0], device=V.device), cand[: a - 1]]) if a > 1 \
            else torch.tensor([0], device=V.device)
        return torch.sort(idx).values

    # nu_t = mean_p || V_t - V_{t-1} ||^2
    diff = (V[1:] - V[:-1]).pow(2).sum(dim=-1).mean(dim=-1)   # [F-1]
    novelty = torch.empty(F_, device=V.device, dtype=diff.dtype)
    novelty[0] = float("inf")                                 # force frame 0
    novelty[1:] = diff
    top_idx = torch.topk(novelty, a, sorted=False).indices
    return torch.sort(top_idx).values


# --------------------------------------------------------------------------- #
# R2 — predictive residual scoring                                            #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def knn_residual_norm(targets: torch.Tensor, pool: torch.Tensor, cfg: PreViCConfig) -> torch.Tensor:
    """Ridge-reconstruction residual norm of each target token from its k-NN in the pool.

    For each target x: find k nearest anchor tokens (cosine sim), solve
    c* = (B B^T + lambda I)^-1 B x, and return || x - B^T c* ||_2.

    targets: [N_t, D']   pool: [N_p, D']  ->  residuals: [N_t]
    """
    N_p = pool.shape[0]
    if N_p == 0:
        return targets.norm(dim=-1)
    k = min(cfg.k_nn, N_p)

    # cosine k-NN
    sims = F.normalize(targets, dim=-1) @ F.normalize(pool, dim=-1).t()   # [N_t, N_p]
    knn_idx = torch.topk(sims, k, dim=-1).indices                        # [N_t, k]
    basis = pool[knn_idx].float()                                        # [N_t, k, D']
    x = targets.float()

    # closed-form ridge least squares: c = (B B^T + lambda I)^-1 B x
    BBt = torch.matmul(basis, basis.transpose(-1, -2))                   # [N_t, k, k]
    Bx = (basis * x.unsqueeze(1)).sum(dim=-1)                            # [N_t, k]
    eye = torch.eye(k, device=basis.device, dtype=basis.dtype).unsqueeze(0)
    if cfg.lam == "adaptive" or cfg.lam == "":
        diag = torch.diagonal(BBt, dim1=-2, dim2=-1).mean(-1, keepdim=True).unsqueeze(-1)
        lam = (cfg.eps * diag).clamp_min(cfg.eps)
    else:
        lam = float(cfg.lam)
    BBt = BBt + lam * eye
    try:
        c = torch.linalg.solve(BBt, Bx.unsqueeze(-1)).squeeze(-1)
    except (torch._C._LinAlgError, RuntimeError):
        c = (torch.linalg.pinv(BBt) @ Bx.unsqueeze(-1)).squeeze(-1)
    x_hat = (c.unsqueeze(-1) * basis).sum(dim=1)                         # [N_t, D']
    return (x - x_hat).norm(dim=-1).to(targets.dtype)


@torch.no_grad()
def _residual_score(targets, pool, V_f, nonanchor_frames, F_, T, cfg):
    """Dispatch R2 scoring for non-anchor tokens (default: knn_residual)."""
    if not cfg.r2:  # ablation: random scores
        gen = torch.Generator(device=targets.device).manual_seed(cfg.seed + 13)
        return torch.rand(targets.shape[0], generator=gen, device=targets.device)
    s = cfg.r2_signal
    if s == "pool_mean":                    # distance to pool centroid
        return (targets - pool.mean(0, keepdim=True)).norm(dim=-1)
    if s == "nearest_anchor":               # 1 - max cosine to any anchor
        sims = F.normalize(targets, dim=-1) @ F.normalize(pool, dim=-1).t()
        return (1.0 - sims.max(dim=-1).values).clamp_min(0.0)
    if s == "cosine_anchor":                # 1 - cosine to pool mean
        cos = (F.normalize(targets, dim=-1) *
               F.normalize(pool.mean(0, keepdim=True), dim=-1)).sum(-1)
        return (1.0 - cos).clamp_min(0.0)
    if s == "frame_pair_cos":               # 1 - cos(V_t, V_{t-1}) per token
        Vn = F.normalize(V_f, dim=-1)
        pair = torch.empty(F_, T, device=V_f.device, dtype=V_f.dtype)
        pair[0] = 0.0
        if F_ > 1:
            pair[1:] = (Vn[1:] * Vn[:-1]).sum(-1)
        return (1.0 - pair[nonanchor_frames]).clamp_min(0.0).reshape(-1)
    return knn_residual_norm(targets, pool, cfg)   # default PreViC R2


# --------------------------------------------------------------------------- #
# Optional Johnson-Lindenstrauss projection (ablation only)                   #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _jl_project(V: torch.Tensor, d_prime: int, seed: int) -> torch.Tensor:
    *prefix, D = V.shape
    if d_prime >= D:
        return V
    gen = torch.Generator(device=V.device).manual_seed(seed)
    P = torch.randn(D, d_prime, generator=gen, device=V.device, dtype=V.dtype) / (d_prime ** 0.5)
    return V.reshape(-1, D).matmul(P).reshape(*prefix, d_prime)


# --------------------------------------------------------------------------- #
# R3 — anchor-pool overflow scoring                                           #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _select_anchor_tokens(pool: torch.Tensor, k_anchor: int, cfg: PreViCConfig) -> torch.Tensor:
    """Pick k_anchor anchor tokens (flat pool indices) when the pool exceeds the budget."""
    N_p = pool.shape[0]
    if k_anchor >= N_p:
        return torch.arange(N_p, device=pool.device)
    if not cfg.r3:  # ablation: random overflow
        gen = torch.Generator(device=pool.device).manual_seed(cfg.seed + 11)
        return torch.topk(torch.rand(N_p, generator=gen, device=pool.device), k_anchor).indices
    if cfg.anchor_overflow in ("k_center", "farthest_point"):
        if cfg.anchor_overflow == "farthest_point":
            gen = torch.Generator(device=pool.device).manual_seed(cfg.seed + 17)
            first = int(torch.randint(0, N_p, (1,), generator=gen, device=pool.device))
        else:
            first = int(torch.argmax((pool - pool.mean(0, keepdim=True)).norm(dim=-1)))
        kept = [first]
        min_d = (pool - pool[first].unsqueeze(0)).norm(dim=-1)
        for _ in range(k_anchor - 1):
            nxt = int(torch.argmax(min_d))
            kept.append(nxt)
            min_d = torch.minimum(min_d, (pool - pool[nxt].unsqueeze(0)).norm(dim=-1))
            min_d[nxt] = -1.0
        return torch.tensor(kept, device=pool.device, dtype=torch.long)
    # default: distance to anchor-pool mean
    scores = (pool - pool.mean(0, keepdim=True)).norm(dim=-1)
    return torch.topk(scores, k_anchor, sorted=False).indices


# --------------------------------------------------------------------------- #
# Main entry point                                                            #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def previc(
    visual_tokens: torch.Tensor,
    keep_ratio: float,
    num_tokens_per_frame: int = 196,
    cfg: PreViCConfig | None = None,
) -> torch.Tensor:
    """Compress post-projector visual tokens with PreViC.

    Args:
        visual_tokens: [F, T, D] or flat [F*T, D] post-projector tokens.
        keep_ratio:    alpha in (0, 1]; fraction of tokens to keep.
        num_tokens_per_frame: T, used only when `visual_tokens` is flat.
        cfg: PreViCConfig (defaults reproduce the paper's headline setting).

    Returns:
        [m, D] compressed tokens in original frame order, where the pre-top-up budget is
        m = min(F*T, max(4*F, round(alpha*F*T))) and the per-frame floor may add a few more.
    """
    cfg = cfg or PreViCConfig()

    # normalise input to [F, T, D]
    if visual_tokens.dim() == 2:
        N, D = visual_tokens.shape
        T = num_tokens_per_frame
        if T <= 0 or N % T != 0:
            return visual_tokens
        V = visual_tokens.reshape(N // T, T, D)
    elif visual_tokens.dim() == 3:
        V = visual_tokens
    else:
        return visual_tokens
    F_, T, D = V.shape

    alpha = max(0.05, min(1.0, float(keep_ratio)))
    m = max(cfg.min_keep_per_frame * F_, int(round(F_ * T * alpha)))
    m = min(m, F_ * T)
    if m >= F_ * T:
        return visual_tokens

    device = V.device
    V_f = V.float()

    # optional feature normalisation (ablation)
    if cfg.norm_variant == "mean_center_video":
        V_f = V_f - V_f.reshape(-1, D).mean(0).view(1, 1, -1)
    elif cfg.norm_variant == "channel_standardize":
        flat = V_f.reshape(-1, D)
        V_f = (V_f - flat.mean(0).view(1, 1, -1)) / flat.std(0).clamp_min(1e-6).view(1, 1, -1)
    elif cfg.norm_variant == "l2_normalize":
        V_f = F.normalize(V_f, dim=-1)

    # single frame: spatial-outlier fallback (no temporal signal)
    if F_ == 1:
        cos = (F.normalize(V_f[0], dim=-1) *
               F.normalize(V_f[0].mean(0, keepdim=True), dim=-1)).sum(-1)
        score = (1.0 - cos).clamp_min(0.0)
        idx = torch.sort(torch.topk(score, min(max(cfg.min_keep_per_frame, m), T),
                                    sorted=False).indices).values
        return V[0, idx]

    # R1: anchor frames
    anchor_idx = select_anchor_frames(V_f, cfg)
    a = anchor_idx.shape[0]
    is_anchor = torch.zeros(F_, dtype=torch.bool, device=device)
    is_anchor[anchor_idx] = True

    # feature space for scoring (full-D by default; JL is an ablation)
    if cfg.use_jl:
        d_prime = min(D, max(cfg.jl_min, int(D * cfg.jl_ratio)))
        V_s = _jl_project(V_f, d_prime, cfg.seed)
    else:
        V_s = V_f

    # R3: budget split
    beta = 0.5 if not cfg.r3 else cfg.beta
    pool = V_s[anchor_idx].reshape(-1, V_s.shape[-1])       # [a*T, D']
    pool_size = a * T
    k_anchor = min(int(round(beta * m)), pool_size)
    k_pred = m - k_anchor

    keep = torch.zeros(F_, T, dtype=torch.bool, device=device)

    # R3: emit anchor tokens
    sel = _select_anchor_tokens(pool, k_anchor, cfg)
    af = anchor_idx[(sel // T).long()]
    at = (sel % T).long()
    keep[af, at] = True

    # R2: score & emit non-anchor tokens
    nonanchor = torch.nonzero(~is_anchor, as_tuple=True)[0]
    if k_pred > 0 and nonanchor.numel() > 0:
        targets = V_s[nonanchor].reshape(-1, V_s.shape[-1])
        residuals = _residual_score(targets, pool, V_f, nonanchor, F_, T, cfg)
        k_pred_safe = min(k_pred, residuals.numel())
        top = torch.topk(residuals, k_pred_safe, sorted=False).indices
        nf = nonanchor[(top // T).long()]
        nt = (top % T).long()
        keep[nf, nt] = True

    # per-frame floor
    for fi in range(F_):
        n = int(keep[fi].sum())
        if n < cfg.min_keep_per_frame and n < T:
            unkept = torch.nonzero(~keep[fi], as_tuple=True)[0]
            keep[fi, unkept[: cfg.min_keep_per_frame - n]] = True

    # assemble in frame order
    parts = [V[fi, torch.nonzero(keep[fi], as_tuple=True)[0]]
             for fi in range(F_) if keep[fi].any()]
    return torch.cat(parts, dim=0) if parts else V[0, : cfg.min_keep_per_frame]
