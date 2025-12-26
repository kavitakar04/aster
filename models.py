import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Dict


@dataclass(frozen=True)
class ProbGrid:
    """Fixed grid of probabilities for discrete distributions."""
    values: torch.Tensor  # shape: (K,)
    K: int


def make_default_grid(K: int = 21) -> ProbGrid:
    """Default 5-cent probability grid for Kalshi-style binary markets (0.05–0.95)."""
    # math assumption: true event prob is well-approximated on {0.05, ..., 0.95} (tails < 5% negligible)
    # math assumption: 5c spacing is sufficient resolution for Π on [0,1]
    return ProbGrid(values=torch.linspace(0.05, 0.95, K), K=K)


def build_gaussian_targets(
    grid: ProbGrid,
    p_mid: torch.Tensor,
    tau: torch.Tensor,
) -> torch.Tensor:
    """
    Build Gaussian targets on the grid:

      Q_k ∝ exp(-(p_k - p_mid)^2 / (2 τ^2)).
    """
    pk = grid.values.to(p_mid.device).view(1, -1)  # (1, K)
    p_mid = p_mid.view(-1, 1)                      # (B, 1)
    tau = tau.view(-1, 1)                          # (B, 1)

    # math assumption: conditional distribution of p around p_mid is symmetric, unimodal, and approx. Gaussian
    sq = (pk - p_mid) ** 2                         # (B, K)

    # math assumption: τ(t) > 0; adding eps just avoids division by 0, doesn't change model class
    Q_unnorm = torch.exp(-sq / (2.0 * (tau ** 2 + 1e-12)))  # (B, K)

    # math assumption: continuous density is represented by its restriction to the fixed discrete grid {p_k}
    Q = Q_unnorm / (Q_unnorm.sum(dim=-1, keepdim=True) + 1e-12)
    return Q


class ProbDistributionMLP(nn.Module):
    """MLP: feature vector -> logits over probability grid."""

    def __init__(
        self,
        d_in: int,
        K: int,
        hidden_dims: Tuple[int, ...] = (64, 64),
        dropout: float = 0.1,  # math assumption: nonzero but small dropout is enough to approximate posterior variability via MC-dropout
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = d_in
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        # math assumption: Π is parameterized via unconstrained logits in ℝ^K and softmax (i.e., categorical on {p_k})
        layers.append(nn.Linear(in_dim, K))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits over the grid; x has shape (B, d_in)."""
        return self.net(x)


def train_model(
    model: ProbDistributionMLP,
    grid: ProbGrid,  # kept for API consistency, not used directly here
    loader,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
) -> float:
    """
    One epoch of training with cross-entropy vs target distributions.

    DataLoader yields (X, P, W):
      - X: (B, d_in)
      - P: (B, K)  target dist on grid (e.g. from build_gaussian_targets)
      - W: (B,)    sample weights
    """
    model.to(device)
    model.train()

    total_loss = 0.0
    n_samples = 0

    for X, P, W in loader:
        X = X.to(device)
        P = P.to(device)
        W = W.to(device)

        logits = model(X)
        log_probs = F.log_softmax(logits, dim=-1)

        # math assumption: training objective is CE(P, Π) ≡ E_P[-log Π], i.e. KL(P || Π) up to constant H(P)
        loss_per = -(P * log_probs).sum(dim=-1)

        # math assumption: importance weights W reweight the empirical risk (weighted empirical measure)
        loss = (loss_per * W).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = X.size(0)
        total_loss += float(loss.item()) * batch_size
        n_samples += batch_size

    return total_loss / max(n_samples, 1)


def distribution_diagnostics(probs: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Diagnostics for discrete distributions on the grid.

    probs: (N, K) or (K,)
    Returns:
      - is_pathological: True if > 1 local peak
      - peak_count: number of local maxima
      - entropy: Shannon entropy
    """
    if probs.dim() == 1:
        probs = probs.unsqueeze(0)

    padded = F.pad(probs, (1, 1), mode="replicate")
    left = padded[:, :-2]
    mid = padded[:, 1:-1]
    right = padded[:, 2:]
    peaks = ((mid > left) & (mid > right)).sum(dim=-1)

    # math assumption: "well-behaved" market beliefs are approximately unimodal in p; multiple strict local maxima are treated as pathologies
    entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)

    return {
        "is_pathological": peaks > 1,
        "peak_count": peaks,
        "entropy": entropy,
    }


def apply_smoothing(probs: torch.Tensor, passes: int = 1) -> torch.Tensor:
    """1D convolutional smoothing along the grid axis."""
    if probs.dim() == 1:
        probs = probs.unsqueeze(0)

    # math assumption: local averaging with a short symmetric kernel approximates a mild low-pass filter on Π over {p_k}
    kernel = torch.tensor(
        [0.2, 0.6, 0.2],
        dtype=probs.dtype,
        device=probs.device,
    ).view(1, 1, 3)

    p = probs.unsqueeze(1)  # (N, 1, K)
    for _ in range(passes):
        # math assumption: Neumann-type boundary condition (replicate) is acceptable for Π at p_1, p_K
        p = F.conv1d(F.pad(p, (1, 1), mode="replicate"), kernel)
    return p.squeeze(1)


def logits_to_probs(
    logits: torch.Tensor,
    smooth_passes: int = 1,
) -> torch.Tensor:
    """Map logits -> (optionally smoothed) probabilities along the grid axis."""
    probs = F.softmax(logits, dim=-1)
    # math assumption: Π is fully captured by its grid probabilities; smoothing enforces a prior of local regularity in p
    return apply_smoothing(probs, passes=smooth_passes)


def probs_to_scalar(
    grid: ProbGrid,
    probs: torch.Tensor,
) -> torch.Tensor:
    """
    Compute scalar expectation E[p] given probs over the probability grid.
    """
    if probs.dim() == 1:
        probs = probs.unsqueeze(0)

    pk = grid.values.to(probs.device)  # (K,)
    # math assumption: downstream decisions only depend on E[p] under Π, not higher moments of Π
    return torch.sum(pk * probs, dim=-1)


def enable_dropout_only(model: nn.Module) -> None:
    """Enable Dropout layers while leaving the rest effectively in eval mode."""
    # math assumption: MC-dropout approximates integrating over a posterior on weights (variational Bayesian view)
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()


@torch.no_grad()
def predict_with_uncertainty(
    model: ProbDistributionMLP,
    grid: ProbGrid,
    features: torch.Tensor,
    n_samples: int = 10,
    smooth_passes: int = 1,
) -> Tuple[torch.Tensor, float, float]:
    """
    MC-dropout prediction: return (mean_probs, mean_p, var_p).
    """
    model.eval()
    enable_dropout_only(model)

    x = features
    if x.dim() == 1:
        x = x.unsqueeze(0)

    probs_samples = []
    p_scalar_samples = []

    for _ in range(n_samples):
        logits = model(x)
        probs = logits_to_probs(logits, smooth_passes=smooth_passes)  # (B, K)
        probs_samples.append(probs)
        p_scalar_samples.append(probs_to_scalar(grid, probs.squeeze(0)))

    probs_stack = torch.stack(probs_samples, dim=0).squeeze(1)  # (n_samples, K)
    p_stack = torch.stack(p_scalar_samples, dim=0)              # (n_samples,)

    mean_probs = probs_stack.mean(dim=0).cpu()
    mean_p = float(p_stack.mean().cpu())
    # math assumption: Var(E[p] | dropout mask) ≈ epistemic variance of predicted probability (σ_i^2 in spec)
    var_p = float(p_stack.var(unbiased=False).cpu())

    return mean_probs, mean_p, var_p
