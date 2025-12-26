import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, Union
from torch.utils.data import DataLoader

# --- Core Probability Components ---

@dataclass(frozen=True)
class ProbGrid:
    """Fixed grid of probabilities for discrete distributions."""
    values: torch.Tensor  # shape: (K,)
    K: int

def make_default_grid(K: int = 21) -> ProbGrid:
    """Default 5-cent probability grid (0.05–0.95)."""
    return ProbGrid(values=torch.linspace(0.05, 0.95, K), K=K)

def build_gaussian_targets(grid: ProbGrid, p_mid: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """Build Gaussian targets on the grid: Q_k ∝ exp(-(p_k - p_mid)^2 / (2 τ^2))."""
    pk = grid.values.to(p_mid.device).view(1, -1)
    p_mid = p_mid.view(-1, 1)
    tau = tau.view(-1, 1)
    sq = (pk - p_mid) ** 2
    Q_unnorm = torch.exp(-sq / (2.0 * (tau ** 2 + 1e-12)))
    return Q_unnorm / (Q_unnorm.sum(dim=-1, keepdim=True) + 1e-12)

# --- Neural Network ---

class ProbDistributionMLP(nn.Module):
    """MLP: feature vector -> logits over probability grid."""
    def __init__(self, d_in: int, K: int, hidden_dims: Tuple[int, ...] = (64, 64), dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = d_in
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, K))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# --- Market Agent ---

class MarketAgent:
    """
    Encapsulates the Model, Optimizer, and Normalization stats for a specific market.
    """
    def __init__(self, market_id: str, input_dim: int = 9, device: str = "cpu", lr: float = 1e-4):
        self.market_id = market_id
        self.device = device
        self.grid = make_default_grid(K=21)
        
        # 1. Model
        self.model = ProbDistributionMLP(d_in=input_dim, K=self.grid.K).to(device)
        
        # 2. Optimizer (Persistent state!)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # 3. Normalization Stats (Initialized to identity)
        self.norm_mean = torch.zeros(input_dim).to(device)
        self.norm_std = torch.ones(input_dim).to(device)
        self.is_normalized = False

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor):
        """Update normalization stats."""
        self.norm_mean = mean.to(self.device)
        self.norm_std = std.to(self.device)
        self.is_normalized = True

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply stored normalization."""
        if not self.is_normalized:
            return x
        return (x.to(self.device) - self.norm_mean) / (self.norm_std + 1e-6)

    def save(self, base_dir: str = "models_ckpts", norm_dir: str = "normalization"):
        """Save model weights and normalization stats."""
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(norm_dir, exist_ok=True)
        
        # Save Model
        torch.save(self.model.state_dict(), f"{base_dir}/{self.market_id}.pt")
        
        # Save Norms
        torch.save({
            "mean": self.norm_mean.cpu(),
            "std": self.norm_std.cpu()
        }, f"{norm_dir}/{self.market_id}_norm.pt")

    def load(self, base_dir: str = "models_ckpts", norm_dir: str = "normalization") -> bool:
        """Load model weights and stats if they exist. Returns True if successful."""
        model_path = f"{base_dir}/{self.market_id}.pt"
        norm_path = f"{norm_dir}/{self.market_id}_norm.pt"
        
        if not (os.path.exists(model_path) and os.path.exists(norm_path)):
            return False
            
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            norms = torch.load(norm_path, map_location=self.device)
            self.set_normalization(norms["mean"], norms["std"])
            self.model.eval() # Default to eval
            return True
        except Exception as e:
            print(f"[{self.market_id}] Failed to load agent: {e}")
            return False

    def train_epoch(self, loader: DataLoader) -> float:
        """Runs one epoch of training/fine-tuning."""
        self.model.train()
        total_loss = 0.0
        n_samples = 0
        
        for X, P, W in loader:
            X = X.to(self.device)
            P = P.to(self.device) # Target distributions
            W = W.to(self.device)

            logits = self.model(X)
            log_probs = F.log_softmax(logits, dim=-1)

            # KL-Divergence
            loss_per = -(P * log_probs).sum(dim=-1)
            accuracy_loss = (loss_per * W).mean()

            # Stability Regularization
            loss = accuracy_loss + 0.01 * (logits ** 2).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_size = X.size(0)
            total_loss += float(loss.item()) * batch_size
            n_samples += batch_size

        self.model.eval()
        return total_loss / max(n_samples, 1)

    @torch.no_grad()
    def predict(self, features: torch.Tensor, n_samples: int = 10) -> Tuple[torch.Tensor, float, float]:
        """
        Inference with MC-Dropout for uncertainty.
        Returns: (mean_probs_tensor, mean_scalar_price, scalar_variance)
        """
        self.model.eval()
        
        # Enable dropout specifically for MC sampling
        for m in self.model.modules():
            if m.__class__.__name__.startswith("Dropout"):
                m.train()

        x = self.normalize(features)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        probs_samples = []
        scalar_samples = []
        pk = self.grid.values.to(self.device)

        for _ in range(n_samples):
            logits = self.model(x)
            # Smooth + Softmax
            probs = F.softmax(logits, dim=-1)
            # TODO: Experiment with smoothing kernel
            kernel = torch.tensor([0.2, 0.6, 0.2], device=self.device).view(1, 1, 3)
            probs = F.conv1d(F.pad(probs.unsqueeze(1), (1, 1), mode="replicate"), kernel).squeeze(1)
            
            probs_samples.append(probs)
            scalar_samples.append((pk * probs).sum(dim=-1))

        probs_stack = torch.stack(probs_samples, dim=0).squeeze(1) # (N, K)
        p_stack = torch.stack(scalar_samples, dim=0)               # (N,)

        mean_probs = probs_stack.mean(dim=0).cpu()
        mean_p = float(p_stack.mean().cpu())
        var_p = float(p_stack.var(unbiased=False).cpu())

        return mean_probs, mean_p, var_p