"""
src/models/pinn.py

Physics-Informed Neural Network for grid inertia estimation.

Architecture
------------
- Shared MLP backbone (configurable depth + width, GELU activations)
- Three output heads:
    H_sys  : system inertia constant  (MWs/MVA)  — softplus → strictly positive
    D      : damping coefficient      (MW/Hz)     — softplus → strictly positive
    RoCoF  : rate of change of freq   (Hz/s)      — unbounded (signed)
- MC Dropout disabled at inference by default; enable via eval_with_dropout()
- DeepEnsemble wrapper trains N independent members and aggregates predictions

Input features (see GridInertiaDataset)
----------------------------------------
Power signals   : P_load, P_solar, P_wind_on, P_wind_off, P_thermal, P_hydro,
                  P_total, renewables_fraction, delta_P (imbalance)
Cyclic time     : sin/cos of hour-of-day, sin/cos of month
Year trend      : (year - year_min) / year_range   (linear normalised)
Total           : 14 features by default (INPUT_DIM = 14)

Usage
-----
    from src.models.pinn import GridInertiaPINN, DeepEnsemble

    model = GridInertiaPINN()
    H, D, RoCoF = model(x)               # deterministic forward pass

    ensemble = DeepEnsemble(n_members=5)
    mean, std = ensemble(x)               # mean/std across members  (3, B) each
    ensemble.save("checkpoints/")
    ensemble.load("checkpoints/")
"""

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INPUT_DIM   = 14    # see feature list above
HIDDEN_DIM  = 128
N_LAYERS    = 6     # shared backbone depth
DROPOUT_P   = 0.15
F0          = 50.0  # nominal grid frequency (Hz)

# Output head clamp bounds (physics-grounded)
H_SYS_MIN   = 0.1   # MWs/MVA  — grid never fully loses inertia
H_SYS_MAX   = 10.0  # MWs/MVA  — realistic upper bound for DE grid
D_MIN       = 0.01  # MW/Hz
D_MAX       = 5.0   # MW/Hz


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Two-layer residual block with GELU + dropout.

    Using residual connections in the backbone helps gradient flow through
    deeper networks and stabilises physics-loss training.
    """

    def __init__(self, dim: int, dropout_p: float = DROPOUT_P):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.block(x))


class OutputHead(nn.Module):
    """Single-output projection with optional soft clamping.

    soft_clamp=True applies  min + softplus(x) * scale  so the output
    is strictly within (min, max) while remaining smooth + differentiable
    everywhere — important for physics gradient computation.
    """

    def __init__(
        self,
        in_dim: int,
        out_min: float | None = None,
        out_max: float | None = None,
    ):
        super().__init__()
        self.fc   = nn.Linear(in_dim, 1)
        self.min  = out_min
        self.max  = out_max
        self._use_clamp = (out_min is not None) and (out_max is not None)

        if self._use_clamp:
            self._range = out_max - out_min

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc(x).squeeze(-1)           # (B,)
        if self._use_clamp:
            # Differentiable soft clamp: maps ℝ → (min, max)
            out = self.min + self._range * torch.sigmoid(out)
        return out


# ---------------------------------------------------------------------------
# Core PINN
# ---------------------------------------------------------------------------

class GridInertiaPINN(nn.Module):
    """Physics-Informed Neural Network for grid inertia estimation.

    Forward pass returns three tensors:
        H_sys  (B,)  — system inertia constant  [MWs/MVA]
        D      (B,)  — damping coefficient       [MW/Hz]
        RoCoF  (B,)  — rate of change of freq    [Hz/s]

    The swing equation residual is computed externally in losses.py using
    these three outputs together with the power imbalance ΔP from the batch.

    Parameters
    ----------
    input_dim   : number of input features       (default 14)
    hidden_dim  : neurons per hidden layer        (default 128)
    n_layers    : number of residual blocks       (default 6)
    dropout_p   : dropout probability             (default 0.15)
    """

    def __init__(
        self,
        input_dim:  int   = INPUT_DIM,
        hidden_dim: int   = HIDDEN_DIM,
        n_layers:   int   = N_LAYERS,
        dropout_p:  float = DROPOUT_P,
    ):
        super().__init__()

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
        )

        # Shared backbone — residual blocks
        self.backbone = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout_p) for _ in range(n_layers)]
        )

        # Three output heads
        self.head_H     = OutputHead(hidden_dim, H_SYS_MIN, H_SYS_MAX)   # H_sys
        self.head_D     = OutputHead(hidden_dim, D_MIN,     D_MAX)        # D(t)
        self.head_RoCoF = OutputHead(hidden_dim)                           # RoCoF (unbounded)

        self._init_weights()

    def _init_weights(self):
        """Kaiming init for linear layers; zeros for biases in heads."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, input_dim)

        Returns
        -------
        H_sys  : (B,)
        D      : (B,)
        RoCoF  : (B,)
        """
        z = self.input_proj(x)
        z = self.backbone(z)
        return self.head_H(z), self.head_D(z), self.head_RoCoF(z)

    def forward_with_grad(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Same as forward but ensures input requires grad.

        Call this when computing the physics loss so autograd can
        differentiate the swing equation residual w.r.t. inputs.
        """
        x = x.requires_grad_(True)
        return self.forward(x)

    def enable_dropout(self):
        """Switch all Dropout layers to training mode for MC Dropout inference."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Deep Ensemble wrapper
# ---------------------------------------------------------------------------

class DeepEnsemble(nn.Module):
    """Ensemble of N independently initialised GridInertiaPINN members.

    Each member is trained from a different random seed (handled in train.py).
    At inference, predictions from all members are aggregated to produce
    mean and std estimates — these form the uncertainty bounds.

    Outputs
    -------
    mean : (3, B)  — [H_sys_mean, D_mean, RoCoF_mean]
    std  : (3, B)  — [H_sys_std,  D_std,  RoCoF_std ]

    Parameters
    ----------
    n_members   : number of ensemble members     (default 5)
    **kwargs    : passed to each GridInertiaPINN
    """

    def __init__(self, n_members: int = 5, **kwargs):
        super().__init__()
        self.n_members = n_members
        self.members   = nn.ModuleList(
            [GridInertiaPINN(**kwargs) for _ in range(n_members)]
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, input_dim)

        Returns
        -------
        mean : (3, B)
        std  : (3, B)
        """
        preds = []
        for member in self.members:
            H, D, R = member(x)
            preds.append(torch.stack([H, D, R], dim=0))   # (3, B)

        stacked = torch.stack(preds, dim=0)   # (M, 3, B)
        mean    = stacked.mean(dim=0)         # (3, B)
        std     = stacked.std(dim=0)          # (3, B)
        return mean, std

    def forward_all(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """Return raw predictions from all members.

        Returns
        -------
        all_preds : (M, 3, B)  — useful for calibration analysis
        """
        preds = []
        for member in self.members:
            H, D, R = member(x)
            preds.append(torch.stack([H, D, R], dim=0))
        return torch.stack(preds, dim=0)

    def save(self, directory: str | Path):
        """Save each member as member_{i}.pt inside directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for i, member in enumerate(self.members):
            torch.save(member.state_dict(), directory / f"member_{i}.pt")
        print(f"Saved {self.n_members} ensemble members to {directory}/")

    def load(self, directory: str | Path, **kwargs):
        """Load member weights from directory."""
        directory = Path(directory)
        for i, member in enumerate(self.members):
            path = directory / f"member_{i}.pt"
            member.load_state_dict(torch.load(path, map_location="cpu"))
        print(f"Loaded {self.n_members} ensemble members from {directory}/")

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Feature engineering helper (used by dataset + notebook)
# ---------------------------------------------------------------------------

def build_time_features(
    hours:  torch.Tensor,   # (B,)  integer 0–23
    months: torch.Tensor,   # (B,)  integer 1–12
    years:  torch.Tensor,   # (B,)  integer e.g. 2015–2020
    year_min: float = 2015.0,
    year_range: float = 5.0,
) -> torch.Tensor:
    """Encode time as cyclic + linear trend features.

    Returns
    -------
    features : (B, 5)
        [sin_hour, cos_hour, sin_month, cos_month, year_trend]
    """
    TWO_PI = 2.0 * math.pi
    sin_h  = torch.sin(TWO_PI * hours.float()  / 24.0)
    cos_h  = torch.cos(TWO_PI * hours.float()  / 24.0)
    sin_m  = torch.sin(TWO_PI * (months.float() - 1) / 12.0)
    cos_m  = torch.cos(TWO_PI * (months.float() - 1) / 12.0)
    trend  = (years.float() - year_min) / year_range

    return torch.stack([sin_h, cos_h, sin_m, cos_m, trend], dim=1)  # (B, 5)


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    B = 32

    model    = GridInertiaPINN()
    ensemble = DeepEnsemble(n_members=5)

    x = torch.randn(B, INPUT_DIM)

    # Single model
    H, D, R = model(x)
    print(f"Single model  — H: {H.shape}  D: {D.shape}  RoCoF: {R.shape}")
    print(f"  H range : [{H.min():.3f}, {H.max():.3f}]  (expect {H_SYS_MIN}–{H_SYS_MAX})")
    print(f"  D range : [{D.min():.3f}, {D.max():.3f}]  (expect {D_MIN}–{D_MAX})")
    print(f"  Params  : {model.count_parameters():,}")

    # Ensemble
    mean, std = ensemble(x)
    print(f"\nEnsemble (5)  — mean: {mean.shape}  std: {std.shape}")
    print(f"  H_sys mean  : {mean[0].mean():.3f} ± {std[0].mean():.4f}")
    print(f"  D     mean  : {mean[1].mean():.3f} ± {std[1].mean():.4f}")
    print(f"  RoCoF mean  : {mean[2].mean():.4f} ± {std[2].mean():.4f}")
    print(f"  Total params: {ensemble.count_parameters():,}")
