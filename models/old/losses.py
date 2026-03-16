"""
src/losses.py

Loss functions for the Grid Inertia PINN.

Three components, combined into PINNLoss:

    L = L_data + λ_phys · L_swing + λ_reg · L_smooth

1. DataLoss        — supervised MSE on H_sys proxy target
2. SwingLoss       — physics residual from the swing equation
3. SmoothnessLoss  — TV regularisation on H_sys and D(t) predictions
4. PINNLoss        — weighted combination with adaptive λ scheduling

Swing equation (normalised form used internally):
    residual = 2·H·(df/dt) − ΔP / (P_total · f₀)
    L_swing  = mean(residual²)

Usage
-----
    from src.losses import PINNLoss

    criterion = PINNLoss(lambda_phys=1.0, lambda_smooth=0.01)

    loss, breakdown = criterion(
        H_pred     = H,            # (B,)  model output
        D_pred     = D,            # (B,)  model output
        RoCoF_pred = RoCoF,        # (B,)  model output
        H_target   = batch["H_sys"],
        delta_P    = batch["delta_P"],
        P_total    = batch["P_total"],
    )

    loss.backward()

    # breakdown is a dict with per-term scalar values for logging
    print(breakdown)
    # {'loss_total': 0.142, 'loss_data': 0.031, 'loss_swing': 0.098, 'loss_smooth': 0.013}
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
F0          = 50.0    # nominal frequency Hz
DT          = 900.0   # 15-min timestep in seconds (OPSD resolution)
EPS         = 1e-6    # numerical stability guard


# ---------------------------------------------------------------------------
# 1. Data loss
# ---------------------------------------------------------------------------

class DataLoss(nn.Module):
    """Supervised MSE between predicted H_sys and proxy target.

    Uses Huber loss (smooth L1) instead of plain MSE — more robust to
    the outlier H_sys values that occur during extreme renewable events.

    Parameters
    ----------
    delta : float
        Huber threshold. Residuals below delta are L2; above are L1.
        Default 0.5 MWs/MVA is roughly 10% of the typical H_sys range.
    """

    def __init__(self, delta: float = 0.5):
        super().__init__()
        self.delta = delta

    def forward(
        self,
        H_pred:   torch.Tensor,   # (B,)
        H_target: torch.Tensor,   # (B,)
        mask:     torch.Tensor | None = None,   # (B,) bool — skip NaN rows
    ) -> torch.Tensor:
        loss = F.huber_loss(H_pred, H_target, reduction="none", delta=self.delta)
        if mask is not None:
            loss = loss[mask]
        return loss.mean()


# ---------------------------------------------------------------------------
# 2. Swing equation physics loss
# ---------------------------------------------------------------------------

class SwingLoss(nn.Module):
    """Physics residual from the normalised swing equation.

    Swing equation:
        2·H(t)·df/dt = ΔP(t) / (P_total(t) · f₀)

    Rearranged residual (what the network should drive to zero):
        r(t) = 2·H_pred·RoCoF_pred − ΔP / (P_total · f₀)

    where:
        ΔP      = P_load − P_gen   (positive = demand exceeds supply)
        P_total = total online generation capacity
        f₀      = 50 Hz

    The loss is mean(r²), weighted optionally by |ΔP| to focus on
    large-imbalance events where the physics constraint matters most.

    Parameters
    ----------
    weighted : bool
        If True, weight each residual by normalised |ΔP| — this puts
        more emphasis on high-stress timesteps.
    """

    def __init__(self, weighted: bool = True):
        super().__init__()
        self.weighted = weighted

    def forward(
        self,
        H_pred:     torch.Tensor,   # (B,)  predicted H_sys
        RoCoF_pred: torch.Tensor,   # (B,)  predicted df/dt  [Hz/s]
        delta_P:    torch.Tensor,   # (B,)  power imbalance  [MW]
        P_total:    torch.Tensor,   # (B,)  total generation [MW]
    ) -> torch.Tensor:

        # Normalised imbalance term  [Hz/s]
        imbalance_norm = delta_P / (P_total.clamp(min=EPS) * F0)

        # Swing equation residual  [Hz/s]
        residual = 2.0 * H_pred * RoCoF_pred - imbalance_norm

        if self.weighted:
            # Weight by normalised |ΔP| — focus on high-stress events
            weights = (delta_P.abs() / (delta_P.abs().mean() + EPS)).detach()
            loss    = (weights * residual.pow(2)).mean()
        else:
            loss = residual.pow(2).mean()

        return loss

    def residuals(
        self,
        H_pred:     torch.Tensor,
        RoCoF_pred: torch.Tensor,
        delta_P:    torch.Tensor,
        P_total:    torch.Tensor,
    ) -> torch.Tensor:
        """Return per-sample residuals (B,) for diagnostics / plotting."""
        imbalance_norm = delta_P / (P_total.clamp(min=EPS) * F0)
        return 2.0 * H_pred * RoCoF_pred - imbalance_norm


# ---------------------------------------------------------------------------
# 3. Smoothness / temporal regularisation loss
# ---------------------------------------------------------------------------

class SmoothnessLoss(nn.Module):
    """Total-variation regularisation on H_sys and D(t) predictions.

    Physical inertia changes slowly — large step-changes between adjacent
    15-min timesteps are unphysical. This loss penalises rapid variation.

    Only meaningful when the batch is a contiguous time window (which is
    how GridInertiaDataset constructs batches in train.py).

    Parameters
    ----------
    h_weight : float   relative weight for H_sys smoothness
    d_weight : float   relative weight for D(t) smoothness
    """

    def __init__(self, h_weight: float = 1.0, d_weight: float = 0.5):
        super().__init__()
        self.h_weight = h_weight
        self.d_weight = d_weight

    def forward(
        self,
        H_pred: torch.Tensor,   # (B,)
        D_pred: torch.Tensor,   # (B,)
    ) -> torch.Tensor:
        if H_pred.shape[0] < 2:
            return torch.tensor(0.0, device=H_pred.device)

        dH = H_pred[1:] - H_pred[:-1]   # first differences
        dD = D_pred[1:] - D_pred[:-1]

        loss_H = dH.pow(2).mean()
        loss_D = dD.pow(2).mean()

        return self.h_weight * loss_H + self.d_weight * loss_D


# ---------------------------------------------------------------------------
# 4. Combined PINN loss
# ---------------------------------------------------------------------------

class PINNLoss(nn.Module):
    """Weighted combination of all three loss terms.

        L = L_data + λ_phys · L_swing + λ_smooth · L_smooth

    Supports adaptive λ scheduling: call step() after each epoch to
    gradually increase the physics weight as the data loss converges.
    This is the recommended warm-up strategy for PINNs — starting with
    a high physics weight often prevents the model from fitting the data.

    Parameters
    ----------
    lambda_phys    : initial physics loss weight    (default 1.0)
    lambda_smooth  : smoothness regularisation weight (default 0.01)
    huber_delta    : Huber loss threshold for DataLoss
    weighted_swing : weight swing residuals by |ΔP|
    phys_warmup    : if True, ramp lambda_phys from 0 → target over
                     `warmup_epochs` epochs
    warmup_epochs  : number of epochs for physics warm-up
    """

    def __init__(
        self,
        lambda_phys:    float = 1.0,
        lambda_smooth:  float = 0.01,
        huber_delta:    float = 0.5,
        weighted_swing: bool  = True,
        phys_warmup:    bool  = True,
        warmup_epochs:  int   = 10,
    ):
        super().__init__()
        self.lambda_phys_target = lambda_phys
        self.lambda_smooth      = lambda_smooth
        self.phys_warmup        = phys_warmup
        self.warmup_epochs      = warmup_epochs

        # Current (possibly ramping) physics weight
        self._lambda_phys = 0.0 if phys_warmup else lambda_phys
        self._epoch       = 0

        self.data_loss   = DataLoss(delta=huber_delta)
        self.swing_loss  = SwingLoss(weighted=weighted_swing)
        self.smooth_loss = SmoothnessLoss()

    @property
    def lambda_phys(self) -> float:
        return self._lambda_phys

    def step(self):
        """Call once per epoch to advance the physics warm-up schedule."""
        self._epoch += 1
        if self.phys_warmup and self._epoch <= self.warmup_epochs:
            # Linear ramp from 0 → lambda_phys_target
            self._lambda_phys = (
                self.lambda_phys_target * self._epoch / self.warmup_epochs
            )
        else:
            self._lambda_phys = self.lambda_phys_target

    def forward(
        self,
        H_pred:     torch.Tensor,             # (B,)
        D_pred:     torch.Tensor,             # (B,)
        RoCoF_pred: torch.Tensor,             # (B,)
        H_target:   torch.Tensor,             # (B,)  proxy from notebook 02
        delta_P:    torch.Tensor,             # (B,)  P_load − P_gen
        P_total:    torch.Tensor,             # (B,)
        mask:       torch.Tensor | None = None,  # (B,) bool
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Returns
        -------
        loss      : scalar tensor  — call .backward() on this
        breakdown : dict of float  — individual terms for logging
        """
        l_data   = self.data_loss(H_pred, H_target, mask)
        l_swing  = self.swing_loss(H_pred, RoCoF_pred, delta_P, P_total)
        l_smooth = self.smooth_loss(H_pred, D_pred)

        loss = (
            l_data
            + self._lambda_phys * l_swing
            + self.lambda_smooth * l_smooth
        )

        breakdown = {
            "loss_total":  loss.item(),
            "loss_data":   l_data.item(),
            "loss_swing":  l_swing.item(),
            "loss_smooth": l_smooth.item(),
            "lambda_phys": self._lambda_phys,
        }

        return loss, breakdown


# ---------------------------------------------------------------------------
# Jitter Index
# ---------------------------------------------------------------------------

def jitter_index(
    RoCoF_series: torch.Tensor,   # (T,)  time series of RoCoF predictions
    H_series:     torch.Tensor,   # (T,)  corresponding H_sys predictions
    window:       int = 96,       # rolling window size (96 × 15min = 24h)
) -> torch.Tensor:
    """Compute the rolling Jitter Index J(t) = Var(df/dt) / H_sys.

    Parameters
    ----------
    RoCoF_series : (T,)
    H_series     : (T,)
    window       : rolling window in timesteps (default 96 = 24 hours)

    Returns
    -------
    J : (T,)  — NaN for the first `window-1` timesteps
    """
    T    = RoCoF_series.shape[0]
    J    = torch.full((T,), float("nan"), device=RoCoF_series.device)

    for t in range(window - 1, T):
        window_RoCoF = RoCoF_series[t - window + 1 : t + 1]
        var_RoCoF    = window_RoCoF.var()
        H_mean       = H_series[t - window + 1 : t + 1].mean().clamp(min=EPS)
        J[t]         = var_RoCoF / H_mean

    return J


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    B = 64

    H      = torch.rand(B) * 4 + 1      # 1–5 MWs/MVA
    D      = torch.rand(B) * 2 + 0.1    # 0.1–2.1 MW/Hz
    RoCoF  = (torch.rand(B) - 0.5) * 0.4   # -0.2 to 0.2 Hz/s
    H_tgt  = H + 0.1 * torch.randn(B)
    dP     = (torch.rand(B) - 0.5) * 5000   # ±2500 MW imbalance
    P_tot  = torch.rand(B) * 30000 + 20000  # 20–50 GW

    criterion = PINNLoss(lambda_phys=1.0, lambda_smooth=0.01,
                         phys_warmup=True, warmup_epochs=5)

    print("=== Loss warm-up progression ===")
    for epoch in range(7):
        loss, bd = criterion(H, D, RoCoF, H_tgt, dP, P_tot)
        print(f"  epoch {epoch+1}: total={bd['loss_total']:.4f}  "
              f"data={bd['loss_data']:.4f}  "
              f"swing={bd['loss_swing']:.4f}  "
              f"smooth={bd['loss_smooth']:.4f}  "
              f"λ_phys={bd['lambda_phys']:.3f}")
        criterion.step()

    print("\n=== Swing equation residuals (first 5) ===")
    sl  = SwingLoss(weighted=False)
    res = sl.residuals(H, RoCoF, dP, P_tot)
    print(f"  {res[:5].tolist()}")

    print("\n=== Jitter Index (last 5 timesteps) ===")
    J = jitter_index(RoCoF, H, window=32)
    valid = J[~J.isnan()]
    print(f"  J mean={valid.mean():.6f}  max={valid.max():.6f}")
