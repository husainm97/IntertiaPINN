"""
models/losses.py

Loss for stochastic swing equation PINN.

L_data    : MSE between smooth learned f_s(t) and measured f_s
            Network must fit the frequency trajectory

L_white   : Sum of squared autocorrelations of R at multiple lags
            R = M*df/dt + D*(f-f0) must be white noise
            This is what identifies M and D — only the correct pair
            produces a residual with zero autocorrelation

L_prior   : Weak Gaussian prior on M — prevents collapse during warmup
            Decays to near-zero so physics dominates at the end

Why whiteness identifies M and D
---------------------------------
If M is too large: df/dt term dominates, R oscillates → AC at lag 1 non-zero
If M is too small: D term dominates, R mean-reverts too fast → AC structure
If D is too large: R over-damps → negative AC at short lags
If D is too small: R has long memory → positive AC at long lags

Only the correct (M, D) pair whitens R.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def autocorr(x: torch.Tensor, lag: int) -> torch.Tensor:
    x = x.squeeze()
    n = len(x)
    if n <= lag:
        return torch.tensor(0.0, device=x.device)
    # Use unbiased centered autocorrelation
    x = x - x.mean()
    var = (x * x).sum()
    if var < 1e-10:
        return torch.tensor(0.0, device=x.device)
    return (x[:-lag] * x[lag:]).sum() / var


class PINNLoss(nn.Module):
    """
    Parameters
    ----------
    alpha        : data loss weight (constant)
    beta_max     : whiteness loss weight (ramped up after warmup_data)
    gamma_max    : prior weight initial
    gamma_min    : prior weight final (near zero)
    M_prior      : prior centre for M [physical MWs/MVA]
    M_std        : prior std [physical MWs/MVA] — wide = weak
    lags         : autocorrelation lags to enforce whiteness at
    warmup_data  : epochs before whiteness loss starts
    warmup_phys  : epochs to reach full beta_max
    """

    def __init__(
        self,
        alpha:       float = 1.0,
        beta_max:    float = 5.0,
        gamma_max:   float = 1.0,
        gamma_min:   float = 0.01,
        M_prior:     float = 5.0,
        M_std:       float = 3.0,
        lags:        list  = [1, 2, 5, 10, 20, 30],
        warmup_data: int   = 200,
        warmup_phys: int   = 1500,
    ):
        super().__init__()
        self.alpha       = alpha
        self.beta_max    = beta_max
        self.gamma_max   = gamma_max
        self.gamma_min   = gamma_min
        self.M_prior     = M_prior
        self.M_std       = M_std
        self.lags        = lags
        self.warmup_data = warmup_data
        self.warmup_phys = warmup_phys
        self.beta        = 0.0
        self.gamma       = gamma_max

    def schedule(self, epoch: int):
        if epoch < self.warmup_data:
            self.beta = 0.0
        elif epoch < self.warmup_phys:
            frac = (epoch - self.warmup_data) / (self.warmup_phys - self.warmup_data)
            self.beta = self.beta_max * frac
        else:
            self.beta = self.beta_max

        frac_decay = min(1.0, epoch / self.warmup_phys)
        self.gamma = self.gamma_max * (1 - frac_decay) + self.gamma_min * frac_decay

    def forward(
        self,
        f_s_pred: torch.Tensor,   # (N,1) network output in scaled space
        f_s_true: torch.Tensor,   # (N,1) measured f in scaled space
        R:        torch.Tensor,   # (N,1) residual M*df/dt + D*(f-f0)
        M:        torch.Tensor,   # scalar M [physical]
    ) -> tuple[torch.Tensor, dict]:

        l_data = F.mse_loss(f_s_pred, f_s_true)

        # Normalise R to unit variance before computing autocorrelations
        # so the whiteness loss is scale-invariant
        R_norm = (R - R.mean()) / (R.std() + 1e-8)
        l_white = sum(autocorr(R_norm, lag).pow(2) for lag in self.lags)
        l_white = l_white / len(self.lags)

        l_prior = ((M - self.M_prior) / self.M_std).pow(2)

        total = self.alpha * l_data + self.beta * l_white + self.gamma * l_prior

        return total, {
            'loss_total': total.item(),
            'loss_data':  l_data.item(),
            'loss_white': l_white.item() if isinstance(l_white, torch.Tensor) else l_white,
            'loss_prior': l_prior.item(),
            'beta':       self.beta,
            'gamma':      self.gamma,
        }
    

# ══════════════════════════════════════════════════════════════════════════════
# InertiaNetLoss — batch whiteness loss for generalisable training
# ══════════════════════════════════════════════════════════════════════════════
 
class InertiaNetLoss(nn.Module):
    """
    Batch physics loss for InertiaNet.
 
    For each window in the batch, computes R = M*dfdt + D*fdev
    and penalises autocorrelation in R — enforcing whiteness.
 
    Also includes a soft prior on M to prevent collapse.
 
    Parameters
    ----------
    lags     : autocorrelation lags to enforce whiteness at
    M_prior  : soft prior centre for M [MWs/MVA]
    M_std    : prior std — wide = weak
    lambda_p : prior weight
    """
 
    def __init__(
        self,
        lags:     list  = [1, 2, 3, 4, 5, 10, 15, 20, 30, 60],
        M_prior:  float = 5.0,
        M_std:    float = 3.0,
        lambda_p: float = 0.05,
    ):
        super().__init__()
        self.lags     = lags
        self.M_prior  = M_prior
        self.M_std    = M_std
        self.lambda_p = lambda_p
 
    def forward(
        self,
        M:    torch.Tensor,   # (B,) predicted inertia
        D:    torch.Tensor,   # (B,) predicted damping
        dfdt: torch.Tensor,   # (B, W) smooth df/dt [Hz/s]
        fdev: torch.Tensor,   # (B, W) f - f0 [Hz]
    ) -> tuple[torch.Tensor, dict]:
 
        # Residual: (B, W)
        R = M.unsqueeze(1) * dfdt + D.unsqueeze(1) * fdev
 
        # Normalise each window's residual to unit variance
        R_norm = (R - R.mean(dim=1, keepdim=True)) / (R.std(dim=1, keepdim=True) + 1e-8)
 
        # Whiteness loss: mean squared autocorrelation across lags and batch
        l_white = torch.tensor(0.0, device=M.device)
        for lag in self.lags:
            if R_norm.shape[1] > lag:
                x  = R_norm
                c0 = (x * x).sum(dim=1)
                cl = (x[:, :-lag] * x[:, lag:]).sum(dim=1)
                ac = cl / (c0 + 1e-10)
                l_white = l_white + ac.pow(2).mean()
        l_white = l_white / len(self.lags)
 
        # Soft prior on M
        l_prior = ((M - self.M_prior) / self.M_std).pow(2).mean()
 
        loss = l_white + self.lambda_p * l_prior
 
        return loss, {
            'loss_total': loss.item(),
            'loss_white': l_white.item(),
            'loss_prior': l_prior.item(),
            'M_mean':     M.mean().item(),
            'D_mean':     D.mean().item(),
            'M_std':      M.std().item(),
        }