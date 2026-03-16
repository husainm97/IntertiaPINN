"""
models/pinn.py

PINN for grid inertia and damping estimation from frequency data alone.

Physics
-------
The stochastic swing equation:

    M * df/dt + D * (f - f0) = xi(t)

where xi(t) is an unknown stochastic power imbalance process.
We do NOT observe xi(t) or ΔP directly.

Identification
--------------
M and D are identified because only the correct pair produces a residual
R(t) = M * df/dt + D * (f - f0) that is white noise (zero autocorrelation).

Wrong M or D leaves structure in R — either too much mean-reversion
(D too high) or oscillation (M too low). The physics loss enforces
whiteness of R, identifying both parameters from frequency alone.

Scaling
-------
- Network operates on scaled f_s (StandardScaler) — O(1) inputs/outputs
- M and D learned in physical units directly
- R is normalised by its own std before autocorrelation is computed
- All public outputs are in physical units (Hz, Hz/s, MWs/MVA, MW/Hz)
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

F0            = 50.0
WINDOW_S      = 3600
AUTOCORR_LAGS = [1, 2, 3, 4, 5, 10, 15, 20, 30, 60]
 
 

@dataclass
class WindowScalers:
    f: StandardScaler

    @property
    def mu_f(self):  return float(self.f.mean_[0])
    @property
    def sig_f(self): return float(self.f.scale_[0])


def prepare_window(df_freq_full, df_15min, start_idx, window_s=3600):
    """
    Extract one window. Only f is scaled — no power data needed.

    Returns
    -------
    t_norm    : (N,1) float32  normalised time [0,1]
    f_s       : (N,1) float32  scaled frequency — network target
    f_dev     : (N,1) float32  f - f0 in Hz — for residual
    f_raw_hz  : np.ndarray     raw Hz — for plots
    scalers   : WindowScalers
    t_scale   : float          window seconds
    index     : DatetimeIndex
    """
    win = df_freq_full.iloc[start_idx:start_idx + window_s]
    n   = len(win)
    idx = win.index

    f_hz  = win['f_hz'].fillna(F0).values.astype(np.float64)
    f_dev = (f_hz - F0).astype(np.float32)

    sc = StandardScaler()
    f_s = sc.fit_transform(f_hz.reshape(-1, 1)).astype(np.float32)

    scalers = WindowScalers(f=sc)

    def t(x): return torch.from_numpy(x).unsqueeze(1)

    return dict(
        t_norm   = torch.from_numpy(np.linspace(0, 1, n, dtype=np.float32)).unsqueeze(1),
        f_s      = t(f_s.squeeze()),
        f_dev    = t(f_dev),
        scalers  = scalers,
        t_scale  = float(n),
        index    = idx,
        f_raw_hz = f_hz.astype(np.float32),
    )


def make_colloc(window, n_colloc=1000, device='cpu'):
    """Collocation points — t and f_dev only."""
    t_c = torch.linspace(0, 1, n_colloc).unsqueeze(1).to(device)
    t_c.requires_grad_(True)
    idx    = torch.linspace(0, len(window['f_dev'])-1, n_colloc).long()
    fdev_c = window['f_dev'][idx].to(device)
    return t_c, fdev_c


class InertiaPINN(nn.Module):
    """
    Learns smooth f(t) and identifies M, D such that
    R(t) = M * df/dt + D * (f - f0) is white noise.
    """

    def __init__(
        self,
        scalers:    WindowScalers,
        hidden_dim: int   = 64,
        n_layers:   int   = 5,
        t_scale:    float = 3600.0,
        M_init:     float = 5.0,
        D_init:     float = 2.0,
    ):
        super().__init__()
        self.scalers  = scalers
        self.t_scale  = t_scale

        layers = [nn.Linear(1, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers += [nn.Linear(hidden_dim, 1)]
        self.net = nn.Sequential(*layers)

        # Physical units — softplus keeps positive
        self._M_raw = nn.Parameter(
            torch.tensor(float(np.log(np.exp(M_init) - 1)))
        )
        self._D_raw = nn.Parameter(
            torch.tensor(float(np.log(np.exp(D_init) - 1)))
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    # ── Physical parameters ───────────────────────────────────────────────

    @property
    def M(self) -> float:
        return float(nn.functional.softplus(self._M_raw))

    @property
    def D(self) -> float:
        return float(nn.functional.softplus(self._D_raw))

    @property
    def _M_t(self) -> torch.Tensor:
        return nn.functional.softplus(self._M_raw)

    @property
    def _D_t(self) -> torch.Tensor:
        return nn.functional.softplus(self._D_raw)

    # ── Internal scaled forward ───────────────────────────────────────────

    def _forward_scaled(self, t_norm: torch.Tensor) -> torch.Tensor:
        """Scaled f_s(t) in [-3, 3] approx. Internal use."""
        return self.net(t_norm)

    # ── Public physical outputs ───────────────────────────────────────────

    def forward(self, t_norm: torch.Tensor) -> torch.Tensor:
        """f(t) in Hz."""
        f_s = self._forward_scaled(t_norm)
        mu  = torch.tensor(self.scalers.mu_f,  dtype=f_s.dtype, device=f_s.device)
        sig = torch.tensor(self.scalers.sig_f, dtype=f_s.dtype, device=f_s.device)
        return f_s * sig + mu

    def dfdt_hz(self, t_norm: torch.Tensor) -> torch.Tensor:
        """
        df/dt in Hz/s via autograd on smooth f_s(t).
        Key contribution — smooth derivative, not noisy finite differences.
        """
        f_s = self._forward_scaled(t_norm)
        dfdt_norm = torch.autograd.grad(
            f_s, t_norm,
            grad_outputs=torch.ones_like(f_s),
            create_graph=False, retain_graph=True,
        )[0]
        # d(f_s)/dt_norm = df_hz/dt * t_scale / sig_f
        # => df_hz/dt = dfdt_norm * sig_f / t_scale
        return dfdt_norm * self.scalers.sig_f / self.t_scale

    def residual(
        self,
        t_norm: torch.Tensor,   # (N,1) requires_grad=True
        f_dev:  torch.Tensor,   # (N,1) f - f0 in Hz
    ) -> torch.Tensor:
        """
        R(t) = M * df/dt + D * (f - f0)

        For correct M and D this is white noise.
        For wrong M or D it has autocorrelation structure.
        """
        f_s = self._forward_scaled(t_norm)
        dfdt_norm = torch.autograd.grad(
            f_s, t_norm,
            grad_outputs=torch.ones_like(f_s),
            create_graph=True, retain_graph=True,
        )[0]
        dfdt_phys = dfdt_norm * self.scalers.sig_f / self.t_scale
        return self._M_t * dfdt_phys + self._D_t * f_dev

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
# ══════════════════════════════════════════════════════════════════════════════
# InertiaNet — generalisable real-time estimator
# ══════════════════════════════════════════════════════════════════════════════
 
from scipy.signal import savgol_filter
 
 
def preprocess_window(
    f_hz:      np.ndarray,
    sg_window: int = 61,
    sg_poly:   int = 3,
) -> dict:
    """
    Preprocess one raw frequency window for InertiaNet.
 
    Returns
    -------
    f_norm   : (W,) float32  StandardScaler-normalised f — network input
    dfdt     : (W,) float32  smooth df/dt [Hz/s] via Savitzky-Golay
    f_dev    : (W,) float32  f - f0 [Hz]
    f_sc_std : float         scale factor for back-converting if needed
    """
    f = f_hz.astype(np.float64)
 
    # Smooth and differentiate
    f_smooth = savgol_filter(f, window_length=sg_window, polyorder=sg_poly)
    dfdt     = np.gradient(f_smooth).astype(np.float32)   # Hz/s at 1-second
    f_dev    = (f - F0).astype(np.float32)
 
    # Normalise f for network input
    mu     = f.mean(); sig = f.std() + 1e-10
    f_norm = ((f - mu) / sig).astype(np.float32)
 
    return dict(f_norm=f_norm, dfdt=dfdt, f_dev=f_dev, f_sc_std=float(sig))
 
 
class InertiaNet(nn.Module):
    """
    Generalisable real-time inertia estimator.
 
    Trained ONCE on many windows. Inference is a single forward pass —
    no per-window training required.
 
    Input  : (B, W) StandardScaler-normalised frequency window
    Output : M (B,) [MWs/MVA],  D (B,) [MW/Hz]
 
    The network learns the mapping from frequency window shape to (M, D)
    by being trained with the whiteness loss across a large dataset.
    """
 
    def __init__(
        self,
        window_s:   int   = WINDOW_S,
        hidden_dim: int   = 128,
        n_layers:   int   = 4,
        M_min:      float = 1.0,
        M_max:      float = 15.0,
        D_min:      float = 0.1,
        D_max:      float = 10.0,
    ):
        super().__init__()
        self.window_s = window_s
        self.M_min = M_min; self.M_max = M_max
        self.D_min = D_min; self.D_max = D_max
 
        # 1D CNN extracts temporal features from f(t) shape
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=60, stride=30), nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=10, stride=5),  nn.GELU(),
            nn.AdaptiveAvgPool1d(16),
        )
 
        # MLP head
        layers = [nn.Linear(64*16, hidden_dim), nn.GELU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        self.mlp = nn.Sequential(*layers)
 
        self.head_M = nn.Linear(hidden_dim, 1)
        self.head_D = nn.Linear(hidden_dim, 1)
 
        self._init_weights()
 
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
 
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x : (B, W) normalised frequency windows
        Returns M : (B,), D : (B,) in physical units
        """
        z = self.cnn(x.unsqueeze(1)).flatten(1)
        z = self.mlp(z)
        M = self.M_min + (self.M_max - self.M_min) * torch.sigmoid(self.head_M(z).squeeze(1))
        D = self.D_min + (self.D_max - self.D_min) * torch.sigmoid(self.head_D(z).squeeze(1))
        return M, D
 
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
 