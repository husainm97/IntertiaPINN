# ⚡ Grid Inertia PINN

![PyTorch](https://img.shields.io/badge/PyTorch-deep_learning-orange?logo=pytorch)
![Physics Informed ML](https://img.shields.io/badge/Physics--Informed-ML-blue)
![Stochastic Modeling](https://img.shields.io/badge/Stochastic-Process%20Modeling-green)
![Signal Processing](https://img.shields.io/badge/Signal%20Processing-Savitzky--Golay-purple)
![Time Series](https://img.shields.io/badge/Time%20Series-Analysis-teal)
![Scientific Python](https://img.shields.io/badge/Scientific-Python-yellow?logo=python)

An experimental **Physics-Informed Neural Network (PINN)** for estimating effective grid inertia from publicly available frequency data — without proprietary generator dispatch information or known power disturbances.

> ⚠️ **This is exploratory research.** Results are preliminary. The methodology has not been validated against independent ground-truth inertia measurements. Claims should be treated as physically plausible hypotheses, not established findings.

---

## 🔬 The Problem

As renewable penetration increases across European grids, synchronous generators are displaced by inverter-based resources. This reduces rotational inertia — the physical resistance of the grid to frequency changes — and increases vulnerability to disturbances.

Grid operators need to know how much inertia the system has at any moment in order to decide how much synthetic inertia compensation (from batteries and flywheels) to procure. The standard approach — summing generator inertia constants weighted by dispatch — systematically undercounts because it ignores load-side inertia from industrial motors and rotating machinery, assigns zero to wind and solar, and is only available at 15-minute resolution from delayed dispatch reports.

This project asks: **can effective system inertia be estimated in real-time from frequency measurements alone?**

---

## 🧮 Physics

The stochastic swing equation governs grid frequency dynamics near equilibrium:

$$M \frac{df}{dt} + D(f - f_0) = \xi(t)$$

where $M$ is effective inertia, $D$ is damping, $f_0 = 50$ Hz, and $\xi(t)$ is an unobserved stochastic power imbalance process.

The key insight: if $\xi(t)$ is physically reasonable — uncorrelated in time, since load switching events have no memory — then $M$ and $D$ are the unique parameter pair for which the residual

$$R(t) = M \frac{df}{dt} + D(f - f_0)$$

has zero autocorrelation at all lags. The model identifies $M$ and $D$ by finding the pair that whitens $R$. By 'whitening' the residual, the expected outcome is that all time-dependent physical dynamics (inertia and damping) have been fully captured by the model, leaving behind only the unpredictable, non-autocorrelated noise of the underlying power demand.

**Important caveat:** this identification is theoretically sound but numerically weak on CE grid data. The gradient signal driving $M$ and $D$ toward the correct values is small because $df/dt$ on the stable CE grid is O(1e-3) Hz/s. The model finds physically plausible values but convergence is not guaranteed to a unique solution. See Limitations.

---

## 💡 Two Approaches

### 🔍 `InertiaPINN` — per-window analysis (notebook 03)

Fits a **Tanh MLP** $\hat{f}(t)$ to one window of frequency data, taking normalised time $t \in [0,1]$ as input. The derivative $d\hat{f}/dt$ is computed via **automatic differentiation** on the smooth learned function — not finite differences on noisy data.

$M$ and $D$ are `nn.Parameter` values trained jointly with the network weights via the whiteness loss. A curvature penalty on $d^2f/dt^2$ suppresses jittery derivatives. Hyperparameters are tuned with Optuna on a per-window basis, jointly optimising residual whiteness and derivative smoothness.

**Status:** produces physically plausible $M$ estimates. Residuals approach but do not fully achieve whiteness on single windows. Cannot generalise — requires retraining per window.

**Purpose:** validation tool and slow reference estimator. Not the operational product.

### 🚀 `InertiaNet` — real-time generalisable estimator (notebook 04)

Trained once on multiple years of data. A single forward pass on any new frequency window produces $M$ and $D$ in under 20ms. This is the operational model.

Uses a 1D-CNN to extract temporal features from the standardised frequency window, followed by an MLP with sigmoid-bounded output heads for $M$ and $D$. The Savitzky-Golay filter provides smooth $df/dt$ as a preprocessed input feature.

**Status:** trained on 2015–2017, validated on 2018–2019. Produces a diurnal inertia cycle (higher overnight, lower midday) consistent with physical expectations. Shows a small but correct decrease in $M$ as renewable penetration increases from 38% (2018) to 42% (2019). Residuals are not fully white — the whiteness loss identifies the correct direction but the identification problem is numerically underdetermined on ambient CE frequency data.

---

## 📊 Results

Trained on 2015–2017 German grid frequency data. Validated on 2018–2019 (unseen years with higher renewable penetration).

![image](images/04_inference.png)

| Metric | Value | Notes |
|--------|-------|-------|
| M_PINN mean | 6.17 ± 0.71 MWs/MVA | Full test set mean |
| M_table mean | 3.41 MWs/MVA | Generation-side only — known underestimate |
| Load-side ΔM | ~2.76 MWs/MVA | Excess not attributable to generation mix |
| D mean | 0.21 ± 0.52 MW/Hz | High variance — poorly constrained |
| Night M | 6.30 MWs/MVA | 00:00–06:00 UTC |
| Day M | 5.92 MWs/MVA | 12:00–18:00 UTC |
| Night−Day Δ | 0.38 MWs/MVA | Diurnal industrial load cycle |
| M drop 2018→2019 | 0.08 MWs/MVA | Correct direction, modest magnitude |
| Inference time | <20ms | Single forward pass |

**What these numbers mean and do not mean:**

The ~2.76 MWs/MVA gap between M_PINN and M_table is physically interpretable as load-side inertia — real rotational inertia from motors and industrial machinery that the generation table cannot see. Literature estimates for CE load-side inertia are in the 2–4 MWs/MVA range, making this consistent. However, without validation against event-based ground truth, this remains a hypothesis.

The diurnal pattern is the strongest result: M is higher overnight without the model being told the time of day. This is consistent with overnight industrial processes contributing more rotational inertia than daytime electronic loads. The pattern appeared on both test years without being present in the training signal explicitly.

The year-on-year drop (0.08 MWs/MVA from 2018 to 2019) is in the correct physical direction — more renewables should mean less inertia — but is small relative to the standard deviation of M estimates. It is consistent with physics but not a strong signal.

D is poorly constrained and should not be interpreted quantitatively. The damping term is difficult to identify from ambient frequency data because $f - f_0$ varies slowly compared to $df/dt$.

---

## 🧪 Validation Summary

| Test | M (MWs/MVA) | Notes |
|------|-------------|-------|
| Original inference | 6.17 ± 0.71 | Diurnal pattern, load-side contribution visible |
| Phase-randomised | 6.13 ± 0.69 | Residual whitening preserved, M robust to phase randomisation |
| df/dt scaling | 6.17 | Network robust to linear scaling of df/dt |
| Renewable correlation | r=-0.098 | Negative correlation with increased renewable penetration, consistent with expectations |

---

## ⚙️ Why Not Just Use the Table?

The generation-weighted formula:

$$H_{\text{sys}}(t) = \frac{\sum_i H_i \cdot P_i(t)}{P_{\text{total}}(t)}$$

has three systematic problems this model addresses:

1. **Missing load-side inertia** — motors, pumps, compressors, and industrial flywheels all contribute synchronous inertia. The formula assigns zero to all of them.
2. **Zero for renewables** — correct for inverter-based generation, but obscures the total system picture.
3. **Latency and resolution** — generation dispatch data is available at 15-minute resolution with reporting delays. The model runs on real-time 1-second frequency data.

---

## 🏗️ Architecture

### InertiaPINN (per-window)
```
normalised time t ∈ [0,1]
    ↓  Tanh MLP (hidden_dim × n_layers)
    ↓  scaled f_s(t)  →  autograd df/dt  →  physical df/dt [Hz/s]
    ↓  R = M·df/dt + D·(f-f0)
    M, D: nn.Parameter (physical units, Softplus-positive)
    Loss: whiteness of R + curvature penalty on d²f/dt²
    Tuned: Optuna over lr, beta, delta, architecture, epochs
```

### InertiaNet (generalisable)
```
raw f(t) — 3600s window
    ↓  Savitzky-Golay smooth (window=61, poly=3) → df/dt
    ↓  StandardScaler normalise → f_norm
    ↓  1D-CNN: Conv1d(1→32, k=60, s=30) → Conv1d(32→64, k=10, s=5) → AdaptiveAvgPool(16)
    ↓  MLP: Linear(1024→128) × 4 layers, GELU
    ↓  Output heads with sigmoid bounds
    M ∈ [1, 15] MWs/MVA      D ∈ [0.1, 10] MW/Hz
    Loss: whiteness of R across batch + weak Gaussian prior on M
```

---

## 📁 Project Structure

```
grid-inertia-pinn/
├── data/
│   ├── raw/
│   │   └── de_frequency_1s_{year}.csv     ← 1-second TransnetBW frequency
│   └── processed/
│       ├── de_load_15min.csv
│       ├── de_solar_15min.csv
│       ├── de_wind_15min.csv
│       └── de_inertia_15min.csv
│
├── models/
│   ├── pinn.py                            ← InertiaPINN + InertiaNet
│   └── losses.py                          ← PINNLoss + InertiaNetLoss
│
├── notebooks/
│   ├── 03_pinn_training.ipynb             ← per-window PINN, Optuna tuning
│   └── 04_inference.ipynb                 ← InertiaNet training + validation
│
├── data/
│   ├── build_data.py                      ← builds processed CSVs from OPSD
│   └── fetch_frequency_1s.py              ← downloads TransnetBW frequency data
│
└── checkpoints/
    ├── pinn/                              ← InertiaPINN weights
    └── inertianet/                        ← InertiaNet trained weights
```

---

## 🚀 Setup

```bash
pip install torch pandas numpy matplotlib seaborn scikit-learn scipy optuna
```

---

## 📋 Usage

```bash
# 1. Download 1-second frequency data
python data/fetch_frequency_1s.py --years 2015 2016 2017 2018 2019

# 2. Build processed CSVs from OPSD
python data/build_data.py

# 3. Per-window PINN with Optuna tuning
jupyter notebook notebooks/03_pinn_training.ipynb

# 4. Train InertiaNet + validate on unseen years
jupyter notebook notebooks/04_inference.ipynb
```

---

## 🔭 Limitations

- **No ground truth validation** — M_PINN has not been compared against event-based RoCoF estimates from known generator trips. This is the critical missing validation step.
- **Weak identification signal** — the whiteness criterion is theoretically correct but numerically underdetermined on CE ambient frequency data. The gradient driving M toward the true value is ~O(1e-6) per window. Multi-year training strengthens this but does not fully resolve it.
- **D is not reliably estimated** — damping is harder to identify than inertia from ambient data. D estimates have high variance and should not be interpreted quantitatively without further validation.
- **CE grid stability** — the CE synchronous area is one of the most stable in the world. Inertia variation is subtle year-on-year. Results would likely be more pronounced on GB or Nordic grids where inertia swings more dramatically.
- **Stationarity assumption** — the model assumes M and D are constant within each 1-hour window. This breaks during rapid renewable ramps.
- **SG filter trade-off** — the Savitzky-Golay smoothing reduces df/dt noise ~54x but also removes genuine fast dynamics. This limits the whiteness of R achievable in principle.

---

## 🔮 Potential Extensions

- **Event-based validation** — use ENTSO-E transparency event logs or National Grid ESO disturbance records to compute ground-truth M from known ΔP events, then compare against InertiaNet estimates at the same timestamps
- **Nordic or GB grid** — apply the same methodology to Fingrid or National Grid ESO data where inertia variation is more pronounced and event data is publicly available
- **Multi-year trend analysis** — extend training to 2015–2022 to track the full German energy transition
- **Synthetic inertia detection** — as battery-based synthetic inertia services are deployed, track whether M_PINN captures their contribution
- **Live inference pipeline** — connect to ENTSO-E Transparency Platform API for real-time M estimation

---

## 📚 Data Sources

| Source | Description | Resolution |
|--------|-------------|------------|
| [OPSD Time Series](https://data.open-power-system-data.org/time_series/) | Load, wind, solar generation | 15-min |
| [TransnetBW / OSF](https://osf.io/) | German grid frequency (CE synchronous area) | 1-second |

---

## 📄 License

MIT — experimental research code, use at your own risk.