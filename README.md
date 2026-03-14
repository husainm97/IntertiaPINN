# Grid Inertia PINN

A **Physics-Informed Neural Network (PINN)** for estimating the effective rotational inertia of the German power grid from publicly available OPSD generation and demand data.

---

## Scientific Motivation

Grid inertia determines how resistant power system frequency is to disturbances. As renewable penetration increases, synchronous generators are progressively displaced by inverter-based resources — reducing inertia and increasing frequency volatility.

Traditional inertia estimation relies on controlled disturbance experiments, detailed generator dispatch data, or proprietary system operator models. These approaches are typically non-public and static.

This project introduces a PINN-based inference method that combines frequency dynamics equations, demand and generation data, and machine learning to estimate **real-time inertia and system stability metrics** from open data alone.

---

## Novel Contributions

1. Latent inertia inference without proprietary generator dispatch data
2. Physics-informed learning of frequency dynamics via the swing equation
3. Uncertainty-aware stability metric (Jitter Index)
4. Demand shock vulnerability analysis via Monte Carlo simulation
5. Fully open and reproducible framework for grid inertia estimation

---

## Core Physics

Grid frequency dynamics follow the **swing equation**:

```
M(t) · df/dt = P_mech(t) − P_elec(t) − D(t) · (f − f₀)
```

where `M(t)` is the time-varying system inertia constant, `D(t)` is the damping coefficient, and `f₀ = 50 Hz`.

The system inertia is estimated as a generation-weighted average:

```
H_sys(t) = Σ [ H_i · P_i(t) ] / P_total(t)
```

Inverter-based resources (solar PV, offshore wind) contribute zero synchronous inertia.

---

## Grid Vulnerability Metric

The **Jitter Index** quantifies instantaneous vulnerability to disturbances:

```
J(t) = Var(df/dt) / M(t)
```

Higher values indicate reduced stability margins. Monte Carlo demand shocks are used to simulate frequency response under perturbation.

---

## Model Architecture

**Inputs:**
- Demand (actual + forecast)
- Solar, wind onshore, wind offshore generation
- Dispatchable generation (thermal + hydro residual)
- Net power imbalance proxy
- Time features (hour, month, day-of-week)

**Outputs:**
- Estimated inertia `M(t)` / `H_sys(t)`
- Damping coefficient `D(t)`
- RoCoF proxy `df/dt`

**Loss function:**
```
L = L_data + λ_phys · L_swing + λ_reg · L_uncertainty
```

| Term | Description |
|---|---|
| `L_data` | MSE between predicted and proxy-estimated `H_sys` |
| `L_swing` | Swing equation residual (physics constraint) |
| `L_uncertainty` | Smoothness regularisation on inertia estimates |

---

## Data Sources

**Primary:** [Open Power System Data (OPSD)](https://open-power-system-data.org/) — 15-min resolution, 2015–2020.

| Column | Description |
|---|---|
| `DE_load_actual_entsoe_transparency` | German demand (MW) |
| `DE_solar_generation_actual` | Solar PV generation |
| `DE_wind_onshore_generation_actual` | Onshore wind |
| `DE_wind_offshore_generation_actual` | Offshore wind |
| `DE_solar_capacity` | Installed solar capacity |
| `DE_wind_capacity` | Installed wind capacity |

**Optional:** ENTSO-E frequency measurement data (enables direct RoCoF supervision).

---

## Project Structure

```
grid-inertia-pinn/
├── data/
│   └── processed/
│       ├── de_load_15min.csv
│       ├── de_solar_15min.csv
│       ├── de_wind_15min.csv
│       └── de_inertia_15min.csv          ← H_sys proxy + risk indicators
│
├── notebooks/
│   ├── 01_eda.ipynb                      ← load / solar / wind exploration
│   ├── 02_inertia_proxy.ipynb            ← build H_sys from generation mix  ✅
│   ├── 03_pinn_training.ipynb            ← model definition + training loop
│   └── 04_insights.ipynb                 ← RoCoF, Jitter Index, risk windows
│
├── src/
│   ├── inertia.py                        ← H_sys & Jitter Index computation
│   ├── losses.py                         ← swing equation + data + uncertainty loss
│   │
│   ├── models/
│   │   └── pinn.py                       ← PyTorch PINN (MLP + physics head)
│   │
│   └── training/
│       ├── train.py                      ← training loop
│       └── config.py                     ← hyperparameters & λ weights
│
├── configs/
│   └── experiment_v1.yaml                ← reproducible experiment config
│
├── reports/
│   └── figures/                          ← generated plots
│
└── README.md
```

---

## Setup

```bash
pip install torch pandas numpy matplotlib seaborn scikit-learn pyyaml
```

---

## Usage

Run notebooks in order:

```
01_eda  →  02_inertia_proxy  →  03_pinn_training  →  04_insights
```

---

## Expected Outputs

- `H_sys(t)` — estimated inertia time series
- `J(t)` — grid vulnerability (Jitter) index
- Shock-response simulations under Monte Carlo demand perturbations
- Research-grade figures and whitepaper

---

## Future Extensions

- Incorporate battery storage and inverter synthetic inertia
- Extend to EU-wide grid (ENTSO-E interconnections)
- Real-time inertia forecasting pipeline
- Integration with live ENTSO-E Transparency Platform API
