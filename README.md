# ⚡ Grid Inertia PINN

![PyTorch](https://img.shields.io/badge/PyTorch-deep_learning-orange?logo=pytorch)
![Physics Informed ML](https://img.shields.io/badge/Physics--Informed-ML-blue)
![Stochastic Modeling](https://img.shields.io/badge/Stochastic-Process%20Modeling-green)
![Signal Processing](https://img.shields.io/badge/Signal%20Processing-Savitzky--Golay-purple)
![Time Series](https://img.shields.io/badge/Time%20Series-Analysis-teal)
![Scientific Python](https://img.shields.io/badge/Scientific-Python-yellow?logo=python)

An experimental **Physics-Informed Neural Network (PINN)** for estimating effective grid inertia from publicly available frequency and generation data — without proprietary generator dispatch information.

> ⚠️ **This is exploratory research.** Results are preliminary and should be interpreted with appropriate scepticism. The methodology is novel and not yet validated against ground-truth inertia measurements.

---

## 🔬 What This Is

As renewable penetration increases across European grids, synchronous generators are progressively displaced by inverter-based resources. This reduces rotational inertia and increases sensitivity to frequency disturbances — a growing concern for grid operators.

Standard inertia estimation typically requires controlled disturbance experiments or proprietary system operator models. This project explores whether inertia can be estimated from open data alone, using physics-informed machine learning.

The approach is based on the **stochastic swing equation**:

```
M · df/dt + D · (f − f₀) = ξ(t)
```

where `ξ(t)` is an unobserved stochastic power imbalance process. Rather than requiring `ξ(t)` directly, the model identifies `M` and `D` as the parameter pair for which the residual `R = M·df/dt + D·(f−f₀)` exhibits the statistical properties of white noise — i.e. zero autocorrelation at all lags.

---

## 💡 What the Model Does

Two complementary approaches are implemented:

### 🔍 `InertiaPINN` — per-window analysis (notebook 03)
Trains a small network on a single frequency window to find the `(M, D)` pair that whitens the residual for that window. Slow (requires training per estimate) but useful for detailed analysis of specific time periods.

### 🚀 `InertiaNet` — generalisable real-time estimator (notebook 04)
Trained once on a full year of data. Performs inference on any new frequency window in a single forward pass — sub-millisecond per estimate. This is the intended production-style model.

---

## 📊 Preliminary Findings

These are observations from running the model on 2018–2019 German grid data. They are interesting but **not conclusive without further validation**.  

![InertiaNet inference results](images/04_inference.png)

| Metric | Value |
|--------|-------|
| M_PINN (2019 mean) | 6.23 ± 0.62 MWs/MVA |
| M_table (generation side only) | 3.31 MWs/MVA |
| Apparent load-side contribution | ~2.9 MWs/MVA |
| D (damping) | 0.44 ± 1.09 MW/Hz |
| Inference time | <1ms per window |

**Observations worth noting:**
- 🌙 Inferred M is consistently higher overnight than during peak afternoon hours (Δ ≈ 0.3 MWs/MVA in 2019) — consistent with industrial rotating loads contributing more inertia at night
- 📈 M_PINN is always greater than M_table — the excess (~2.9 MWs/MVA) may represent load-side inertia that the generation table cannot capture
- 📉 A weak negative correlation between M_PINN and renewable fraction is observed — physically expected, though not strongly pronounced in a single year of CE grid data
- 🔄 D has high variance across windows — harder to identify than M under normal grid conditions

These patterns are **physically plausible** but the model has not been validated against independent inertia measurements, so caution is warranted.

---

## ⚙️ Core Physics

**Why not just use the table method?**

The generation-weighted H_sys formula:
```
H_sys(t) = Σ [ H_i · P_i(t) ] / P_total(t)
```

...only counts synchronous generators. It assigns zero to wind, solar, and all load-side rotating machinery. The PINN attempts to recover the full effective inertia from frequency dynamics, without needing generation data at all.

**Why is df/dt hard?**

Finite differences on 1-second PMU data amplify noise by ~50x. The model smooths the frequency trajectory using a Savitzky-Golay filter before computing df/dt — giving a clean derivative without a per-window learned smoother.

---

## 🏗️ Architecture

```
raw f(t) — 3600s window
    ↓  Savitzky-Golay smooth → df/dt  (~54x noise reduction)
    ↓  StandardScaler normalise
    ↓  1D-CNN feature extraction
    ↓  MLP
    ↓
  M (MWs/MVA)    D (MW/Hz)
```

**Training signal:** whiteness of `R = M·df/dt + D·(f−f₀)` across a batch of windows. No labels. No ΔP. Frequency data only.

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
│   ├── 03_pinn_training.ipynb             ← per-window PINN analysis
│   └── 04_inference.ipynb                 ← generalisable real-time model
│
├── data/
│   ├── build_data.py                      ← builds processed CSVs from OPSD
│   └── fetch_frequency_1s.py              ← downloads TransnetBW frequency data
│
└── checkpoints/
    ├── pinn/                              ← InertiaPINN ensemble weights
    └── inertianet/                        ← InertiaNet trained weights
```

---

## 🚀 Setup

```bash
pip install torch pandas numpy matplotlib seaborn scikit-learn scipy
```

---

## 📋 Usage

```bash
# 1. Download 1-second frequency data
python data/fetch_frequency_1s.py --years 2018 2019

# 2. Build processed CSVs from OPSD
python data/build_data.py

# 3. Per-window analysis (slow, detailed)
jupyter notebook notebooks/03_pinn_training.ipynb

# 4. Train generalisable model + inference (fast)
jupyter notebook notebooks/04_inference.ipynb
```

---

## 🔭 Limitations and Open Questions

- **No ground truth validation** — M_PINN has not been compared against event-based inertia estimates from actual frequency disturbances
- **D is poorly constrained** — damping is harder to identify than inertia from ambient frequency data under normal grid conditions
- **Single synchronous area** — the CE grid is large and well-coupled; results may differ for smaller, weaker systems (GB, Nordic) where inertia variation is more pronounced
- **Stationarity assumption** — the stochastic swing equation assumes slowly-varying M and D within each window; this may not hold during rapid renewable ramps
- **Single year of training** — the model was trained on 2018 data only; multi-year training may improve stability of estimates

---

## 🔮 Potential Extensions

- Training hyperparameter tuning for better modelling of f(t) and white noise, current frequency response is somewhat overfit, and residuals have slight bias 
- Validate against ENTSO-E frequency event database (known ΔP + observed RoCoF)
- Extend training to 2015–2020 to capture the full renewable transition
- Compare CE grid results against Nordic/GB grids where inertia variation is larger
- Incorporate battery storage synthetic inertia signals
- Build a live inference pipeline against the ENTSO-E Transparency Platform API

---

## 📚 Data Sources

| Source | Description | Resolution |
|--------|-------------|------------|
| [OPSD Time Series](https://data.open-power-system-data.org/time_series/) | Load, wind, solar generation | 15-min |
| [TransnetBW / OSF](https://osf.io/) | German grid frequency | 1-second |

---

## 📄 License

MIT — experimental research code, use at your own risk.