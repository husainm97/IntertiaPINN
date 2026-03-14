# Physics-Informed Neural Network for Estimating German Grid Inertia

## Project Overview

This project develops a **Physics-Informed Neural Network (PINN)** to estimate the *effective rotational inertia of the German power grid* using open data from the **Open Power System Data (OPSD)** platform.

The project aims to:

1. Infer **time-varying grid inertia** from publicly available operational data.
2. Quantify **grid vulnerability to demand shocks** using uncertainty-aware modeling.
3. Provide a reproducible open-source framework for **data-driven power system stability analysis**.

---

## Scientific Motivation

Grid inertia determines how resistant the power system frequency is to disturbances. As renewable penetration increases, synchronous generators are replaced by inverter-based resources, reducing inertia and increasing frequency volatility.

Traditional inertia estimation relies on:

- Controlled disturbance experiments
- Detailed generator dispatch data
- Proprietary system operator models

These approaches are often **non-public and static**.

This project introduces a **PINN-based inference method** that combines:

- Frequency dynamics equations
- Demand and generation data
- Machine learning inference

to estimate **real-time inertia and system stability metrics**.

---

## Data Sources

Primary dataset:

Open Power System Data (OPSD)

Relevant tables:

- time_series_60min_singleindex.csv
- time_series_15min_singleindex.csv

Key columns used:

| Column | Description |
|------|-------------|
| load_actual_entsoe_transparency | German demand |
| generation_wind_onshore | Wind production |
| generation_wind_offshore | Offshore wind |
| generation_solar | Solar generation |
| generation_lignite | Dispatchable generation |
| generation_hard_coal | Dispatchable generation |
| generation_gas | Dispatchable generation |
| generation_nuclear | Baseline synchronous generation |

Additional dataset (optional):

ENTSO‑E frequency data.

---

## Core Idea

Grid frequency dynamics approximately follow the **swing equation**:

M df/dt = P_m − P_e − D(f − f₀)

where

M = system inertia constant  
P_m = mechanical power  
P_e = electrical demand  
D = damping constant

The neural network learns **latent time-varying inertia M(t)** while being constrained by this equation.

---

## Model Architecture

Inputs:

- Demand
- Renewable generation
- Dispatchable generation
- Net imbalance proxy
- Time features

Network outputs:

- Estimated inertia M(t)
- Damping coefficient D(t)
- Frequency deviation prediction

Loss function:

L = L_data + λ_phys L_swing + λ_reg L_uncertainty

Where:

- Data loss compares predicted frequency deviations
- Physics loss enforces swing equation consistency
- Regularization controls smoothness of inertia estimates

---

## Novel Contributions

1. **Latent inertia inference without generator dispatch data**
2. **Physics-informed learning of frequency dynamics**
3. **Uncertainty-aware stability metric**
4. **Demand shock vulnerability analysis**
5. **Open reproducible framework for grid inertia estimation**

---

## Grid Vulnerability Metric

Define the **Jitter Index**:

J(t) = Var(df/dt) / M(t)

Higher values indicate increased vulnerability to disturbances.

Monte Carlo demand shocks simulate stability margins.

---

## Project Structure

src/
    data/            data loaders
    models/          PINN architecture
    training/        training loops
    evaluation/      inertia metrics

notebooks/

    exploratory analysis
    baseline models

reports/

    figures and results

configs/

    experiment configurations

---

## Version 1.0 Goal

A reproducible pipeline that:

1. Downloads OPSD data
2. Processes generation + demand features
3. Trains a PINN model
4. Estimates grid inertia over time
5. Computes a vulnerability metric
6. Produces a public dashboard of results

---

## Expected Outputs

- Estimated inertia time series
- Grid vulnerability index
- Shock-response simulations
- Research-grade whitepaper

---

## Future Extensions

- Incorporate battery and inverter dynamics
- Extend to EU-wide grids
- Real-time inertia forecasting