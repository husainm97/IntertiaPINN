"""
data/build_data.py

Builds all processed CSVs needed for pinn_training.ipynb from the raw
OPSD time series download.

Outputs (all in data/processed/)
---------------------------------
de_load_15min.csv       — actual + forecast load
de_solar_15min.csv      — solar capacity + generation
de_wind_15min.csv       — wind capacity + generation (on/offshore)
de_inertia_15min.csv    — H_sys proxy, P_total, P_load, delta_P

Usage
-----
    python data/build_data.py

Requires the OPSD loader to be importable (paste your load_opsd() function
into opsd_loader.py in the same directory, or run this after load_opsd()
is already in scope).
"""

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ── Constants ─────────────────────────────────────────────────────────────────
F0 = 50.0

# Inertia constants per technology [MWs/MVA]
H_CONSTANTS = {
    "nuclear":         6.0,
    "lignite":         4.5,
    "hard_coal":       4.0,
    "gas":             4.0,
    "hydro":           3.0,
    "pumped_storage":  3.0,
    "wind_onshore":    0.0,
    "wind_offshore":   0.0,
    "solar":           0.0,
    "other":           2.0,
}

DATA_DIR = Path(__file__).parent / "processed"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ── OPSD downloader (self-contained) ─────────────────────────────────────────

def _download_zip(url: str) -> bytes:
    print(f"  Downloading {url.split('/')[-1]}...")
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    return r.content


def _csv_from_zip(zip_bytes: bytes, pattern: str) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        matches = [f for f in z.namelist()
                   if f.endswith(".csv") and pattern in f]
        if not matches:
            raise RuntimeError(f"No CSV matching '{pattern}' in zip")
        with z.open(matches[0]) as f:
            return pd.read_csv(f, low_memory=False)


def load_time_series() -> pd.DataFrame:
    url = ("https://data.open-power-system-data.org/time_series/"
           "opsd-time_series-2020-10-06.zip")
    zb  = _download_zip(url)
    df  = _csv_from_zip(zb, "15min")
    df["utc_timestamp"] = pd.to_datetime(df["utc_timestamp"], utc=True)
    df  = df.set_index("utc_timestamp").sort_index()
    # Keep only DE national columns (not TSO sub-regions)
    keep = [c for c in df.columns if c.startswith("DE_")
            and not any(tso in c for tso in
                        ["50hertz","amprion","tennet","transnetbw","LU"])]
    return df[keep]


# ── Build processed files ─────────────────────────────────────────────────────

def build_load(df: pd.DataFrame) -> pd.DataFrame:
    cols = {
        "DE_load_actual_entsoe_transparency":   "P_load",
        "DE_load_forecast_entsoe_transparency": "P_load_forecast",
    }
    out = df[[c for c in cols if c in df.columns]].rename(columns=cols)
    out.to_csv(DATA_DIR / "de_load_15min.csv")
    print(f"  de_load_15min.csv          {len(out):,} rows")
    return out


def build_solar(df: pd.DataFrame) -> pd.DataFrame:
    cols = {
        "DE_solar_capacity":          "solar_capacity_mw",
        "DE_solar_generation_actual": "P_solar",
        "DE_solar_profile":           "solar_profile",
    }
    out = df[[c for c in cols if c in df.columns]].rename(columns=cols)
    out.to_csv(DATA_DIR / "de_solar_15min.csv")
    print(f"  de_solar_15min.csv         {len(out):,} rows")
    return out


def build_wind(df: pd.DataFrame) -> pd.DataFrame:
    cols = {
        "DE_wind_capacity":                  "wind_capacity_mw",
        "DE_wind_generation_actual":         "P_wind",
        "DE_wind_onshore_capacity":          "wind_onshore_capacity_mw",
        "DE_wind_onshore_generation_actual": "P_wind_onshore",
        "DE_wind_offshore_capacity":         "wind_offshore_capacity_mw",
        "DE_wind_offshore_generation_actual":"P_wind_offshore",
    }
    out = df[[c for c in cols if c in df.columns]].rename(columns=cols)
    out.to_csv(DATA_DIR / "de_wind_15min.csv")
    print(f"  de_wind_15min.csv          {len(out):,} rows")
    return out


def build_inertia(
    df:     pd.DataFrame,
    load:   pd.DataFrame,
    solar:  pd.DataFrame,
    wind:   pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the inertia proxy CSV needed by pinn_training.ipynb.

    Columns produced
    ----------------
    P_load        : actual load [MW]
    P_solar       : solar generation [MW]
    P_wind_onshore: onshore wind [MW]
    P_wind_offshore: offshore wind [MW]
    P_thermal     : residual thermal (load - renewable - hydro) [MW]
    P_hydro       : hydro proxy [MW]  (0 — not in OPSD DE national)
    P_total       : total generation estimate [MW]
    delta_P       : P_total - P_load [MW]
    renewables_fraction : (solar + wind) / P_total
    H_sys         : generation-mix inertia proxy [MWs/MVA]
    RoCoF_proxy   : estimated RoCoF from swing eq [Hz/s]
    f_dev_hz      : placeholder 0 (filled from frequency data later)
    """
    out = pd.DataFrame(index=df.index)

    out["P_load"]         = load.get("P_load",         pd.Series(np.nan, index=df.index))
    out["P_solar"]        = solar.get("P_solar",        pd.Series(0.0,   index=df.index)).fillna(0)
    out["P_wind_onshore"] = wind.get("P_wind_onshore",  pd.Series(0.0,   index=df.index)).fillna(0)
    out["P_wind_offshore"]= wind.get("P_wind_offshore", pd.Series(0.0,   index=df.index)).fillna(0)
    out["P_hydro"]        = 0.0

    # Thermal = load minus all known renewables (floor at 0)
    out["P_thermal"] = (
        out["P_load"].fillna(0)
        - out["P_solar"]
        - out["P_wind_onshore"]
        - out["P_wind_offshore"]
    ).clip(lower=0)

    out["P_total"] = (
        out["P_thermal"]
        + out["P_solar"]
        + out["P_wind_onshore"]
        + out["P_wind_offshore"]
    ).replace(0, np.nan)

    out["delta_P"] = out["P_total"] - out["P_load"].fillna(out["P_total"])

    out["renewables_fraction"] = (
        (out["P_solar"] + out["P_wind_onshore"] + out["P_wind_offshore"])
        / out["P_total"].replace(0, np.nan)
    ).clip(0, 1)

    # H_sys proxy: weighted average of inertia constants
    H_weighted = (
        H_CONSTANTS["gas"]           * out["P_thermal"]       +
        H_CONSTANTS["solar"]         * out["P_solar"]         +
        H_CONSTANTS["wind_onshore"]  * out["P_wind_onshore"]  +
        H_CONSTANTS["wind_offshore"] * out["P_wind_offshore"]
    )
    out["H_sys"] = (H_weighted / out["P_total"].replace(0, np.nan)).clip(0, 15)

    # RoCoF proxy from swing equation (ΔP known, H assumed)
    out["RoCoF_proxy"] = (
        out["delta_P"]
        / (2.0 * out["H_sys"].replace(0, np.nan)
           * out["P_total"].replace(0, np.nan)
           * F0)
    ).fillna(0)

    out["f_dev_hz"] = 0.0   # placeholder — filled from 1s frequency data at runtime

    out.to_csv(DATA_DIR / "de_inertia_15min.csv")
    print(f"  de_inertia_15min.csv       {len(out):,} rows")
    print(f"    H_sys  : {out['H_sys'].mean():.3f} ± {out['H_sys'].std():.3f} MWs/MVA")
    print(f"    P_total: {out['P_total'].mean():.0f} ± {out['P_total'].std():.0f} MW")
    print(f"    ren%   : {out['renewables_fraction'].mean()*100:.1f}%")
    return out


def main():
    print("Building processed data files...")
    print("="*50)

    print("\n[1/5] Downloading OPSD time series...")
    df = load_time_series()
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} DE columns")

    print("\n[2/5] Building load CSV...")
    load = build_load(df)

    print("\n[3/5] Building solar CSV...")
    solar = build_solar(df)

    print("\n[4/5] Building wind CSV...")
    wind = build_wind(df)

    print("\n[5/5] Building inertia proxy CSV...")
    build_inertia(df, load, solar, wind)

    print("\n" + "="*50)
    print("All files saved to data/processed/:")
    for f in sorted(DATA_DIR.glob("*.csv")):
        print(f"  {f.name:<35} {f.stat().st_size/1024:>8.1f} KB")
    print("\nNext: python data/fetch_frequency_1s.py --years 2019")


if __name__ == "__main__":
    main()