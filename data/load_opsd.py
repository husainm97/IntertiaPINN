import pandas as pd

import io
import zipfile
import tempfile
import requests
import pandas as pd


OPSD_DATASETS = {
    "time_series": "https://data.open-power-system-data.org/time_series/opsd-time_series-2020-10-06.zip",
    "conventional_power_plants": "https://data.open-power-system-data.org/conventional_power_plants/opsd-conventional_power_plants-2020-10-01.zip",
    "renewable_power_plants": "https://data.open-power-system-data.org/renewable_power_plants/opsd-renewable_power_plants-2020-08-25.zip",
}


def _download_zip(url: str) -> bytes:
    r = requests.get(url)
    r.raise_for_status()
    return r.content


def _extract_csv_from_zip(zip_bytes: bytes, pattern: str | None = None) -> pd.DataFrame:
    """
    Extract a CSV file from an OPSD zip archive.
    If pattern is given, select the CSV containing that pattern.
    """
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:

        csv_files = [f for f in z.namelist() if f.endswith(".csv")]

        if pattern:
            matches = [f for f in csv_files if pattern in f]
            if not matches:
                raise RuntimeError(f"No CSV matching {pattern}")
            csv_file = matches[0]
        else:
            csv_file = csv_files[0]

        with z.open(csv_file) as f:
            df = pd.read_csv(f, low_memory=False)

    return df


def load_opsd_time_series_15min() -> pd.DataFrame:
    """
    Fetch OPSD time_series dataset and load the 15-minute resolution dataframe.
    """
    zip_bytes = _download_zip(OPSD_DATASETS["time_series"])

    df = _extract_csv_from_zip(zip_bytes, pattern="15min")

    df["utc_timestamp"] = pd.to_datetime(df["utc_timestamp"])
    df = df.set_index("utc_timestamp").sort_index()

    return df


def load_opsd_power_plants():
    """
    Load renewable and conventional power plant datasets.
    """
    renewable_zip = _download_zip(OPSD_DATASETS["renewable_power_plants"])
    conventional_zip = _download_zip(OPSD_DATASETS["conventional_power_plants"])

    renewable_df = _extract_csv_from_zip(renewable_zip)
    conventional_df = _extract_csv_from_zip(conventional_zip)

    return renewable_df, conventional_df


def load_opsd():
    """
    Convenience loader returning all datasets used in the project.
    """

    ts = load_opsd_time_series_15min()
    renewable, conventional = load_opsd_power_plants()

    return {
        "time_series_15min": ts,
        "renewable_power_plants": renewable,
        "conventional_power_plants": conventional,
    }




data = load_opsd()

ts = data["time_series_15min"]
renewable = data["renewable_power_plants"]
conventional = data["conventional_power_plants"]

print(ts)
print(renewable)
print(conventional)