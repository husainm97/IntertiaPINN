"""
data/fetch_frequency_1s.py

Downloads raw 1-second German grid frequency data from the Power-Grid-Frequency
database (TransnetBW / OSF) and saves yearly CSVs at full 1-second resolution.

Output : data/raw/de_frequency_1s_{year}.csv
         columns: utc_timestamp (index), f_hz, f_dev_mhz

Usage
-----
    python data/fetch_frequency_1s.py --years 2019
    python data/fetch_frequency_1s.py --years 2015 2016 2017 2018 2019 2020
"""

import argparse
import io
import time
from pathlib import Path

import pandas as pd
import requests

F0 = 50.0

URLS = {
    (2015,  1): "https://osf.io/download/5eecfc146598280108cf0bc0/",
    (2015,  2): "https://osf.io/download/5eecfc4b65982800fbcf4963/",
    (2015,  3): "https://osf.io/download/5eecfc6c145b1a011352fad0/",
    (2015,  4): "https://osf.io/download/5eecfc586598280107cf0b84/",
    (2015,  5): "https://osf.io/download/5eecfc5976ebd80110cd6025/",
    (2015,  6): "https://osf.io/download/5eecfc74145b1a011352fadf/",
    (2015,  7): "https://osf.io/download/5eecfccc6598280102cf2dc4/",
    (2015,  8): "https://osf.io/download/5eecfcd576ebd8010bcd6b31/",
    (2015,  9): "https://osf.io/download/5eecfcdb76ebd8010ccd6fe1/",
    (2015, 10): "https://osf.io/download/5eecfce765982800fbcf49c7/",
    (2015, 11): "https://osf.io/download/5eecfcea6598280108cf0de5/",
    (2015, 12): "https://osf.io/download/5eecfcfc65982800facf3968/",
    (2016,  1): "https://osf.io/download/5eecff4876ebd8010ccd729a/",
    (2016,  2): "https://osf.io/download/5eecff5b145b1a011252ecb8/",
    (2016,  3): "https://osf.io/download/5eecff756598280108cf1349/",
    (2016,  4): "https://osf.io/download/5eecff7565982800fbcf4bb1/",
    (2016,  5): "https://osf.io/download/5eecff80145b1a010c52ef60/",
    (2016,  6): "https://osf.io/download/5eecffc565982800fbcf4c31/",
    (2016,  7): "https://osf.io/download/5eecffdc65982800fbcf4c62/",
    (2016,  8): "https://osf.io/download/5eecfff3145b1a011a52bf6d/",
    (2016,  9): "https://osf.io/download/5eecfffb145b1a011a52bf84/",
    (2016, 10): "https://osf.io/download/5eed00026598280107cf1119/",
    (2016, 11): "https://osf.io/download/5eed001c145b1a011a52bfda/",
    (2016, 12): "https://osf.io/download/5eed002176ebd8010ccd7427/",
    (2017,  1): "https://osf.io/download/5eed010476ebd80111cd6b07/",
    (2017,  2): "https://osf.io/download/5eed010e76ebd80111cd6b2f/",
    (2017,  3): "https://osf.io/download/5eed01276598280107cf14a4/",
    (2017,  4): "https://osf.io/download/5eed013876ebd80110cd6866/",
    (2017,  5): "https://osf.io/download/5eed01426598280108cf19f6/",
    (2017,  6): "https://osf.io/download/5eed018376ebd80110cd68c3/",
    (2017,  7): "https://osf.io/download/5eed0192145b1a011252effb/",
    (2017,  8): "https://osf.io/download/5eed01a9145b1a011a52c2f6/",
    (2017,  9): "https://osf.io/download/5eed01b476ebd80110cd68ee/",
    (2017, 10): "https://osf.io/download/5eed01be145b1a011b52c4f4/",
    (2017, 11): "https://osf.io/download/5eed01d676ebd8010ccd76ed/",
    (2017, 12): "https://osf.io/download/5eed01da6598280108cf1b0e/",
    (2018,  1): "https://osf.io/download/5eef68cf145b1a017152fb5b/",
    (2018,  2): "https://osf.io/download/5eef68d66598280144cf81ce/",
    (2018,  3): "https://osf.io/download/5eef68de76ebd80167ce0314/",
    (2018,  4): "https://osf.io/download/5eef6aed6598280144cf8409/",
    (2018,  5): "https://osf.io/download/5eef6b12145b1a017152fe2b/",
    (2018,  6): "https://osf.io/download/5eef6b16659828014ccf3f81/",
    (2018,  7): "https://osf.io/download/5eef6b1f145b1a0172532152/",
    (2018,  8): "https://osf.io/download/5eef6b28145b1a016e52f394/",
    (2018,  9): "https://osf.io/download/5eef6b6176ebd8015fce125c/",
    (2018, 10): "https://osf.io/download/5eef6b78145b1a0169531540/",
    (2018, 11): "https://osf.io/download/5eef6b79659828013bcfd24f/",
    (2018, 12): "https://osf.io/download/5eef6b7b76ebd80164cdcb3e/",
    (2019,  1): "https://osf.io/download/5eef6c026598280144cf84f5/",
    (2019,  2): "https://osf.io/download/5eef6c18145b1a0172532349/",
    (2019,  3): "https://osf.io/download/5eef6c3276ebd80167ce081e/",
    (2019,  4): "https://osf.io/download/5eef6c356598280144cf8529/",
    (2019,  5): "https://osf.io/download/5eef6c4076ebd80167ce0837/",
    (2019,  6): "https://osf.io/download/5eef6c7876ebd80166cdde55/",
    (2019,  7): "https://osf.io/download/5eef6c88659828014ccf4191/",
    (2019,  8): "https://osf.io/download/5eef6cb2145b1a017253244e/",
    (2019,  9): "https://osf.io/download/5eef6cb7659828014ccf41e1/",
    (2019, 10): "https://osf.io/download/5eef6cbc659828014ccf41f6/",
    (2019, 11): "https://osf.io/download/5eef6cf676ebd80164cdcc98/",
    (2019, 12): "https://osf.io/download/5eef6d086598280144cf85f3/",
    (2020,  1): "https://osf.io/download/5eef6d1d76ebd8015fce13df/",
    (2020,  2): "https://osf.io/download/5eef6d1b76ebd80166cddf16/",
    (2020,  3): "https://osf.io/download/5eef6d21145b1a017253251c/",
}

AVAILABLE_YEARS = sorted(set(y for y, _ in URLS))


def fetch_month(year: int, month: int) -> pd.DataFrame | None:
    url = URLS.get((year, month))
    if url is None:
        print(f"  {year}-{month:02d} — no URL, skipping")
        return None
    print(f"  {year}-{month:02d} ...", end=" ", flush=True)
    try:
        r = requests.get(url, timeout=120, allow_redirects=True)
        r.raise_for_status()
        raw = pd.read_csv(
            io.BytesIO(r.content), compression="zip",
            index_col=0, parse_dates=True,
            names=["utc_timestamp", "f_dev_mhz"], header=0,
        )
        raw.index.name = "utc_timestamp"
        raw["f_hz"] = F0 + raw["f_dev_mhz"] / 1000.0
        raw = raw.sort_index()[["f_hz", "f_dev_mhz"]]
        print(f"OK ({len(raw):,} rows)")
        return raw
    except Exception as e:
        print(f"FAILED ({e})")
        return None


def fetch_year(year: int, out_dir: Path) -> Path | None:
    out_path = out_dir / f"de_frequency_1s_{year}.csv"
    if out_path.exists():
        print(f"  {out_path.name} already exists — skipping")
        return out_path

    print(f"\nFetching {year}...")
    chunks = []
    for month in range(1, 13):
        df = fetch_month(year, month)
        if df is not None:
            chunks.append(df)
        time.sleep(0.3)

    if not chunks:
        return None

    df_year = pd.concat(chunks).sort_index()
    if df_year.index.tz is None:
        df_year.index = df_year.index.tz_localize("UTC")
    df_year = df_year[~df_year.index.duplicated(keep="first")]
    df_year.to_csv(out_path)
    print(f"  Saved {out_path.name} ({len(df_year):,} rows, "
          f"{out_path.stat().st_size/1e6:.1f} MB)")
    return out_path


def main(years: list[int]):
    out_dir = Path(__file__).parent / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    for year in years:
        if year not in AVAILABLE_YEARS:
            print(f"Year {year} not available")
            continue
        fetch_year(year, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=int, default=[2019])
    args = parser.parse_args()
    main(args.years)