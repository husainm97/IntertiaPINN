import pandas as pd

def load_opsd(path):
    df = pd.read_csv(path, parse_dates=["utc_timestamp"])
    df = df.set_index("utc_timestamp")
    return df