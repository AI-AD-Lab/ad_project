import pandas as pd
import numpy as np

def rolling(data:pd.Series, rolling_seconds:int=2, sampling_hz:int=50) -> pd.Series:
    # data smoothing, reduce noise
    if sampling_hz <=0:
        raise ValueError("sampling_hz must be greater than 0")
    if rolling_seconds <0:
        raise ValueError("rolling_seconds must be non-negative")

    df_copy = data.loc[:, ~data.columns.isin(['Entity'])].copy()
    window_size = rolling_seconds * sampling_hz if rolling_seconds > 0 else 1
    df_rolling = df_copy.rolling(int(window_size)).mean().bfill()
    df_rolling['time (sec)'] = df_rolling.index * (1/sampling_hz) # index * 0.02

    return df_rolling