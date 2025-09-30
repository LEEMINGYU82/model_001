from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Literal, Optional
from ..common.logger import get_logger

logger = get_logger()

def aggregate_to_hourly(df: pd.DataFrame, time_col: str, rule: str = "1H", agg: str = "mean") -> pd.DataFrame:
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        raise ValueError(f"{time_col} must be datetime dtype")
    df = df.set_index(time_col).sort_index()
    if agg == "mean":
        out = df.resample(rule).mean(numeric_only=True)
    elif agg == "sum":
        out = df.resample(rule).sum(numeric_only=True)
    else:
        out = df.resample(rule).agg(agg)
    return out.reset_index()

def aggregate_on_duplicate_timestamps(df: pd.DataFrame, time_col: str, agg: str = "mean") -> pd.DataFrame:
    if agg == "mean":
        return (df.groupby(time_col, as_index=False).mean(numeric_only=True))
    elif agg == "sum":
        return (df.groupby(time_col, as_index=False).sum(numeric_only=True))
    else:
        return (df.groupby(time_col, as_index=False).agg(agg))

def ensure_hourly_index(df: pd.DataFrame, time_col: str, rule: str="1H") -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(time_col)
    idx = pd.date_range(df[time_col].min(), df[time_col].max(), freq=rule)
    out = df.set_index(time_col).reindex(idx).rename_axis(time_col).reset_index()
    return out

def add_dt_features(df: pd.DataFrame, dt_column: str = "datetime") -> pd.DataFrame:
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[dt_column]):
        df[dt_column] = pd.to_datetime(df[dt_column], errors='coerce')
    df["hour"] = df[dt_column].dt.hour
    df["dow"] = df[dt_column].dt.dayofweek
    df["month"] = df[dt_column].dt.month
    return df
