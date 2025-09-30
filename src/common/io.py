from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional, Sequence, Tuple, List
from .logger import get_logger

logger = get_logger()

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    return df

def pick_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    cols = set(df.columns)
    for cand in candidates:
        if cand in cols:
            return cand
    return None

def detect_timestamp_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            try:
                pd.to_datetime(df[c])
                return c
            except Exception:
                continue
    # fall back: first datetime-like column
    for c in df.columns:
        try:
            pd.to_datetime(df[c])
            return c
        except Exception:
            pass
    return None

def load_csv(path: str, parse_dates: bool = True) -> pd.DataFrame:
    logger.info(f"Loading CSV: {path}")
    df = pd.read_csv(path)
    return standardize_columns(df)

def parse_time_index(df: pd.DataFrame, time_col: str, tz: Optional[str] = None) -> pd.DataFrame:
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    df = df.dropna(subset=[time_col])
    if tz:
        # assume naive is local tz; cast to tz-aware
        df[time_col] = df[time_col].dt.tz_localize(tz, nonexistent='NaT', ambiguous='NaT')
        df = df.dropna(subset=[time_col])
    df = df.sort_values(time_col)
    return df

def coerce_numeric(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df
