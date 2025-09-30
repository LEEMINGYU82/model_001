from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, Any

def default_day_mask(times: pd.Series, start_hour: int = 6, end_hour: int = 18) -> np.ndarray:
    hours = pd.to_datetime(times).dt.hour
    return (hours >= start_hour) & (hours < end_hour)

def compute_denom(y: np.ndarray, strategy: Union[str, float] = "p95-p5") -> float:
    y = np.asarray(y, dtype=float)
    if isinstance(strategy, (int, float)):
        return float(strategy) if strategy != 0 else 1.0
    s = str(strategy).lower()
    if s == "p95-p5":
        d = np.nanpercentile(y, 95) - np.nanpercentile(y, 5)
        return d if d != 0 else 1.0
    if s in ("mean>0", "mean_pos", "mean_pos_only"):
        pos = y[y > 0]
        m = np.nanmean(pos) if pos.size else np.nanmean(y)
        return m if m != 0 else 1.0
    if s == "range":
        d = np.nanmax(y) - np.nanmin(y)
        return d if d != 0 else 1.0
    # fallback
    return 1.0

def nmae_with_denom(y_true: np.ndarray, y_pred: np.ndarray, denom: Union[str,float]="p95-p5", mask: Optional[np.ndarray]=None) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if mask is not None:
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    d = compute_denom(y_true, denom)
    mae = np.nanmean(np.abs(y_true - y_pred))
    return float(mae / d * 100.0)

def nmae_kpx(y_true: np.ndarray, y_pred: np.ndarray, threshold: float=0.10, denom: Union[str,float]="p95-p5") -> float:
    y_true = np.asarray(y_true, dtype=float)
    mask = y_true >= threshold
    return nmae_with_denom(y_true, y_pred, denom=denom, mask=mask)
