from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from .preprocess import add_dt_features
from ..common.logger import get_logger

logger = get_logger()

def make_features(df: pd.DataFrame, time_col: str, target_col: str, group_col: Optional[str]=None) -> Tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    df = add_dt_features(df, dt_column=time_col)
    # 간단 베이스라인: 시각만 사용 + 이전 시점 타깃 랙(24h)
    df = df.sort_values(time_col)
    if group_col and group_col in df.columns:
        df["y_lag24"] = df.groupby(group_col)[target_col].shift(24)
    else:
        df["y_lag24"] = df[target_col].shift(24)
    # 결측 제거
    feat_cols = ["hour","dow","month","y_lag24"]
    X = df[feat_cols].copy()
    y = df[target_col].copy()
    m = X.notna().all(axis=1) & y.notna()
    X = X[m]
    y = y[m]
    X.columns = [str(c) for c in X.columns]
    return X, y
