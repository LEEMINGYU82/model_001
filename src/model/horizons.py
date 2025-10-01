# src/model/horizons.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from ..data_process.features import make_features
from ..common.logger import get_logger

logger = get_logger()

def persistence_series(df: pd.DataFrame, time_col: str, target_col: str, H: int, group_col: Optional[str] = None) -> pd.Series:
    if group_col and group_col in df.columns:
        return df.groupby(group_col)[target_col].shift(H)
    return df[target_col].shift(H)

def make_residual_training(
    df: pd.DataFrame,
    time_col: str,
    target_col: str,
    H: int,
    group_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Δy = y - y_pers(H)를 타깃으로 학습 데이터셋 생성"""
    y = df[target_col]
    y_pers = persistence_series(df, time_col, target_col, H, group_col)
    dy = y - y_pers
    X, _ = make_features(df, time_col=time_col, target_col=target_col, group_col=group_col)
    m = (~dy.isna()) & X.notna().all(axis=1)
    return X.loc[m], dy.loc[m]

def make_residual_predict_inputs(
    df: pd.DataFrame,
    time_col: str,
    target_col: str,
    H: int,
    group_col: Optional[str] = None,
):
    """예측시 Δŷ 입력 X와 y_pers(H) 반환(인덱스 정렬)"""
    X, _ = make_features(df, time_col=time_col, target_col=target_col, group_col=group_col)
    y_pers = persistence_series(df, time_col, target_col, H, group_col)
    y_pers = y_pers.loc[X.index]
    return X, y_pers
