# src/data_process/features.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Iterable
from .preprocess import add_dt_features
from ..common.logger import get_logger

logger = get_logger()

def make_features(
    df: pd.DataFrame,
    time_col: str,
    target_col: str,
    group_col: Optional[str] = None,
    *,
    # 잔차 파이프라인에 맞춘 기본값
    target_lags: Iterable[int] = (24,),            # 타깃 랙: 24h
    exog_candidates: Optional[Iterable[str]] = None,
    exog_lags: Iterable[int] = (1, 3),             # 외생 변수 얕은 랙: 1h, 3h
    include_current_exog: bool = True              # 외생 변수의 현재값 포함
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    - 기본: hour, dow, month + y_lag24
    - 외생 변수(있으면 자동) + 얕은 랙(1,3) 생성
    - 그룹별 랙 계산 지원
    - 반환: (X, y)  — NaN 정리 및 인덱스 정렬 일관화
    """
    df = df.copy()

    # 시간 파생
    df = add_dt_features(df, dt_column=time_col)
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.sort_values(time_col)

    # 외생 변수 후보 (없으면 기본 셋)
    if exog_candidates is None:
        exog_candidates = [
            # 태양/복사
            "altitude", "azimuth", "ghi", "dhi",
            # 기상
            "temp", "humidity", "pressure",
            "wind_speed", "wind_u", "wind_v",
            "cloud_total", "cloud_low", "cloud_med", "cloud_high", "cloud_vlow"
        ]

    # 숫자 변환(있을 때만)
    for c in [target_col, *exog_candidates]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 타깃 랙 생성
    def _shift(series: pd.Series, k: int) -> pd.Series:
        if group_col and group_col in df.columns:
            return series.groupby(df[group_col]).shift(k)
        return series.shift(k)

    for lag in target_lags:
        df[f"{target_col}_lag{lag}"] = _shift(df[target_col], lag)

    # 외생 변수 현재/랙 생성
    exog_used = []
    for c in exog_candidates:
        if c not in df.columns:
            continue
        if include_current_exog:
            exog_used.append(c)
        for lag in exog_lags:
            name = f"{c}_lag{lag}"
            df[name] = _shift(df[c], lag)
            exog_used.append(name)

    # 기본 피처 + 타깃 랙 + 외생 변수/랙
    base_feats = ["hour", "dow", "month"]
    tgt_feats = [f"{target_col}_lag{lag}" for lag in target_lags]
    feat_cols = base_feats + tgt_feats + exog_used

    # 존재하는 컬럼만 선택(안전)
    feat_cols = [c for c in feat_cols if c in df.columns]

    # X, y 생성 + NaN 마스킹
    X = df[feat_cols].copy()
    y = df[target_col].copy()
    m = X.notna().all(axis=1) & y.notna()
    X = X[m]
    y = y[m]

    # 컬럼명을 문자열로 고정(스케일러/모델 호환)
    X.columns = [str(c) for c in X.columns]

    logger.info(f"[make_features] n_rows={len(X)} | n_feats={len(X.columns)} | "
                f"target_lags={list(target_lags)} | exog_included={len(exog_used)}")
    return X, y
