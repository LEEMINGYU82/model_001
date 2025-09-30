from __future__ import annotations
import os, json, pickle
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from ..common.logger import get_logger

logger = get_logger()

def fit_xgb(X: pd.DataFrame, y: pd.Series, params: Dict[str, Any], use_scaler: bool=True):
    scaler = None
    X_train = X
    if use_scaler:
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)
        # store feature names for later
        scaler.feature_names_in_ = list(X.columns)
    model = XGBRegressor(**params)
    model.fit(X_train, y)
    # store feature names
    model.feature_names_in_ = list(X.columns)
    return model, scaler

def save_model(model, scaler, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "xgb_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    if scaler is not None:
        with open(os.path.join(out_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)
    logger.info(f"Saved model/scaler to: {out_dir}")

def load_model(model_dir: str):
    with open(os.path.join(model_dir, "xgb_model.pkl"), "rb") as f:
        model = pickle.load(f)
    scaler = None
    sp = os.path.join(model_dir, "scaler.pkl")
    if os.path.exists(sp):
        with open(sp, "rb") as f:
            scaler = pickle.load(f)
    return model, scaler

def predict(model, scaler, X: pd.DataFrame) -> np.ndarray:
    Xp = X
    if scaler is not None:
        Xp = pd.DataFrame(scaler.transform(X), index=X.index, columns=X.columns)
    return model.predict(Xp)
