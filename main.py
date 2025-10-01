# main.py
from __future__ import annotations
import argparse, json, os
import numpy as np
import pandas as pd
import yaml

from src.common.logger import get_logger
from src.common.io import (
    load_csv, pick_col, detect_timestamp_column,
    parse_time_index, coerce_numeric
)
from src.common.metrics import compute_denom, nmae_with_denom

from src.data_process.preprocess import (
    aggregate_on_duplicate_timestamps, aggregate_to_hourly, ensure_hourly_index
)
from src.data_process.features import make_features
from src.common.metrics import default_day_mask, nmae_with_denom, nmae_kpx
from src.model.train_xgb import fit_xgb, save_model, load_model, predict
from src.model.horizons import (
    make_residual_training, make_residual_predict_inputs
)

logger = get_logger()

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _resolve_columns(df: pd.DataFrame, cfg: dict):
    time_col = detect_timestamp_column(df, cfg["data"]["time_col_candidates"])
    if time_col is None:
        raise ValueError(f"시간 컬럼을 찾지 못했습니다. 후보: {cfg['data']['time_col_candidates']}")
    target_col = pick_col(df, cfg["data"]["target_col_candidates"])
    if target_col is None:
        raise ValueError("타깃(y) 컬럼을 찾지 못했습니다. 후보: " + ", ".join(cfg["data"]["target_col_candidates"]))
    group_col = pick_col(df, cfg["data"]["group_col_candidates"])
    return time_col, target_col, group_col

# ---------------------------
# Baseline train / predict / eval (기존)
# ---------------------------
def _prep_df(cfg, df, time_col):
    df = parse_time_index(df, time_col, tz=cfg["data"]["tz"])
    df = aggregate_on_duplicate_timestamps(df, time_col, agg=cfg["data"]["duplicate_agg"])
    df = aggregate_to_hourly(df, time_col, rule=cfg["data"]["resample_rule"], agg="mean")
    df = ensure_hourly_index(df, time_col, rule=cfg["data"]["resample_rule"])
    return df

def cmd_train(args):
    cfg = load_config(args.config)
    df = load_csv(args.train_csv, parse_dates=cfg["data"]["parse_dates"])
    time_col, target_col, group_col = _resolve_columns(df, cfg)
    df = _prep_df(cfg, df, time_col)
    df = coerce_numeric(df, [target_col])

    X, y = make_features(df, time_col=time_col, target_col=target_col, group_col=group_col)
    model, scaler = fit_xgb(X, y, params=cfg["train"]["params"], use_scaler=cfg["train"]["scaler"])
    out_dir = args.out_dir or "./models"
    save_model(model, scaler, out_dir)

def cmd_predict(args):
    cfg = load_config(args.config)
    df = load_csv(args.test_csv, parse_dates=cfg["data"]["parse_dates"])
    time_col, target_col, group_col = _resolve_columns(df, cfg)
    df = _prep_df(cfg, df, time_col)

    model, scaler = load_model(args.model_dir)
    X, y = make_features(df, time_col=time_col, target_col=target_col, group_col=group_col)

    y_hat = predict(model, scaler, X)
    out = df.loc[X.index, [time_col]].copy()
    out["y_true"] = y.values
    out["y_pred"] = y_hat
    if group_col and group_col in df.columns:
        out[group_col] = df.loc[X.index, group_col].values
    out_path = args.out_csv or "./outputs/preds.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8")
    logger.info(f"Saved predictions to {out_path}")

def cmd_eval(args):
    cfg = load_config(args.config)
    pred = pd.read_csv(args.pred_csv)
    time_col = pick_col(pred, cfg["data"]["time_col_candidates"]) or "datetime"
    mask = None
    if cfg["eval"]["day_only"]["enabled"]:
        mask = default_day_mask(pred[time_col], start_hour=cfg["eval"]["day_only"]["start_hour"], end_hour=cfg["eval"]["day_only"]["end_hour"])
    nmae = nmae_with_denom(pred["y_true"].values, pred["y_pred"].values, denom=cfg["eval"]["denom"], mask=mask)
    nmae_k = nmae_kpx(pred["y_true"].values, pred["y_pred"].values, threshold=cfg["eval"]["kpx_threshold"], denom=cfg["eval"]["denom"])
    report = {"nMAE_%": nmae, "nMAE_KPX_%": nmae_k}
    out = args.report or "./outputs/metrics.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved metrics to {out}: {report}")

# ---------------------------
# Residual pipeline (노트북 로직과 동일)
# ---------------------------
def _predict_residual_one(df, time_col, target_col, H, model_dir, group_col=None, clip=False, w=1.0):
    """퍼시스턴스 y_pers(H) + Δŷ(H)로 y_res, y_blend 생성"""
    model, scaler = load_model(model_dir)
    X, y_pers = make_residual_predict_inputs(df, time_col, target_col, H, group_col=group_col)
    dy_hat = predict(model, scaler, X)
    y_pers_v = y_pers.values
    y_res   = y_pers_v + dy_hat
    y_blend = y_pers_v + float(w) * dy_hat
    if clip:
        y_pers_v = np.clip(y_pers_v, 0.0, 1.0)
        y_res    = np.clip(y_res,   0.0, 1.0)
        y_blend  = np.clip(y_blend, 0.0, 1.0)
    return X.index, y_pers_v, y_res, y_blend

def cmd_train_resid(args):
    cfg = load_config(args.config)
    df = load_csv(args.train_csv, parse_dates=cfg["data"]["parse_dates"])
    time_col, target_col, group_col = _resolve_columns(df, cfg)
    df = _prep_df(cfg, df, time_col)

    H = int(args.horizon)
    X, dy = make_residual_training(df, time_col, target_col, H, group_col=group_col)
    model, scaler = fit_xgb(X, dy, params=cfg["train"]["params"], use_scaler=cfg["train"]["scaler"])
    out_dir = args.out_dir or f"./models/H{H}"
    os.makedirs(out_dir, exist_ok=True)
    save_model(model, scaler, out_dir)
    logger.info(f"Saved residual Δy model for H={H} to {out_dir}")

def cmd_predict_resid(args):
    cfg = load_config(args.config)
    df = load_csv(args.test_csv, parse_dates=cfg["data"]["parse_dates"])
    time_col, target_col, group_col = _resolve_columns(df, cfg)
    df = _prep_df(cfg, df, time_col)

    out = df[[time_col]].copy()
    if group_col and group_col in df.columns:
        out[group_col] = df[group_col].values
    if target_col in df.columns:
        out["y_true"] = df[target_col].values

    # H=1
    if args.h1_model_dir:
        idx, pers, res, blend = _predict_residual_one(
            df, time_col, target_col, 1, args.h1_model_dir, group_col=group_col, clip=args.clip, w=args.blend_w1
        )
        out.loc[idx, "y_pred_pers_h1"]  = pers
        out.loc[idx, "y_pred_res_h1"]   = res
        out.loc[idx, "y_pred_blend_h1"] = blend

    # H=3
    if args.h3_model_dir:
        idx, pers, res, blend = _predict_residual_one(
            df, time_col, target_col, 3, args.h3_model_dir, group_col=group_col, clip=args.clip, w=args.blend_w3
        )
        out.loc[idx, "y_pred_pers_h3"]  = pers
        out.loc[idx, "y_pred_res_h3"]   = res
        out.loc[idx, "y_pred_blend_h3"] = blend

    out_path = args.out_csv or "./outputs/preds_h13.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8")
    logger.info(f"Saved residual predictions to {out_path}")

# ---------------------------
# parser
# ---------------------------
def build_parser():
    p = argparse.ArgumentParser(description="gen_rate_ported CLI")
    p.add_argument("--config", default="configs/config.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train", help="학습")
    pt.add_argument("--train_csv", required=True)
    pt.add_argument("--out_dir", default="./models")
    pt.set_defaults(func=cmd_train)

    pp = sub.add_parser("predict", help="예측")
    pp.add_argument("--test_csv", required=True)
    pp.add_argument("--model_dir", default="./models")
    pp.add_argument("--out_csv", default="./outputs/preds.csv")
    pp.set_defaults(func=cmd_predict)

    pe = sub.add_parser("eval", help="평가")
    pe.add_argument("--pred_csv", required=True)
    pe.add_argument("--report", default="./outputs/metrics.json")
    pe.set_defaults(func=cmd_eval)

    tr = sub.add_parser("train_resid", help="잔차(Δy) 학습 - horizon별")
    tr.add_argument("--train_csv", required=True)
    tr.add_argument("--horizon", type=int, choices=[1,3], required=True)
    tr.add_argument("--out_dir", default=None)
    tr.set_defaults(func=cmd_train_resid)

    pr = sub.add_parser("predict_resid", help="퍼시스턴스/잔차/블렌드 예측 저장")
    pr.add_argument("--test_csv", required=True)
    pr.add_argument("--h1_model_dir", default=None)
    pr.add_argument("--h3_model_dir", default=None)
    pr.add_argument("--blend_w1", type=float, default=1.0)
    pr.add_argument("--blend_w3", type=float, default=1.0)
    pr.add_argument("--clip", action="store_true")
    pr.add_argument("--out_csv", default="./outputs/preds_h13.csv")
    pr.set_defaults(func=cmd_predict_resid)

    return p

def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
