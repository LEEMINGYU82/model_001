from __future__ import annotations
import argparse, json, os
import pandas as pd
import yaml

from src.common.logger import get_logger
from src.common.io import load_csv, pick_col, detect_timestamp_column, parse_time_index, coerce_numeric
from src.data_process.preprocess import aggregate_on_duplicate_timestamps, aggregate_to_hourly, ensure_hourly_index
from src.data_process.features import make_features
from src.common.metrics import default_day_mask, nmae_with_denom, nmae_kpx
from src.model.train_xgb import fit_xgb, save_model, load_model, predict

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
        raise ValueError(f"타깃(y) 컬럼을 찾지 못했습니다. 후보: {', '.join(cfg['data']['target_col_candidates'])}")
    group_col = pick_col(df, cfg["data"]["group_col_candidates"])
    return time_col, target_col, group_col

def cmd_train(args):
    cfg = load_config(args.config)
    df = load_csv(args.train_csv, parse_dates=cfg["data"]["parse_dates"])
    time_col, target_col, group_col = _resolve_columns(df, cfg)
    df = parse_time_index(df, time_col, tz=cfg["data"]["tz"])
    df = aggregate_on_duplicate_timestamps(df, time_col, agg=cfg["data"]["duplicate_agg"])
    df = aggregate_to_hourly(df, time_col, rule=cfg["data"]["resample_rule"], agg="mean")
    df = ensure_hourly_index(df, time_col, rule=cfg["data"]["resample_rule"])

    # 안전하게 숫자화
    df = coerce_numeric(df, [target_col])

    X, y = make_features(df, time_col=time_col, target_col=target_col, group_col=group_col)
    model, scaler = fit_xgb(X, y, params=cfg["train"]["params"], use_scaler=cfg["train"]["scaler"])
    out_dir = args.out_dir or "./models"
    save_model(model, scaler, out_dir)

def cmd_predict(args):
    cfg = load_config(args.config)
    df = load_csv(args.test_csv, parse_dates=cfg["data"]["parse_dates"])
    time_col, target_col, group_col = _resolve_columns(df, cfg)
    df = parse_time_index(df, time_col, tz=cfg["data"]["tz"])
    df = aggregate_on_duplicate_timestamps(df, time_col, agg=cfg["data"]["duplicate_agg"])
    df = aggregate_to_hourly(df, time_col, rule=cfg["data"]["resample_rule"], agg="mean")
    df = ensure_hourly_index(df, time_col, rule=cfg["data"]["resample_rule"])

    model, scaler = load_model(args.model_dir)
    from src.data_process.features import make_features
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
    # try detect time column
    time_col = pick_col(pred, cfg["data"]["time_col_candidates"]) or "datetime"
    # compute masks
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

    return p

def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
