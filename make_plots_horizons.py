from __future__ import annotations
import argparse, re
import pandas as pd
from typing import List
from src.visualize.plots import plot_actual_vs_pred
from src.common.io import pick_col

def _parse_list(s: str|None)->List[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]

def _autodetect_cols(cols, keys):
    pat = re.compile("|".join([re.escape(k) for k in keys]), re.I)
    return [c for c in cols if pat.search(c)]

def _pretty(name: str):
    n = name.replace("_", " ")
    n = re.sub(r"\bh1\b","H=1h",n, flags=re.I)
    n = re.sub(r"\bh3\b","H=3h",n, flags=re.I)
    n = n.replace("pers","persistence").replace("res","residual")
    return n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", required=True)
    ap.add_argument("--out_dir", default="./outputs/plots")
    ap.add_argument("--time_col", default=None)
    ap.add_argument("--title_prefix", default="[TRAIN] overlay")
    ap.add_argument("--denom", default="p95-p5")
    ap.add_argument("--kpx_threshold", type=float, default=0.10)
    ap.add_argument("--day_only", action="store_true")
    ap.add_argument("--start_hour", type=int, default=6)
    ap.add_argument("--end_hour", type=int, default=18)
    ap.add_argument("--fig_w", type=float, default=22.0)
    ap.add_argument("--fig_h", type=float, default=3.5)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--locator", choices=["auto","day","week","month"], default="week")
    ap.add_argument("--max_points", type=int, default=8000)
    # 컬럼 지정(없으면 자동 탐지)
    ap.add_argument("--h1_cols", type=str, default="")
    ap.add_argument("--h3_cols", type=str, default="")
    # 기간 제한(옵션)
    ap.add_argument("--x_from", type=str, default=None)
    ap.add_argument("--x_to", type=str, default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.pred_csv)
    time_col = args.time_col or pick_col(df, ["datetime","ts","time","date"]) or "datetime"
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    # horizon별 예측 컬럼 수집
    cols = df.columns.tolist()
    h1_list = _parse_list(args.h1_cols) or _autodetect_cols(cols, ["h1","1h"])
    h3_list = _parse_list(args.h3_cols) or _autodetect_cols(cols, ["h3","3h"])

    def run_once(h_label: str, pred_cols: List[str], file_suffix: str):
        # y_pred가 없고 pred_cols가 비면 건너뜀
        base_pred = pred_cols[0] if pred_cols else ("y_pred" if "y_pred" in df.columns else None)
        if base_pred is None:
            return
        extras = pred_cols[1:] if pred_cols else []

        labels = {"y_true":"Actual"}
        labels[base_pred] = f"Pred ({h_label}, {_pretty(base_pred)})" if pred_cols else f"Pred ({h_label})"
        for c in extras:
            labels[c] = f"Pred ({h_label}, {_pretty(c)})"

        day = {"start_hour": args.start_hour, "end_hour": args.end_hour} if args.day_only else None

        plot_actual_vs_pred(
            df=df,
            time_col=time_col,
            y_true="y_true",
            y_pred=base_pred,
            extra_pred_cols=extras,
            labels=labels,
            out_dir=args.out_dir,
            filename_prefix=f"overlay_{file_suffix}",
            day_only=day,
            title=f"{args.title_prefix} • H={h_label} • norm={args.denom} • full",
            figsize=(args.fig_w, args.fig_h),
            dpi=args.dpi,
            max_points=args.max_points,
            x_from=args.x_from,
            x_to=args.x_to,
            locator=args.locator,
            annotate=True,
            denom=args.denom,
            kpx_threshold=args.kpx_threshold,
        )

    run_once("1h", h1_list, "H1")
    run_once("3h", h3_list, "H3")

if __name__ == "__main__":
    main()
