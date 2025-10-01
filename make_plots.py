from __future__ import annotations
import argparse, pandas as pd
from src.visualize.plots import plot_actual_vs_pred
from src.common.io import pick_col

def parse_labels(s: str | None):
    if not s: return {}
    out={}
    for pair in s.split(","):
        if "=" in pair:
            k,v = pair.split("=",1)
            out[k.strip()] = v.strip()
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", required=True)
    ap.add_argument("--out_dir", default="./outputs/plots")
    ap.add_argument("--time_col", default=None)
    ap.add_argument("--group_col", default=None)  # 유지: 현재는 rto_id 고정 사용
    ap.add_argument("--day_only", action="store_true")
    ap.add_argument("--start_hour", type=int, default=6)
    ap.add_argument("--end_hour", type=int, default=18)
    # 보기 옵션
    ap.add_argument("--fig_w", type=float, default=22.0)
    ap.add_argument("--fig_h", type=float, default=3.5)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--max_points", type=int, default=8000)
    ap.add_argument("--x_from", type=str, default=None)
    ap.add_argument("--x_to", type=str, default=None)
    ap.add_argument("--locator", type=str, default="week", choices=["auto","day","week","month"])
    # 지표/표시 옵션
    ap.add_argument("--annotate", action="store_true")
    ap.add_argument("--denom", default="p95-p5")
    ap.add_argument("--kpx_threshold", type=float, default=0.10)
    ap.add_argument("--extra_pred_cols", type=str, default="")  # 콤마구분
    ap.add_argument("--labels", type=str, default="")           # "y_pred=Pred (H=3h),pers=Pred(H=3h,persistence)"
    ap.add_argument("--title", type=str, default="Actual vs Predicted")
    args = ap.parse_args()

    df = pd.read_csv(args.pred_csv)
    time_col = args.time_col or pick_col(df, ["datetime","ts","time","date"]) or "datetime"
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    day = {"start_hour": args.start_hour, "end_hour": args.end_hour} if args.day_only else None
    extra_cols = [c.strip() for c in args.extra_pred_cols.split(",") if c.strip()]
    labels = parse_labels(args.labels)

    plot_actual_vs_pred(
        df=df,
        time_col=time_col,
        y_true="y_true",
        y_pred="y_pred",
        extra_pred_cols=extra_cols,
        labels=labels,
        out_dir=args.out_dir,
        filename_prefix="actual_vs_pred",
        day_only=day,
        title=args.title,
        figsize=(args.fig_w, args.fig_h),
        dpi=args.dpi,
        max_points=args.max_points,
        x_from=args.x_from,
        x_to=args.x_to,
        locator=args.locator,
        annotate=args.annotate,
        denom=args.denom,
        kpx_threshold=args.kpx_threshold,
    )

if __name__ == "__main__":
    main()
