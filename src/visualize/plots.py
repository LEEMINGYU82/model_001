from __future__ import annotations
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Iterable, Dict, Optional, Tuple, List
from ..common.metrics import default_day_mask, nmae_with_denom, nmae_kpx

def _apply_time_axis(ax, locator: str = "auto"):
    if locator == "day":
        loc = mdates.DayLocator(interval=1)
    elif locator == "week":
        loc = mdates.WeekdayLocator(byweekday=mdates.MO, interval=1)
    elif locator == "month":
        loc = mdates.MonthLocator()
    else:
        loc = mdates.AutoDateLocator(minticks=6, maxticks=12)
    fmt = mdates.ConciseDateFormatter(loc)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(fmt)

def plot_actual_vs_pred(
    df: pd.DataFrame,
    time_col: str,
    y_true: str = "y_true",
    y_pred: str = "y_pred",
    extra_pred_cols: Optional[Iterable[str]] = None,
    labels: Optional[Dict[str, str]] = None,
    out_dir: str = "./outputs/plots",
    filename_prefix: str = "actual_vs_pred",
    day_only: dict | None = None,
    title: str | None = None,
    # 보기 옵션
    figsize: Tuple[float, float] = (18, 3.2),
    dpi: int = 150,
    max_points: Optional[int] = 6000,
    x_from: Optional[str] = None,
    x_to: Optional[str] = None,
    locator: str = "auto",
    # 지표 옵션
    annotate: bool = True,
    denom: str | float = "p95-p5",
    kpx_threshold: float = 0.10,
):
    """여러 예측 열을 함께 그리며 그래프 제목에 nMAE/ nMAE(KPX) 표기"""
    os.makedirs(out_dir, exist_ok=True)
    labels = labels or {}
    all_preds = [y_pred] + [c for c in (extra_pred_cols or []) if c]

    groups = [None]
    if "rto_id" in df.columns:
        groups = df["rto_id"].dropna().unique()

    for g in groups:
        sub = df if g is None else df[df["rto_id"] == g]

        # 시간/필수컬럼 정리
        if not pd.api.types.is_datetime64_any_dtype(sub[time_col]):
            sub = sub.copy()
            sub[time_col] = pd.to_datetime(sub[time_col], errors="coerce")
            sub = sub.dropna(subset=[time_col])

        if x_from:
            sub = sub[sub[time_col] >= pd.to_datetime(x_from)]
        if x_to:
            sub = sub[sub[time_col] <= pd.to_datetime(x_to)]

        if day_only is not None:
            mask = default_day_mask(
                sub[time_col],
                start_hour=day_only.get("start_hour", 6),
                end_hour=day_only.get("end_hour", 18),
            )
            sub = sub.loc[mask]

        if sub.empty or (y_true not in sub.columns):
            continue

        # 포인트 너무 많으면 샘플링
        if max_points is not None and len(sub) > max_points:
            stride = max(1, len(sub) // max_points)
            sub = sub.iloc[::stride]

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.plot(sub[time_col], sub[y_true], label=labels.get(y_true, y_true), linewidth=1.2, alpha=0.95, zorder=3)

        metrics_strs: List[str] = []
        for pc in all_preds:
            if pc not in sub.columns:
                continue
            ax.plot(sub[time_col], sub[pc], label=labels.get(pc, pc), linewidth=1.0, alpha=0.85, zorder=2)
            if annotate:
                m_all = nmae_with_denom(sub[y_true].values, sub[pc].values, denom=denom, mask=None)
                m_kpx = nmae_kpx(sub[y_true].values, sub[pc].values, threshold=kpx_threshold, denom=denom)
                metrics_strs.append(f"{labels.get(pc, pc)}={m_all:.2f}% | KPX={m_kpx:.2f}%")

        ax.legend()
        ax.set_xlabel(time_col)
        ax.set_ylabel("gen_rate")
        base = title or "Actual vs Predicted"
        ctx = []
        if day_only is not None: ctx.append(f"Day-only({day_only.get('start_hour',6)}–{day_only.get('end_hour',18)})")
        ctx.append(f"denom={denom}")
        if x_from or x_to: ctx.append(f"range={x_from or ''}..{x_to or ''}")
        head = f"{base} • " + " • ".join(ctx)
        tail = ""
        if annotate and metrics_strs:
            tail = "\n" + " | ".join(metrics_strs)
        ax.set_title(head + tail)

        _apply_time_axis(ax, locator=locator)
        ax.margins(x=0)
        ax.grid(True, alpha=0.25, which="both")
        fig.tight_layout()

        fn = f"{filename_prefix}_{'all' if g is None else str(g)}.png"
        fp = os.path.join(out_dir, fn)
        fig.savefig(fp, bbox_inches="tight")
        plt.close(fig)
