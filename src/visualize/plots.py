from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt

def plot_range_df(df: pd.DataFrame, time_col: str, y_cols, title: str, save_path: str=None):
    plt.figure()
    for c in y_cols:
        if c in df.columns:
            plt.plot(df[time_col], df[c], label=c)
    plt.legend()
    plt.title(title)
    plt.xlabel(time_col)
    plt.ylabel("value")
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    return plt.gca()
