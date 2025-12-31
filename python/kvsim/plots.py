import matplotlib.pyplot as plt
import pandas as pd

def plot_vram_timeseries(df, out_path=None):
    plt.figure()
    plt.plot(df["time_ms"], df["vram_used"])
    plt.xlabel("time (ms)")
    plt.ylabel("VRAM used (bytes)")
    plt.title("VRAM over time")
    if out_path:
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
    else:
        plt.show()

def plot_queue(df: pd.DataFrame, out_path=None):
    plt.figure()
    plt.plot(df["time_ms"], df["queue_depth"])
    plt.xlabel("time (ms)")
    plt.ylabel("queue depth")
    plt.title("Queue depth over time")
    if out_path:
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
    else:
        plt.show()

def plot_tokens_per_sec(df: pd.DataFrame, out_path=None):
    if "time_ms" not in df or "tokens_generated_delta" not in df:
        return
    dt = df["time_ms"].diff() / 1000.0
    # Avoid divide-by-zero and NA
    valid = dt > 0
    tps = pd.Series([pd.NA] * len(df))
    tps.loc[valid] = df.loc[valid, "tokens_generated_delta"] / dt.loc[valid]
    tps = tps.fillna(0)
    plt.figure()
    plt.plot(df["time_ms"], tps)
    plt.xlabel("time (ms)")
    plt.ylabel("tokens/sec")
    plt.title("Tokens per second over time")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=120)
    else:
        plt.show()

def plot_bar_metric(df: pd.DataFrame, metric: str, label_col: str, out_path=None, title=None):
    if metric not in df or label_col not in df:
        return
    plt.figure()
    plt.bar(df[label_col], df[metric])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(metric)
    plt.title(title or metric)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=120)
    else:
        plt.show()
