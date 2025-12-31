import matplotlib.pyplot as plt

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