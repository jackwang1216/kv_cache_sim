import os
import sys
import argparse

# Ensure python/ is on sys.path before kvsim imports
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
PY_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PY_ROOT not in sys.path:
    sys.path.insert(0, PY_ROOT)

from kvsim.io import read_summary, read_timeseries
from kvsim.plots import plot_vram_timeseries, plot_queue, plot_tokens_per_sec
from kvsim.reports import write_simple_report


def main():
    parser = argparse.ArgumentParser(description="Generate quick plots and a markdown report for one run_dir.")
    parser.add_argument("run_dir", help="Path to a run directory containing summary.json, timeseries.csv, events.jsonl")
    parser.add_argument("--out", help="Output directory for plots/report (default: run_dir)", default=None)
    parser.add_argument("--no-show", action="store_true", help="Do not display plots, only save to files")
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    out_dir = os.path.abspath(args.out) if args.out else run_dir
    os.makedirs(out_dir, exist_ok=True)

    summary_path = os.path.join(run_dir, "summary.json")
    ts_path = os.path.join(run_dir, "timeseries.csv")

    if not os.path.exists(summary_path):
        print(f"summary.json not found in {run_dir}")
        return 1

    summary = read_summary(summary_path)
    write_simple_report(summary, os.path.join(out_dir, "report.md"))

    if os.path.exists(ts_path):
        df = read_timeseries(ts_path)
        plot_vram_timeseries(df, out_path=os.path.join(out_dir, "vram.png"))
        plot_queue(df, out_path=os.path.join(out_dir, "queue.png"))
        plot_tokens_per_sec(df, out_path=os.path.join(out_dir, "tokens_per_sec.png"))
        if not args.no_show:
            # Show the last plot; others are saved. To show all, users can open the PNGs.
            import matplotlib.pyplot as plt
            plt.show()
    else:
        print(f"timeseries.csv not found in {run_dir}, skipping plots.")

    print(f"Report written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

