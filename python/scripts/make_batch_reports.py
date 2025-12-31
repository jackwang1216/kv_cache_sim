import os
import sys
import argparse

# Ensure python/ on path before kvsim imports
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
PY_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PY_ROOT not in sys.path:
    sys.path.insert(0, PY_ROOT)

from kvsim.io import load_summaries
from kvsim.plots import plot_compare_metric

def main():
    parser = argparse.ArgumentParser(description="Batch report/plots for multiple run dirs.")
    parser.add_argument("run_dirs", nargs="+", help="List of run directories (each with summary.json)")
    parser.add_argument("--label", nargs="*", help="Optional labels matching run_dirs")
    parser.add_argument("--out", default="runs/batch_report", help="Output directory for plots")
    args = parser.parse_args()

    df = load_summaries(args.run_dirs)
    if df.empty:
        print("No summaries loaded.")
        return 1
    labels = args.label if args.label and len(args.label) == len(args.run_dirs) else args.run_dirs
    df["label"] = labels

    os.makedirs(args.out, exist_ok=True)
    # Example metrics to compare; extend as needed
    metrics = ["p95_latency_ms", "p99_latency_ms", "throughput_tokens_per_sec", "evictions", "reject_rate"]
    for m in metrics:
        plot_compare_metric(df, m, "label", out_path=os.path.join(args.out, f"{m}.png"),
                            title=f"{m} by run")

    # Write a simple table
    table_path = os.path.join(args.out, "summary_table.csv")
    df.to_csv(table_path, index=False)
    print(f"Wrote plots and table to {args.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())