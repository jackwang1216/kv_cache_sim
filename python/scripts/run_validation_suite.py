import os
import sys
import argparse

# Ensure python/ is on sys.path before kvsim imports
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
PY_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PY_ROOT not in sys.path:
    sys.path.insert(0, PY_ROOT)

from kvsim.runner import run_sim
from kvsim.io import read_summary

BIN = os.path.join(REPO_ROOT, "cpp", "build", "kv_sim")

def check_monotonicity(low_summary, high_summary):
    # Higher VRAM should not have more evictions/rejects
    low_ev = low_summary.get("evictions", 0)
    high_ev = high_summary.get("evictions", 0)
    low_rej = low_summary.get("rejected", 0)
    high_rej = high_summary.get("rejected", 0)
    return (high_ev <= low_ev) and (high_rej <= low_rej)

def main():
    parser = argparse.ArgumentParser(description="Run validation sweeps and optional monotonicity check.")
    parser.add_argument("--run-sweeps", action="store_true", help="Run predefined validation cases.")
    parser.add_argument("--check-low", help="run_dir for low VRAM scenario (for monotonicity check)")
    parser.add_argument("--check-high", help="run_dir for high VRAM scenario (for monotonicity check)")
    args = parser.parse_args()

    if args.run_sweeps:
        cases = [
            ("configs/validation_low_load.txt", "data/validation_low_load.txt", "runs/val_low_load_py"),
            ("configs/validation_saturation.txt", "data/validation_saturation.txt", "runs/val_saturation_py"),
            ("configs/validation_vram_low.txt", "data/validation_vram_base.txt", "runs/val_vram_low_py"),
            ("configs/validation_vram_high.txt", "data/validation_vram_base.txt", "runs/val_vram_high_py"),
            ("configs/validation_long_tail.txt", "data/validation_long_tail.txt", "runs/val_long_tail_py"),
        ]
        for cfg, trace, out_dir in cases:
            cfg_path = os.path.join(REPO_ROOT, cfg)
            trace_path = os.path.join(REPO_ROOT, trace)
            out_path = os.path.join(REPO_ROOT, out_dir)
            rc = run_sim(BIN, cfg_path, trace_path, out_path)
            print(cfg, trace, "rc=", rc)
            if rc == 0:
                summary = read_summary(os.path.join(out_path, "summary.json"))
                print(summary)

    if args.check_low and args.check_high:
        low = read_summary(os.path.join(args.check_low, "summary.json"))
        high = read_summary(os.path.join(args.check_high, "summary.json"))
        ok = check_monotonicity(low, high)
        if not ok:
            print("VRAM monotonicity check FAILED (evictions/rejects not improved).")
            return 1
        print("VRAM monotonicity check passed.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())