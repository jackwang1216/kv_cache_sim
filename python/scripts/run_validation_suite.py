import os
import sys

# Ensure python/ is on sys.path before kvsim imports
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
PY_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PY_ROOT not in sys.path:
    sys.path.insert(0, PY_ROOT)

from kvsim.runner import run_sim
from kvsim.io import read_summary

BIN = os.path.join(REPO_ROOT, "cpp", "build", "kv_sim")

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