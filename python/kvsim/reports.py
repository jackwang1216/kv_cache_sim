import os
import json
from typing import Dict, Any

def write_simple_report(summary:Dict[str, Any], out_path: str):
    lines = []
    lines.append(f"# Run Report")
    lines.append("")
    for k in ["throughput_tokens_per_sec", "p50_latency_ms", "p95_latency_ms", "p99_latency_ms",
              "completion_rate", "reject_rate", "avg_vram_bytes", "gpu_busy_ms", "evictions"]:
        if k in summary:
            lines.append(f"- **{k}**: {summary[k]}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))