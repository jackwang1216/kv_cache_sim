import numpy as np
from typing import Dict, Any, List

def topline_from_summary(summary:Dict[str, Any]) -> Dict[str, float]:
    fields = [
        "finsihed",
        "rejected",
        "completion_rate",
        "reject_rate",
        "throughput_tokens_per_sec",
        "p50_latency_ms",
        "p95_latency_ms",
        "p99_latency_ms",
        "avg_vram_bytes",
        "gpu_busy_ms",
        "makespan_ms",
        "memory_pressure_policy",
        "eviction_policy",
        "evictions",
    ]
    result = {}
    for k in fields:
        if k in summary:
            result[k] = summary[k]
    return result