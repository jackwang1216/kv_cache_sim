import os
import uuid
from typing import List, Dict, Any
from .runner import run_sim
from .io import read_summary

def sweep(
    binary_path: str,
    configs: List[str],
    traces: List[str],
    out_root: str,
    seed: List[int]
) -> List[Dict[str, Any]]:
    result = []
    for cfg in configs:
        for tr in traces:
            for seed in seeds:
                run_id = f"{uuid.uuid4().hex[:8]}"
                out_dir = os.path.join(out_root, run_id)
                code = run_sim(binary_path, cfg, tr, out_dir, seed=seed)
                if code == 0:
                    summary = read_summary(os.path.join(out_dir, "summary.json"))
                    result.append({
                        "run_id": run_id,
                        "config": cfg,
                        "trace": tr,
                        "seed": seed,
                        **summary
                    })

    return result