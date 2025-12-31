import json
import pandas as pd
from typing import Dict, Any
import os

def read_summary(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)

def read_run_meta(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)

def read_events(path: str):
    events = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))
    return events

def read_timeseries(path: str):
    return pd.read_csv(path)

def load_summaries(run_dirs) -> pd.DataFrame:
    rows = []
    for rd in run_dirs:
        path = os.path.join(rd, "summary.json")
        if not os.path.exists(path):
            continue
        with open(path, "r") as f:
            j = json.load(f)
        j["run_dir"] = rd
        rows.append(j)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)