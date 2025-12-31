import json
import pandas as pd
from typing import Dict, Any

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