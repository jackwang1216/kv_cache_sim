import os
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, JSONResponse
from pydantic import BaseModel


ROOT = Path(__file__).resolve().parents[2]  # repo root
BIN_DEFAULT = ROOT / "cpp" / "build" / "kv_sim"
RUNS_ROOT = ROOT / "runs" / "api"


class RunRequest(BaseModel):
    trace_path: Optional[str] = None       
    trace_content: Optional[str] = None     
    trace_name: Optional[str] = "trace.txt"  
    out_dir: Optional[str] = None
    seed: Optional[int] = None
    config_options: Optional[Dict[str, Any]] = None


app = FastAPI(title="kv-sim backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def resolve_path(p: str) -> Path:
    cp = Path(p)
    if not cp.is_absolute():
        cp = ROOT / p
    return cp


def write_config_temp(config_options: Dict[str, Any]) -> Path:
    lines = []
    for k, v in config_options.items():
        lines.append(f"{k} {v}")
    fd, tmp_path = tempfile.mkstemp(suffix=".txt", prefix="kv_cfg_")
    with os.fdopen(fd, "w") as f:
        f.write("\n".join(lines))
    return Path(tmp_path)


def write_trace_temp(content: str, name: str = "trace.txt") -> Path:
    # create a temp file with provided content
    fd, tmp_path = tempfile.mkstemp(suffix=Path(name).suffix or ".txt", prefix="kv_trace_")
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return Path(tmp_path)


@app.post("/run")
def run_sim(req: RunRequest):
    bin_path = os.environ.get("KV_SIM_BIN", str(BIN_DEFAULT))
    bin_path = resolve_path(bin_path)
    if not bin_path.exists():
        raise HTTPException(status_code=400, detail=f"Binary not found: {bin_path}")

    trace_path = None
    if req.trace_content:
        trace_path = write_trace_temp(req.trace_content, req.trace_name or "trace.txt")
    elif req.trace_path:
        trace_path = resolve_path(req.trace_path)
        if not trace_path.exists():
            raise HTTPException(status_code=400, detail=f"Trace not found: {trace_path}")
    else:
        raise HTTPException(status_code=400, detail="Provide trace_content or trace_path")

    config_path = None
    try:
        cfg_opts = req.config_options or {}
        if cfg_opts:
            config_path = write_config_temp(cfg_opts)
        # out dir
        run_id = uuid.uuid4().hex[:8]
        out_dir = resolve_path(req.out_dir) if req.out_dir else RUNS_ROOT / run_id
        out_dir.parent.mkdir(parents=True, exist_ok=True)

        cmd = [str(bin_path), "--trace", str(trace_path), "--out", str(out_dir)]
        if config_path:
            cmd.extend(["--config", str(config_path)])
        if req.seed is not None:
            cmd.extend(["--seed", str(req.seed)])

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            detail = proc.stderr or proc.stdout or "simulator failed"
            raise HTTPException(status_code=500, detail=detail)

        summary_file = out_dir / "summary.json"
        ts_file = out_dir / "timeseries.csv"
        events_file = out_dir / "events.jsonl"
        if not summary_file.exists():
            raise HTTPException(status_code=500, detail="summary.json not produced")

        import json
        with open(summary_file, "r") as f:
            summary = json.load(f)

        base_url = "/runs/api"
        # if out_dir under RUNS_ROOT, expose relative path
        if str(out_dir).startswith(str(RUNS_ROOT)):
            rel = out_dir.relative_to(ROOT)
            run_dir_rel = str(rel)
        else:
            run_dir_rel = str(out_dir)

        return {
            "run_id": run_id,
            "run_dir": run_dir_rel,
            "summary": summary,
            "summary_url": f"/runs/{run_id}/summary",
            "timeseries_url": f"/runs/{run_id}/timeseries",
            "events_url": f"/runs/{run_id}/events",
        }
    finally:
        if config_path and config_path.exists():
            try:
                config_path.unlink()
            except OSError:
                pass
        if req.trace_content and trace_path and trace_path.exists():
            try:
                trace_path.unlink()
            except OSError:
                pass


def _resolve_run_file(run_id: str, filename: str) -> Path:
    p = RUNS_ROOT / run_id / filename
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"{filename} not found for run {run_id}")
    return p


@app.get("/runs/{run_id}/summary")
def get_summary(run_id: str):
    p = _resolve_run_file(run_id, "summary.json")
    return FileResponse(p)


@app.get("/runs/{run_id}/timeseries")
def get_timeseries(run_id: str):
    p = _resolve_run_file(run_id, "timeseries.csv")
    return PlainTextResponse(p.read_text(), media_type="text/csv")


@app.get("/runs/{run_id}/events")
def get_events(run_id: str):
    p = _resolve_run_file(run_id, "events.jsonl")
    return PlainTextResponse(p.read_text(), media_type="application/jsonl")

