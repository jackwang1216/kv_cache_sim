import subprocess
import os
from typing import Optional, Dict

def run_sim(
    binary_path: str,
    config_path: str,
    trace_path: str,
    out_dir: str, 
    seed: Optional[int] = None,
    extra_env: Optional[Dict[str, str]] = None,
) -> int:
    """
    Launch kv_sim as a subprocess, returns exit code. 
    """

    cmd = [binary_path, "--config", config_path, "--trace", trace_path, "--out", out_dir]

    if seed is not None:
        cmd += ["--seed", str(seed)]

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    
    os.makedirs(out_dir, exist_ok=True)
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        print("STDOUT:\n", proc.stdout)
        print("STDERR:\n", proc.stderr)
    return proc.returncode