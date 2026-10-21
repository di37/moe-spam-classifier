"""
Run records: which machine and software produced a result, and how long it took.
train_moe.py writes its records to results/, and baselines.py reads them back.
"""
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from importlib import metadata

import torch

RESULTS_DIR = "results"
PACKAGES = ("torch", "numpy", "matplotlib", "pillow", "imageio-ffmpeg")


def cpu_name():
    if sys.platform == "darwin":
        try:
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return platform.processor() or "unknown"


def environment_info():
    return {
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "cpu": cpu_name(),
        "python": platform.python_version(),
        "torch_threads": torch.get_num_threads(),
        "device": "cpu",
        "packages": {name: metadata.version(name) for name in PACKAGES},
    }


def file_sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _ensure_parent(path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def write_json(path, record):
    _ensure_parent(path)
    with open(path, "w") as f:
        json.dump(record, f, indent=2, sort_keys=True)
        f.write("\n")


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_csv(path, rows):
    """Rows are dicts with the same keys; the first row's key order sets the columns."""
    if not rows:
        raise ValueError("write_csv needs at least one row")
    _ensure_parent(path)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_training_record(name, args, seeds, recorder, final_metrics, timings, checkpoint,
                          frames):
    """results/<name>.csv: the logged history; results/<name>.json: everything else."""
    history = recorder.history_rows()
    write_csv(os.path.join(RESULTS_DIR, f"{name}.csv"), history)
    write_json(os.path.join(RESULTS_DIR, f"{name}.json"), {
        "run": name,
        "command": " ".join(["python", *sys.argv]),
        "args": vars(args),
        "seeds": seeds,
        "final_metrics": final_metrics,
        "logged_steps": len(history),
        "frames": frames,
        "wall_clock_s": timings,
        "checkpoint": {"path": checkpoint, "sha256": file_sha256(checkpoint)},
        "environment": environment_info(),
    })
