"""Run records: what a run measured, and which machine and software produced it."""

import csv
import hashlib
import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import torch

from moe_spam.paths import PATHS

PACKAGES = ("torch", "numpy", "matplotlib", "pillow", "imageio-ffmpeg")


class RecordStore:
    """Reads and writes JSON and CSV records in one folder.

    Attributes:
        directory: The folder the records live in.
    """

    def __init__(self, directory=PATHS.results):
        """Creates a store for the records in ``directory`` (``results/`` by default)."""
        self.directory = Path(directory)

    def path(self, name, suffix):
        """Returns the path of record ``name`` with the given suffix, e.g. ``.json``."""
        return self.directory / f"{name}{suffix}"

    def write_json(self, name, record):
        """Writes ``record`` as ``<name>.json``, with sorted keys and a final newline."""
        path = self.path(name, ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(record, f, indent=2, sort_keys=True)
            f.write("\n")

    def read_json(self, name):
        """Returns the contents of ``<name>.json``."""
        with open(self.path(name, ".json")) as f:
            return json.load(f)

    def write_csv(self, name, rows):
        """Writes ``rows`` as ``<name>.csv``.

        Args:
            name: The record name.
            rows: Dicts with the same keys; the first row's key order sets the columns.

        Raises:
            ValueError: If there are no rows.
        """
        if not rows:
            raise ValueError("write_csv needs at least one row")
        path = self.path(name, ".csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def read_csv(self, name):
        """Returns the rows of ``<name>.csv`` as dicts of strings."""
        with open(self.path(name, ".csv"), newline="") as f:
            return list(csv.DictReader(f))

    def read_all(self, pattern):
        """Returns ``{name: record}`` for every JSON record whose name matches ``pattern``."""
        paths = sorted(self.directory.glob(f"{pattern}.json"))
        return {p.stem: self.read_json(p.stem) for p in paths}

    @staticmethod
    def sha256(path):
        """Returns the SHA-256 of a file, as hex."""
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    @staticmethod
    def environment():
        """Returns the machine and package versions a result was produced with."""
        return {
            "os": f"{platform.system()} {platform.release()}",
            "machine": platform.machine(),
            "cpu": _cpu_name(),
            "python": platform.python_version(),
            "torch_threads": torch.get_num_threads(),
            "device": "cpu",
            "packages": {name: metadata.version(name) for name in PACKAGES},
        }

    @staticmethod
    def command():
        """Returns the ``moe-spam`` command line this process was started with.

        Only the arguments are kept, so no absolute path ends up in a record.
        """
        return " ".join(["moe-spam", *sys.argv[1:]])


def _cpu_name():
    if sys.platform == "darwin":
        try:
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return platform.processor() or "unknown"
