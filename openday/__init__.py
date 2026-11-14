"""OpenDay "Who's walking?": a two-person Wi-Fi walk demo built beside the thesis code, never in it.

This file, in order:
  1. Start-up switches (no bytecode, one BLAS thread), set before anything else is imported.
  2. Where things live: OpenDay's own folders, the read-only thesis Code/, the config.
  3. The participant codes (P1-P6) and the laptop-only list that joins them to the recordings' names.

Other files use these as `import openday` and then `openday.STATE`, `openday.load_config()` and so on.
"""

from __future__ import annotations

import os
import sys

# --- 1. Start-up switches ---

# Before anything can import `pipeline`: no bytecode may be written into Code/, which is only ever read. The
# environment variable carries this into any child process (multiprocessing).
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
# Small matrices: one BLAS thread is faster and keeps the CPU budget honest (the thesis runs OMP_NUM_THREADS=1).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

# Imported only now, so the switches above apply to these imports too.
from functools import lru_cache
from typing import Any

import yaml

# --- 2. Where things live ---

OPENDAY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.path.normpath(os.path.join(OPENDAY, "..", "Code"))
CONFIG = os.path.join(OPENDAY, "configs", "openday.yaml")
STATE = os.path.join(OPENDAY, "state")
REPORTS = os.path.join(OPENDAY, "reports")
LOGS = os.path.join(OPENDAY, "logs")
SCRIPTS = os.path.join(OPENDAY, "scripts")
ROOM_B_ID = os.path.join(CODE, "data", "room_B", "identification")


def use_code() -> None:
    """Put Code/ on sys.path so the light `pipeline.io` modules import. Code/ is read, never written (the start-up
    switches above have already turned bytecode off)."""
    if not os.path.isdir(os.path.join(CODE, "pipeline")):
        raise SystemExit(f"The private thesis code is not at {CODE}: live, replay, eval --bench and bench-check "
                         "need it (see the README).")
    if CODE not in sys.path:
        sys.path.append(CODE)


def load_config(path: str = CONFIG) -> dict[str, Any]:
    """The YAML config as a plain dict."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs() -> None:
    """Create OpenDay's writable folders (all inside OpenDay/)."""
    for d in (STATE, REPORTS, LOGS):
        os.makedirs(d, exist_ok=True)


# --- 3. The participant codes ---

# The six thesis participants, by code only. The thesis recordings are filed under first names; the list that joins
# the two (configs/people.local.yaml) is git-ignored and stays on the owner's laptop.
PEOPLE = ("P1", "P2", "P3", "P4", "P5", "P6")
PEOPLE_FILE = os.path.join(OPENDAY, "configs", "people.local.yaml")


@lru_cache(maxsize=1)
def _people() -> dict:
    """configs/people.local.yaml: {"codes": {code: first name}, "excluded": [names recorded but never used]}."""
    if not os.path.exists(PEOPLE_FILE):
        raise SystemExit(f"{PEOPLE_FILE} is missing: the thesis recordings are filed by first name, and only the "
                         "owner's laptop has the list (README, Files).")
    with open(PEOPLE_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f)


def recording_name(code: str) -> str:
    """The first name a participant's thesis recordings are filed under (P1-P6 only; no fallback to raw names)."""
    name = _people()["codes"].get(code) if code in PEOPLE else None
    if not name:
        raise SystemExit(f"Unknown participant code {code!r}: use P1-P6")
    return str(name)


def have_people() -> bool:
    """Is the code list on this machine?"""
    return os.path.exists(PEOPLE_FILE)
