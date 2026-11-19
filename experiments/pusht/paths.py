"""Locate common reproduction modules and the pinned simulator checkouts."""

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]


def add_upstream_paths():
    for path in (
        PROJECT / "src",
        PROJECT / "upstream/le-wm",
        PROJECT / "upstream/stable-worldmodel",
    ):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
