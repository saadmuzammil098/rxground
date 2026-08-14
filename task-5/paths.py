"""Makes ../task-1, ../task-2, and ../task-4 importable so task-5 can
reuse the index, the provider-agnostic generators, and the advanced
retrieval pipeline without duplicating or rebuilding any of them.
"""

from __future__ import annotations

import sys
from pathlib import Path

TASK1_DIR = Path(__file__).resolve().parent.parent / "task-1"
TASK2_DIR = Path(__file__).resolve().parent.parent / "task-2"
TASK4_DIR = Path(__file__).resolve().parent.parent / "task-4"

for _dir in (TASK1_DIR, TASK2_DIR, TASK4_DIR):
    if str(_dir) not in sys.path:
        sys.path.append(str(_dir))
