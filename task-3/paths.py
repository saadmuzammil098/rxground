"""Makes ../task-1 and ../task-2 importable so task-3 can reuse the index,
the embedding model, and the provider-agnostic generators without
duplicating or rebuilding any of them. Import this before importing
anything from task-1 or task-2.
"""

from __future__ import annotations

import sys
from pathlib import Path

TASK1_DIR = Path(__file__).resolve().parent.parent / "task-1"
TASK2_DIR = Path(__file__).resolve().parent.parent / "task-2"

# Appended, not inserted at the front: task-3 has its own prompts.py and
# retrieve.py, and they must win over task-1's/task-2's same-named modules.
for _dir in (TASK1_DIR, TASK2_DIR):
    if str(_dir) not in sys.path:
        sys.path.append(str(_dir))
