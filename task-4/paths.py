"""Makes ../task-1 importable so task-4 can reuse the section-aware chunks,
the embedding model, and the Chroma index without rebuilding any of them.
Import this before importing anything from task-1.
"""

from __future__ import annotations

import sys
from pathlib import Path

TASK1_DIR = Path(__file__).resolve().parent.parent / "task-1"

# Appended, not inserted at the front: task-4 has its own module names that
# must win over task-1's same-named modules (see task-3's paths.py for the
# same reasoning).
if str(TASK1_DIR) not in sys.path:
    sys.path.append(str(TASK1_DIR))
