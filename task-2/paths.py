"""Makes ../task-1 importable so task-2 can reuse its Chroma index and
embedding model without rebuilding either. Import this before importing
anything from task-1.
"""

from __future__ import annotations

import sys
from pathlib import Path

TASK1_DIR = Path(__file__).resolve().parent.parent / "task-1"

if str(TASK1_DIR) not in sys.path:
    sys.path.insert(0, str(TASK1_DIR))
