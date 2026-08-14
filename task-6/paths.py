"""Makes ../task-1 importable so task-6 can reuse the section-aware
chunking function and the embedding model without duplicating either.
"""

from __future__ import annotations

import sys
from pathlib import Path

TASK1_DIR = Path(__file__).resolve().parent.parent / "task-1"

if str(TASK1_DIR) not in sys.path:
    sys.path.append(str(TASK1_DIR))
