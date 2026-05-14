from __future__ import annotations

from pathlib import Path
from typing import List, Protocol


class RawTrialSource(Protocol):
    """Adapter that materialises data into canonical raw trial folders."""

    def attach(self, out_dir: Path) -> List[Path]:
        """Write raw trial folders and return their paths."""
        ...
