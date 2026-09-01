from __future__ import annotations

import os
from pathlib import Path
import tempfile

import pandas as pd


def atomic_write_csv(frame: pd.DataFrame, path: str | Path, **kwargs: object) -> Path:
    """Stage a CSV beside its destination and atomically promote it."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".staging", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(temporary, **kwargs)
        if not temporary.exists() or temporary.stat().st_size == 0:
            raise IOError(f"Staged CSV is empty: {temporary}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
