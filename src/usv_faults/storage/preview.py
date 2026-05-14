from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from usv_faults.storage.trials import read_manifest


def write_preview_csv(trial_dir: Path, out: Optional[Path] = None, max_rows: int = 1000) -> Path:
    manifest = read_manifest(trial_dir)
    telemetry = pd.read_parquet(trial_dir / "telemetry.parquet")
    stride = max(1, len(telemetry) // max_rows)
    preview = telemetry.iloc[::stride].head(max_rows).copy()
    if out is None:
        out = trial_dir / "telemetry_preview.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    preview.to_csv(out, index=False)

    summary_path = out.with_name(out.stem + "_summary.csv")
    summary_rows = []
    for column in telemetry.columns:
        values = telemetry[column]
        summary_rows.append(
            {
                "trial_id": manifest.trial_id,
                "channel": column,
                "sample_count": int(values.count()),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    return out
