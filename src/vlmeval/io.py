from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .schema import COLUMNS


def append_records(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: r.get(k) for k in COLUMNS}, ensure_ascii=False) + "\n")


def load_records(path: str | Path) -> pd.DataFrame:
    df = pd.read_json(path, lines=True, dtype={"sample_id": str})
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"records missing columns: {sorted(missing)}")
    return df


def done_keys(path: str | Path) -> set[tuple]:
    """Keys already written, so an interrupted run can resume."""
    path = Path(path)
    if not path.exists():
        return set()
    df = load_records(path)
    return set(zip(df.benchmark, df.method, df.ratio.astype(float), df.sample_id.astype(str)))
