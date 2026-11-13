"""Append-only results log (one CSV row per run) and summary helpers."""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path


def append_result(csv_path: str | Path, row: dict) -> None:
    """Append a row; if new columns appear, the file is rewritten with the union header."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"timestamp": datetime.now().isoformat(timespec="seconds"), **row}
    if not path.exists():
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            w.writeheader()
            w.writerow(row)
        return
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or [])
        existing = list(reader)
    new_cols = [k for k in row if k not in header]
    if new_cols:
        header += new_cols
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            w.writerows(existing)
            w.writerow(row)
    else:
        with path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=header).writerow(row)


def load_results(csv_path: str | Path):
    import pandas as pd
    return pd.read_csv(csv_path)


def domain_summary(df, by=("backbone", "layers", "pc_coreset_ratio", "pc_reducer_dim")):
    """Mean metrics per configuration and domain (texture vs object)."""
    cols = [c for c in ("image_auroc", "pixel_auroc", "pro", "bank_mb", "fps") if c in df]
    by = [c for c in by if c in df] + ["domain"]
    return df.groupby(by, dropna=False)[cols].mean().round(4)
