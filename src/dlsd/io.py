"""Result storage. Row-level generations are JSON Lines (local only). Aggregates are written as CSV and JSON with identical content."""
import datetime as dt
import json
import platform
import subprocess
from pathlib import Path
import pandas as pd

SCHEMA_VERSION = "1.0"
RAW = Path("results/raw"); SCORED = Path("results/scored"); AGG = Path("results/aggregate")

def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"

def write_jsonl(rows, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows: f.write(json.dumps(r, default=str) + "\n")

def read_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]

def write_table(df: pd.DataFrame, name: str, meta: dict, out_dir=AGG):
    """Write results/aggregate/<name>.csv and <name>.json (same records). meta is merged into the JSON header."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df.to_csv(out_dir / f"{name}.csv", index=False)
    header = {"schema_version": SCHEMA_VERSION, "table": name, "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
              "git_commit": git_commit(), "n_records": int(len(df)), "columns": list(df.columns), **meta}
    payload = {**header, "records": json.loads(df.to_json(orient="records"))}
    (out_dir / f"{name}.json").write_text(json.dumps(payload, indent=2))

def write_manifest(cfg: dict, extra: dict, path="results/run_manifest.json"):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": SCHEMA_VERSION, "date_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "git_commit": git_commit(), "python": platform.python_version(), "platform": platform.platform(),
                "seed": cfg.get("seed"), "decoding": cfg.get("decoding"), "uncertainty": cfg.get("uncertainty"),
                "models": cfg.get("models"), "classifiers": cfg.get("classifiers"), "judge": cfg.get("judge"),
                "renderer": cfg.get("rendering"), **extra}
    path.write_text(json.dumps(manifest, indent=2, default=str))
    return manifest

METRIC_COLS = ["accuracy", "mean_conf", "ece", "brier", "auroc", "sel_acc", "overconf_err"]

def to_long(df: pd.DataFrame, id_cols, metric_cols) -> pd.DataFrame:
    """Tidy format: one row per (cell, metric) with columns metric, value."""
    return df.melt(id_vars=list(id_cols), value_vars=[c for c in metric_cols if c in df.columns],
                   var_name="metric", value_name="value")
