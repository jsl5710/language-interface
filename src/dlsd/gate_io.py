"""Round bookkeeping for the render -> judge -> gate -> re-render loop. All files are JSON Lines, one row per (item, round)."""
import json
from pathlib import Path

LLM_DIR, JUDGE_DIR, GATE_DIR = Path("data/llm"), Path("data/judge"), Path("data/gate")


def read_rows(path) -> list:
    p = Path(path)
    return [json.loads(l) for l in open(p) if l.strip()] if p.exists() else []


def latest_by_item(rows: list) -> dict:
    """Highest round wins; later rows win ties (resumed runs may append)."""
    out: dict = {}
    for r in rows:
        k = r["item_id"]
        if k not in out or r.get("round", 1) >= out[k].get("round", 1):
            out[k] = r
    return out


def codeswitch_shards() -> list:
    return sorted(LLM_DIR.glob("rendered_codeswitch_*.jsonl"))


def failed_with_feedback(shard: Path, round_no: int) -> dict:
    """Items whose round (round_no - 1) rendering failed the gate -> the labels the judges gave (feedback for re-rendering)."""
    gate_rows = latest_by_item([r for r in read_rows(GATE_DIR / shard.name.replace("rendered_", "gate_")) if r["round"] == round_no - 1])
    return {k: r for k, r in gate_rows.items() if not r["passed"] and not r.get("not_renderable")}
