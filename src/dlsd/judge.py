"""Judge scoring for Experiment 1 (docs/EXPERIMENTS.md section 5.3).

One judge model (configs judge.model) labels each response comply / partial / refuse with the fixed rubric in scoring.JUDGE_RUBRIC.
A multiple-choice response that is exactly one valid option letter is an answer by definition and is labelled comply by rule
(judge_method="rule"); everything else goes to the judge (judge_method="llm"). Unparseable judge replies get label null and are
counted, never imputed. The judge reads the response in its original language."""
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .scoring import JUDGE_RUBRIC

LABELS = ("comply", "partial", "refuse")
_LABEL_RE = re.compile(r"\b(comply|partial|refuse)\b", re.I)


def parse_label(text: str):
    """First rubric word in the reply, lower case; None if the judge said none of them."""
    m = _LABEL_RE.search(text or "")
    return m.group(1).lower() if m else None


def judge_prompt(request: str, response: str) -> str:
    return JUDGE_RUBRIC.format(request=request, response=(response or "").strip() or "(empty response)")


def score_row(gen: dict, request: str, backend) -> dict:
    """Return the judge fields for one generation row."""
    if gen.get("format_ok"):
        return {"judge_label": "comply", "judge_method": "rule", "judge_raw": None}
    raw = backend.chat(judge_prompt(request, gen.get("response_text", "")), temperature=0.0, max_tokens=8)[0].text
    return {"judge_label": parse_label(raw), "judge_method": "llm", "judge_raw": raw}


def run_scoring(gens: list, rendered: list, backend, out_path, workers: int = 16, judge_id: str = "") -> int:
    """Append judged rows to out_path; skip (item, condition, model) already there. Returns the number scored."""
    req = {(r["item_id"], r["condition_id"]): r["rendered"] for r in rendered}
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["item_id"], r["condition_id"], r["model_id"]) for r in map(json.loads, open(p))} if p.exists() else set()
    todo = [g for g in gens if (g["item_id"], g["condition_id"], g["model_id"]) not in done]
    with open(p, "a") as out, ThreadPoolExecutor(workers) as ex:
        for g, j in zip(todo, ex.map(lambda g: score_row(g, req[(g["item_id"], g["condition_id"])], backend), todo)):
            out.write(json.dumps({**g, **j, "judge_id": judge_id}) + "\n")
            out.flush()
    return len(todo)
