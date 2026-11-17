"""Uncertainty estimators and metrics for Experiment 2."""
import math
from collections import Counter
import numpy as np

def sampling_confidence(answers):
    """Share agreeing with the modal answer, plus entropy of the answer distribution (nats)."""
    answers = [a for a in answers if a is not None]
    if not answers: return None, None, None
    c = Counter(answers); n = len(answers); mode, k = c.most_common(1)[0]
    ent = -sum((v / n) * math.log(v / n) for v in c.values())
    return mode, k / n, ent

import re as _re
_LETTER = _re.compile(r"^[\s\*\(\[]*([A-J])(?=\s*(?:[\.\)\]:\*]|$))")                      # letter at the start of the first line, then punctuation or end of line
_BARE = _re.compile(r"^[\s\*]*(?:confidence\s*(?:level)?\s*[:=\-]?\s*)?(\d{1,3}(?:\.\d+)?)\s*%?[\s\*\.]*$", _re.I)   # a line that is only a number (optionally labelled or with %)
_LABELLED = _re.compile(r"confidence[^\d\n]{0,25}?(\d{1,3}(?:\.\d+)?)\s*%?", _re.I)


def parse_verbalized(resp: str):
    """Model-independent parser for 'letter on the first line, confidence 0-100 next' (docs/EXPERIMENTS.md section 8.2).
    The letter must start the first line (alone, or followed by punctuation and any text, e.g. 'B.' or 'B. Aortic regurgitation').
    The confidence is the first number in 0-100 that is (a) alone on a later line, (b) alone after the letter on the first line ('B. 80'),
    or (c) labelled 'confidence' on a later line. Numbers inside option text are never used. Returns (letter, conf in [0,1]) or (None, None).
    Revised 2026-11-14 (docs/DEVIATIONS.md): the first version required the letter alone on line 1 and rejected the common 'B. option text' format."""
    lines = [l.strip() for l in (resp or "").strip().splitlines() if l.strip()]
    if not lines:
        return None, None
    m = _LETTER.match(lines[0])
    if not m:
        return None, None
    candidates = []
    rest = _re.sub(r"^[\s\*\(\[]*[A-J]\s*[\.\)\]:\*]*", "", lines[0])
    if rest and _BARE.match(rest):
        candidates.append(_BARE.match(rest).group(1))
    for l in lines[1:]:
        b = _BARE.match(l) or _LABELLED.search(l)
        if b:
            candidates.append(b.group(1))
    for c in candidates:
        v = float(c)
        if 0 <= v <= 100:
            return m.group(1), v / 100.0
    return None, None


def ece(conf, correct, bins=10):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    if len(conf) == 0: return float("nan")
    edges = np.linspace(0, 1, bins + 1); total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any(): total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)

def brier(conf, correct):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    return float(np.mean((conf - correct) ** 2)) if len(conf) else float("nan")

def auroc(conf, correct):
    from sklearn.metrics import roc_auc_score
    correct = np.asarray(correct)
    if len(set(correct.tolist())) < 2: return float("nan")
    return float(roc_auc_score(correct, conf))

def selective_accuracy(conf, correct, coverage=0.8):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    if len(conf) == 0: return float("nan")
    k = max(1, int(round(coverage * len(conf))))
    idx = np.argsort(-conf, kind="stable")[:k]
    return float(correct[idx].mean())

def overconfident_error_rate(conf, correct, threshold=0.8):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    return float(np.mean((conf >= threshold) & (correct == 0))) if len(conf) else float("nan")


def apply_verbalized(df, regen_rows):
    """Recompute verbalized_letter and conf_verbalized for every row with the CURRENT parser from the stored raw reply (the row's own
    verbalized_raw, else the re-asked reply in regen_rows). Rows with no raw reply keep their stored values and are counted in the return value."""
    lookup = {(r["item_id"], r["condition_id"], r["model_id"]): r["verbalized_raw"] for r in regen_rows}
    letters, confs, missing = [], [], 0
    for r in df.to_dict("records"):
        raw = r.get("verbalized_raw")
        if not isinstance(raw, str):
            raw = lookup.get((r["item_id"], r["condition_id"], r["model_id"]))
        if raw is None:
            missing += 1
            letters.append(r.get("verbalized_letter")); confs.append(r.get("conf_verbalized"))
        else:
            l, c = parse_verbalized(raw); letters.append(l); confs.append(c)
    out = df.copy()
    out["verbalized_letter"], out["conf_verbalized"] = letters, confs
    return out, missing
