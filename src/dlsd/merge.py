"""Merge the baseline, dialect, translation and code-switch shards into one row per (item, condition): data/rendered.jsonl.

Every item gets exactly one row for each condition (27 with the native-base extension). A rendering that failed its checks or the judge gate is kept with
render_ok=False and a status, never dropped, so later stages can count and report it (docs/EXPERIMENTS.md section 17, check 5)."""
import re
from pathlib import Path

from .conditions import Condition, build_conditions
from .gate_io import latest_by_item, read_rows


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def llm_slug(cond: Condition) -> str:
    return "_".join(x for x in (cond.axis, cond.level, cond.language) if x).lower()


def _row(item: dict, cond: Condition, **kw) -> dict:
    base = {"item_id": item["item_id"], "source": item["source"], "principle": item.get("principle"), "answer": item.get("answer"),
            "condition_id": cond.cid, "condition_axis": cond.axis, "condition_level": cond.level, "language_pair": cond.language,
            "switch_type": cond.switch_type, "rendered": None, "options": None, "method": None, "render_ok": False,
            "render_status": "missing", "gate_passed": None, "round": None}
    return {**base, **kw}


def load_review(path="data/review/decisions.jsonl") -> dict:
    """Bilingual reviewer decisions for renderings that scored below the similarity threshold: {(condition_id, item_id): 'pass'|'fail'}."""
    p = Path(path)
    if not p.exists():
        return {}
    import json
    return {(d["condition_id"], d["item_id"]): d["decision"] for d in map(json.loads, p.read_text().splitlines()) if d.get("decision") in ("pass", "fail")}


def merge(items: list, cfg: dict, dialect_dir="data/dialect", llm_dir="data/llm", gate_dir="data/gate", review_path="data/review/decisions.jsonl") -> list:
    review = load_review(review_path)
    flips = set()                                   # false failures found by 02g_triage_failures.py under the corrected checks
    tp = Path(llm_dir) / "triage.jsonl"
    if tp.exists():
        import json
        flips = {(x["file"], x["item_id"]) for x in map(json.loads, tp.read_text().splitlines()) if x["decision"] == "flip"}
    sim_min = cfg.get("rendering", {}).get("semantic_similarity_min", 0.80)
    conds = build_conditions(cfg)
    dialect = {c.level: {r["item_id"]: r for r in read_rows(Path(dialect_dir) / f"rendered_{slug(c.level)}.jsonl")}
               for c in conds if c.axis == "dialect"}
    llm, gate = {}, {}
    for c in conds:
        if c.axis in ("resource", "codeswitch", "codeswitch_native"):
            f = f"rendered_{llm_slug(c)}.jsonl"
            llm[c.cid] = latest_by_item(read_rows(Path(llm_dir) / f))
            gate[c.cid] = latest_by_item(read_rows(Path(gate_dir) / f.replace("rendered_", "gate_")))
    rows = []
    for it in items:
        for c in conds:
            if c.axis == "baseline":
                rows.append(_row(it, c, rendered=it["text"], options=it["options"], method="original", render_ok=True, render_status="ok"))
            elif c.axis == "dialect":
                r = dialect[c.level].get(it["item_id"])
                if r is None:
                    rows.append(_row(it, c))
                elif r.get("error") or r.get("rendered") is None:
                    rows.append(_row(it, c, method="multivalue", render_status="error"))
                else:
                    rows.append(_row(it, c, rendered=r["rendered"], options=r.get("rendered_options"), method="multivalue",
                                     render_ok=True, render_status="ok"))
            else:
                r = llm[c.cid].get(it["item_id"])
                if r is None or r.get("rendered") is None:
                    rows.append(_row(it, c, method="llm", render_status="missing" if r is None else "error", round=(r or {}).get("round")))
                    continue
                g = gate[c.cid].get(it["item_id"])
                gate_passed = None if (c.axis == "resource" or g is None or g.get("round") != r.get("round")) else bool(g["passed"])
                ok = bool(r["passed"]) and gate_passed is not False
                status = "ok" if ok else ("failed_gate" if gate_passed is False else "failed_checks")
                if not r["passed"] and gate_passed is not False and (f"rendered_{llm_slug(c)}.jsonl", it["item_id"]) in flips:
                    ok, status = True, "ok_triage"
                if not r["passed"] and gate_passed is not False and _below_sim_only(r, sim_min):     # only the similarity threshold failed: a person decides
                    verdict = review.get((c.cid, it["item_id"]))
                    ok, status = (True, "ok_reviewed") if verdict == "pass" else (False, "failed_review" if verdict == "fail" else "needs_review")
                rows.append(_row(it, c, rendered=r["rendered"], options=r.get("rendered_options"), method=r.get("translator") or "llm",
                                 render_ok=ok, render_status=status, gate_passed=gate_passed, round=r.get("round")))
    return rows


def _below_sim_only(r: dict, sim_min: float) -> bool:
    ck = r.get("checks") or {}
    sim = ck.get("semantic_sim")
    others = [k for k in ("key_terms_ok", "langid_ok", "changed_ok", "no_repeated_content", "options_letters_ok") if ck.get(k) is False]
    return bool(r.get("rendered")) and sim is not None and sim < sim_min and not others


def quality_log(rows: list) -> list:
    out, by = [], {}
    for r in rows:
        by.setdefault(r["condition_id"], []).append(r)
    for cid, g in by.items():
        n = len(g)
        statuses = {s: sum(r["render_status"] == s for r in g) for s in ("ok", "ok_reviewed", "ok_triage", "needs_review", "failed_review", "failed_checks", "failed_gate", "error", "missing")}
        out.append({"condition_id": cid, "condition_axis": g[0]["condition_axis"], "condition_level": g[0]["condition_level"],
                    "language_pair": g[0]["language_pair"], "n": n, "share_ok": (statuses["ok"] + statuses["ok_reviewed"] + statuses["ok_triage"]) / n,
                    **{f"n_{k}": v for k, v in statuses.items()},
                    "gate_checked": sum(r["gate_passed"] is not None for r in g),
                    "flag_over_15pct_unrendered": (1 - (statuses["ok"] + statuses["ok_reviewed"] + statuses["ok_triage"]) / n) > 0.15})
    return out
