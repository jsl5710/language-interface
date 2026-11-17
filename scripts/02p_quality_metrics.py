"""Quality metrics for the rendered conditions (CPU only; reads data/llm, data/gate, data/items.jsonl).

Code-switch conditions (per language pair x switch type x base):
  mix error rate   share of rendered items whose realised switch type, as labelled by the judge panel, is not the intended one
                   (1 - gate pass rate, round 1 and latest round; items that could not be rendered are counted separately)
  CMI              Code-Mixing Index (Das & Gamback 2014), 0-100: 100 * (1 - max_lang_tokens / tagged_tokens)
  M-index          Barnett et al. 2000 language-entropy index, 0 (monolingual) to 1 (balanced)
  I-index          Guzman et al. 2017: switch points / (tagged tokens - 1)
  mixed-sentence share, target-language token share
  Token language is a proxy (no gold tags): a Latin token is English if it occurs in the English source text, any other word is the
  partner language; Devanagari tokens are Hindi. Cognates and shared medical terms therefore count as English.
Translation conditions (resource level):
  LaBSE cosine with the source (stored by the renderer), numbers preserved, named terms preserved, length ratio (chars, target/source),
  copy rate (target shares >= 80% of its tokens with the source), pass rate of the automatic checks.
Writes results/aggregate/quality_metrics.{csv,json} and paper/tables/quality_*.tex.
"""
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dlsd import checks
from dlsd.gate_io import GATE_DIR, LLM_DIR, latest_by_item, read_rows

WORD = re.compile(r"[A-Za-zÀ-ɏ]+|[ऀ-ॿ]+")
OUT = Path("results/aggregate")
TAB = Path("paper/tables")


def tag_tokens(text: str, source: str, partner_is_hindi: bool) -> list:
    src = {w.lower() for w in WORD.findall(source)}
    tags = []
    for w in WORD.findall(text):
        if "ऀ" <= w[0] <= "ॿ":
            tags.append("T")
        elif partner_is_hindi:
            tags.append("E")                                  # Latin word inside Hindi text is English
        else:
            tags.append("E" if w.lower() in src else "T")
    return tags


def indices(tags: list) -> dict:
    n = len(tags)
    if n < 2:
        return {"cmi": 0.0, "m_index": 0.0, "i_index": 0.0, "share_target": 0.0}
    e, t = tags.count("E"), tags.count("T")
    cmi = 100 * (1 - max(e, t) / n)
    pe, pt = e / n, t / n
    m = (1 - pe ** 2 - pt ** 2) / (pe ** 2 + pt ** 2)         # Barnett et al. 2000 with k = 2 languages: (1 - sum p^2) / ((k - 1) sum p^2)
    switches = sum(a != b for a, b in zip(tags, tags[1:]))
    return {"cmi": cmi, "m_index": m, "i_index": switches / (n - 1), "share_target": pt}


def mixed_sentence_share(text: str, source: str, hindi: bool) -> float:
    sents = [s for s in checks.SENT_SPLIT.split(text) if s.strip()]
    if not sents:
        return 0.0
    return sum(len(set(tag_tokens(s, source, hindi))) > 1 for s in sents) / len(sents)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    items = {r["item_id"]: r for r in read_rows("data/items.jsonl")}
    rows = []
    for shard in sorted(LLM_DIR.glob("rendered_*.jsonl")):
        allrows = read_rows(shard)
        if not allrows:
            continue
        first = {r["item_id"]: r for r in allrows if r.get("round", 1) == 1}
        last = latest_by_item(allrows)
        gate = Path(GATE_DIR / shard.name.replace("rendered_", "gate_"))
        g1 = latest_by_item([r for r in read_rows(gate) if r["round"] == 1])
        gl = latest_by_item(read_rows(gate))
        axis = allrows[0]["condition_axis"]
        lang, level = allrows[0]["language_pair"], allrows[0]["condition_level"]
        hindi = lang == "Hindi"
        cond = {"axis": axis, "language": lang, "level": level, "n_items": len(last)}
        if axis.startswith("codeswitch"):
            rend = [r for r in last.values() if not r.get("not_renderable") and r.get("rendered")]
            cond["n_rendered"] = len(rend)
            cond["not_renderable"] = len(last) - len(rend)
            if g1:
                cond["mix_error_round1"] = 1 - mean([g1[k]["passed"] for k in g1 if not g1[k].get("not_renderable")])
            if gl:
                cond["mix_error_latest"] = 1 - mean([gl[k]["passed"] for k in gl if not gl[k].get("not_renderable")])
            per = []
            for r in rend:
                src = items[r["item_id"]]["text"]
                idx = indices(tag_tokens(r["rendered"], src, hindi))
                idx["mixed_sentences"] = mixed_sentence_share(r["rendered"], src, hindi)
                per.append(idx)
            for k in ("cmi", "m_index", "i_index", "share_target", "mixed_sentences"):
                cond[k] = mean([p[k] for p in per])
        else:
            rend = [r for r in last.values() if not r.get("not_renderable") and r.get("rendered")]
            cond["n_rendered"] = len(rend)
            cond["sim_labse"] = mean([r["checks"].get("semantic_sim") for r in rend if r.get("checks")])
            cond["pass_rate"] = mean([bool(r.get("passed")) for r in rend])
            cond["numbers_ok"] = mean([checks.numbers_preserved(items[r["item_id"]]["text"], r["rendered"], lang) for r in rend])
            cond["terms_ok"] = mean([not checks.missing_terms(items[r["item_id"]]["text"], r["rendered"], lang) for r in rend])
            cond["length_ratio"] = mean([len(r["rendered"]) / max(len(items[r["item_id"]]["text"]), 1) for r in rend])
            cond["copy_rate"] = mean([len(set(WORD.findall(r["rendered"].lower())) & set(WORD.findall(items[r["item_id"]]["text"].lower())))
                                      / max(len(set(WORD.findall(r["rendered"].lower()))), 1) >= 0.8 for r in rend])
        rows.append(cond)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(OUT / "quality_metrics.json", "w"), indent=1)
    import csv
    keys = sorted({k for r in rows for k in r})
    with open(OUT / "quality_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    write_tables(rows)
    for r in rows:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})


def fmt(x, d=2):
    return "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def write_tables(rows):
    TAB.mkdir(parents=True, exist_ok=True)
    cs = [r for r in rows if r["axis"].startswith("codeswitch")]
    lines = [r"\begin{tabular}{llllrrrrr}", r"\toprule",
             r"Base & Language & Type & Mix err. r1 & Mix err. final & CMI & M & I & Target share \\", r"\midrule"]
    for r in sorted(cs, key=lambda r: (r["axis"], r["language"], r["level"])):
        base = "native" if r["axis"].endswith("native") else "English"
        lines.append(f"{base} & {r['language']} & {r['level']} & {fmt(r.get('mix_error_round1'))} & {fmt(r.get('mix_error_latest'))} & "
                     f"{fmt(r.get('cmi'), 1)} & {fmt(r.get('m_index'))} & {fmt(r.get('i_index'))} & {fmt(r.get('share_target'))} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "quality_codeswitch.tex").write_text("\n".join(lines) + "\n")
    tr = [r for r in rows if r["axis"] == "resource"]
    lines = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r"Resource & Language & LaBSE & Numbers kept & Names kept & Length ratio & Copy rate & Checks passed \\", r"\midrule"]
    for r in tr:
        lines.append(f"{r['level']} & {r['language']} & {fmt(r['sim_labse'], 3)} & {fmt(r['numbers_ok'])} & {fmt(r['terms_ok'])} & "
                     f"{fmt(r['length_ratio'])} & {fmt(r['copy_rate'])} & {fmt(r['pass_rate'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "quality_translation.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
