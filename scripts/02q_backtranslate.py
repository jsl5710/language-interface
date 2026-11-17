"""Stage 2q: back-translation quality of the resource-level renderings (NLLB-200 3.3B, target -> English) and chrF++ against the source.

    python scripts/02q_backtranslate.py translate     # GPU: back-translate each rendered stem -> data/backtranslation/<language>.jsonl (resumable)
    python scripts/02q_backtranslate.py score         # CPU: chrF++ and BLEU of the back-translation against the original English
                                                      #   -> results/aggregate/backtranslation.{csv,json}, paper/tables/quality_backtranslation.tex

Back-translation chrF++ is a reference-based proxy for meaning preservation: the English original is the reference. It is lower than the
forward quality (two translation steps, and NLLB is the same system in both directions, so errors can cancel); it is reported as a
sanity check beside the LaBSE similarity, not as a ground-truth accuracy. Only the question stem or prompt is back-translated; answer options are not.
Code-switched renderings are not back-translated (the partner-language spans are partial sentences).
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dlsd.checks import SENT_SPLIT
from dlsd.gate_io import LLM_DIR, latest_by_item, read_rows
from dlsd.translator import NLLB_CODE

OUT = Path("data/backtranslation")
SHARDS = {"Spanish": "rendered_resource_high_spanish.jsonl", "Hindi": "rendered_resource_mid_hindi.jsonl", "Swahili": "rendered_resource_low_swahili.jsonl"}


def translate():
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    path = Path(os.environ["DLSD_MODELS_DIR"]) / "nllb-200-3.3B"
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSeq2SeqLM.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()
    OUT.mkdir(parents=True, exist_ok=True)
    for lang, shard in SHARDS.items():
        out = OUT / f"{lang}.jsonl"
        done = {r["item_id"] for r in read_rows(out)}
        rows = [r for r in latest_by_item(read_rows(LLM_DIR / shard)).values() if r.get("rendered") and not r.get("not_renderable") and r["item_id"] not in done]
        print(f"{lang}: {len(done)} done, {len(rows)} to do", flush=True)
        tok.src_lang = NLLB_CODE[lang]
        sents = [(i, s) for i, r in enumerate(rows) for s in ([x for x in SENT_SPLIT.split(r["rendered"]) if x.strip()] or [r["rendered"]])]
        order = sorted(range(len(sents)), key=lambda k: len(sents[k][1]))
        back = [""] * len(sents)
        for b in range(0, len(order), 16):
            ids = order[b:b + 16]
            enc = tok([sents[k][1] for k in ids], return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, forced_bos_token_id=tok.convert_tokens_to_ids("eng_Latn"), num_beams=4, do_sample=False, max_new_tokens=512)
            for k, t in zip(ids, tok.batch_decode(gen, skip_special_tokens=True)):
                back[k] = t.strip()
        joined = {}
        for (i, _), t in zip(sents, back):
            joined.setdefault(i, []).append(t)
        with open(out, "a", encoding="utf-8") as f:
            for i, r in enumerate(rows):
                f.write(json.dumps({"item_id": r["item_id"], "language": lang, "rendered": r["rendered"], "back": " ".join(joined[i])}, ensure_ascii=False) + "\n")


def score():
    import csv
    import random
    import sacrebleu
    items = {r["item_id"]: r for r in read_rows("data/items.jsonl")}
    chrf, bleu = sacrebleu.metrics.CHRF(word_order=2), sacrebleu.metrics.BLEU()
    rng, rows = random.Random(13), []
    for lang in SHARDS:
        recs = read_rows(OUT / f"{lang}.jsonl")
        hyp = [r["back"] for r in recs]
        ref = [items[r["item_id"]]["text"] for r in recs]
        per = [chrf.sentence_score(h, [x]).score for h, x in zip(hyp, ref)]
        boots = []
        for _ in range(300):
            idx = [rng.randrange(len(recs)) for _ in recs]
            boots.append(chrf.corpus_score([hyp[i] for i in idx], [[ref[i] for i in idx]]).score)
        boots.sort()
        rows.append({"language": lang, "n": len(recs), "chrfpp": chrf.corpus_score(hyp, [ref]).score, "chrfpp_lo": boots[7], "chrfpp_hi": boots[292],
                     "bleu": bleu.corpus_score(hyp, [ref]).score, "item_chrfpp_median": sorted(per)[len(per) // 2],
                     "share_below_40": sum(p < 40 for p in per) / len(per)})
    Path("results/aggregate").mkdir(parents=True, exist_ok=True)
    json.dump(rows, open("results/aggregate/backtranslation.json", "w"), indent=1)
    with open("results/aggregate/backtranslation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule", r"Language & n & chrF++ (95\% CI) & BLEU & Median item chrF++ & Items below 40 \\", r"\midrule"]
    for r in rows:
        lines.append(f"{r['language']} & {r['n']} & {r['chrfpp']:.1f} ({r['chrfpp_lo']:.1f}--{r['chrfpp_hi']:.1f}) & {r['bleu']:.1f} & "
                     f"{r['item_chrfpp_median']:.1f} & {100 * r['share_below_40']:.0f}\\% \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    Path("paper/tables/quality_backtranslation.tex").write_text("\n".join(lines) + "\n")
    for r in rows:
        print({k: round(v, 2) if isinstance(v, float) else v for k, v in r.items()})


if __name__ == "__main__":
    {"translate": translate, "score": score}[sys.argv[1]]()
