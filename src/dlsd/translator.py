"""Dedicated machine-translation system for the resource-level and code-switch pieces (NLLB-200 3.3B, chosen by the
pre-registered bake-off, results/aggregate/translator_bakeoff). Deterministic (beam search, no sampling) and it never refuses.

Multi-sentence text is translated sentence by sentence so sentence counts are preserved; sentences are batched together."""
import os
import re
from pathlib import Path

from .checks import SENT_SPLIT

NLLB_CODE = {"Spanish": "spa_Latn", "Hindi": "hin_Deva", "Swahili": "swh_Latn"}


_PAREN = re.compile(r"\s*\(([^()]*\d[^()]*)\)")
_ANCHOR = re.compile(r"(\d[\d.,]*\s?[°%A-Za-z/]*)$")


def strip_value_parens(text: str):
    """Remove parentheticals that hold a number and follow a quantity ("101.8°F (38.8°C)", "6 kg (13.2 lb)"). NLLB tends to drop
    such conversions. Returns (text without them, [(anchor, parenthetical)]) so they can be put back verbatim after the anchor."""
    kept, out, pos = [], [], 0
    for m in _PAREN.finditer(text):
        a = _ANCHOR.search(text[:m.start()])
        if not a:
            continue
        kept.append(text[pos:m.start()])
        pos = m.end()
        out.append((a.group(1).strip(), m.group(0).strip()))
    kept.append(text[pos:])
    return "".join(kept), out


def restore_value_parens(translated: str, parens: list) -> str:
    """Re-insert each parenthetical right after its anchor quantity in the translation: the number (decimal comma accepted), the unit
    attached to it ("°F", "%"), and, if the source anchor had a separate unit, the next word (a translated unit such as "किलो").
    If an anchor cannot be found the translation is returned unrepaired, so the number check reports the loss instead of hiding it."""
    out = translated
    for anchor, paren in parens:
        num = re.match(r"\d[\d.,]*", anchor).group(0)
        unit = anchor[len(num):]
        placed = False
        for cand in dict.fromkeys([num, re.sub(r"(?<=\d)\.(?=\d)", ",", num)]):
            for m in reversed(list(re.finditer(re.escape(cand) + r"(?![\d])", out))):
                end = m.end()
                tail = re.match(r"[°%]\w*", out[end:])
                if tail:
                    end += tail.end()
                elif unit.strip():
                    nxt = re.match(r"\s?[^\s\d.,;:()]{1,15}", out[end:])
                    if nxt and not re.match(r"\s?[,.;:]", out[end:]):
                        end += nxt.end()
                out = out[:end] + " " + paren + out[end:]
                placed = True
                break
            if placed:
                break
        if not placed:
            return translated
    return out


class NLLBTranslator:
    def __init__(self, path: str | None = None, batch_size: int = 16, num_beams: int = 4):
        self.path = Path(path) if path else Path(os.environ["DLSD_MODELS_DIR"]) / "nllb-200-3.3B"
        self.batch_size, self.num_beams, self._m = batch_size, num_beams, None

    def _load(self):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self._tok = AutoTokenizer.from_pretrained(self.path, src_lang="eng_Latn")
        self._m = AutoModelForSeq2SeqLM.from_pretrained(self.path, dtype=torch.bfloat16).cuda().eval()

    def _translate_sentences(self, sents: list, lang: str, n_best: int = 1):
        """n_best=1 returns one string per sentence; n_best>1 returns a list of candidate strings per sentence."""
        import torch
        if self._m is None:
            self._load()
        order = sorted(range(len(sents)), key=lambda i: len(sents[i]))
        out = [""] * len(sents) if n_best == 1 else [[] for _ in sents]
        for b in range(0, len(order), self.batch_size):
            ids = order[b:b + self.batch_size]
            enc = self._tok([sents[i] for i in ids], return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
            with torch.no_grad():
                gen = self._m.generate(**enc, forced_bos_token_id=self._tok.convert_tokens_to_ids(NLLB_CODE[lang]),
                                       num_beams=max(self.num_beams, n_best), num_return_sequences=n_best, do_sample=False, max_new_tokens=512)
            dec = [t.strip() for t in self._tok.batch_decode(gen, skip_special_tokens=True)]
            for k, i in enumerate(ids):
                out[i] = dec[k] if n_best == 1 else list(dict.fromkeys(dec[k * n_best:(k + 1) * n_best]))
        return out

    def _plain(self, texts: list, lang: str) -> list:
        split = [[x for x in SENT_SPLIT.split(t) if x.strip()] or [t] for t in texts]
        flat = self._translate_sentences([x for ss in split for x in ss], lang)
        res, k = [], 0
        for ss in split:
            res.append(" ".join(flat[k:k + len(ss)]))
            k += len(ss)
        return res

    def translate(self, texts: list, lang: str) -> list:
        """English texts -> `lang`, same order. If NLLB drops the number in a parenthetical conversion ("101.8°F (38.8°C)"), the text is
        translated again with the parenthetical held out and it is put back verbatim after its quantity. Never worse than the plain
        translation: the repaired text is used only when it keeps every number the plain one lost."""
        from .checks import numbers_preserved
        plain = self._plain(texts, lang)
        redo = [i for i, t in enumerate(texts) if strip_value_parens(t)[1] and not numbers_preserved(t, plain[i], lang)]
        if redo:
            held = [strip_value_parens(texts[i]) for i in redo]
            fixed = self._plain([h[0] for h in held], lang)
            for i, (_, parens), f in zip(redo, held, fixed):
                cand = restore_value_parens(f, parens)
                if numbers_preserved(texts[i], cand, lang):
                    plain[i] = cand
        return plain

    def translate_nbest(self, texts: list, lang: str, n: int = 4) -> list:
        """Candidate translations per short piece (one sentence each), best first, duplicates removed."""
        return self._translate_sentences(texts, lang, n_best=n)
