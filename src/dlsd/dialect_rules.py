"""Rule-based dialect rendering with Multi-VALUE (Ziems et al. 2023; https://github.com/SALT-NLP/multi-value).
Feature definitions come from eWAVE (https://ewave-atlas.org/). No language model generates the text, so output is
deterministic for a fixed commit and fixed seed (the library draws random numbers for rule application, so we seed it).

Multi-VALUE is loaded from a checkout at the pinned commit (cfg rendering.dialect.checkout), not from PyPI.
Each Multi-VALUE dialect object builds its own Stanza coref pipeline (about 2 GB), so we build one and share it.
"""
import os
import re
from collections import Counter
import sys
from dataclasses import dataclass, field

_DIALECTS: dict = {}
_SHARED_COREF = None


@dataclass
class DialectResult:
    text: str
    rules_fired: list = field(default_factory=list)   # eWAVE-derived rule names reported by Multi-VALUE
    changed: bool = False                             # False means the dialect rendering equals the SAE source
    protected: bool = False                           # True if hyphenated/slashed terms were masked to avoid a tokenizer mismatch
    mode: str = "whole"                               # whole | masked | sentence (per-sentence fallback)
    n_sentence_failures: int = 0                      # sentences left unchanged because Multi-VALUE failed on them


def _checkout_path(cfg: dict) -> str:
    d = cfg["rendering"]["dialect"]
    return os.path.abspath(os.path.expanduser(d.get("checkout", "../external/multi-value"))) + "/src"


def _get_dialect(name: str, cfg: dict):
    """Build (once) the Multi-VALUE dialect object for a paper dialect name."""
    global _SHARED_COREF
    if name in _DIALECTS:
        return _DIALECTS[name]
    src = _checkout_path(cfg)
    if src not in sys.path:
        sys.path.insert(0, src)
    import stanza
    from multivalue import Dialects

    cls_name = cfg["rendering"]["dialect"]["dialect_map"][name]
    if _SHARED_COREF is None:
        real_pipeline = stanza.Pipeline
        obj = getattr(Dialects, cls_name)(seed=cfg["seed"])
        _SHARED_COREF = obj.coref
    else:
        # Reuse the coref pipeline instead of loading another copy.
        real_pipeline = stanza.Pipeline
        stanza.Pipeline = lambda *a, **k: _SHARED_COREF
        try:
            obj = getattr(Dialects, cls_name)(seed=cfg["seed"])
        finally:
            stanza.Pipeline = real_pipeline
    _DIALECTS[name] = obj
    return obj


_COMPOUND = re.compile(r"\b[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)+\b")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
_NUMERIC_EXPR = re.compile(r"\d+(?:[.,]\d+)?(?:\s?[-\u2013/]\s?\d+(?:[.,]\d+)?)*%?|" + _COMPOUND.pattern)   # ranges, ratios, percentages, compounds


def numbers_intact(src: str, out: str) -> bool:
    """Every number token of the source occurs at least as often in the output (a rule must not turn '2-3' into '2-')."""
    need, have = Counter(_NUMBER.findall(src)), Counter(_NUMBER.findall(out))
    return not (need - have)


def _mask(text: str, numbers: bool = False):
    """Swap hyphenated or slashed terms (COVID-19, beta-blocker, and/or) for letter-only placeholders. With numbers=True,
    numeric expressions (2-3, 120/80, 5%) are masked too."""
    terms: list = []

    def sub(m):
        terms.append(m.group(0))
        return "Zqx" + "".join(chr(97 + int(d)) for d in str(len(terms) - 1)) + "xqz"
    return (_NUMERIC_EXPR if numbers else _COMPOUND).sub(sub, text), terms


def _unmask(text: str, terms: list):
    for i, t in enumerate(terms):
        tag = "Zqx" + "".join(chr(97 + int(d)) for d in str(i)) + "xqz"
        if tag not in text:
            return None            # a rule changed or dropped the placeholder: treat as failure, do not guess
        text = text.replace(tag, t)
    return text


def _run(d, text: str):
    out = d.transform(text)
    return out, sorted({v["type"] for v in dict(d.executed_rules).values()})


_ABBREV = ("Dr", "Mr", "Mrs", "Ms", "vs", "St", "No", "Fig", "approx", "e.g", "i.e", "etc")
_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def split_sentences(text: str) -> list:
    """Deterministic splitter for the fallback: break after . ! ? before a capital/digit, but not after common abbreviations."""
    parts, start = [], 0
    for m in _BOUNDARY.finditer(text):
        head = text[start:m.start()]
        if head.split(" ")[-1].rstrip(".") in _ABBREV:
            continue
        parts.append((head, text[m.start():m.end()]))
        start = m.end()
    parts.append((text[start:], ""))
    return parts          # list of (sentence, trailing whitespace) so the text can be rebuilt exactly


def _try(d, text: str, seed: int):
    """Transform whole `text` and return (out, rules, protected), or None. Passes: plain, compounds masked, compounds and numbers
    masked. A pass whose output changed a number is skipped (the first such output is kept only if no pass keeps numbers)."""
    best = None
    for mode in ("plain", "compounds", "numbers"):
        t, terms = (text, []) if mode == "plain" else _mask(text, numbers=(mode == "numbers"))
        if mode != "plain" and not terms:
            continue
        d.set_seed(seed)
        try:
            out, fired = _run(d, t)
        except Exception:  # noqa: BLE001 Multi-VALUE raises AssertionError, KeyError and others on odd input
            continue
        out = _unmask(out, terms) if mode != "plain" else out
        if out is None:
            continue
        if numbers_intact(text, out):
            return out, fired, mode != "plain"
        best = best or (out, fired, mode != "plain")
    return best


def transform(text: str, dialect: str, cfg: dict) -> DialectResult:
    """Return the dialect rendering of `text` (inputs: SAE text, paper dialect name, config).
    1. whole text; 2. whole text with hyphenated/slashed compounds masked (Multi-VALUE asserts that spaCy and Stanza
    tokenise alike, which COVID-19 breaks); 3. per-sentence fallback for long multi-sentence items, where a sentence that
    still fails stays unchanged and is counted. Raises only if every sentence fails."""
    d, seed = _get_dialect(dialect, cfg), cfg["seed"]
    r = _try(d, text, seed)
    if r:
        return DialectResult(text=r[0], rules_fired=r[1], changed=(r[0] != text), protected=r[2], mode="masked" if r[2] else "whole")
    pieces, out, fired, fails = split_sentences(text), [], set(), 0
    for sent, gap in pieces:
        rs = _try(d, sent, seed) if sent.strip() else None
        if rs:
            out.append(rs[0] + gap)
            fired |= set(rs[1])
        else:
            out.append(sent + gap)
            fails += bool(sent.strip())
    if fails == sum(bool(s.strip()) for s, _ in pieces):
        raise ValueError("Multi-VALUE failed on every sentence")
    joined = "".join(out)
    return DialectResult(text=joined, rules_fired=sorted(fired), changed=(joined != text), mode="sentence", n_sentence_failures=fails)
