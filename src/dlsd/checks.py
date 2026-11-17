"""Automatic checks for translated and code-switched renderings (docs/EXPERIMENTS.md section 3.2).

All checks return plain values that are stored next to each rendering. None of them edit the rendering.
The code-switch type check is a heuristic (word-window language ID plus sentence split); it is stored as a flag,
not used to claim the switch type is verified.
"""
import re
import unicodedata

LANG_CODE = {"English": "en", "Spanish": "es", "Hindi": "hi", "Swahili": "sw"}
NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")
_SPLIT_RE = re.compile(r"(?<=[.!?।¿?])\s+")
_ABBREV = re.compile(r"(?:^|\s)(?:[A-Z]|Dr|Mr|Mrs|Ms|Prof|St|vs|spp?|ca|approx|Fig|No|e\.g|i\.e|cf)\.$")   # a period here does not end a sentence


class _SentenceSplitter:
    """Splits after . ! ? । but not after initials or abbreviations ("B. anthracis", "Dr. Smith", "e.g."), which would cut a sentence."""

    @staticmethod
    def split(text: str) -> list:
        out = []
        for part in _SPLIT_RE.split(text):
            if out and _ABBREV.search(out[-1]):
                out[-1] = out[-1] + " " + part
            else:
                out.append(part)
        return out


SENT_SPLIT = _SentenceSplitter()
REFUSAL_PATTERNS = [
    r"\bi (can(?:no|')t|cannot|am unable|'m unable|won't|will not)\b", r"\bi'?m sorry\b", r"\bas an ai\b",
    r"\bcan(?:no|')t (assist|help|comply|fulfill|translate|rewrite)\b",
    r"\blo siento\b", r"\bno puedo\b", r"\bsamahani\b", r"\bsiwezi\b",
    "मैं .{0,40}नहीं कर सक", "क्षमा", "माफ़? कीजिये",
]
_REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)


def normalize_digits(s: str) -> str:
    """Map any Unicode decimal digit (for example Devanagari) to ASCII so number checks are script independent."""
    return "".join(str(unicodedata.digit(c)) if c.isdigit() and not c.isascii() else c for c in s)


def _canon_numbers(s: str) -> str:
    """Decimal comma and decimal point are the same number (0.8 / 0,8); Spanish and Swahili text writes the comma."""
    return re.sub(r"(?<=\d)[.,](?=\d)", ".", normalize_digits(s))


NUMBER_WORDS = {      # small numbers are often written out in translation ("1 month" -> "un mes"); only 1 to 10 are accepted as words
    "Spanish": "uno un una dos tres cuatro cinco seis siete ocho nueve diez".split(),
    "Swahili": "moja mmoja mbili tatu nne tano sita saba nane tisa kumi".split(),
    "Hindi": "एक दो तीन चार पांच पाँच छह छः सात आठ नौ दस".split(),
}
_WORD_VALUES = {"Spanish": [{"uno", "un", "una"}, {"dos"}, {"tres"}, {"cuatro"}, {"cinco"}, {"seis"}, {"siete"}, {"ocho"}, {"nueve"}, {"diez"}],
                "Swahili": [{"moja", "mmoja", "mmoja"}, {"mbili", "miwili", "wawili", "mawili"}, {"tatu", "watatu", "mitatu", "matatu"}, {"nne", "wanne", "minne", "manne"}, {"tano", "watano", "mitano", "matano"}, {"sita"}, {"saba"}, {"nane"}, {"tisa"}, {"kumi"}],
                "Hindi": [{"एक"}, {"दो"}, {"तीन"}, {"चार"}, {"पांच", "पाँच"}, {"छह", "छः"}, {"सात"}, {"आठ"}, {"नौ"}, {"दस"}]}


def _spelled(n: str, out: str, lang: str | None) -> bool:
    if lang not in _WORD_VALUES or not n.isdigit() or not 1 <= int(n) <= 10:
        return False
    words = set(re.findall(r"[\w\u0900-\u097F]+", out.lower()))      # Devanagari vowel signs are not \w
    return bool(words & _WORD_VALUES[lang][int(n) - 1])


def numbers_preserved(src: str, out: str, lang: str | None = None) -> bool:
    """Every number of `src` is in `out` (decimal comma = point). With `lang`, a bare 1 to 10 may also appear as a word."""
    out_n = _canon_numbers(out)
    return all(n in out_n or _spelled(n, out_n, lang) for n in NUM_RE.findall(_canon_numbers(src)))


def key_terms(src: str) -> set:
    """Capitalised words that are not sentence-initial and not acronyms (names, brands). Acronyms localise (RNA -> ARN,
    HIV -> VIH), so they are excluded here and only recorded by `acronyms`."""
    terms = set()
    for sent in SENT_SPLIT.split(src):
        words = re.findall(r"[A-Za-z][A-Za-z\-]+", sent)
        terms |= {w for w in words[1:] if w[0].isupper() and not w.isupper()}
    return terms


def acronyms(src: str) -> set:
    return set(re.findall(r"\b[A-Z]{2,}\b", src))


def missing_terms(src, out: str, target_lang: str | None) -> list:
    """Named terms (capitalised, mid-sentence, not acronyms) of the pieces that do not appear in `out`. Hindi has none (script differs)."""
    if target_lang == "Hindi":
        return []
    pieces = [src] if isinstance(src, str) else list(src)
    low = out.lower()
    return sorted({t for p in pieces for t in key_terms(p) if t.lower() not in low})


def key_terms_preserved(src, out: str, target_lang: str | None, similar_enough: bool = False) -> bool:
    """`src` is a string or a list of pieces (stem, each option). Terms are taken per piece so an option's first word
    is not mistaken for a mid-sentence proper noun. Numbers always (decimal comma and spelled-out 1 to 10 accepted); named terms
    only for Latin-script targets. A missing term is tolerated when the whole text is semantically close (`similar_enough`):
    terms such as "X-ray" or "VRs" are legitimately localised ("radiografia")."""
    pieces = [src] if isinstance(src, str) else list(src)
    if not all(numbers_preserved(p, out, target_lang) for p in pieces):
        return False
    return similar_enough or not missing_terms(src, out, target_lang)


def looks_like_refusal(text: str, src: str) -> bool:
    """Renderer refused or answered instead of rewriting: refusal phrases, or output far longer than the source."""
    phrase = bool(_REFUSAL_RE.search(text or "")) and not _REFUSAL_RE.search(src or "")   # source may itself say "I can't sleep"
    return phrase or len(text or "") > 4 * max(len(src), 40)


_langid = None


def _classifier(langs):
    global _langid
    if _langid is None:
        import langid
        from langid.langid import LanguageIdentifier, model
        _langid = LanguageIdentifier.from_modelstring(model, norm_probs=True)
    _langid.set_languages(sorted(set(langs)))
    return _langid


def detect(text: str, langs=("en", "es", "hi", "sw")) -> str:
    return _classifier(langs).classify(text)[0]


def langid_ok_translation(rendered: str, target: str) -> bool:
    return detect(rendered) == LANG_CODE[target]


def _windows(text: str, size: int = 4):
    """Label consecutive word windows as en or the other language. Returns list of (sentence_idx, label)."""
    out = []
    for si, sent in enumerate(SENT_SPLIT.split(text)):
        words = sent.split()
        for i in range(0, max(len(words), 1), size):
            chunk = " ".join(words[i:i + size])
            if chunk:
                out.append((si, chunk))
    return out


def codeswitch_profile(rendered: str, target: str) -> dict:
    """Word-window language ID: share of windows in each language and whether any sentence mixes both."""
    t = LANG_CODE[target]
    labelled = [(si, detect(c, ("en", t))) for si, c in _windows(rendered)]
    if not labelled:
        return {"both_present": False, "share_target": 0.0, "mixed_sentence": False, "sentences": 0}
    share_t = sum(l == t for _, l in labelled) / len(labelled)
    by_sent: dict = {}
    for si, l in labelled:
        by_sent.setdefault(si, set()).add(l)
    return {"both_present": 0 < share_t < 1, "share_target": share_t,
            "mixed_sentence": any(len(v) > 1 for v in by_sent.values()), "sentences": len(by_sent)}


def switch_type_ok(rendered: str, target: str, switch_type: str) -> bool:
    """Heuristic match of the intended switch type (Poplack 1980); see the module docstring."""
    p = codeswitch_profile(rendered, target)
    if not p["both_present"]:
        return False
    if switch_type == "intra":
        return p["mixed_sentence"]
    if switch_type == "inter":
        return p["sentences"] >= 2 and not p["mixed_sentence"]
    if switch_type == "extra":
        return 0 < p["share_target"] <= 0.35
    raise ValueError(switch_type)


class Similarity:
    """Cosine similarity with a multilingual sentence encoder (default LaBSE). Loaded lazily so tests can skip it."""

    def __init__(self, model_name: str = "sentence-transformers/LaBSE", device: str | None = None):
        self.model_name, self.device, self._m = model_name, device, None

    def __call__(self, a: str, b: str) -> float:
        if self._m is None:
            from sentence_transformers import SentenceTransformer
            self._m = SentenceTransformer(self.model_name, device=self.device)
        ea, eb = self._m.encode([a, b], normalize_embeddings=True)
        return float(ea @ eb)

    def max_pair(self, sentences: list) -> float:
        """Highest cosine similarity between any two different sentences (detects content repeated in two languages)."""
        if len(sentences) < 2:
            return 0.0
        if self._m is None:
            from sentence_transformers import SentenceTransformer
            self._m = SentenceTransformer(self.model_name, device=self.device)
        e = self._m.encode(sentences, normalize_embeddings=True)
        m = e @ e.T
        return float(max(m[i, j] for i in range(len(sentences)) for j in range(i + 1, len(sentences))))
