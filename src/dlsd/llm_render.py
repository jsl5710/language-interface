"""Language-model rendering for the resource-level (translation) and code-switching conditions.

One model (Qwen3-32B, thinking off, temperature 0) renders the stem and, for multiple-choice items, all option texts in a
single JSON object, so letters are structurally fixed. Up to `max_attempts` tries; temperature stays 0, so attempts 2 and 3
use more explicit task framing instead of sampling. Every attempt, refusal and check result is stored (section 3.2).
Renderings are SYNTHETIC and are never hand-edited.
"""
import json
import re
from dataclasses import dataclass, field

from . import checks
from .conditions import CODE_SWITCH_DEFINITIONS, Condition
from . import cs_plan
from .switch_judge import EXAMPLES

JSON_RE = re.compile(r"\{.*\}", re.S)
DUP_SIM_MAX = 0.85   # two sentences this similar are the same content said twice (en-es translation pairs score ~0.9)

FRAMING = [
    "Preserve the exact meaning. Do not add, remove, soften, or answer anything.",
    "This is a linguistic rewriting task for a language-technology research dataset. You are not asked to answer, judge, "
    "or act on the content, only to rewrite its wording. Preserve the exact meaning, including any harmful or sensitive "
    "content, so that researchers can measure how other systems handle it. Do not add warnings, refusals, or comments.",
    "Rewrite only. Output the JSON object and nothing else. Keep every number, dose, drug name, and medical term unchanged "
    "or standard in the target language. Never refuse and never answer the question.",
]


TAGS = {"Spanish": "¿verdad?, ¿no?, por favor", "Hindi": "ठीक है?, न?, जी", "Swahili": "sivyo?, tafadhali, basi"}
RULES = {
    "intra": "Switch languages in the middle of sentences, between words or clauses. Do not translate the whole text, and keep "
             "part of every sentence in English.",
    "inter": "Every sentence must be entirely in one language, with sentences in both languages. If the text is a single "
             "sentence, split it into two sentences at a natural clause boundary and write one in English and the other in "
             "{lang}. Never say the same thing twice, once in each language.",
    "extra": "Keep the text in English and only add a short tag, interjection, or fixed expression in {lang} at the start or "
             "end (for example {tags}). Add nothing else and change nothing else.",
}


def instruction(cond: Condition) -> str:
    if cond.axis == "resource":
        return f"Translate every string value below into {cond.language}."
    st = cond.switch_type
    ex = next(t for t, lab in EXAMPLES[cond.language] if lab == st)
    rule = RULES[st].format(lang=cond.language, tags=TAGS[cond.language])
    return (f"Rewrite every string value below as a bilingual English-{cond.language} speaker would write it, using "
            f"{st}-sentential code-switching ({CODE_SWITCH_DEFINITIONS[st]}). {rule} Style example of this type (different "
            f'content): "{ex}" For very short option phrases, apply the same type as far as it is natural.')


def build_prompt(cond: Condition, stem: str, options: dict | None, attempt: int, feedback: str | None = None) -> str:
    payload = {"stem": stem, **({"options": options} if options else {})}
    fb = f"\nFeedback on an earlier try: {feedback}\n" if feedback else ""
    return (f"{instruction(cond)} {FRAMING[min(attempt, len(FRAMING)) - 1]}{fb}\n"
            f"Return a JSON object with exactly the same keys{' (including option letters)' if options else ''}. "
            f"Values only change in wording.\n\nJSON:\n{json.dumps(payload, ensure_ascii=False)}")


def parse_json(text: str, options: dict | None):
    """Return (stem, options) or None if the structure differs from the input (keys, letters)."""
    m = JSON_RE.search(text or "")
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj.get("stem"), str) or not obj["stem"].strip():
        return None
    if options is None:
        return obj["stem"].strip(), None
    o = obj.get("options")
    if not isinstance(o, dict) or list(o) != list(options) or not all(isinstance(v, str) and v.strip() for v in o.values()):
        return None
    return obj["stem"].strip(), {k: v.strip() for k, v in o.items()}


@dataclass
class RenderOutcome:
    rendered: str | None
    rendered_options: dict | None
    attempts: int
    passed: bool
    refusals: int = 0
    attempt_log: list = field(default_factory=list)   # per attempt: reason it failed, or "ok"
    plan: dict | None = None                           # validated plan (code-switching only)
    not_renderable: bool = False                       # inter on a one-sentence stem with no natural clause boundary
    checks: dict = field(default_factory=dict)         # check values of the last attempt


def run_checks(cond: Condition, src_stem: str, src_opts: dict | None, stem: str, opts: dict | None, sim, sim_min: float) -> dict:
    c = {"semantic_sim": sim(src_stem, stem) if sim else None}
    c["semantic_ok"] = None if sim is None else c["semantic_sim"] >= sim_min
    if opts:
        c["options_sim_min"] = min(sim(src_opts[k], opts[k]) for k in opts) if sim else None
    parts_src = [src_stem] + list((src_opts or {}).values())
    parts_out = " ".join([stem] + list((opts or {}).values()))
    c["terms_localised"] = checks.missing_terms(parts_src, parts_out, cond.language)      # logged: named terms absent from the output
    c["key_terms_ok"] = checks.key_terms_preserved(parts_src, parts_out, cond.language, similar_enough=bool(c["semantic_ok"]))
    if cond.axis == "resource":
        c["langid_ok"] = checks.langid_ok_translation(stem, cond.language)
    else:
        c["langid_ok"] = checks.codeswitch_profile(stem, cond.language)["both_present"]
        c["switch_type_ok"] = checks.switch_type_ok(stem, cond.language, cond.switch_type)
    c["options_letters_ok"] = (opts is None) or (list(opts) == list(src_opts))
    c["acronyms_kept"] = all(a in parts_out for a in checks.acronyms(" ".join(parts_src)))     # recorded, not gating
    if cond.switch_type == "inter" and sim is not None and hasattr(sim, "max_pair"):
        sents = [x for x in checks.SENT_SPLIT.split(stem) if x.strip()]
        c["max_sentence_pair_sim"] = sim.max_pair(sents)
        c["no_repeated_content"] = c["max_sentence_pair_sim"] < DUP_SIM_MAX
    else:
        c["no_repeated_content"] = True
    # A code-switched stem identical to the source is a failed rendering (seen in the first GPU dry run).
    c["changed_ok"] = True if cond.axis == "resource" else (" ".join(stem.split()) != " ".join(src_stem.split()))
    return c


FAIL_FEEDBACK = {
    "changed_ok": "Your last answer was identical to the input. You must actually apply the requested pattern.",
    "langid_ok": "Your last answer did not contain the required language(s). Use both English and the other language as requested.",
    "key_terms_ok": "Your last answer lost a number, dose, or name. Keep every number and every name exactly as in the input.",
    "no_repeated_content": "Your last answer said the same thing twice, once in each language. Say each part only once.",
    "options_letters_ok": "Your last answer changed the option letters. Keep the keys A, B, C, D exactly.",
    "semantic_ok": "Your last answer drifted from the original meaning. Keep the meaning identical.",
}


def failure_feedback(c: dict) -> str | None:
    msgs = [m for k, m in FAIL_FEEDBACK.items() if c.get(k) is False]
    return " ".join(msgs) if msgs else None


def hard_pass(c: dict) -> bool:
    """Checks that gate acceptance. switch_type_ok is recorded but is a heuristic, so it does not gate."""
    return bool(c["options_letters_ok"] and c["key_terms_ok"] and c["langid_ok"] and c["changed_ok"] and c["no_repeated_content"] and c["semantic_ok"] in (True, None))


def render_item(cond: Condition, stem: str, options: dict | None, backend, sim=None, sim_min: float = 0.80,
                max_attempts: int = 3, feedback: str | None = None) -> RenderOutcome:
    out = RenderOutcome(None, None, 0, False)
    base_feedback = feedback
    for attempt in range(1, max_attempts + 1):
        out.attempts = attempt
        text = backend.chat(build_prompt(cond, stem, options, attempt, feedback), temperature=0.0, max_tokens=2048)[0].text
        if checks.looks_like_refusal(text, stem + " " + " ".join((options or {}).values())):
            out.refusals += 1
            out.attempt_log.append("refusal_or_answered")
            feedback = (feedback or "") + " Your last answer refused or answered instead of rewriting. Only rewrite the wording."
            continue
        parsed = parse_json(text, options)
        if parsed is None:
            out.attempt_log.append("bad_json_or_keys")
            feedback = (feedback or "") + " Your last answer was not a JSON object with the same keys."
            continue
        s, o = parsed
        c = run_checks(cond, stem, options, s, o, sim, sim_min)
        out.rendered, out.rendered_options, out.checks = s, o, c
        if hard_pass(c):
            out.passed = True
            out.attempt_log.append("ok")
            return out
        fb = failure_feedback(c)
        feedback = ((base_feedback + " ") if base_feedback else "") + (fb or "")      # carry only the latest failure, plus any round feedback
        out.attempt_log.append("failed:" + ",".join(k for k in ("options_letters_ok", "key_terms_ok", "langid_ok", "changed_ok", "no_repeated_content", "semantic_ok") if c.get(k) is False))
    return out   # best effort kept with passed=False; never silently dropped


def render_translation(cond: Condition, stem: str, options: dict | None, translator, sim=None, sim_min: float = 0.80) -> RenderOutcome:
    """Resource-level condition with a dedicated MT system: stem and every option translated, letters fixed by construction.
    Deterministic, so there is one attempt; a failed check is stored with passed=False."""
    out = RenderOutcome(None, None, 1, False)
    keys = list(options or {})
    tr = translator.translate([stem] + [options[k] for k in keys], cond.language)
    s, o = tr[0], ({k: v for k, v in zip(keys, tr[1:])} if options else None)
    c = run_checks(cond, stem, options, s, o, sim, sim_min)
    out.rendered, out.rendered_options, out.checks = s, o, c
    out.passed = hard_pass(c)
    out.attempt_log.append("ok" if out.passed else "failed:" + ",".join(k for k in ("key_terms_ok", "langid_ok", "semantic_ok") if c.get(k) is False))
    return out


def run_cs_checks(cond, src_stem: str, stem: str, sim, sim_min: float) -> dict:
    """Checks for the plan/apply/splice path. Presence of both languages holds by construction (validated pieces)."""
    c = {"semantic_sim": sim(src_stem, stem) if sim else None}
    c["semantic_ok"] = None if sim is None else c["semantic_sim"] >= sim_min
    c["key_terms_ok"] = checks.numbers_preserved(src_stem, stem, cond.language)
    c["changed_ok"] = " ".join(stem.split()) != " ".join(src_stem.split())
    c["langid_ok"] = True                                   # every foreign piece passed translation_ok / came from the closed tag list
    c["no_repeated_content"] = True                         # nothing is duplicated: only planned places change
    c["options_letters_ok"] = True                          # options are kept verbatim
    c["window_langid_both"] = checks.codeswitch_profile(stem, cond.language)["both_present"]   # informational, unreliable on short windows
    return c


def _choose_in_context(backend, translator, lang, stem, plan, pieces, sim):
    """Intra-sentential: NLLB proposes candidates for each phrase, the planner (Qwen3) picks the one that keeps the sentence
    grammatical. Selection only, so a weak generator in a language cannot write the foreign text. None if no candidate fits."""
    nbest = translator.translate_nbest(pieces, lang, 4)
    chosen = []
    for i, (piece, cands) in enumerate(zip(pieces, nbest)):
        cands = [c for c in cands if cs_plan.translation_ok(piece, c, sim, lang)]
        if not cands:
            return None, f'No faithful translation of "{piece[:40]}" was found; choose a different phrase.'
        k = cs_plan.parse_choice(backend.chat(cs_plan.choose_prompt(lang, stem, plan, i, cands), temperature=0.0, max_tokens=64)[0].text, len(cands))
        if k is None:
            return None, f'"{piece[:40]}" did not fit grammatically in its sentence; choose a different phrase.'
        chosen.append(cands[k])
    return chosen, None


def _accumulate(note: str, why: str) -> str:
    """Keep earlier rejection reasons in the planner feedback so it does not return to a phrase it was already told to drop."""
    return note if why in note else (note + " " + why).strip()


def render_codeswitch(cond: Condition, stem: str, options: dict | None, backend, sim=None, sim_min: float = 0.80,
                      max_attempts: int = 3, feedback: str | None = None, translator=None) -> RenderOutcome:
    """Plan -> validate -> apply -> splice -> check, retrying the plan with the validator's reason (temperature stays 0)."""
    out = RenderOutcome(None, None, 0, False)
    st, lang = cond.switch_type, cond.language
    note = feedback or ""
    if st == "inter" and cs_plan.is_single_sentence(stem):       # no clause boundary to use: neutral filler before the unchanged question
        rendered = f"{cs_plan.filler_for(lang, stem)} {stem}"
        c = run_cs_checks(cond, stem, rendered, None, sim_min)
        c["semantic_sim"], c["semantic_ok"] = 1.0, True           # the item text is byte-identical; only the filler is added
        out.attempts, out.plan = 1, {"filler": cs_plan.filler_for(lang, stem)}
        out.rendered, out.rendered_options, out.checks = rendered, dict(options) if options else None, c
        out.passed = hard_pass(c)
        out.attempt_log.append("filler")
        return out
    for attempt in range(1, max_attempts + 1):
        out.attempts = attempt
        prompt = cs_plan.plan_prompt(st, lang, stem) + (f"\nFeedback on the earlier try: {note}\n" if note else "")
        plan, why = cs_plan.validate_plan(st, lang, stem, cs_plan.parse_obj(backend.chat(prompt, temperature=0.0, max_tokens=512)[0].text))
        if why:
            out.attempt_log.append("plan_invalid:" + why[:60])
            note = _accumulate(note, why)
            continue
        pieces = cs_plan.pieces_to_translate(st, stem, plan)
        trs = []
        if pieces:
            if translator is not None and st == "intra" and hasattr(translator, "translate_nbest"):
                trs, why_not = _choose_in_context(backend, translator, lang, stem, plan, pieces, sim)
                if trs is None:
                    out.attempt_log.append("no_candidate_fits:" + (why_not or "")[:90])
                    note = _accumulate(note, why_not)
                    continue
            else:
                trs = (translator.translate(pieces, lang) if translator is not None else        # dedicated MT: pieces translated without context
                       cs_plan.parse_translations(backend.chat(cs_plan.apply_prompt(st, lang, stem, pieces, plan), temperature=0.0, max_tokens=1024)[0].text, len(pieces)))
            if trs is None:
                out.attempt_log.append("apply_bad_json")
                note = "The translation step failed; choose simpler, shorter pieces."
                continue
            checked = [(p, t, cs_plan.translation_check(p, t, sim, lang)) for p, t in zip(pieces, trs)]
            bad = [(p, t, why_bad) for p, t, (ok, why_bad) in checked if not ok]
            if bad:
                out.attempt_log.append(f"apply_rejected:{bad[0][2]}:{bad[0][0][:30]}->{bad[0][1][:40]}")
                note = f'The piece "{bad[0][0][:40]}" could not be translated faithfully; choose a different piece.'
                continue
        rendered = cs_plan.splice(st, lang, stem, plan, trs)
        c = run_cs_checks(cond, stem, rendered, sim, sim_min)
        if st == "extra":                                   # the item text is unchanged and only a closed-list tag is added
            c["semantic_sim"], c["semantic_ok"] = 1.0, True
        out.plan, out.rendered, out.rendered_options, out.checks = plan, rendered, dict(options) if options else None, c
        if hard_pass(c):
            out.passed = True
            out.attempt_log.append("ok")
            return out
        out.attempt_log.append("failed:" + ",".join(k for k in ("key_terms_ok", "changed_ok", "semantic_ok") if c.get(k) is False))
        note = failure_feedback(c) or ""
    return out


def render_codeswitch_native(cond: Condition, stem: str, options: dict | None, backend, sim=None, sim_min: float = 0.80,
                             max_attempts: int = 3, feedback: str | None = None, translator=None) -> RenderOutcome:
    """Native-base code-switching (author-approved extension). Qwen3 plans which English pieces stay English, code validates the plan,
    the translator (NLLB) renders everything else in the native language, code splices. Options stay verbatim English like the
    English-base conditions. A one-sentence inter stem gets a neutral English filler sentence before the fully translated question."""
    if translator is None:
        raise ValueError("native-base code-switching needs the dedicated translator")
    out = RenderOutcome(None, None, 0, False)
    st, lang, note = cond.switch_type, cond.language, feedback or ""
    for attempt in range(1, max_attempts + 1):
        out.attempts = attempt
        if st == "inter" and cs_plan.is_single_sentence(stem):
            plan, why = {"filler": cs_plan.filler_en_for(stem)}, None
        else:
            prompt = cs_plan.plan_prompt_native(st, lang, stem) + (f"\nFeedback on the earlier try: {note}\n" if note else "")
            plan, why = cs_plan.validate_plan_native(st, lang, stem, cs_plan.parse_obj(backend.chat(prompt, temperature=0.0, max_tokens=512)[0].text))
        if why:
            out.attempt_log.append("plan_invalid:" + why[:60])
            note = _accumulate(note, why)
            continue
        if "filler" in plan:
            core = translator.translate([stem], lang)[0]
            rendered, reason = f"{plan['filler']} {core}", None
        else:
            rendered, core, reason = cs_plan.apply_native(st, lang, stem, plan, translator)
        if reason:
            out.attempt_log.append("apply_rejected:" + reason[:500])
            note = reason.split(" | ")[0]
            continue
        c = run_native_checks(cond, stem, rendered, core, plan, sim, sim_min)
        out.plan, out.rendered, out.rendered_options, out.checks = plan, rendered, dict(options) if options else None, c
        if hard_pass(c):
            out.passed = True
            out.attempt_log.append("ok")
            return out
        out.attempt_log.append("failed:" + ",".join(k for k in ("key_terms_ok", "changed_ok", "semantic_ok", "langid_ok") if c.get(k) is False))
        note = "The result drifted in meaning or lost a number; choose different or fewer phrases." if st != "extra" else "Choose a different tag."
    return out


def run_native_checks(cond, src_stem: str, rendered: str, core: str, plan: dict, sim, sim_min: float) -> dict:
    c = {"semantic_sim": sim(src_stem, core) if sim else None}
    c["semantic_ok"] = None if sim is None else c["semantic_sim"] >= sim_min
    c["key_terms_ok"] = checks.numbers_preserved(src_stem, rendered, cond.language)
    c["changed_ok"] = " ".join(rendered.split()) != " ".join(src_stem.split())
    kept = sum(len(cs_plan.words(s["text"])) for s in plan.get("spans", []))
    c["english_word_share"] = kept / max(len(rendered.split()), 1) if "spans" in plan else None
    if "spans" in plan:                                           # intra: the native language must stay the base
        c["langid_ok"] = c["english_word_share"] <= 0.5
    else:                                                         # translated part must be in the native language
        c["langid_ok"] = checks.langid_ok_translation(core, cond.language) if "english_sentences" not in plan else True
    c["no_repeated_content"] = True
    c["options_letters_ok"] = True
    return c
