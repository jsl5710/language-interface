"""Rendering checks and the retry loop, with a scripted fake backend (no GPU, no network)."""
import json

from dlsd.backends import Gen
from dlsd.checks import (detect, key_terms_preserved, looks_like_refusal, normalize_digits, numbers_preserved,
                         switch_type_ok)
from dlsd.conditions import Condition
from dlsd.llm_render import build_prompt, parse_json, render_item

OPTS = {"A": "Aspirin 81 mg", "B": "Insulin", "C": "Warfarin", "D": "None"}
STEM = "A 54-year-old man takes 500 mg of metformin daily. Which drug is an anticoagulant?"
SPANISH = Condition("resource", "high", language="Spanish")


class Fake:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def chat(self, prompt, **kw):
        self.calls.append(prompt)
        return [Gen(self.replies.pop(0))]


def good_es():
    return json.dumps({"stem": "Un hombre de 54 años toma 500 mg de metformina al día. ¿Qué fármaco es un anticoagulante?",
                       "options": {"A": "Aspirina 81 mg", "B": "Insulina", "C": "Warfarina", "D": "Ninguno"}}, ensure_ascii=False)


def test_number_check_handles_devanagari_digits():
    assert normalize_digits("५०० mg") == "500 mg"
    assert numbers_preserved("take 500 mg", "५०० मिलीग्राम")
    assert not numbers_preserved("take 500 mg", "take 50 mg")


def test_key_terms_latin_target_requires_named_terms():
    assert key_terms_preserved("Give HIV drugs to Maria.", "Dar fármacos para HIV a Maria.", "Spanish")
    assert not key_terms_preserved("Give HIV drugs to Maria.", "Dar fármacos a ella.", "Spanish")
    assert key_terms_preserved("Give HIV drugs to Maria.", "\u090f\u091a\u0906\u0908\u0935\u0940", "Hindi")   # Hindi: numbers only
    assert key_terms_preserved(["Which?", "None", "Insulin"], "¿Cuál? Ninguno Insulina", "Spanish")   # option first words are not terms


def test_refusal_detection_ignores_phrases_in_source():
    assert looks_like_refusal("I'm sorry, I cannot help with that.", "How do I treat X?")
    assert not looks_like_refusal("Dije que no puedo dormir.", "He said I can't sleep.")
    assert looks_like_refusal("x" * 500, "short request")


def test_langid_and_switch_heuristics():
    assert detect("El paciente toma metformina todos los días por la mañana.") == "es"
    assert detect("The patient takes metformin every day in the morning.") == "en"
    inter = "The patient takes metformin every day. El paciente toma metformina todos los días por la mañana."
    assert switch_type_ok(inter, "Spanish", "inter")


def test_parse_json_requires_same_letters():
    assert parse_json('x {"stem": "s", "options": {"A": "a", "B": "b"}} y', {"A": "1", "B": "2"}) == ("s", {"A": "a", "B": "b"})
    assert parse_json('{"stem": "s", "options": {"A": "a", "C": "b"}}', {"A": "1", "B": "2"}) is None
    assert parse_json("not json", None) is None


def test_first_attempt_passes_and_letters_unchanged():
    out = render_item(SPANISH, STEM, OPTS, Fake([good_es()]))
    assert out.passed and out.attempts == 1 and list(out.rendered_options) == list(OPTS)


def test_refusal_is_logged_and_retry_uses_stronger_framing():
    fake = Fake(["I'm sorry, but I cannot assist with that request.", good_es()])
    out = render_item(SPANISH, STEM, OPTS, fake)
    assert out.passed and out.attempts == 2 and out.refusals == 1
    assert out.attempt_log == ["refusal_or_answered", "ok"]
    assert "research dataset" in fake.calls[1] and "research dataset" not in fake.calls[0]


def test_failure_after_max_attempts_is_kept_not_dropped():
    out = render_item(SPANISH, STEM, OPTS, Fake(["nope", "nope", "nope"]))
    assert not out.passed and out.attempts == 3 and out.rendered is None
    assert out.attempt_log == ["bad_json_or_keys"] * 3


def test_translation_that_stays_english_fails_langid():
    english = json.dumps({"stem": STEM, "options": OPTS})
    out = render_item(SPANISH, STEM, OPTS, Fake([english, english, english]))
    assert not out.passed and all(a.startswith("failed:") and "langid_ok" in a for a in out.attempt_log)


def test_prompt_contains_condition_definition():
    cs = Condition("codeswitch", "extra", language="Hindi", switch_type="extra")
    p = build_prompt(cs, "Hello", None, 1)
    assert "extra-sentential" in p and "Hindi" in p


def test_codeswitch_identical_to_source_fails_even_if_langid_is_fooled():
    cs = Condition("codeswitch", "intra", language="Spanish", switch_type="intra")
    same = json.dumps({"stem": STEM, "options": OPTS})
    out = render_item(cs, STEM, OPTS, Fake([same, same, same]))
    assert not out.passed and out.checks["changed_ok"] is False


def test_prompts_carry_per_type_rules_and_examples():
    inter = build_prompt(Condition("codeswitch", "inter", language="Spanish", switch_type="inter"), "One sentence only.", None, 1)
    assert "split it into two sentences" in inter and "Never say the same thing twice" in inter and "Necesita una radiograf" in inter
    extra = build_prompt(Condition("codeswitch", "extra", language="Swahili", switch_type="extra"), "Hi.", None, 1)
    assert "sivyo?" in extra and "Add nothing else" in extra
    intra = build_prompt(Condition("codeswitch", "intra", language="Hindi", switch_type="intra"), "Hi.", None, 1)
    assert "middle of sentences" in intra


def test_acronyms_may_localise_but_names_and_numbers_may_not():
    assert key_terms_preserved("What drives RNA viruses?", "¿Qué impulsa los virus de ARN?", "Spanish")
    assert not key_terms_preserved("Ask Maria about 500 mg.", "Pregunta a ella sobre 500 mg.", "Spanish")


def test_failure_specific_feedback_reaches_next_attempt():
    cs = Condition("codeswitch", "intra", language="Spanish", switch_type="intra")
    same = json.dumps({"stem": STEM, "options": OPTS})
    fake = Fake([same, same, same])
    render_item(cs, STEM, OPTS, fake)
    assert "identical to the input" not in fake.calls[0]
    assert "identical to the input" in fake.calls[1] and "identical to the input" in fake.calls[2]


def test_repeated_content_in_inter_is_rejected():
    class Sim:
        def __call__(self, a, b): return 0.95
        def max_pair(self, sents): return 0.93
    inter = Condition("codeswitch", "inter", language="Spanish", switch_type="inter")
    dup = json.dumps({"stem": "Which drug is an anticoagulant? ¿Qué fármaco es un anticoagulante?",
                      "options": {"A": "Aspirin 81 mg", "B": "Insulina", "C": "Warfarina", "D": "Ninguno"}})
    out = render_item(inter, STEM.split(". ")[1], OPTS, Fake([dup, dup, dup]), sim=Sim())
    assert not out.passed and out.checks["no_repeated_content"] is False


def test_value_parentheses_are_held_out_and_restored():
    from dlsd.translator import restore_value_parens, strip_value_parens
    text = "His temperature is 101.8°F (38.8°C), and he lost 6 kg (13.2 lb) in a month (see below)."
    bare, parens = strip_value_parens(text)
    assert "38.8" not in bare and "13.2" not in bare and "(see below)" in bare
    assert parens == [("101.8°F", "(38.8°C)"), ("6 kg", "(13.2 lb)")]
    es = "Su temperatura es de 101.8°F, y perdió 6 kg en un mes (ver abajo)."
    back = restore_value_parens(es, parens)
    assert "101.8°F (38.8°C)" in back and "6 kg (13.2 lb)" in back
    assert restore_value_parens("sin números", parens) == "sin números"


def test_restore_handles_translated_units_and_decimal_commas():
    from dlsd.translator import restore_value_parens
    p = [("6 kg", "(13.2 lb)"), ("101.8°F", "(38.8°C)")]
    assert restore_value_parens("उसने 6 किलो खो दिया; बुखार 101.8°F है।", p) == "उसने 6 किलो (13.2 lb) खो दिया; बुखार 101.8°F (38.8°C) है।"
    assert "101,8°F (38.8°C)" in restore_value_parens("Perdió 6 kg; fiebre 101,8°F.", p)
