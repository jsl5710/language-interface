from dlsd.dialect_rules import _mask, _unmask


def test_mask_roundtrip_and_no_digits_or_hyphens_in_placeholder():
    text = "Doctors in COVID-19 wards use beta-blockers and/or ACE-inhibitors."
    masked, terms = _mask(text)
    assert terms == ["COVID-19", "beta-blockers", "and/or", "ACE-inhibitors"]
    assert "-" not in masked and "/" not in masked and not any(c.isdigit() for c in masked)
    assert _unmask(masked, terms) == text


def test_unmask_fails_when_placeholder_dropped():
    masked, terms = _mask("a COVID-19 case")
    assert _unmask(masked.replace("Zqxaxqz", ""), terms) is None


def test_plain_text_has_nothing_to_mask():
    assert _mask("A plain sentence.") == ("A plain sentence.", [])


from dlsd.dialect_rules import split_sentences, _try


def test_split_rebuilds_text_exactly_and_respects_abbreviations():
    t = "Dr. Lee saw a 54-year-old man. Temp is 36.6 C. He has pain! Does it radiate? Yes."
    parts = split_sentences(t)
    assert "".join(s + g for s, g in parts) == t
    assert [s for s, _ in parts][0].startswith("Dr. Lee saw")      # no split after the abbreviation
    assert any(s == "Temp is 36.6 C." for s, _ in parts)            # decimal point is not a boundary


class FakeDialect:
    def __init__(self, bad=()):
        self.bad, self.executed_rules = bad, {}

    def set_seed(self, s):
        pass

    def transform(self, text):
        if any(b in text for b in self.bad):
            raise AssertionError("Spacy and Stanza word mismatch")
        self.executed_rules = {(0, 1): {"type": "r1"}}
        return text.replace("see", "SEE")


def test_try_masks_compounds_when_whole_text_fails():
    class D(FakeDialect):
        def transform(self, text):
            if "-" in text:
                raise AssertionError("mismatch")
            return super().transform(text)
    out, fired, protected = _try(D(), "see COVID-19 now", 13)
    assert protected and "COVID-19" in out and fired == ["r1"]


def test_try_returns_none_when_unfixable():
    assert _try(FakeDialect(bad=("x",)), "an x here", 13) is None


def test_numeric_ranges_are_masked_and_a_number_changing_pass_is_skipped():
    from dlsd.dialect_rules import _mask, _unmask, numbers_intact
    t, terms = _mask("irregular every 2-3 months, BP 120/80 mmHg, 5% risk", numbers=True)
    assert terms == ["2-3", "120/80", "5%"] and "2-3" not in t
    assert _unmask(t, terms) == "irregular every 2-3 months, BP 120/80 mmHg, 5% risk"
    assert not numbers_intact("every 2-3 months", "every 2- months")
    assert numbers_intact("every 2-3 months", "every 2-3 months")
