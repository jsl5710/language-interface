"""The 27 input conditions applied to every item: the 18 of the spec plus 9 author-approved native-base code-switch conditions."""
from dataclasses import dataclass
from typing import List, Optional

@dataclass(frozen=True)
class Condition:
    axis: str                      # baseline | dialect | resource | codeswitch (English base) | codeswitch_native (native-language base, English insertions)
    level: str                     # e.g. SAE, AAVE, high, intra
    language: Optional[str] = None # L2 or target language
    switch_type: Optional[str] = None

    @property
    def cid(self) -> str:
        parts = [self.axis, self.level]
        if self.language: parts.append(self.language)
        return "|".join(parts)

def is_codeswitch(axis: str) -> bool:
    return axis in ("codeswitch", "codeswitch_native")


CODE_SWITCH_DEFINITIONS = {
    "intra": "switch within a sentence or clause",
    "inter": "switch at a sentence boundary",
    "extra": "a tag, interjection, or fixed expression from the other language inserted into an otherwise single-language sentence",
}

def build_conditions(cfg: dict) -> List[Condition]:
    c = cfg["conditions"]
    out = [Condition("baseline", "SAE")]
    out += [Condition("dialect", d) for d in c["dialects"]]
    for level, lang in c["languages"].items():
        out.append(Condition("resource", level, language=lang))
    for sw in c["code_switch_types"]:
        for level, lang in c["languages"].items():
            out.append(Condition("codeswitch", sw, language=lang, switch_type=sw))
    if c.get("native_base_code_switch", False):          # author-approved extension: same three types with the native language as the base
        for sw in c["code_switch_types"]:
            for level, lang in c["languages"].items():
                out.append(Condition("codeswitch_native", sw, language=lang, switch_type=sw))
    assert len(out) == (27 if c.get("native_base_code_switch", False) else 18), len(out)
    return out
