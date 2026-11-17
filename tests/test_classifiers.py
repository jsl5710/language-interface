"""Official templates and output parsers for the four safety classifiers, using outputs copied from the model cards."""
from dlsd.classifiers import (GuardClassifier, parse_llama_guard, parse_qwen3guard, parse_wildguard_like, polyguard_messages,
                              qwen3guard_messages, llama_guard_messages, wildguard_prompt)


def test_wildguard_prompt_matches_the_model_card_exactly():
    p = wildguard_prompt("How can I rob the bank?", "I can not respond to that.")
    assert p.startswith("<s><|user|>\n[INST] You are a classifier for analyzing interactions between humans and LLMs.")
    assert "Human user:\nHow can I rob the bank?\n\nAI assistant:\nI can not respond to that.\n\n---\n\nAnswers: [/INST]\n<|assistant|>\n" in p
    assert wildguard_prompt("q").count("AI assistant:\n\n\n---") == 1          # input-only leaves the response slot empty


def test_message_builders():
    assert llama_guard_messages("hi")[0]["content"] == [{"type": "text", "text": "hi"}] and len(llama_guard_messages("hi", "yo")) == 2
    assert qwen3guard_messages("hi") == [{"role": "user", "content": "hi"}]
    pg = polyguard_messages("q", "a")
    assert pg[0]["role"] == "system" and "S14: Code Interpreter Abuse." in pg[0]["content"] and "Human user:\nq\n\nAI assistant:\na" in pg[1]["content"]


def test_parsers_on_card_examples():
    assert parse_llama_guard("unsafe\nS1,S2").flagged is True and parse_llama_guard("unsafe\nS1,S2").detail["categories"] == ["S1", "S2"]
    assert parse_llama_guard("\n\nsafe").flagged is False
    bad = parse_llama_guard("I think this is fine")
    assert bad.flagged is None and not bad.parsed_ok
    wg = "Harmful request: yes\nResponse refusal: yes\nHarmful response: no"
    assert parse_wildguard_like(wg, "input").flagged is True and parse_wildguard_like(wg, "output").flagged is False
    assert parse_wildguard_like(wg).detail["response_refusal"] is True
    assert parse_wildguard_like("nothing useful").flagged is None
    q = "Safety: Unsafe\nCategories: Violent"
    assert parse_qwen3guard(q).flagged is True and parse_qwen3guard(q).detail["categories"] == ["Violent"]
    assert parse_qwen3guard("Safety: Safe\nCategories: None\nRefusal: Yes").detail["refusal"] is True
    assert parse_qwen3guard("Safety: Controversial").flagged is False and parse_qwen3guard("Safety: Controversial", controversial_flags=True).flagged is True


class FakeBackend:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def chat_messages(self, messages, **kw):
        self.calls.append(messages)
        return self.reply

    def complete(self, prompt, **kw):
        self.calls.append((prompt, kw))
        return self.reply


def test_wrappers_route_to_the_right_endpoint_and_scope():
    wg = GuardClassifier("wildguard", FakeBackend("Harmful request: no\nResponse refusal: no\nHarmful response: yes"))
    assert wg.classify_input("q").flagged is False and wg.classify_output("q", "a").flagged is True
    prompt, kw = wg.backend.calls[0]
    assert kw["extra_body"] == {"add_special_tokens": False} and "AI assistant:\n\n\n---" in prompt     # input-only call has an empty response
    lg = GuardClassifier("llama_guard", FakeBackend("unsafe\nS6"))
    assert lg.classify_input("q").flagged is True and len(lg.backend.calls[0]) == 1
    lg.classify_output("q", "a")
    assert len(lg.backend.calls[1]) == 2
    assert GuardClassifier("polyguard", FakeBackend("garbage")).classify_input("q").parsed_ok is False
