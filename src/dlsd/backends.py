"""OpenAI-compatible chat backend. Works with vLLM, Together, OpenAI, and similar servers."""
import math, os
from dataclasses import dataclass
from typing import List, Optional
from openai import OpenAI

@dataclass
class Gen:
    text: str
    top_logprobs: Optional[list] = None   # first-token top logprobs [(token, logprob), ...]

class Backend:
    def __init__(self, model_id: str, base_url: str, api_key: Optional[str] = None, extra_body: Optional[dict] = None):
        self.model_id = model_id
        self.extra_body = extra_body   # e.g. {"chat_template_kwargs": {"enable_thinking": False}} for Qwen3
        self.client = OpenAI(base_url=base_url, api_key=api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"))

    def chat(self, prompt: str, system: Optional[str] = None, temperature=0.0, max_tokens=512,
             n=1, logprobs=False) -> List[Gen]:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        r = self.client.chat.completions.create(model=self.model_id, messages=msgs, temperature=temperature,
                                                max_tokens=max_tokens, n=n, logprobs=logprobs or None,
                                                top_logprobs=20 if logprobs else None,
                                                extra_body=self.extra_body)
        outs = []
        for ch in r.choices:
            tl = None
            if logprobs and ch.logprobs and ch.logprobs.content:
                tl = [(t.token, t.logprob) for t in ch.logprobs.content[0].top_logprobs]
            outs.append(Gen(ch.message.content or "", tl))
        return outs

def _chat_messages(self, messages: list, temperature=0.0, max_tokens=128) -> str:
    """Chat call with explicit messages (the server applies the model's own chat template). Returns the reply text."""
    r = self.client.chat.completions.create(model=self.model_id, messages=messages, temperature=temperature, max_tokens=max_tokens,
                                            extra_body=self.extra_body)
    return r.choices[0].message.content or ""


def _complete(self, prompt: str, temperature=0.0, max_tokens=64, extra_body: Optional[dict] = None) -> str:
    """Raw completion call for models without a chat template (WildGuard, Meditron). Returns the reply text."""
    r = self.client.completions.create(model=self.model_id, prompt=prompt, temperature=temperature, max_tokens=max_tokens,
                                       extra_body={**(self.extra_body or {}), **(extra_body or {})} or None)
    return r.choices[0].text or ""


def _complete_gen(self, prompt: str, temperature=0.0, max_tokens=64, n=1, logprobs=False, stop=None) -> List[Gen]:
    """Raw completion with optional first-token top log probabilities (base models such as Meditron)."""
    r = self.client.completions.create(model=self.model_id, prompt=prompt, temperature=temperature, max_tokens=max_tokens, n=n,
                                       logprobs=20 if logprobs else None, stop=stop, extra_body=self.extra_body)
    outs = []
    for ch in r.choices:
        tl = None
        if logprobs and ch.logprobs and ch.logprobs.top_logprobs:
            tl = list(ch.logprobs.top_logprobs[0].items())
        outs.append(Gen(ch.text or "", tl))
    return outs


Backend.complete_gen = _complete_gen
Backend.chat_messages = _chat_messages
Backend.complete = _complete


def letter_probs(top_logprobs, letters="ABCDEFGHIJ"):
    """Normalized probability over option letters from first-token top logprobs."""
    if not top_logprobs: return None
    p = {}
    for tok, lp in top_logprobs:
        t = tok.strip().upper()
        if len(t) == 1 and t in letters: p[t] = p.get(t, 0.0) + math.exp(lp)
    z = sum(p.values())
    return {k: v / z for k, v in p.items()} if z > 0 else None
