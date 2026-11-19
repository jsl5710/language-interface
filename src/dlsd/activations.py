"""Residual-stream activations at the last prompt token (E1, docs/EXPERIMENTS_LANGUAGE_INTERFACE.md).

The position is the final token of the chat-templated prompt including the generation prompt, which is the post-instruction position
used for refusal directions (Arditi et al. 2024). Layer index l means the output of block l (hidden_states[l]; 0 is the embedding).
Left padding with explicit position ids, so a prompt gets the same activation alone or in a batch.
"""
import numpy as np

# Which rows are harmful and which are benign for the direction. MedSafetyBench requests are harmful. MMedBench items are benign
# health questions, but they are multiple choice, so harmful-versus-benign is confounded with prompt format: see the spec (open item).
LABEL_BY_SOURCE = {"medsafetybench": 1, "mmedbench": 0, "medsafetybench_benign": 0}   # the matched rewrite (scripts/01b) removes the format confound


def chat_text(tokenizer, user_text: str) -> str:
    return tokenizer.apply_chat_template([{"role": "user", "content": user_text}], add_generation_prompt=True, tokenize=False)


def last_token_hidden(model, tokenizer, texts: list, layers: list, batch_size: int = 8, max_length: int = 1024) -> dict:
    """Return {layer: float32 array (len(texts), d_model)}."""
    import torch
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    out = {l: [] for l in layers}
    dev = next(model.parameters()).device
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i:i + batch_size], return_tensors="pt", padding=True, truncation=True, max_length=max_length,
                        add_special_tokens=False)           # the chat template already carries the special tokens
        mask = enc["attention_mask"]
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        with torch.no_grad():
            hs = model(input_ids=enc["input_ids"].to(dev), attention_mask=mask.to(dev), position_ids=pos.to(dev),
                       output_hidden_states=True).hidden_states
        for l in layers:
            out[l].append(hs[l][:, -1, :].float().cpu().numpy())
    return {l: np.concatenate(v, axis=0) for l, v in out.items()}


def labelled_rows(rows: list, benign_source: str | None = None) -> list:
    """Rows usable for E1: rendered ok, from a labelled source. With `benign_source`, keep only harmful rows and that benign source
    ("medsafetybench_benign" = matched rewrite, "mmedbench" = the format-confounded contrast)."""
    ok = {"medsafetybench"} | ({benign_source} if benign_source else set(LABEL_BY_SOURCE))
    return [r for r in rows if r.get("render_ok") and r.get("rendered") is not None and r["source"] in ok and r["source"] in LABEL_BY_SOURCE]
