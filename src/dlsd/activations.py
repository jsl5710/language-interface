"""Residual-stream activations for E1 (docs/EXPERIMENTS_LANGUAGE_INTERFACE.md, stage map).

`last_token_hidden` (shard keys L<l>): the final token of the chat-templated prompt including the generation prompt, the
post-instruction position (t_post-inst in Zhao et al. 2025, where they read the refusal direction; also the position of Arditi et al. 2024).
`capture` adds, from the same forward pass:
  I<l>  the last token of the user instruction (t_inst in Zhao et al. 2025, where they read the harmfulness direction);
  M<l>  the mean over the user-content tokens only (template tokens excluded, since they are identical across conditions);
  embedding norms per user-content token from hidden_states[0] (flat array plus offsets).
Layer index l means the output of block l (hidden_states[l]; 0 is the embedding).
Left padding with explicit position ids, so a prompt gets the same activation alone or in a batch.
The user-content span is found by tokenizing the template prefix and prefix + content separately; `span_exact` records whether those ids are
a prefix of the full prompt's ids (a merge across the boundary would make the span approximate).
"""
import numpy as np

# Which rows are harmful and which are benign for the direction. MedSafetyBench requests are harmful. MMedBench items are benign
# health questions, but they are multiple choice, so harmful-versus-benign is confounded with prompt format: see the spec (open item).
LABEL_BY_SOURCE = {"medsafetybench": 1, "mmedbench": 0, "medsafetybench_benign": 0}   # the matched rewrite (scripts/01b) removes the format confound


def chat_text(tokenizer, user_text: str) -> str:
    return tokenizer.apply_chat_template([{"role": "user", "content": user_text}], add_generation_prompt=True, tokenize=False)


SENTINEL = "\x01\x02SPLIT\x02\x01"


def template_parts(tokenizer, user_text: str) -> tuple:
    """(prefix, suffix) of the chat template around the user content."""
    full = chat_text(tokenizer, SENTINEL)
    pre, post = full.split(SENTINEL)
    return pre, post


def _ids(tokenizer, text: str, max_length: int) -> list:
    enc = tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=max_length, add_special_tokens=False)
    ids, mask = enc["input_ids"][0].tolist(), enc["attention_mask"][0].tolist()
    return [i for i, m in zip(ids, mask) if m]


def content_span(tokenizer, user_text: str, max_length: int = 1024) -> dict:
    """Token positions (in the unpadded templated prompt) of the user content: [start, end). end - 1 is the instruction token."""
    pre, post = template_parts(tokenizer, user_text)
    a, b, full = _ids(tokenizer, pre, max_length), _ids(tokenizer, pre + user_text, max_length), _ids(tokenizer, pre + user_text + post, max_length)
    start, end = len(a), min(len(b), len(full))
    exact = full[:len(b)] == b and b[:len(a)] == a
    if end <= start:                                   # empty content or truncated away: fall back to the last prompt token
        start, end = len(full) - 1, len(full)
    return {"start": start, "end": end, "n": len(full), "exact": bool(exact)}


def capture(model, tokenizer, user_texts: list, layers: list, batch_size: int = 8, max_length: int = 1024) -> dict:
    """One forward pass per batch; returns {"last": {l: arr}, "inst": {l: arr}, "mean": {l: arr}, "emb_norms": flat float32,
    "emb_offsets": int64 (len n+1), "span_exact": bool array}. `user_texts` are the user messages (the template is applied here)."""
    import torch
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    out = {k: {l: [] for l in layers} for k in ("last", "inst", "mean")}
    norms, offsets, exact = [], [0], []
    dev = next(model.parameters()).device
    for i in range(0, len(user_texts), batch_size):
        batch = user_texts[i:i + batch_size]
        texts = [chat_text(tokenizer, t) for t in batch]
        spans = [content_span(tokenizer, t, max_length) for t in batch]
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length, add_special_tokens=False)
        mask = enc["attention_mask"]
        width = mask.shape[1]
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        with torch.no_grad():
            hs = model(input_ids=enc["input_ids"].to(dev), attention_mask=mask.to(dev), position_ids=pos.to(dev),
                       output_hidden_states=True).hidden_states
        lens = mask.sum(-1).tolist()
        for j, sp in enumerate(spans):
            pad = width - lens[j]
            s, e = pad + sp["start"], pad + min(sp["end"], lens[j])
            n = hs[0][j, s:e, :].float().norm(dim=-1).cpu().numpy()
            norms.append(n)
            offsets.append(offsets[-1] + len(n))
            exact.append(sp["exact"] and sp["n"] == lens[j])
            for l in layers:
                h = hs[l][j]
                out["last"][l].append(h[-1].float().cpu().numpy())
                out["inst"][l].append(h[e - 1].float().cpu().numpy())
                out["mean"][l].append(h[s:e].float().mean(0).cpu().numpy())
    res = {k: {l: np.stack(v) for l, v in d.items()} for k, d in out.items()}
    res |= {"emb_norms": np.concatenate(norms).astype(np.float32) if norms else np.zeros(0, np.float32),
            "emb_offsets": np.array(offsets, dtype=np.int64), "span_exact": np.array(exact, dtype=bool)}
    return res


def has_extended(shard, layers: list) -> bool:
    """True if an npz shard already holds the extended capture (I<l>, M<l>, embedding norms) for every requested layer."""
    keys = set(shard.files) if hasattr(shard, "files") else set(shard)
    return {"emb_norms", "emb_offsets"} <= keys and all(f"{p}{l}" in keys for l in layers for p in "LIM")


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


def load_shards(act_dir: str) -> dict:
    """{condition_id: npz} for every finished shard in a directory (temporary files from an interrupted write are ignored)."""
    import glob
    import os
    files = [f for f in sorted(glob.glob(os.path.join(act_dir, "*.npz"))) if not f.endswith(".tmp.npz")]
    return {str(np.load(f)["condition_id"]): np.load(f) for f in files}
