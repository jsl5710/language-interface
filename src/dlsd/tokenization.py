"""Tokenization covariates for E1 and a decomposition of condition gaps into a tokenization part and a remainder.

`tokenize` is any callable text -> list of token strings (for a Hugging Face tokenizer: tok.convert_ids_to_tokens(tok(text, add_special_tokens=False)["input_ids"])).
Measures follow Petrov et al. 2023 (premium = token length relative to a reference text) and Rust et al. 2021 (fertility = tokens per word).
"""
import re

import numpy as np
import pandas as pd

BYTE_TOKEN = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")


def words(text: str) -> list:
    return text.split()


def fertility(text: str, tokenize) -> float:
    w = words(text)
    return len(tokenize(text)) / max(len(w), 1)


def premium(text: str, reference: str, tokenize) -> float:
    """Token length of `text` over token length of the matched English text (Petrov et al. 2023)."""
    return len(tokenize(text)) / max(len(tokenize(reference)), 1)


def byte_fallback_share(text: str, tokenize, unk: str = "<unk>") -> float:
    toks = tokenize(text)
    return sum(1 for t in toks if BYTE_TOKEN.match(t) or t == unk) / max(len(toks), 1)


def split_word_share(text: str, tokenize, k: int = 3) -> float:
    """Share of whitespace words that the tokenizer breaks into k or more pieces."""
    w = words(text)
    return sum(1 for x in w if len(tokenize(x)) >= k) / max(len(w), 1)


def prompt_measures(text: str, reference: str, tokenize) -> dict:
    return {"n_tokens": len(tokenize(text)), "fertility": fertility(text, tokenize), "premium": premium(text, reference, tokenize),
            "byte_fallback_share": byte_fallback_share(text, tokenize), "split3_share": split_word_share(text, tokenize)}


def adjusted_gaps(df: pd.DataFrame, y: str, x: str, cond: str, ref: str, n_boot: int = 0, seed: int = 13, item: str = "item_id") -> pd.DataFrame:
    """Condition gaps from `ref` in outcome y, raw and adjusted for log(x), with a common within-condition slope.

    Model: y = a_cond + b * log(x). The slope b is identified from variation inside each condition (the within estimator), so it is not
    contaminated by differences between conditions. Adjusted gap = (a_cond - a_ref) evaluated at the reference condition's mean log(x).
    Caveat: if x barely varies inside a condition, b is poorly identified and the adjustment is an extrapolation. Report b with its interval.
    Bootstrap resamples items (rows of the same item stay together).
    """
    d = df[[item, cond, y, x]].dropna().copy()
    d["lx"] = np.log(d[x].astype(float))

    def fit(dd):
        c = dd.groupby(cond)
        yc, xc = dd[y] - c[y].transform("mean"), dd["lx"] - c["lx"].transform("mean")
        b = float((xc * yc).sum() / (xc ** 2).sum()) if (xc ** 2).sum() > 0 else float("nan")
        means = c[[y, "lx"]].mean()
        ref_lx = means.loc[ref, "lx"]
        raw = means[y] - means.loc[ref, y]
        adj = (means[y] - b * (means["lx"] - ref_lx)) - means.loc[ref, y]
        return b, raw, adj

    b, raw, adj = fit(d)
    out = pd.DataFrame({"condition": raw.index, "raw_gap": raw.values, "adjusted_gap": adj.values, "slope_per_log_unit": b})
    out["share_explained"] = 1 - out["adjusted_gap"] / out["raw_gap"].replace(0, np.nan)
    if n_boot:
        rng = np.random.default_rng(seed)
        ids = d[item].unique()
        groups = {i: g for i, g in d.groupby(item)}
        bs = []
        for _ in range(n_boot):
            samp = pd.concat([groups[i] for i in rng.choice(ids, len(ids))], ignore_index=True)
            if samp[cond].nunique() < d[cond].nunique() or ref not in set(samp[cond]):
                continue
            bb, rr, aa = fit(samp)
            bs.append(pd.DataFrame({"condition": rr.index, "raw": rr.values, "adj": aa.values, "b": bb}))
        if bs:
            B = pd.concat(bs)
            q = B.groupby("condition").agg(raw_lo=("raw", lambda v: np.quantile(v, .025)), raw_hi=("raw", lambda v: np.quantile(v, .975)),
                                           adj_lo=("adj", lambda v: np.quantile(v, .025)), adj_hi=("adj", lambda v: np.quantile(v, .975)))
            out = out.merge(q.reset_index(), on="condition", how="left")
            out["slope_lo"], out["slope_hi"] = np.quantile(B["b"], .025), np.quantile(B["b"], .975)
    return out


def item_table(rows: list, tokenize, baseline_id: str = "baseline|SAE") -> pd.DataFrame:
    """One row per rendered (item, condition) with tokenization measures against the same item's English baseline text."""
    base = {r["item_id"]: r["rendered"] for r in rows if r["condition_id"] == baseline_id}
    out = []
    for r in rows:
        if r["item_id"] not in base or r.get("rendered") is None:
            continue
        out.append({"item_id": r["item_id"], "condition_id": r["condition_id"], **prompt_measures(r["rendered"], base[r["item_id"]], tokenize)})
    return pd.DataFrame(out)
