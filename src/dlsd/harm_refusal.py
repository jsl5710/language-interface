"""Stages 6 to 8 of E1: harmfulness at the instruction token, refusal at the post-instruction token, and the output.

Zhao et al. 2025 (NeurIPS 2025) read harmfulness at t_inst, the last token of the user instruction, and refusal at t_post-inst, the last
of the post-instruction template tokens; shard keys I<l> and L<l> (dlsd.activations). Here:
  harmfulness direction: English harmful minus benign, difference in means at I<l>, fit on the English half A;
  refusal direction: English harmful prompts the model refused minus those it complied with, difference in means at L<l>, half A.
Refusal labels come from the existing judge outputs of the generation pipeline (local, row-level, never committed); only aggregates leave
this module. Per condition, for harmful prompts: refusal rate, harmfulness retention (share of the English harmful-versus-benign
separation kept at I<l>) and refusal position (mean projection on the refusal direction rescaled so English complied = 0 and English
refused = 1). Reading (descriptive, no tests): harmfulness kept while refusal position falls points to a recognition-to-refusal gap;
both falling points to a recognition gap (subject to the stage 5 comprehension check).
"""
import numpy as np
import pandas as pd

from dlsd import mechanism as M
from dlsd.representation import boot_ci


def refusal_labels(scored_rows, model_id: str, refused=("refuse",)) -> dict:
    """{(item_id, condition_id): 1 refused / 0 not} for one model; rows with no judge label are left out."""
    out = {}
    for r in scored_rows:
        if r.get("model_id") != model_id or r.get("judge_label") is None:
            continue
        out[(str(r["item_id"]), r["condition_id"])] = int(r["judge_label"] in set(refused))
    return out


def english_halves(item_ids, seed: int) -> set:
    """Half A of the English items, same seeded split as mechanism.pick_layer."""
    ids = sorted(set(item_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    return set(ids[: len(ids) // 2])


def fit_directions(eng: dict, layer: int, ref_labels: dict, base: str, seed: int, min_per_class: int) -> dict:
    """Directions and English reference values. `eng` is the English shard (mapping with item_ids, labels, I<l>, L<l>)."""
    ids = [str(i) for i in eng["item_ids"]]
    lab = np.asarray(eng["labels"]).astype(bool)
    half_a = english_halves(ids, seed)
    in_a = np.array([i in half_a for i in ids])
    xi, xl = eng[f"I{layer}"], eng[f"L{layer}"]
    d_h = M.diff_in_means(xi[in_a & lab], xi[in_a & ~lab])
    sep_h = M.separation(xi[~in_a & lab], xi[~in_a & ~lab], d_h)
    r = np.array([ref_labels.get((i, base), -1) for i in ids])
    ref_a, com_a = in_a & lab & (r == 1), in_a & lab & (r == 0)
    out = {"layer": layer, "d_harm": d_h, "sep_harm_english": sep_h, "half_a": half_a,
           "n_en_refused_a": int(ref_a.sum()), "n_en_complied_a": int(com_a.sum()), "d_refusal": None}
    if ref_a.sum() >= min_per_class and com_a.sum() >= min_per_class:
        d_r = M.diff_in_means(xl[ref_a], xl[com_a])
        p = M.project(xl, d_r)
        ref_b, com_b = ~in_a & lab & (r == 1), ~in_a & lab & (r == 0)
        src_r, src_c = (ref_b, com_b) if ref_b.sum() and com_b.sum() else (ref_a, com_a)   # held-out half when it has both classes
        out |= {"d_refusal": d_r, "en_refused_mean": float(p[src_r].mean()), "en_complied_mean": float(p[src_c].mean()),
                "cosine_harm_refusal": M.cosine(d_h, d_r)}
    return out


def condition_rows(z: dict, dirs: dict, ref_labels: dict, cid: str, base: str, n_boot: int, seed: int, min_per_class: int) -> dict:
    ids = np.array([str(i) for i in z["item_ids"]])
    lab = np.asarray(z["labels"]).astype(bool)
    keep = np.ones(len(ids), bool) if cid != base else np.array([i not in dirs["half_a"] for i in ids])   # English: held-out half only
    ids, lab = ids[keep], lab[keep]
    L = dirs["layer"]
    xi, xl = z[f"I{L}"][keep], z[f"L{L}"][keep]
    ph = M.project(xi, dirs["d_harm"])
    r = np.array([ref_labels.get((i, cid), -1) for i in ids])
    labelled = lab & (r >= 0)
    row = {"condition_id": cid, "layer": L, "n_harmful": int(lab.sum()), "n_benign": int((~lab).sum()), "n_harmful_with_label": int(labelled.sum()),
           "refusal_rate": float(r[labelled].mean()) if labelled.any() else np.nan}
    if lab.sum() >= 2 and (~lab).sum() >= 2:
        sep = float(ph[lab].mean() - ph[~lab].mean())
        row |= {"harm_retention": M.retention(sep, dirs["sep_harm_english"]), "harm_auroc": M.auroc(ph[lab], ph[~lab])}
        rng = np.random.default_rng(seed)
        hp, bp = ph[lab], ph[~lab]
        if n_boot > 0:
            bs = [(hp[rng.integers(0, len(hp), len(hp))].mean() - bp[rng.integers(0, len(bp), len(bp))].mean()) / dirs["sep_harm_english"]
                  for _ in range(n_boot)]
            row |= {"harm_retention_lo": float(np.quantile(bs, 0.025)), "harm_retention_hi": float(np.quantile(bs, 0.975))}
    if dirs.get("d_refusal") is not None and lab.any():
        span = dirs["en_refused_mean"] - dirs["en_complied_mean"]
        pr = (M.project(xl[lab], dirs["d_refusal"]) - dirs["en_complied_mean"]) / span if span else np.full(lab.sum(), np.nan)
        lo, hi = boot_ci(pr, n_boot, seed)
        row |= {"refusal_position": float(np.nanmean(pr)), "refusal_position_lo": lo, "refusal_position_hi": hi}
        rr = r[lab]
        if (rr == 1).sum() >= min_per_class and (rr == 0).sum() >= min_per_class:
            row["refusal_auroc_within"] = M.auroc(pr[rr == 1], pr[rr == 0])
    return row


def item_projections(z: dict, dirs: dict, ref_labels: dict, cid: str) -> pd.DataFrame:
    """Row-level projections of harmful prompts (for joining in memory only; never written)."""
    ids = np.array([str(i) for i in z["item_ids"]])
    lab = np.asarray(z["labels"]).astype(bool)
    L = dirs["layer"]
    d = pd.DataFrame({"item_id": ids[lab], "condition_id": cid, "proj_harm": M.project(z[f"I{L}"][lab], dirs["d_harm"])})
    if dirs.get("d_refusal") is not None:
        d["proj_refusal"] = M.project(z[f"L{L}"][lab], dirs["d_refusal"])
    d["refused"] = [ref_labels.get((i, cid), np.nan) for i in d["item_id"]]
    return d


def output_aggregates(df: pd.DataFrame, expected_language: dict) -> pd.DataFrame:
    """Per condition: response-language shares, refusal rate, and mean projections by refusal and by response language.
    df columns: condition_id, refused, response_lang, proj_harm and optionally proj_refusal. Aggregates only."""
    rows = []
    for cid, g in df.groupby("condition_id"):
        exp = expected_language.get(cid)
        row = {"condition_id": cid, "n": len(g), "refusal_rate": float(g["refused"].mean()),
               "share_response_en": float((g["response_lang"] == "en").mean()),
               "share_response_prompt_language": float((g["response_lang"] == exp).mean()) if exp else np.nan,
               "share_response_empty": float((g["response_lang"] == "empty").mean())}
        for lang in sorted(set(g["response_lang"]) - {"en", "empty"}):
            row[f"share_response_{lang}"] = float((g["response_lang"] == lang).mean())
        for col in [c for c in ("proj_harm", "proj_refusal") if c in g]:
            for val, name in ((1, "refused"), (0, "complied")):
                sub = g.loc[g["refused"] == val, col]
                row[f"{col}_{name}"] = float(sub.mean()) if len(sub) else np.nan
            for flag, name in ((True, "en_response"), (False, "non_en_response")):
                sub = g.loc[(g["response_lang"] == "en") == flag, col]
                row[f"{col}_{name}"] = float(sub.mean()) if len(sub) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
