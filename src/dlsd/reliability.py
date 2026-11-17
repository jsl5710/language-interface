"""Inter-rater reliability for nominal ratings: percent agreement, Fleiss' kappa, Krippendorff's alpha, Gwet's AC1, bootstrap CIs.

Ratings are an items x raters integer matrix with -1 for a missing rating. Kappa and alpha collapse when one category dominates
(the prevalence paradox: almost all examples are rated "correct"), so Gwet's AC1 and the raw agreement are always reported next to them.
"""
import numpy as np


def _counts(m, k):
    m = np.asarray(m)
    return np.stack([(m == c).sum(1) for c in range(k)], 1)          # items x categories


def percent_agreement(m, k):
    """Mean over items of the share of rater pairs that agree (items with fewer than 2 ratings are skipped)."""
    n = _counts(m, k)
    r = n.sum(1)
    ok = r >= 2
    return float(((n[ok] * (n[ok] - 1)).sum(1) / (r[ok] * (r[ok] - 1))).mean())


def fleiss_kappa(m, k):
    n = _counts(m, k)
    pa = percent_agreement(m, k)
    p = n.sum(0) / n.sum()
    pe = float((p ** 2).sum())
    return float("nan") if pe == 1 else (pa - pe) / (1 - pe)


def gwet_ac1(m, k):
    n = _counts(m, k)
    pa = percent_agreement(m, k)
    p = n.sum(0) / n.sum()
    pe = float((p * (1 - p)).sum() / (k - 1))
    return float("nan") if pe == 1 else (pa - pe) / (1 - pe)


def krippendorff_alpha(m, k):
    """Nominal Krippendorff's alpha with missing ratings allowed."""
    n = _counts(m, k).astype(float)
    r = n.sum(1)
    keep = r >= 2
    n, r = n[keep], r[keep]
    w = 1.0 / (r - 1)                                                   # coincidence matrix, vectorised over items
    o = (n.T * w) @ n - np.diag((n * w[:, None]).sum(0))
    nc = o.sum(1)
    total = nc.sum()
    do = (o.sum() - np.trace(o)) / total
    de = (nc.sum() ** 2 - (nc ** 2).sum()) / (total * (total - 1))
    return float("nan") if de == 0 else 1 - do / de


def pairwise_cohen(m, k):
    """Mean and range of Cohen's kappa over rater pairs (items rated by both)."""
    m = np.asarray(m)
    out = []
    for i in range(m.shape[1]):
        for j in range(i + 1, m.shape[1]):
            ok = (m[:, i] >= 0) & (m[:, j] >= 0)
            a, b = m[ok, i], m[ok, j]
            po = float((a == b).mean())
            pe = sum(float((a == c).mean() * (b == c).mean()) for c in range(k))
            out.append(float("nan") if pe == 1 else (po - pe) / (1 - pe))
    v = [x for x in out if x == x]
    return (float(np.mean(v)), float(min(v)), float(max(v))) if v else (float("nan"),) * 3


def bootstrap_ci(fn, m, k, reps=2000, seed=13):
    """95% percentile interval, resampling items."""
    m = np.asarray(m)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        v = fn(m[rng.integers(0, len(m), len(m))], k)
        if v == v:
            vals.append(v)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"), float("nan"))


def encode(rows, categories):
    """rows: list of lists of labels (None = missing) -> int matrix."""
    idx = {c: i for i, c in enumerate(categories)}
    return np.array([[idx.get(x, -1) for x in r] for r in rows])


def summary(m, k, reps=2000):
    out = {"n_items": int(len(m)), "raters": int(np.asarray(m).shape[1]), "percent_agreement": percent_agreement(m, k)}
    for name, fn in (("fleiss_kappa", fleiss_kappa), ("krippendorff_alpha", krippendorff_alpha), ("gwet_ac1", gwet_ac1)):
        out[name] = fn(m, k)
        out[name + "_lo"], out[name + "_hi"] = bootstrap_ci(fn, m, k, reps)
    out["cohen_mean"], out["cohen_min"], out["cohen_max"] = pairwise_cohen(m, k)
    return out


def qualified_annotators(cfg, probe_dir="data/judge_probe"):
    """Annotators whose accuracy on the probe texts reaches annotators.probe_min. Returns ({name: accuracy}, [qualified names])."""
    import json
    from pathlib import Path
    acc = {}
    for a in cfg["annotators"]["panel"]:
        f = Path(probe_dir) / f"annotator_{a['name']}.jsonl"
        if f.exists():
            rows = [json.loads(l) for l in open(f, encoding="utf-8")]
            acc[a["name"]] = sum(r["correct"] for r in rows) / len(rows)
    return acc, [n for n, v in acc.items() if v >= cfg["annotators"]["probe_min"]]
