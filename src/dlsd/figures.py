"""Figures in the paper's style: hollow bars with dashed dark-grey outlines, diagonal then dotted hatching, light grey horizontal
gridlines, no top or right spines, DejaVu Sans 8 pt, 6.2 in wide, grayscale-safe, vector PDF (docs/EXPERIMENTS.md section 10.x).
The 'PLACEHOLDER' stamp of the reference scripts is not drawn: these figures hold computed results."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
DARK = "#5f6368"
HATCH = ["///", "...", "\\\\\\", "xxx"]


def _base(ax, ylabel, title, ylim=(0, 100)):
    ax.set_ylim(*ylim); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=9, loc="left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color="#e0e0e0", lw=0.6); ax.set_axisbelow(True)


def _bars(ax, groups_x, series_values, width_total=0.8):
    """groups_x: positions; series_values: list (one per series) of lists of heights, aligned with groups_x."""
    k = len(series_values)
    w = width_total / k
    for si, vals in enumerate(series_values):
        for x, v in zip(groups_x, vals):
            if v is None or np.isnan(v):
                continue
            x0 = x - width_total / 2 + si * w
            ax.add_patch(Rectangle((x0, 0), w * 0.95, v, fill=False, hatch=HATCH[si], edgecolor=DARK, lw=0.8, ls="--"))


def _legend(ax, labels, ncol=None):
    ax.legend(handles=[Rectangle((0, 0), 1, 1, fill=False, hatch=HATCH[i], edgecolor=DARK, ls="--") for i in range(len(labels))],
              labels=labels, frameon=False, loc="upper right", ncol=ncol or len(labels), fontsize=7)


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


CATS = ["SAE", "D1", "D2", "D3", "D4", "D5", "High-\nresource", "Mid-\nresource", "Low-\nresource", "Intra", "Inter", "Extra",
        "Intra\n(native)", "Inter\n(native)", "Extra\n(native)"]


def condition_series(cells: pd.DataFrame, metric: str, mode: str, model_id: str, classifier=None, dialects=()) -> list:
    """Percent values in CATS order for one metric. Code-switch types are averaged over the three languages."""
    d = cells[(cells.metric == metric) & (cells["mode"] == mode) & (cells.model_id == model_id)]
    if classifier is not None:
        d = d[d.classifier == classifier]
    else:
        d = d.groupby(["condition_axis", "condition_level", "language_pair"], dropna=False)["value"].mean().reset_index()
    pick = lambda ax, lv: d[(d.condition_axis == ax) & (d.condition_level == lv)]["value"]
    vals = [pick("baseline", "SAE").mean()] + [pick("dialect", x).mean() for x in dialects]
    vals += [pick("resource", x).mean() for x in ("high", "mid", "low")] + [pick("codeswitch", x).mean() for x in ("intra", "inter", "extra")] + [pick("codeswitch_native", x).mean() for x in ("intra", "inter", "extra")]
    return [v * 100 for v in vals]


def fig_compliance(cells, dialects, path, model_id="group:all"):
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    _base(ax, "Harmful-compliance rate (%)", "Medical safety prompts, by input condition")
    internal = condition_series(cells, "harmful_compliance_rate", "internal", model_id, dialects=dialects)
    layered = condition_series(cells, "residual_risk_rate", "layered", model_id, dialects=dialects)
    _bars(ax, range(len(CATS)), [internal, layered])
    ax.set_xlim(-0.6, len(CATS) - 0.4); ax.set_xticks(range(len(CATS))); ax.set_xticklabels(CATS)
    _legend(ax, ["Internal alignment", "Layered (+ classifier)"])
    _save(fig, path)
    return pd.DataFrame({"condition": CATS, "internal_pct": internal, "layered_pct": layered})


def fig_residual(cells, path, model_id="group:all"):
    groups = ["SAE", "Dialect", "Resource level", "Code-switch (English base)", "Code-switch (native base)"]
    d = cells[(cells.metric == "residual_risk_rate") & (cells.model_id == model_id)].copy()
    d["group"] = d.condition_axis.map({"baseline": "SAE", "dialect": "Dialect", "resource": "Resource level", "codeswitch": "Code-switch (English base)", "codeswitch_native": "Code-switch (native base)"})
    clfs = sorted(d.classifier.dropna().unique())
    fig, ax = plt.subplots(figsize=(6.2, 2.5))
    _base(ax, "Residual risk (%)", "Model complies and classifier does not flag, by classifier")
    vals = [[d[(d.group == g) & (d.classifier == c)]["value"].mean() * 100 for g in groups] for c in clfs]
    _bars(ax, range(len(groups)), vals)
    ax.set_xlim(-0.6, len(groups) - 0.4); ax.set_xticks(range(len(groups))); ax.set_xticklabels(groups)
    if clfs:
        _legend(ax, clfs)
    _save(fig, path)
    return pd.DataFrame([{"group": g, "classifier": c, "residual_risk_pct": v} for c, vs in zip(clfs, vals) for g, v in zip(groups, vs)])


def fig_gap(cells, path):
    axes = ["dialect", "resource", "codeswitch", "codeswitch_native"]
    names = ["Dialect", "Resource level", "Code-switch\n(English base)", "Code-switch\n(native base)"]
    d = cells[(cells.metric == "harmful_compliance_rate") & (cells["mode"] == "internal") & cells.model_id.isin(["group:general", "group:medical"])]
    vals = [[d[(d.model_id == m) & (d.condition_axis == a)]["gap_from_sae"].mean() * 100 for a in axes] for m in ("group:general", "group:medical")]
    allv = [v for vs in vals for v in vs if not np.isnan(v)] or [0]
    lim = (min(-10, np.floor(min(allv) / 10) * 10), max(10, np.ceil(max(allv) / 10) * 10))
    fig, ax = plt.subplots(figsize=(6.2, 2.5))
    _base(ax, "Gap from SAE (pp)", "Change in harmful-compliance rate versus SAE, general vs medical models", ylim=lim)
    ax.axhline(0, color=DARK, lw=0.8)
    k, w = 2, 0.4
    for si, vs in enumerate(vals):                      # negative gaps need bars below zero
        for x, v in enumerate(vs):
            if not np.isnan(v):
                x0 = x - 0.4 + si * w
                ax.add_patch(Rectangle((x0, 0), w * 0.95, v, fill=False, hatch=HATCH[si], edgecolor=DARK, lw=0.8, ls="--"))
    ax.set_xlim(-0.6, len(axes) - 0.4); ax.set_xticks(range(len(axes))); ax.set_xticklabels(names)
    _legend(ax, ["General-purpose", "Medical / health"])
    _save(fig, path)
    return pd.DataFrame([{"axis": n, "model_group": m, "gap_pp": v} for m, vs in zip(("general", "medical"), vals) for n, v in zip(names, vs)])


def fig_uq(gens: pd.DataFrame, estimator: str, path):
    conds = ["SAE", "Dialect", "High\nres.", "Mid\nres.", "Low\nres.", "Intra", "Inter", "Extra", "Intra\n(nat.)", "Inter\n(nat.)", "Extra\n(nat.)"]
    sel = [("baseline", None), ("dialect", None), ("resource", "high"), ("resource", "mid"), ("resource", "low"),
           ("codeswitch", "intra"), ("codeswitch", "inter"), ("codeswitch", "extra"),
           ("codeswitch_native", "intra"), ("codeswitch_native", "inter"), ("codeswitch_native", "extra")]
    fig, axs = plt.subplots(1, 2, figsize=(6.2, 2.6), sharey=True)
    recs = []
    for ax, (grp, title) in zip(axs, (("general", "General models"), ("medical", "Medical models"))):
        d = gens[gens.model_group == grp].dropna(subset=[estimator, "correct"])
        acc, conf = [], []
        for axis, lv in sel:
            g = d[(d.condition_axis == axis) & ((d.condition_level == lv) if lv else True)]
            acc.append(g["correct"].astype(float).mean() * 100 if len(g) else np.nan)
            conf.append(g[estimator].astype(float).mean() * 100 if len(g) else np.nan)
            recs.append({"model_group": grp, "condition": f"{axis}|{lv or 'all'}", "accuracy_pct": acc[-1], "mean_conf_pct": conf[-1], "n": len(g)})
        _base(ax, "", title)
        _bars(ax, range(len(conds)), [acc, conf])
        ax.set_xlim(-0.6, len(conds) - 0.4); ax.set_xticks(range(len(conds))); ax.set_xticklabels(conds, fontsize=6.5)
    axs[0].set_ylabel("Rate (%)")
    _legend(axs[0], ["Accuracy", "Mean confidence"], ncol=1)
    _save(fig, path)
    return pd.DataFrame(recs)
