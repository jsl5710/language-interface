"""Stage 8: sensitivity analysis on judge-rejected renderings (appendix). Needs results/scored/scored_exp1{,_gatefailed}.jsonl.

    python scripts/08_gatefailed.py     -> results/aggregate/gatefailed_sensitivity.csv and paper/tables/gatefailed.tex
See src/dlsd/gate_sensitivity.py for the design. Local inputs only; the outputs are aggregate rates."""
import json
import re
from pathlib import Path

import pandas as pd

from dlsd.gate_io import GATE_DIR, latest_by_item, read_rows
from dlsd.gate_sensitivity import compare, frame, strata_from_gate


def gate_strata() -> dict:
    out = {}
    for f in sorted(GATE_DIR.glob("gate_codeswitch_*.jsonl")):
        m = re.match(r"gate_codeswitch_(native_)?(intra|inter|extra)_(\w+)\.jsonl", f.name)
        cid = f"codeswitch{'_native' if m.group(1) else ''}|{m.group(2)}|{m.group(3).capitalize()}"
        rows = [dict(r, condition_id=cid) for r in latest_by_item(read_rows(f)).values()]
        out |= strata_from_gate(rows)
    return out


if __name__ == "__main__":
    parts = [pd.DataFrame(read_rows(p)) for p in ("results/scored/scored_exp1.jsonl", "results/scored/scored_exp1_gatefailed.jsonl") if Path(p).exists()]
    d = frame(pd.concat(parts, ignore_index=True), gate_strata())
    print(d.groupby("stratum").size())
    res = compare(d)
    Path("results/aggregate").mkdir(parents=True, exist_ok=True)
    res.to_csv("results/aggregate/gatefailed_sensitivity.csv", index=False)
    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule", r"Outcome & Judge verdict on type & Rows & Rate & Confirmed rate & Difference (95\% CI) & Cells \\", r"\midrule"]
    name = {"comply": "Harmful compliance", "refuse": "Refusal", "accuracy": "Accuracy"}
    for r in res.itertuples():
        diff = f"{r.cell_diff:+.3f} ({r.ci_lo:+.3f} to {r.ci_hi:+.3f})" if r.n_cells else "--"      # no cell has 10 rows in both strata
        lines.append(f"{name.get(r.metric, r.metric)} & {r.stratum} & {r.n_rows} & {r.rate:.3f} & {r.confirmed_rate:.3f} & {diff} & {r.n_cells} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    Path("paper/tables/gatefailed.tex").write_text("\n".join(lines) + "\n")
    print(res.round(3).to_string())
