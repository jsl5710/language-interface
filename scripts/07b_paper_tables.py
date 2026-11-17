"""LaTeX tables for the Results sections, written from results/aggregate (never typed by hand).
    python scripts/07b_paper_tables.py
Writes paper/tables/caseresults.tex (harmful compliance and residual risk by axis), classifiers.tex (all four classifiers),
uqresults.tex (ECE by estimator and axis). Values are means over the cells of each axis, so they are descriptive; intervals are in the
cell-level tables."""
from pathlib import Path

import pandas as pd

AGG, TAB = Path("results/aggregate"), Path("paper/tables")
AXES = ["SAE", "Dialect", "Resource level", "Code-switch (English base)", "Code-switch (native base)"]
GROUPS = [("general", "General"), ("medical", "Medical")]
CLF = {"llama_guard": "Llama Guard 4", "wildguard": "WildGuard", "polyguard": "PolyGuard", "qwen3guard": "Qwen3Guard"}
f = lambda x: "--" if pd.isna(x) else f"{x:.2f}"


def table(rows, header, cols):
    body = [r" & ".join(map(str, r)) + r" \\" for r in rows]
    return "\n".join([r"\begin{tabular}{@{}" + cols + r"@{}}", r"\hline", header + r" \\", r"\hline", *body, r"\hline", r"\end{tabular}"]) + "\n"


t = pd.read_csv(AGG / "tab_caseresults.csv")
t["metric"] = t["metric"].fillna("harmful_compliance_rate")
rows = []
for g, gname in GROUPS:
    for mode, clf, met, label in (("internal", None, "harmful_compliance_rate", "Internal"), ("layered", "llama_guard", "residual_risk_rate", "Layered")):
        d = t[(t.model_group == g) & (t["mode"] == mode) & (t.metric == met)]
        if clf:
            d = d[d.classifier == clf]
        v = d.set_index("group")["value"]
        rows.append([gname, label] + [f(v.get(a)) for a in AXES])
(TAB / "caseresults.tex").write_text(table(rows, "Model group & Mode & " + " & ".join(["SAE", "Dialect", "Resource level", "Code-switch (English base)", "Code-switch (native base)"]), "llccccc"))
rows = []
for g, gname in GROUPS:
    for clf, cname in CLF.items():
        v = t[(t.model_group == g) & (t["mode"] == "layered") & (t.classifier == clf) & (t.metric == "residual_risk_rate")].set_index("group")["value"]
        rows.append([gname, cname] + [f(v.get(a)) for a in AXES])
(TAB / "classifiers.tex").write_text(table(rows, "Model group & Classifier & SAE & Dialect & Resource level & Code-switch (English base) & Code-switch (native base)", "llccccc"))
u = pd.read_csv(AGG / "tab_uqresults.csv")
rows = []
for g, gname in GROUPS:
    for est, ename in (("conf_prob", "Probability"), ("conf_verbalized", "Verbalized"), ("conf_sampling", "Sampling")):
        v = u[(u.model_group == g) & (u.estimator == est)].set_index("group")["ece"]
        rows.append([gname, ename] + [f(v.get(a)) for a in AXES])
(TAB / "uqresults.tex").write_text(table(rows, "Model group & Estimator & SAE & Dialect & Resource level & Code-switch (English base) & Code-switch (native base)", "llccccc"))
print("written", [p.name for p in (TAB / n for n in ("caseresults.tex", "classifiers.tex", "uqresults.tex"))])
