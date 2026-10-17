"""Rule-based starting assumptions: sector benchmarks plus transparent rules.

The benchmark figures are ILLUSTRATIVE mature-firm values in the style of Damodaran's industry averages.
Edit them in the app (Story & assumptions → Rulebook) or here, and update them each year.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# group: (target EBIT margin, sales-to-capital, unlevered beta, terminal ROIC)
BENCHMARKS = {
    "Software": (0.25, 2.0, 1.10, 0.20),
    "IT services": (0.20, 2.5, 0.90, 0.25),
    "Internet platforms": (0.18, 3.0, 1.10, 0.18),
    "Consumer cyclical": (0.10, 2.0, 0.90, 0.13),
    "Consumer staples": (0.15, 2.5, 0.60, 0.18),
    "Healthcare": (0.18, 1.5, 0.90, 0.15),
    "Industrials": (0.12, 1.5, 0.90, 0.13),
    "Materials": (0.14, 1.0, 1.00, 0.11),
    "Energy": (0.12, 0.8, 0.90, 0.10),
    "Utilities": (0.20, 0.4, 0.50, 0.08),
    "Real estate": (0.30, 0.3, 0.70, 0.08),
    "Telecom & media": (0.18, 1.0, 0.80, 0.12),
    "Financial platforms": (0.22, 3.0, 0.90, 0.18),
    "Banks & insurers": (0.25, 2.0, 0.70, 0.12),
    "Other": (0.12, 2.0, 0.90, 0.12),
}

# keyword in Yahoo industry/sector/company text -> group (first match wins; industry is checked first)
KEYWORDS = [
    ("software", "Software"), ("saas", "Software"),
    ("information technology services", "IT services"), ("it services", "IT services"),
    ("internet", "Internet platforms"), ("e-commerce", "Internet platforms"), ("online", "Internet platforms"),
    ("insurance brokers", "Financial platforms"), ("financial data", "Financial platforms"),
    ("capital markets", "Financial platforms"), ("credit services", "Financial platforms"),
    ("bank", "Banks & insurers"), ("insurance", "Banks & insurers"),
    ("consumer defensive", "Consumer staples"), ("fmcg", "Consumer staples"), ("household", "Consumer staples"),
    ("packaged foods", "Consumer staples"), ("beverages", "Consumer staples"), ("tobacco", "Consumer staples"),
    ("consumer cyclical", "Consumer cyclical"), ("auto", "Consumer cyclical"), ("retail", "Consumer cyclical"),
    ("healthcare", "Healthcare"), ("pharma", "Healthcare"), ("drug", "Healthcare"),
    ("industrials", "Industrials"), ("engineering", "Industrials"), ("machinery", "Industrials"),
    ("basic materials", "Materials"), ("chemical", "Materials"), ("steel", "Materials"), ("cement", "Materials"),
    ("energy", "Energy"), ("oil", "Energy"), ("utilities", "Utilities"), ("power", "Utilities"),
    ("real estate", "Real estate"), ("communication", "Telecom & media"), ("telecom", "Telecom & media"),
    ("technology", "Software"), ("financial services", "Financial platforms"),
]

BENCH_COLS = ["Target EBIT margin %", "Sales-to-capital", "Unlevered beta", "Terminal ROIC %"]


def bench_frame() -> pd.DataFrame:
    return pd.DataFrame([[m * 100, s, b, r * 100] for (m, s, b, r) in BENCHMARKS.values()],
                        index=list(BENCHMARKS), columns=BENCH_COLS)


def classify(sector: str, industry: str = "", name: str = "") -> str:
    for text in (industry, sector, name):
        t = (text or "").lower()
        if not t:
            continue
        for kw, grp in KEYWORDS:
            if kw in t:
                return grp
    return "Other"


def apply_rules(c, a, bench: pd.DataFrame | None = None, group: str | None = None):
    """Overwrite the starting assumptions in `a` using the rulebook. Returns (a, log rows, group)."""
    bench = bench if bench is not None else bench_frame()
    group = group or classify(c.sector, getattr(c, "industry", ""), c.name)
    if group not in bench.index:
        group = "Other"
    row = bench.loc[group]
    b_margin, b_s2c = float(row[BENCH_COLS[0]]) / 100, float(row[BENCH_COLS[1]])
    b_beta, b_roic = float(row[BENCH_COLS[2]]), float(row[BENCH_COLS[3]]) / 100
    log = []

    def add(assumption, value, rule):
        log.append({"Assumption": assumption, "Value": value, "Rule applied": rule})

    # growth
    hist = c.history or {}
    revs = [r for r in hist.get("revenue", []) if r]
    cagr = (revs[-1] / revs[0]) ** (1 / (len(revs) - 1)) - 1 if len(revs) >= 2 and revs[0] > 0 else None
    g = float(np.clip(cagr, a.terminal_growth, 0.30)) if cagr is not None else max(a.terminal_growth, 0.08)
    a.growth_y1 = a.growth_stage1 = g
    add("Revenue growth (stage 1)", f"{g:.1%}",
        (f"Historical CAGR {cagr:.1%}" if cagr is not None else "No history: 8%")
        + ", kept between terminal growth and a 30% cap")

    # length of high growth
    if g > 0.25:
        h, n = 10, 12
    elif g > 0.15:
        h, n = 7, 10
    elif g > 0.08:
        h, n = 5, 10
    else:
        h, n = 3, 10
    a.years, a.high_growth_years = n, h
    add("High-growth years", f"{h} (forecast {n} years)",
        "Growth above 25% → 10 years; 15–25% → 7; 8–15% → 5; otherwise 3. Then a linear fade to terminal growth")

    # margins
    cur = c.ebit / c.revenue if c.revenue else 0.0
    a.margin_y1 = float(cur)
    if cur >= b_margin:
        tgt = min(cur, 1.5 * b_margin)
        rule = (f"Today's margin {cur:.1%} is at or above the {group} benchmark {b_margin:.0%}: keep it (a moat), "
                f"capped at 1.5× the benchmark")
        conv = 3
    else:
        tgt = b_margin
        conv = 7 if cur < 0.05 else 5
        rule = (f"Today's margin {cur:.1%} is below the {group} benchmark {b_margin:.0%}: converge to the "
                f"benchmark by year {conv}")
    a.target_margin, a.margin_converge_year = float(tgt), min(conv, a.years)
    add("Target EBIT margin", f"{tgt:.1%} by year {a.margin_converge_year}", rule)

    # sales to capital
    ic = c.ic0()
    own = c.revenue / ic if ic > 0 else None
    if own is not None and 0.5 * b_s2c <= own <= 2 * b_s2c:
        s2c = (own + b_s2c) / 2
        rule = f"Average of the company's own {own:.2f} and the {group} benchmark {b_s2c:.1f}"
    else:
        s2c = b_s2c
        rule = (f"{group} benchmark {b_s2c:.1f}" + (f" (the company's own {own:.2f} is out of range, usually "
                                                     "distorted by cash, investments or goodwill)" if own else ""))
    a.sales_to_capital = float(s2c)
    a.reinvest_method = "sales_to_capital"
    add("Sales-to-capital", f"{s2c:.2f}", rule)

    # beta
    a.wacc.beta_method = "bottom_up"
    a.wacc.unlevered_beta = b_beta
    add("Beta", f"bottom-up: unlevered {b_beta:.2f}, relevered at D/E {a.wacc.target_de:.0%}",
        f"{group} unlevered beta, relevered at the company's own debt (regression beta {c.beta:.2f} shown for "
        "comparison on ③ Cost of capital)")

    # terminal growth
    if a.terminal_growth > a.wacc.rf:
        a.terminal_growth = a.wacc.rf
    add("Terminal growth", f"{a.terminal_growth:.1%}", f"Long-run nominal growth, capped at the risk-free rate "
                                                       f"{a.wacc.rf:.1%}")

    # cost of debt
    if c.ebit > 0:
        a.wacc.kd_method = "synthetic"
        add("Cost of debt", "synthetic rating", "EBIT is positive: rating from interest coverage")
    else:
        a.wacc.kd_method = "direct"
        a.wacc.kd_direct = a.wacc.rf + a.wacc.crp + 0.03
        add("Cost of debt", f"{a.wacc.kd_direct:.1%}", "EBIT is negative: risk-free + country risk + 3%")

    # terminal ROIC: sector benchmark, at least WACC + 2 points
    from lv_wacc import compute_wacc
    w = compute_wacc(a.wacc, c.ebit, c.interest)["wacc"]
    roic = max(b_roic, w + 0.02)
    a.terminal_roic = float(roic)
    add("Terminal ROIC", f"{roic:.1%}", f"{group} benchmark {b_roic:.0%}, but at least WACC ({w:.1%}) + 2 points")

    if group == "Banks & insurers":
        add("Model choice", "use residual income or DDM",
            "Banks and insurers: debt is raw material, not financing, so FCFF DCF is unreliable. Prefer residual "
            "income or the dividend discount model on ⑤ Other models")
    return a, log, group


def rule_defaults(c, bench: pd.DataFrame | None = None, group: str | None = None):
    """History-based defaults, then the rulebook on top. Returns (assumptions, log DataFrame, group)."""
    from lv_engine import defaults_from_company
    a = defaults_from_company(c)
    a, log, group = apply_rules(c, a, bench, group)
    a.notes = [f"{r['Assumption']}: {r['Value']}. {r['Rule applied']}." for r in log]
    return a, pd.DataFrame(log), group
