"""Core FCFF discounted-cash-flow engine.

Money is in the company's reporting unit (for example Rs crore or $ million) and shares are
in the matching unit (crore or million), so equity value / shares gives a per-share price.
All rates are decimals.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

from lv_wacc import WaccInputs, compute_wacc


@dataclass
class Company:
    name: str = "Sample company"
    ticker: str = ""
    sector: str = ""
    industry: str = ""
    currency: str = "₹"
    unit: str = "crore"            # crore | million
    revenue: float = 10000.0        # last full year
    ebit: float = 1500.0
    da: float = 300.0               # depreciation & amortisation
    capex: float = 400.0
    nwc: float = 1000.0             # net working capital (non-cash)
    cash: float = 1000.0
    debt: float = 500.0
    minority: float = 0.0
    non_operating: float = 0.0      # cross-holdings and other non-operating assets
    options: float = 0.0            # value of employee options / ESOPs outstanding
    shares: float = 100.0
    price: float = 100.0
    eff_tax: float = 0.25
    interest: float = 50.0
    net_income: float = 1000.0
    dividends: float = 300.0
    book_equity: float = 6000.0
    invested_capital: float | None = None   # if None: book equity + debt - cash
    beta: float = 1.0
    history: dict = field(default_factory=dict)  # {"year": [...], "revenue": [...], "ebit": [...]}

    def ic0(self) -> float:
        """Operating invested capital: equity + debt - cash - non-operating investments."""
        if self.invested_capital:
            return self.invested_capital
        ic = self.book_equity + self.debt - self.cash - self.non_operating
        return ic if ic > 0 else max(self.revenue * 0.3, 1.0)


@dataclass
class Assumptions:
    years: int = 10
    high_growth_years: int = 5
    growth_y1: float = 0.12
    growth_stage1: float = 0.12        # years 2..H
    terminal_growth: float = 0.05
    margin_y1: float = 0.15
    target_margin: float = 0.18
    margin_converge_year: int = 5
    tax_converge: bool = True           # move from effective to marginal tax by the final year
    reinvest_method: str = "sales_to_capital"   # sales_to_capital | explicit
    sales_to_capital: float = 2.0
    capex_pct: float = 0.04             # explicit method: capex % revenue
    da_pct: float = 0.03                # D&A % revenue (also used for EBITDA)
    nwc_pct: float = 0.10               # NWC % of revenue (explicit method uses change)
    terminal_method: str = "gordon"     # gordon | exit_multiple
    terminal_roic: float = 0.15         # reinvestment in perpetuity = g / ROIC
    exit_multiple: float = 12.0         # EV / EBITDA at end of forecast
    terminal_wacc: float | None = None  # mature WACC; None = same as WACC
    mid_year: bool = False
    failure_prob: float = 0.0
    distress_recovery: float = 0.5      # fraction of operating value recovered in failure
    notes: list = field(default_factory=list)  # why the defaults were chosen (shown in the app)
    growth_path: list | None = None     # optional year-by-year revenue growth (overrides the fade)
    margin_path: list | None = None     # optional year-by-year EBIT margin (overrides convergence)
    wacc: WaccInputs = field(default_factory=WaccInputs)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def _paths(a: Assumptions):
    n, h = a.years, min(a.high_growth_years, a.years)
    g = np.zeros(n)
    for t in range(1, n + 1):
        if t == 1:
            g[t - 1] = a.growth_y1
        elif t <= h:
            g[t - 1] = a.growth_stage1
        else:
            # linear fade from stage-1 growth to terminal growth by year n
            steps = n - h
            k = t - h
            g[t - 1] = a.growth_stage1 + (a.terminal_growth - a.growth_stage1) * k / steps
    m = np.zeros(n)
    c = max(1, min(a.margin_converge_year, n))
    for t in range(1, n + 1):
        if t >= c:
            m[t - 1] = a.target_margin
        elif c == 1:
            m[t - 1] = a.target_margin
        else:
            m[t - 1] = a.margin_y1 + (a.target_margin - a.margin_y1) * (t - 1) / (c - 1)
    if a.growth_path is not None and len(a.growth_path) == n:
        g = np.array(a.growth_path, dtype=float)
    if a.margin_path is not None and len(a.margin_path) == n:
        m = np.array(a.margin_path, dtype=float)
    return g, m


def base_paths(a: "Assumptions"):
    """Growth and margin paths from the fade inputs alone, ignoring any year-by-year override."""
    import copy as _c
    b = _c.copy(a)
    b.growth_path = b.margin_path = None
    return _paths(b)


def set_driver(a: "Assumptions", field_name: str, value: float) -> None:
    """Change one value driver. If a year-by-year path is in use, shift the whole path by the same amount,
    so sensitivity, scenarios, Monte Carlo and reverse DCF keep working on top of the user's own path."""
    if field_name == "growth_stage1":
        d = value - a.growth_stage1
        a.growth_stage1 = value
        a.growth_y1 = value
        if a.growth_path is not None:
            a.growth_path = [x + d for x in a.growth_path]
    elif field_name == "target_margin":
        d = value - a.target_margin
        a.target_margin = value
        if a.margin_path is not None:
            a.margin_path = [x + d for x in a.margin_path]
    else:
        setattr(a, field_name, value)


def run_dcf(c: Company, a: Assumptions, wacc_override: float | None = None) -> dict:
    """FCFF DCF. Returns a dict with the forecast table and the valuation bridge."""
    w = compute_wacc(a.wacc, c.ebit, c.interest)
    wacc0 = wacc_override if wacc_override is not None else w["wacc"]
    wacc_T = a.terminal_wacc if (a.terminal_wacc is not None and wacc_override is None) else wacc0
    n, h = a.years, min(a.high_growth_years, a.years)
    g, m = _paths(a)

    # tax path: effective rate in high growth, converging to marginal by year n
    tax = np.full(n, c.eff_tax)
    if a.tax_converge:
        for t in range(h + 1, n + 1):
            tax[t - 1] = c.eff_tax + (a.wacc.marginal_tax - c.eff_tax) * (t - h) / max(1, n - h)
    # wacc path: constant during high growth, converging to terminal wacc
    wv = np.full(n, wacc0)
    for t in range(h + 1, n + 1):
        wv[t - 1] = wacc0 + (wacc_T - wacc0) * (t - h) / max(1, n - h)

    rev = np.zeros(n)
    prev = c.revenue
    for i in range(n):
        rev[i] = prev * (1 + g[i])
        prev = rev[i]
    d_rev = np.diff(np.concatenate([[c.revenue], rev]))
    ebit = rev * m
    nopat = ebit * (1 - tax)
    da = rev * a.da_pct
    if a.reinvest_method == "explicit":
        capex = rev * a.capex_pct
        d_nwc = d_rev * a.nwc_pct
        reinvest = capex - da + d_nwc
    else:
        reinvest = d_rev / a.sales_to_capital
        capex = reinvest + da          # implied gross capex (NWC folded in)
        d_nwc = np.zeros(n)
    fcff = nopat - reinvest

    # discounting with a time-varying rate
    df = np.zeros(n)
    acc = 1.0
    for i in range(n):
        acc /= (1 + wv[i])
        df[i] = acc
    df_cf = df * np.sqrt(1 + wv) if a.mid_year else df
    pv = fcff * df_cf

    # invested capital and ROIC
    ic = np.zeros(n)
    prev_ic = c.ic0()
    for i in range(n):
        ic[i] = prev_ic + reinvest[i]
        prev_ic = ic[i]
    ic_open = np.concatenate([[c.ic0()], ic[:-1]])
    roic = np.where(ic_open > 0, nopat / ic_open, np.nan)

    gT = a.terminal_growth
    ebitda_n = ebit[-1] + da[-1]
    if a.terminal_method == "exit_multiple":
        tv = ebitda_n * a.exit_multiple
        nopat_n1 = rev[-1] * (1 + gT) * m[-1] * (1 - tax[-1])
        # implied perpetual growth: solve TV = FCFF1/(w-g) with FCFF1 = NOPAT1(1-g/ROIC)
        implied_g = _implied_growth(tv, nopat_n1 / (1 + gT), wacc_T, a.terminal_roic)
        fcff_n1 = np.nan
        reinvest_rate_T = np.nan
    else:
        if gT >= wacc_T:
            raise ValueError("Terminal growth must be below the terminal WACC.")
        nopat_n1 = rev[-1] * (1 + gT) * m[-1] * (1 - tax[-1])
        reinvest_rate_T = gT / a.terminal_roic if a.terminal_roic > 0 else 0.0
        fcff_n1 = nopat_n1 * (1 - reinvest_rate_T)
        tv = fcff_n1 / (wacc_T - gT)
        implied_g = gT
    pv_tv = tv * df[-1]
    implied_multiple = tv / ebitda_n if ebitda_n > 0 else np.nan

    pv_fcff = pv.sum()
    op_value = pv_fcff + pv_tv
    op_value_adj = op_value * (1 - a.failure_prob) + op_value * a.distress_recovery * a.failure_prob
    equity = op_value_adj + c.cash + c.non_operating - c.debt - c.minority - c.options
    per_share = equity / c.shares if c.shares else np.nan

    years = [f"Y{i}" for i in range(1, n + 1)]
    table = pd.DataFrame({
        "Revenue growth": g, "Revenue": rev, "EBIT margin": m, "EBIT": ebit, "Tax rate": tax,
        "NOPAT": nopat, "D&A": da, "Capex": capex, "Change in NWC": d_nwc,
        "Reinvestment": reinvest, "FCFF": fcff, "WACC": wv, "Discount factor": df_cf,
        "PV of FCFF": pv, "Invested capital": ic, "ROIC": roic,
    }, index=years)

    return {
        "table": table, "wacc": w, "wacc_used": wacc0, "wacc_terminal": wacc_T,
        "pv_fcff": pv_fcff, "tv": tv, "pv_tv": pv_tv, "fcff_n1": fcff_n1,
        "reinvest_rate_terminal": reinvest_rate_T, "op_value": op_value,
        "op_value_adj": op_value_adj, "equity": equity, "per_share": per_share,
        "tv_share": pv_tv / op_value if op_value else np.nan,
        "implied_exit_multiple": implied_multiple, "implied_terminal_growth": implied_g,
        "ebitda_n": ebitda_n, "upside": per_share / c.price - 1 if c.price else np.nan,
        "roic_terminal": a.terminal_roic,
    }


def _implied_growth(tv, nopat_n, w, roic):
    """Growth that makes a Gordon TV equal tv: tv = nopat_n(1+g)(1-g/roic)/(w-g)."""
    lo, hi = -0.05, w - 1e-4
    f = lambda g: nopat_n * (1 + g) * (1 - g / roic) / (w - g) - tv
    try:
        if f(lo) * f(hi) > 0:
            return np.nan
        for _ in range(100):
            mid = (lo + hi) / 2
            if f(lo) * f(mid) <= 0:
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2
    except ZeroDivisionError:
        return np.nan


def defaults_from_company(c: Company) -> Assumptions:
    """Sensible starting assumptions from the company's own history.

    Loss-making or very-low-margin firms get a placeholder target margin (the story must set the real one),
    an unusual beta is replaced by a bottom-up beta, and negative EBIT switches the cost of debt to a direct
    estimate (a synthetic rating from negative coverage would say 'default').
    """
    a = Assumptions()
    a.notes = []
    if c.currency == "₹":
        a.wacc.rf, a.wacc.erp, a.wacc.crp = 0.065, 0.045, 0.025
        a.terminal_growth = 0.05
    else:
        a.wacc.rf, a.wacc.erp, a.wacc.crp = 0.043, 0.045, 0.0
        a.terminal_growth = 0.025
    a.wacc.marginal_tax = 0.25 if c.currency == "₹" else 0.21 if c.currency == "$" else 0.25

    hist = c.history or {}
    revs = [r for r in hist.get("revenue", []) if r]
    if len(revs) >= 2 and revs[0] > 0:
        cagr = (revs[-1] / revs[0]) ** (1 / (len(revs) - 1)) - 1
        a.growth_y1 = a.growth_stage1 = float(np.clip(cagr, -0.05, 0.30))
        if cagr > 0.30:
            a.notes.append(f"Past revenue growth was {cagr:.0%} a year; the default caps future growth at 30%.")
    margin = c.ebit / c.revenue if c.revenue else 0.1
    a.margin_y1 = float(margin)
    if margin < 0.05:
        a.target_margin = 0.12
        a.margin_converge_year = 7
        a.notes.append(f"Today's EBIT margin is {margin:.1%}. A loss-making or thin-margin firm cannot be valued on "
                       "today's margin for ever, so the target margin is a 12% placeholder reached in year 7. "
                       "This is the biggest call in your story: set it on ② Story & assumptions.")
    else:
        a.target_margin = float(margin)
    ic = c.ic0()
    s2c_hist = c.revenue / ic if ic > 0 else 2.0
    a.sales_to_capital = float(np.clip(s2c_hist, 0.5, 6.0))
    if s2c_hist < 1.0 and a.growth_stage1 >= 0.15:
        a.sales_to_capital = 2.5
        a.notes.append(f"Revenue ÷ invested capital is only {s2c_hist:.2f} today, which would make growth very "
                       "expensive. For a fast grower this is usually distorted by cash, investments or goodwill on "
                       "the balance sheet, so the default sales-to-capital is a 2.5 placeholder. Asset-light "
                       "businesses often run at 2–5; set yours on ② Story & assumptions.")
    a.da_pct = c.da / c.revenue if c.revenue else 0.03
    a.capex_pct = c.capex / c.revenue if c.revenue else 0.04
    a.nwc_pct = c.nwc / c.revenue if c.revenue else 0.1
    roic = c.ebit * (1 - c.eff_tax) / ic if ic > 0 else 0.15
    if margin < 0.05:
        a.terminal_roic = 0.15
    else:
        a.terminal_roic = float(np.clip(roic, 0.10, 0.40))

    b = c.beta or 1.0
    if b < 0.5 or b > 2.5:
        a.wacc.beta_method = "bottom_up"
        a.wacc.unlevered_beta = 1.0
        a.notes.append(f"The data feed's beta of {b:.2f} looks unreliable, so the default uses a bottom-up beta "
                       "(unlevered 1.0, relevered at the company's debt). Change it on ③ Cost of capital.")
    a.wacc.beta_given = b
    a.wacc.target_de = c.debt / (c.price * c.shares) if c.price and c.shares else 0.1
    if c.ebit <= 0:
        a.wacc.kd_method = "direct"
        a.wacc.kd_direct = a.wacc.rf + a.wacc.crp + 0.03
        a.notes.append("EBIT is negative, so a rating from interest coverage would say 'default'. The default "
                       f"pre-tax cost of debt is {a.wacc.kd_direct:.1%} (risk-free + country risk + 3%).")
    return a
