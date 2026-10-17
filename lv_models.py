"""Alternative valuation models: FCFE, dividend discount, residual income, relative (multiples)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from lv_engine import Company, Assumptions, run_dcf
from lv_wacc import compute_wacc


def run_fcfe(c: Company, a: Assumptions, debt_ratio: float | None = None) -> dict:
    """FCFE = net income - reinvestment + net borrowing, discounted at the cost of equity.

    Debt is kept at a constant share (debt_ratio) of invested capital, so
    net borrowing = debt_ratio x reinvestment, and interest = pre-tax Kd x opening debt.
    """
    base = run_dcf(c, a)
    t = base["table"]
    w = base["wacc"]
    ke, kd = w["ke"], w["kd_pre"]
    ic0 = c.ic0()
    delta = debt_ratio if debt_ratio is not None else (c.debt / ic0 if ic0 > 0 else 0.0)
    delta = float(np.clip(delta, 0.0, 0.9))
    debt_open = c.debt
    rows = []
    for i, (yr, r) in enumerate(t.iterrows()):
        interest = kd * debt_open
        ni = (r["EBIT"] - interest) * (1 - r["Tax rate"])
        net_borrow = delta * r["Reinvestment"]
        fcfe = ni - r["Reinvestment"] + net_borrow
        rows.append({"Net income": ni, "Interest": interest, "Reinvestment": r["Reinvestment"],
                     "Net borrowing": net_borrow, "FCFE": fcfe})
        debt_open += net_borrow
    df = pd.DataFrame(rows, index=t.index)
    disc = np.array([(1 + ke) ** -(i + 1) for i in range(len(df))])
    df["PV of FCFE"] = df["FCFE"] * disc
    gT = a.terminal_growth
    if gT >= ke:
        raise ValueError("Terminal growth must be below the cost of equity.")
    last = t.iloc[-1]
    ebit_n1 = last["EBIT"] * (1 + gT)
    ni_n1 = (ebit_n1 - kd * debt_open) * (1 - last["Tax rate"])
    reinv_n1 = ebit_n1 * (1 - last["Tax rate"]) * (gT / a.terminal_roic if a.terminal_roic > 0 else 0)
    fcfe_n1 = ni_n1 - (1 - delta) * reinv_n1
    tv = fcfe_n1 / (ke - gT)
    pv_tv = tv * disc[-1]
    equity = df["PV of FCFE"].sum() + pv_tv + c.cash + c.non_operating - c.minority - c.options
    return {"table": df, "ke": ke, "tv": tv, "pv_tv": pv_tv, "equity": equity,
            "per_share": equity / c.shares, "debt_ratio": delta}


def run_ddm(dps0: float, g1: float, years: int, g_terminal: float, ke: float) -> dict:
    """Two-stage dividend discount model."""
    if g_terminal >= ke:
        raise ValueError("Terminal growth must be below the cost of equity.")
    divs, pvs = [], []
    d = dps0
    for t in range(1, years + 1):
        d = d * (1 + g1)
        divs.append(d)
        pvs.append(d / (1 + ke) ** t)
    d_next = divs[-1] * (1 + g_terminal) if divs else dps0 * (1 + g_terminal)
    tv = d_next / (ke - g_terminal)
    pv_tv = tv / (1 + ke) ** years
    value = sum(pvs) + pv_tv
    table = pd.DataFrame({"Dividend per share": divs, "PV": pvs},
                         index=[f"Y{i}" for i in range(1, years + 1)])
    return {"table": table, "tv": tv, "pv_tv": pv_tv, "per_share": value,
            "pv_stage1": sum(pvs)}


def run_residual_income(bv0: float, roe1: float, roe_terminal: float, payout: float, ke: float,
                        years: int, g_terminal: float, shares: float) -> dict:
    """Residual income: value = BV0 + PV of (ROE - Ke) x opening BV. ROE fades linearly."""
    if g_terminal >= ke:
        raise ValueError("Terminal growth must be below the cost of equity.")
    bv = bv0
    rows = []
    for t in range(1, years + 1):
        roe = roe1 + (roe_terminal - roe1) * (t - 1) / max(1, years - 1)
        ni = roe * bv
        ri = (roe - ke) * bv
        pv = ri / (1 + ke) ** t
        rows.append({"Opening book value": bv, "ROE": roe, "Net income": ni,
                     "Residual income": ri, "PV of RI": pv})
        bv = bv + ni * (1 - payout)
    df = pd.DataFrame(rows, index=[f"Y{i}" for i in range(1, years + 1)])
    ri_next = (roe_terminal - ke) * bv * 1.0
    tv = ri_next / (ke - g_terminal)
    pv_tv = tv / (1 + ke) ** years
    equity = bv0 + df["PV of RI"].sum() + pv_tv
    return {"table": df, "tv": tv, "pv_tv": pv_tv, "equity": equity, "per_share": equity / shares}


MULTIPLES = ["P/E", "EV/EBITDA", "EV/Sales", "P/B"]


def run_relative(c: Company, peers: pd.DataFrame, stat: str = "median") -> pd.DataFrame:
    """Apply peer multiples to the company's own metrics. Returns implied price per multiple."""
    eps = c.net_income / c.shares
    bvps = c.book_equity / c.shares
    ebitda = c.ebit + c.da
    bridge = c.cash + c.non_operating - c.debt - c.minority
    out = []
    for mcol in MULTIPLES:
        if mcol not in peers.columns:
            continue
        vals = pd.to_numeric(peers[mcol], errors="coerce").dropna()
        vals = vals[vals > 0]
        if vals.empty:
            continue
        if stat == "mean":
            mult = vals.mean()
        elif stat == "harmonic mean":
            mult = len(vals) / (1.0 / vals).sum()
        else:
            mult = vals.median()
        lo, hi = vals.min(), vals.max()

        def price(x):
            if mcol == "P/E":
                return x * eps
            if mcol == "P/B":
                return x * bvps
            metric = ebitda if mcol == "EV/EBITDA" else c.revenue
            return (x * metric + bridge) / c.shares

        own = {"P/E": c.price / eps if eps > 0 else np.nan,
               "P/B": c.price / bvps if bvps > 0 else np.nan,
               "EV/EBITDA": (c.price * c.shares - bridge) / ebitda if ebitda > 0 else np.nan,
               "EV/Sales": (c.price * c.shares - bridge) / c.revenue if c.revenue else np.nan}[mcol]
        out.append({"Multiple": mcol, "Peer value used": mult, "Peer low": lo, "Peer high": hi,
                    "Company today": own, "Implied price": price(mult),
                    "Implied low": price(lo), "Implied high": price(hi)})
    return pd.DataFrame(out)
