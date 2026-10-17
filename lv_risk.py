"""Risk and judgement tools: sensitivity tables, scenarios, Monte Carlo, reverse DCF."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from scipy import stats

from lv_engine import Company, Assumptions, run_dcf, set_driver


def _value(c, a, wacc_override=None):
    try:
        return run_dcf(c, a, wacc_override)["per_share"]
    except ValueError:
        return np.nan


def sensitivity_wacc_growth(c: Company, a: Assumptions, wacc_vals, g_vals) -> pd.DataFrame:
    """Value per share for each (WACC, terminal growth) pair. WACC applied flat over all years."""
    rows = {}
    for gv in g_vals:
        aa = copy.deepcopy(a)
        aa.terminal_growth = gv
        rows[f"{gv:.1%}"] = [(_value(c, aa, wv) if gv < wv else np.nan) for wv in wacc_vals]
    df = pd.DataFrame(rows, index=[f"{w:.1%}" for w in wacc_vals]).T
    df.index.name = "Terminal growth \\ WACC"
    return df


def sensitivity_two(c: Company, a: Assumptions, field1: str, vals1, field2: str, vals2) -> pd.DataFrame:
    rows = {}
    for v1 in vals1:
        out = []
        for v2 in vals2:
            aa = copy.deepcopy(a)
            set_driver(aa, field1, v1)
            set_driver(aa, field2, v2)
            out.append(_value(c, aa))
        rows[f"{v1:.1%}"] = out
    return pd.DataFrame(rows, index=[f"{v:.1%}" for v in vals2]).T


def tornado(c: Company, a: Assumptions, shocks: dict) -> pd.DataFrame:
    """One-at-a-time swings. shocks = {label: (field, low_value, high_value)}."""
    base = _value(c, a)
    out = []
    for label, (fld, lo, hi) in shocks.items():
        vals = []
        for v in (lo, hi):
            aa = copy.deepcopy(a)
            if fld == "wacc_shift":
                vals.append(_value(c, aa, run_dcf(c, a)["wacc_used"] + v))
                continue
            set_driver(aa, fld, v)
            vals.append(_value(c, aa))
        out.append({"Driver": label, "Low input": lo, "High input": hi,
                    "Value at low": vals[0], "Value at high": vals[1],
                    "Swing": abs(vals[1] - vals[0])})
    df = pd.DataFrame(out).sort_values("Swing", ascending=True)
    return df, base


def scenarios(c: Company, a: Assumptions, spec: dict) -> pd.DataFrame:
    """spec = {name: {"prob": p, "growth": g, "margin": m, "wacc_shift": dw, "terminal_growth": gt}}."""
    base_w = run_dcf(c, a)["wacc_used"]
    rows = []
    for name, s in spec.items():
        aa = copy.deepcopy(a)
        set_driver(aa, "growth_stage1", s["growth"])
        set_driver(aa, "target_margin", s["margin"])
        aa.terminal_growth = s["terminal_growth"]
        v = _value(c, aa, base_w + s["wacc_shift"])
        rows.append({"Scenario": name, "Probability": s["prob"], "Stage-1 growth": s["growth"],
                     "Target margin": s["margin"], "WACC": base_w + s["wacc_shift"],
                     "Terminal growth": s["terminal_growth"], "Value per share": v})
    df = pd.DataFrame(rows)
    return df


def _tri_ppf(u, lo, mode, hi):
    if hi <= lo:
        return np.full_like(u, mode)
    cpar = (mode - lo) / (hi - lo)
    return stats.triang.ppf(u, cpar, loc=lo, scale=hi - lo)


def monte_carlo(c: Company, a: Assumptions, n: int, growth: tuple, margin: tuple, wacc_sd: float,
                tg: tuple, rho: float = 0.0, seed: int = 42) -> dict:
    """Simulate value per share.

    growth, margin, tg are (low, mode, high) triangular distributions; WACC ~ Normal(base, sd).
    rho correlates growth and margin through a Gaussian copula.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 4))
    z[:, 1] = rho * z[:, 0] + np.sqrt(max(0.0, 1 - rho ** 2)) * z[:, 1]
    u = stats.norm.cdf(z)
    g_s = _tri_ppf(u[:, 0], *growth)
    m_s = _tri_ppf(u[:, 1], *margin)
    tg_s = _tri_ppf(u[:, 2], *tg)
    base_w = run_dcf(c, a)["wacc_used"]
    w_s = base_w + wacc_sd * z[:, 3]
    vals = np.empty(n)
    for i in range(n):
        aa = copy.deepcopy(a)
        aa.terminal_wacc = None
        set_driver(aa, "growth_stage1", g_s[i])
        set_driver(aa, "target_margin", m_s[i])
        tgi = min(tg_s[i], w_s[i] - 0.01)
        aa.terminal_growth = tgi
        vals[i] = _value(c, aa, w_s[i])
    vals = vals[np.isfinite(vals)]
    pct = np.percentile(vals, [5, 10, 25, 50, 75, 90, 95])
    return {"values": vals, "mean": vals.mean(), "sd": vals.std(), "percentiles": dict(zip(
        ["P5", "P10", "P25", "P50", "P75", "P90", "P95"], pct)),
        "prob_above_price": float((vals > c.price).mean()) if c.price else np.nan,
        "inputs": pd.DataFrame({"growth": g_s, "margin": m_s, "wacc": w_s, "terminal_growth": tg_s})}


def reverse_dcf(c: Company, a: Assumptions, field: str, lo: float, hi: float, target: float | None = None):
    """Find the value of `field` that makes value per share equal the market price."""
    target = c.price if target is None else target

    def f(x):
        aa = copy.deepcopy(a)
        set_driver(aa, field, x)
        v = _value(c, aa)
        return v - target

    xs = np.linspace(lo, hi, 41)
    fs = np.array([f(x) for x in xs])
    for i in range(len(xs) - 1):
        if np.isfinite(fs[i]) and np.isfinite(fs[i + 1]) and fs[i] * fs[i + 1] <= 0:
            a_, b_ = xs[i], xs[i + 1]
            for _ in range(60):
                mid = (a_ + b_) / 2
                if f(a_) * f(mid) <= 0:
                    b_ = mid
                else:
                    a_ = mid
            return (a_ + b_) / 2
    return None
