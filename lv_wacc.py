"""Cost of capital: CAPM cost of equity, cost of debt (direct or synthetic rating), WACC.

All rates are decimals (0.12 = 12%).
"""
from __future__ import annotations

from dataclasses import dataclass

# Interest-coverage -> rating -> default spread (large non-financial firms).
# Illustrative spreads in the style of Damodaran's annual table; update them each year.
RATING_TABLE = [
    # (minimum coverage, rating, spread)
    (8.50, "AAA", 0.0060),
    (6.50, "AA", 0.0080),
    (5.50, "A+", 0.0100),
    (4.25, "A", 0.0110),
    (3.00, "A-", 0.0125),
    (2.50, "BBB", 0.0160),
    (2.25, "BB+", 0.0200),
    (2.00, "BB", 0.0240),
    (1.75, "B+", 0.0290),
    (1.50, "B", 0.0350),
    (1.25, "B-", 0.0440),
    (0.80, "CCC", 0.0600),
    (0.65, "CC", 0.0800),
    (0.20, "C", 0.1100),
    (float("-inf"), "D", 0.1500),
]


def synthetic_rating(ebit: float, interest: float):
    """Return (coverage, rating, spread) from interest coverage = EBIT / interest."""
    if interest is None or interest <= 0:
        cov = float("inf")
    else:
        cov = ebit / interest
    for floor, rating, spread in RATING_TABLE:
        if cov >= floor:
            return cov, rating, spread
    return cov, "D", 0.15


def relever_beta(unlevered: float, de_ratio: float, tax: float) -> float:
    """Hamada: beta_L = beta_U x (1 + (1 - t) x D/E)."""
    return unlevered * (1.0 + (1.0 - tax) * de_ratio)


def unlever_beta(levered: float, de_ratio: float, tax: float) -> float:
    return levered / (1.0 + (1.0 - tax) * de_ratio)


def blume_adjust(raw_beta: float) -> float:
    """Blume adjustment: 2/3 x raw + 1/3 x 1."""
    return 2.0 / 3.0 * raw_beta + 1.0 / 3.0


@dataclass
class WaccInputs:
    rf: float = 0.065               # risk-free rate
    erp: float = 0.045              # mature-market equity risk premium
    crp: float = 0.025              # country risk premium
    crp_scaled_by_beta: bool = True  # Ke = rf + beta x (ERP + CRP)  vs  rf + beta x ERP + CRP
    size_premium: float = 0.0
    alpha: float = 0.0              # company-specific premium (judgement)
    beta_method: str = "given"      # given | blume | bottom_up
    beta_given: float = 1.0
    unlevered_beta: float = 0.9     # for bottom_up (industry average)
    target_de: float = 0.10         # target debt / equity (market values)
    kd_method: str = "synthetic"    # synthetic | direct
    kd_direct: float = 0.085        # pre-tax cost of debt if direct
    add_crp_to_kd: bool = True
    marginal_tax: float = 0.25
    wacc_override: float | None = None


def compute_wacc(w: WaccInputs, ebit: float = 0.0, interest: float = 0.0) -> dict:
    """Build the WACC step by step and return every intermediate number."""
    if w.beta_method == "bottom_up":
        beta = relever_beta(w.unlevered_beta, w.target_de, w.marginal_tax)
    elif w.beta_method == "blume":
        beta = blume_adjust(w.beta_given)
    else:
        beta = w.beta_given

    if w.crp_scaled_by_beta:
        ke = w.rf + beta * (w.erp + w.crp) + w.size_premium + w.alpha
    else:
        ke = w.rf + beta * w.erp + w.crp + w.size_premium + w.alpha

    cov, rating, spread = synthetic_rating(ebit, interest)
    if w.kd_method == "synthetic":
        kd_pre = w.rf + spread + (w.crp if w.add_crp_to_kd else 0.0)
    else:
        kd_pre = w.kd_direct
    kd_post = kd_pre * (1.0 - w.marginal_tax)

    d_v = w.target_de / (1.0 + w.target_de)
    e_v = 1.0 - d_v
    wacc = e_v * ke + d_v * kd_post
    if w.wacc_override is not None:
        wacc = w.wacc_override

    return {
        "beta": beta,
        "ke": ke,
        "coverage": cov,
        "rating": rating,
        "spread": spread,
        "kd_pre": kd_pre,
        "kd_post": kd_post,
        "e_v": e_v,
        "d_v": d_v,
        "wacc": wacc,
        "equity_component": e_v * ke,
        "debt_component": d_v * kd_post,
    }
