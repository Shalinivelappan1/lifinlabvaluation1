"""Assumption health checks: flag numbers that break economic logic or need a story."""
from __future__ import annotations

import numpy as np

from lv_engine import Company, Assumptions


def health_checks(c: Company, a: Assumptions, res: dict, peer_ev_ebitda: float | None = None) -> list:
    """Return a list of (level, title, message). level: red | amber | green."""
    out = []
    w = res["wacc"]
    rf = a.wacc.rf
    gT = a.terminal_growth
    t = res["table"]

    if gT > rf:
        out.append(("red", "Terminal growth above the risk-free rate",
                    f"Terminal growth {gT:.1%} exceeds the risk-free rate {rf:.1%}. In perpetuity no firm can "
                    "outgrow the economy; the risk-free rate is a sensible ceiling."))
    else:
        out.append(("green", "Terminal growth is capped",
                    f"Terminal growth {gT:.1%} is at or below the risk-free rate {rf:.1%}."))

    share = res["tv_share"]
    if share > 0.85:
        out.append(("red", "Terminal value dominates",
                    f"{share:.0%} of operating value comes after year {a.years}. The forecast years barely matter; "
                    "consider a longer explicit forecast or test the terminal assumptions hard."))
    elif share > 0.70:
        out.append(("amber", "Terminal value is large",
                    f"{share:.0%} of operating value is terminal value. Normal for growth firms, but test it."))
    else:
        out.append(("green", "Terminal value share", f"Terminal value is {share:.0%} of operating value."))

    if a.terminal_roic < res["wacc_terminal"]:
        out.append(("amber", "Growth destroys value in perpetuity",
                    f"Terminal ROIC {a.terminal_roic:.1%} is below the terminal WACC {res['wacc_terminal']:.1%}. "
                    "Every rupee reinvested earns less than it costs; growth then lowers value."))
    elif a.terminal_roic > 0.35:
        out.append(("amber", "Very high perpetual ROIC",
                    f"Terminal ROIC {a.terminal_roic:.1%} assumes a durable moat for ever. Competition usually "
                    "pulls excess returns toward the cost of capital."))
    else:
        out.append(("green", "Terminal ROIC", f"Terminal ROIC {a.terminal_roic:.1%} vs WACC {res['wacc_terminal']:.1%}."))

    roic_n = t["ROIC"].iloc[-1]
    if np.isfinite(roic_n) and roic_n > 0.6:
        out.append(("amber", "Forecast ROIC looks extreme",
                    f"ROIC reaches {roic_n:.0%} in year {a.years}. Check the sales-to-capital ratio: the model may be "
                    "growing revenue with too little reinvestment."))

    cur_margin = c.ebit / c.revenue if c.revenue else 0
    if cur_margin > 0 and a.target_margin > 2 * cur_margin and a.target_margin - cur_margin > 0.05:
        out.append(("amber", "Big margin expansion",
                    f"Target margin {a.target_margin:.1%} is more than double today's {cur_margin:.1%}. "
                    "What in your story delivers this: scale, pricing power, mix?"))

    if a.wacc.kd_method == "direct" and w["kd_pre"] < rf:
        out.append(("red", "Cost of debt below risk-free rate",
                    "A company cannot normally borrow below the government rate."))
    if not 0.3 <= w["beta"] <= 2.5:
        out.append(("amber", "Unusual beta", f"Beta {w['beta']:.2f} is outside 0.3–2.5. Consider a bottom-up beta."))
    if w["wacc"] < rf:
        out.append(("red", "WACC below the risk-free rate", "Check the inputs: equity risk cannot be negative."))

    if peer_ev_ebitda and np.isfinite(res["implied_exit_multiple"]):
        ratio = res["implied_exit_multiple"] / peer_ev_ebitda
        if ratio > 1.5 or ratio < 0.5:
            out.append(("amber", "Implied exit multiple far from peers",
                        f"The terminal value implies {res['implied_exit_multiple']:.1f}× EBITDA in year {a.years}, "
                        f"against a peer median of {peer_ev_ebitda:.1f}× today."))
        else:
            out.append(("green", "Implied exit multiple",
                        f"{res['implied_exit_multiple']:.1f}× EBITDA vs peer median {peer_ev_ebitda:.1f}×."))

    if (t["FCFF"] < 0).all():
        out.append(("amber", "No positive free cash flow in the forecast",
                    "All value sits in the terminal value. Typical for young firms, but the story must explain "
                    "when cash turns positive."))

    up = res["upside"]
    if np.isfinite(up) and abs(up) > 0.5:
        out.append(("amber", "Value far from price",
                    f"Your value is {up:+.0%} versus the market price. Either you see something the market does "
                    "not, or an assumption is off. Run the reverse DCF to see what the price implies."))

    if a.failure_prob > 0:
        out.append(("green", "Failure risk included",
                    f"{a.failure_prob:.0%} chance of failure with {a.distress_recovery:.0%} recovery is built in."))
    return out


STORIES = {
    "Keep my numbers": None,
    "Steady compounder (moat)": {
        "blurb": "A market leader with a durable advantage: brand, network or switching costs. Growth is solid, "
                 "margins hold or edge up, and excess returns fade slowly.",
        "growth": 0.10, "margin_delta": 0.02, "converge": 5, "roic": 0.25, "failure": 0.0,
        "s2c_mult": 1.2},
    "High-growth disruptor": {
        "blurb": "A young firm taking share in a big market. Fast growth now, margins far from mature, heavy "
                 "reinvestment, and a real chance the story does not work out.",
        "growth": 0.25, "margin_target": 0.20, "converge": 8, "roic": 0.15, "failure": 0.10,
        "s2c_mult": 0.8},
    "Mature cash cow": {
        "blurb": "A dominant firm in a slow market. Growth near the economy, stable margins, low reinvestment, "
                 "and most profit paid out.",
        "growth": 0.05, "margin_delta": 0.0, "converge": 3, "roic": 0.18, "failure": 0.0, "s2c_mult": 1.5},
    "Cyclical at the peak": {
        "blurb": "Earnings are flattered by the cycle. Expect margins to revert to mid-cycle levels and growth to "
                 "slow before it settles.",
        "growth": 0.03, "margin_delta": -0.04, "converge": 3, "roic": 0.12, "failure": 0.0, "s2c_mult": 1.0},
    "Turnaround": {
        "blurb": "A troubled firm with a credible fix. Margins climb from depressed levels toward the industry, "
                 "but there is a meaningful chance the turnaround fails.",
        "growth": 0.06, "margin_target": 0.12, "converge": 6, "roic": 0.12, "failure": 0.15, "s2c_mult": 1.0},
}
