"""Independent checks of the valuation math. Run: python -m pytest -q tests"""
import copy
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lv_engine import Company, Assumptions, run_dcf, defaults_from_company
from lv_wacc import WaccInputs, compute_wacc, relever_beta, synthetic_rating
from lv_models import run_ddm, run_fcfe, run_residual_income, run_relative
from lv_risk import reverse_dcf, sensitivity_wacc_growth, monte_carlo, scenarios
from lv_data import SAMPLES, SAMPLE_PEERS, sample_company


def simple():
    c = Company(revenue=1000, ebit=100, da=20, cash=50, debt=200, minority=10, options=5,
                non_operating=15, shares=10, price=50, eff_tax=0.25, interest=10, book_equity=600)
    a = Assumptions(years=5, high_growth_years=5, growth_y1=0.10, growth_stage1=0.10,
                    terminal_growth=0.03, margin_y1=0.10, target_margin=0.10, margin_converge_year=1,
                    tax_converge=False, sales_to_capital=2.0, terminal_roic=0.15, da_pct=0.02)
    a.wacc = WaccInputs(wacc_override=0.10, marginal_tax=0.25)
    return c, a


def test_dcf_hand_calc():
    c, a = simple()
    r = run_dcf(c, a)
    rev, prev, pv = [], 1000.0, 0.0
    for t in range(1, 6):
        cur = prev * 1.10
        nopat = cur * 0.10 * 0.75
        reinv = (cur - prev) / 2.0
        pv += (nopat - reinv) / 1.10 ** t
        prev = cur
    nopat6 = prev * 1.03 * 0.10 * 0.75
    tv = nopat6 * (1 - 0.03 / 0.15) / (0.10 - 0.03)
    op = pv + tv / 1.10 ** 5
    equity = op + 50 + 15 - 200 - 10 - 5
    assert math.isclose(r["op_value"], op, rel_tol=1e-9)
    assert math.isclose(r["per_share"], equity / 10, rel_tol=1e-9)


def test_mid_year_raises_value():
    c, a = simple()
    v0 = run_dcf(c, a)["pv_fcff"]
    a.mid_year = True
    v1 = run_dcf(c, a)["pv_fcff"]
    assert math.isclose(v1, v0 * math.sqrt(1.10), rel_tol=1e-9)


def test_exit_multiple_and_implied_growth_roundtrip():
    c, a = simple()
    g = run_dcf(c, a)
    a2 = copy.deepcopy(a)
    a2.terminal_method = "exit_multiple"
    a2.exit_multiple = g["implied_exit_multiple"]
    e = run_dcf(c, a2)
    assert math.isclose(e["tv"], g["tv"], rel_tol=1e-9)
    assert abs(e["implied_terminal_growth"] - 0.03) < 1e-6


def test_wacc_build():
    w = WaccInputs(rf=0.06, erp=0.05, crp=0.02, crp_scaled_by_beta=True, beta_method="bottom_up",
                   unlevered_beta=0.8, target_de=0.25, kd_method="direct", kd_direct=0.09, marginal_tax=0.25)
    r = compute_wacc(w)
    beta = 0.8 * (1 + 0.75 * 0.25)
    ke = 0.06 + beta * 0.07
    wacc = 0.8 * ke + 0.2 * 0.09 * 0.75
    assert math.isclose(r["beta"], beta) and math.isclose(r["ke"], ke) and math.isclose(r["wacc"], wacc)
    assert synthetic_rating(100, 10)[1] == "AAA"
    assert synthetic_rating(100, 60)[1] == "B"


def test_ddm():
    r = run_ddm(10, 0.10, 2, 0.04, 0.12)
    d1, d2 = 11.0, 12.1
    v = d1 / 1.12 + d2 / 1.12 ** 2 + d2 * 1.04 / 0.08 / 1.12 ** 2
    assert math.isclose(r["per_share"], v, rel_tol=1e-12)


def test_residual_income_equals_bv_when_roe_equals_ke():
    r = run_residual_income(1000, 0.12, 0.12, 0.4, 0.12, 5, 0.03, 10)
    assert math.isclose(r["equity"], 1000, rel_tol=1e-9)


def test_reverse_dcf_recovers_price():
    c, a = simple()
    target = run_dcf(c, a)["per_share"]
    a2 = copy.deepcopy(a)
    g = reverse_dcf(c, a2, "growth_stage1", -0.05, 0.4, target=target)
    assert g is not None and abs(g - 0.10) < 1e-5


def test_samples_run_all_models():
    for key in SAMPLES:
        c = sample_company(key)
        a = defaults_from_company(c)
        r = run_dcf(c, a)
        assert np.isfinite(r["per_share"])
        f = run_fcfe(c, a)
        assert np.isfinite(f["per_share"])
        rel = run_relative(c, SAMPLE_PEERS[c.ticker])
        assert len(rel) == 4
        s = sensitivity_wacc_growth(c, a, [0.10, 0.12], [0.03, 0.04])
        assert s.shape == (2, 2)
        mc = monte_carlo(c, a, 200, (0.05, 0.1, 0.15), (0.1, 0.15, 0.2), 0.01, (0.02, 0.03, 0.04))
        assert len(mc["values"]) > 150


def test_failure_probability():
    c, a = simple()
    base = run_dcf(c, a)["op_value"]
    a.failure_prob, a.distress_recovery = 0.2, 0.5
    r = run_dcf(c, a)
    assert math.isclose(r["op_value_adj"], base * (0.8 + 0.2 * 0.5), rel_tol=1e-12)


def test_year_by_year_paths_override_fade():
    c, a = simple()
    a.growth_path = [0.20, 0.15, 0.10, 0.05, 0.03]
    a.margin_path = [0.08, 0.09, 0.10, 0.11, 0.12]
    r = run_dcf(c, a)
    prev, pv = 1000.0, 0.0
    for t, (g, m) in enumerate(zip(a.growth_path, a.margin_path), 1):
        cur = prev * (1 + g)
        pv += (cur * m * 0.75 - (cur - prev) / 2.0) / 1.10 ** t
        prev = cur
    assert math.isclose(r["pv_fcff"], pv, rel_tol=1e-9)
    assert list(r["table"]["Revenue growth"]) == a.growth_path


def test_set_driver_shifts_paths():
    from lv_engine import set_driver
    c, a = simple()
    a.growth_path = [0.20, 0.15, 0.10, 0.05, 0.03]
    set_driver(a, "growth_stage1", a.growth_stage1 + 0.02)
    assert np.allclose(a.growth_path, [0.22, 0.17, 0.12, 0.07, 0.05])
    g = reverse_dcf(c, a, "growth_stage1", -0.2, 0.5, target=run_dcf(c, a)["per_share"])
    assert g is not None and abs(g - a.growth_stage1) < 1e-5


def test_loss_making_defaults_are_sensible():
    c = Company(revenue=50000, ebit=-60, da=1600, capex=1750, nwc=1400, cash=8500, debt=4600, shares=920,
                price=325, eff_tax=0.25, interest=180, net_income=400, book_equity=38000, beta=0.3, currency="₹",
                history={"year": [2023, 2026], "revenue": [7000, 50000], "ebit": [-1400, -60]})
    a = defaults_from_company(c)
    assert a.target_margin == 0.12 and a.wacc.beta_method == "bottom_up" and a.wacc.kd_method == "direct"
    assert len(a.notes) == 4
    assert run_dcf(c, a)["per_share"] > 0


def test_investments_excluded_from_invested_capital_and_s2c_floor():
    c = Company(revenue=5000, ebit=60, cash=900, debt=350, non_operating=5000, book_equity=6900, shares=46,
                price=1070, history={"year": [2022, 2025], "revenue": [1425, 5000], "ebit": [-900, 60]})
    assert c.ic0() == 6900 + 350 - 900 - 5000
    c2 = Company(**{**c.__dict__, "non_operating": 0.0})
    a2 = defaults_from_company(c2)
    assert a2.sales_to_capital == 2.5 and any("2.5 placeholder" in n for n in a2.notes)


def test_rulebook_defaults():
    from lv_rules import rule_defaults, classify
    assert classify("Financial Services", "Insurance Brokers") == "Financial platforms"
    assert classify("Consumer Cyclical", "Internet Retail") == "Internet platforms"
    assert classify("Financial Services", "Banks - Regional") == "Banks & insurers"
    c = Company(name="PB", sector="Financial Services", industry="Insurance Brokers", revenue=4977, ebit=60,
                cash=900, debt=350, book_equity=6900, shares=46, price=1070, eff_tax=0.0, interest=30, beta=0.9,
                history={"year": [2022, 2025], "revenue": [1425, 4977], "ebit": [-900, 60]})
    a, log, grp = rule_defaults(c)
    assert grp == "Financial platforms" and a.high_growth_years == 10 and a.years == 12
    assert abs(a.target_margin - 0.22) < 1e-9 and a.wacc.beta_method == "bottom_up"
    assert a.terminal_growth <= a.wacc.rf and a.terminal_roic >= 0.10
    assert np.isfinite(run_dcf(c, a)["per_share"]) and len(log) >= 8
    a2, _, g2 = rule_defaults(c, group="Utilities")
    assert g2 == "Utilities" and abs(a2.target_margin - 0.20) < 1e-9
