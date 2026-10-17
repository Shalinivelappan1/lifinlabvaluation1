"""LiFin Lab · Equity Valuation Studio (Streamlit).

Run:  streamlit run app.py
"""
from __future__ import annotations

import copy
from dataclasses import asdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Works with either layout: a 'valuation' folder, or flat lv_*.py files.
try:
    from valuation.checks import STORIES, health_checks
    from valuation.data import (EMPTY_PEERS, FIELDS, SAMPLE_PEERS, SAMPLES, company_from_upload,
                                fetch_ticker, sample_company, template_csv)
    from valuation.engine import Assumptions, Company, base_paths, defaults_from_company, run_dcf
    from valuation.explain import TIPS
    from valuation.export import from_json, to_excel, to_json
    from valuation.models import run_ddm, run_fcfe, run_relative, run_residual_income
    from valuation.risk import (monte_carlo, reverse_dcf, scenarios, sensitivity_two,
                                sensitivity_wacc_growth, tornado)
    from valuation.wacc import RATING_TABLE, WaccInputs
except ImportError:
    from lv_checks import STORIES, health_checks
    from lv_data import (EMPTY_PEERS, FIELDS, SAMPLE_PEERS, SAMPLES, company_from_upload,
                                fetch_ticker, sample_company, template_csv)
    from lv_engine import Assumptions, Company, base_paths, defaults_from_company, run_dcf
    from lv_explain import TIPS
    from lv_export import from_json, to_excel, to_json
    from lv_models import run_ddm, run_fcfe, run_relative, run_residual_income
    from lv_risk import (monte_carlo, reverse_dcf, scenarios, sensitivity_two,
                                sensitivity_wacc_growth, tornado)
    from lv_wacc import RATING_TABLE, WaccInputs

from contextlib import contextmanager


@contextmanager
def guard(name):
    """Show an error inside one tab instead of stopping the whole app."""
    try:
        yield
    except Exception as e:  # noqa: BLE001
        st.error(f"Something went wrong in {name}: {type(e).__name__}: {e}. The other tabs still work. "
                 "Check the inputs on the Company tab (shares, price, revenue).")


st.set_page_config(page_title="Equity Valuation Studio · LiFin Lab", page_icon="📈", layout="wide")

NAVY, GOLD, TEAL, RED, GREY = "#15233F", "#C9A227", "#2AA79B", "#B03A48", "#6B7280"
st.markdown(f"""
<style>
  .block-container {{padding-top: 1.4rem;}}
  h1, h2, h3 {{color: {NAVY};}}
  .kpi {{background:#fff;border:1px solid #E5DFD0;border-left:5px solid {GOLD};border-radius:8px;
         padding:.6rem .9rem;}}
  .kpi .l {{font-size:.78rem;color:{GREY};text-transform:uppercase;letter-spacing:.06em}}
  .kpi .v {{font-size:1.5rem;font-weight:700;color:{NAVY}}}
  .chk {{border-radius:8px;padding:.6rem .9rem;margin-bottom:.5rem;border:1px solid #E5DFD0;background:#fff}}
  .chk.red {{border-left:6px solid {RED}}} .chk.amber {{border-left:6px solid #E0A100}}
  .chk.green {{border-left:6px solid {TEAL}}}
  .credit {{font-size:.8rem;color:{GREY};}}
</style>""", unsafe_allow_html=True)

PCT_A = ["growth_y1", "growth_stage1", "terminal_growth", "margin_y1", "target_margin", "capex_pct",
         "da_pct", "nwc_pct", "terminal_roic", "failure_prob", "distress_recovery"]
PCT_W = ["rf", "erp", "crp", "size_premium", "alpha", "kd_direct", "marginal_tax"]
STORY_KEYS = ["n_story", "n_market", "n_edge", "n_risks", "n_catalyst"]


# ---------------------------------------------------------------- state helpers
RESET_PREFIXES = ("ddm_", "ri_", "mc_", "fcfe_", "scn_", "wt_", "peer_editor", "peers_edited", "yby_")


def put_company(c: Company):
    for k in list(st.session_state.keys()):
        if k.startswith(RESET_PREFIXES):
            del st.session_state[k]
    for f, _, kind in FIELDS:
        v = getattr(c, f)
        if kind == "num":
            v = float(v or 0.0)
        st.session_state[f"c_{f}"] = v * 100 if f == "eff_tax" else v
    st.session_state["c_history"] = c.history or {}
    peers = SAMPLE_PEERS.get(c.ticker)
    st.session_state["peers_df"] = (peers if peers is not None else EMPTY_PEERS).copy()
    put_assumptions(defaults_from_company(c))
    st.session_state.pop("mc_res", None)


def put_assumptions(a: Assumptions):
    for f in ["years", "high_growth_years", "margin_converge_year", "tax_converge", "reinvest_method",
              "sales_to_capital", "terminal_method", "exit_multiple", "mid_year"]:
        v = getattr(a, f)
        st.session_state[f"a_{f}"] = float(v) if f in ("sales_to_capital", "exit_multiple") else v
    for f in PCT_A:
        st.session_state[f"a_{f}"] = round(getattr(a, f) * 100, 2)
    st.session_state["a_twacc_on"] = a.terminal_wacc is not None
    st.session_state["a_twacc"] = round((a.terminal_wacc or 0.10) * 100, 2)
    w = a.wacc
    for f in PCT_W:
        st.session_state[f"w_{f}"] = round(getattr(w, f) * 100, 2)
    for f in ["crp_scaled_by_beta", "beta_method", "beta_given", "unlevered_beta", "kd_method", "add_crp_to_kd"]:
        v = getattr(w, f)
        st.session_state[f"w_{f}"] = float(v) if f in ("beta_given", "unlevered_beta") else v
    st.session_state["w_target_de"] = round(w.target_de * 100, 1)
    st.session_state["w_ovr_on"] = w.wacc_override is not None
    for k in [k for k in st.session_state.keys() if str(k).startswith("yby_")]:
        del st.session_state[k]
    if a.growth_path is not None or a.margin_path is not None:
        g0, m0 = base_paths(a)
        gp = a.growth_path if a.growth_path is not None else list(g0)
        mp = a.margin_path if a.margin_path is not None else list(m0)
        st.session_state["yby_on"] = True
        st.session_state["yby_base"] = yby_frame(gp, mp)
    st.session_state["w_ovr"] = round((w.wacc_override or 0.11) * 100, 2)


def get_company() -> Company:
    kw = {}
    for f, _, kind in FIELDS:
        v = st.session_state.get(f"c_{f}")
        kw[f] = (v / 100 if f == "eff_tax" else float(v)) if kind == "num" else str(v)
    kw["history"] = st.session_state.get("c_history", {})
    return Company(**kw)


def get_assumptions() -> Assumptions:
    s = st.session_state
    w = WaccInputs(**{f: s[f"w_{f}"] / 100 for f in PCT_W},
                   crp_scaled_by_beta=s["w_crp_scaled_by_beta"], beta_method=s["w_beta_method"],
                   beta_given=float(s["w_beta_given"]), unlevered_beta=float(s["w_unlevered_beta"]),
                   target_de=s["w_target_de"] / 100, kd_method=s["w_kd_method"],
                   add_crp_to_kd=s["w_add_crp_to_kd"],
                   wacc_override=(s["w_ovr"] / 100 if s["w_ovr_on"] else None))
    a = Assumptions(
        years=int(s["a_years"]), high_growth_years=int(s["a_high_growth_years"]),
        margin_converge_year=int(s["a_margin_converge_year"]), tax_converge=s["a_tax_converge"],
        reinvest_method=s["a_reinvest_method"], sales_to_capital=float(s["a_sales_to_capital"]),
        terminal_method=s["a_terminal_method"], exit_multiple=float(s["a_exit_multiple"]),
        mid_year=s["a_mid_year"], terminal_wacc=(s["a_twacc"] / 100 if s["a_twacc_on"] else None),
        wacc=w, **{f: s[f"a_{f}"] / 100 for f in PCT_A})
    if s.get("yby_on") and isinstance(s.get("yby_base"), pd.DataFrame) and len(s["yby_base"]) == a.years:
        df = s["yby_base"].copy()
        for ri, chg in (s.get("yby_editor") or {}).get("edited_rows", {}).items():
            for col, v in chg.items():
                if v is not None and col in df.columns:
                    df.iloc[int(ri), df.columns.get_loc(col)] = float(v)
        a.growth_path = [float(x) / 100 for x in df["Revenue growth %"]]
        a.margin_path = [float(x) / 100 for x in df["EBIT margin %"]]
    return a


def yby_frame(g, m) -> pd.DataFrame:
    return pd.DataFrame({"Revenue growth %": [round(x * 100, 2) for x in g],
                         "EBIT margin %": [round(x * 100, 2) for x in m]},
                        index=[f"Y{i}" for i in range(1, len(g) + 1)])


def init_state():
    if "c_name" not in st.session_state:
        put_company(sample_company(list(SAMPLES)[0]))
        st.session_state["mode"] = "Learning"
        for k in STORY_KEYS:
            st.session_state.setdefault(k, "")
        st.session_state["msg"] = None


init_state()


def tip(key):
    if st.session_state.get("mode") == "Learning" and key in TIPS:
        st.info(TIPS[key], icon="💡")


def num(label, key, step=1.0, fmt="%.2f", help=None, min_value=None, max_value=None, disabled=False):
    return st.number_input(label, key=key, step=step, format=fmt, help=help,
                           min_value=min_value, max_value=max_value, disabled=disabled)


def money(x, c: Company, dp=0):
    if x is None or not np.isfinite(x):
        return "–"
    return f"{c.currency}{indian(x, dp) if c.currency == '₹' else f'{x:,.{dp}f}'}"


def indian(x, dp=0):
    neg = x < 0
    x = abs(x)
    s = f"{x:.{dp}f}"
    whole, _, frac = s.partition(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        whole = ",".join(parts + [tail])
    return ("-" if neg else "") + whole + ("." + frac if frac else "")


def kpi(col, label, value):
    col.markdown(f"<div class='kpi'><div class='l'>{label}</div><div class='v'>{value}</div></div>",
                 unsafe_allow_html=True)


def pct(x, dp=1):
    return "–" if x is None or not np.isfinite(x) else f"{x * 100:.{dp}f}%"


# ---------------------------------------------------------------- callbacks
def cb_load_sample():
    put_company(sample_company(st.session_state["sample_key"]))
    st.session_state["msg"] = ("ok", "Sample loaded. These are fictional teaching companies.")


def cb_fetch():
    t = st.session_state.get("ticker_in", "").strip()
    if not t:
        st.session_state["msg"] = ("err", "Type a ticker first, e.g. INFY.NS or AAPL.")
        return
    try:
        c, warns = fetch_ticker(t)
        if not c.revenue:
            raise ValueError("no financial statements returned")
        put_company(c)
        st.session_state["msg"] = ("ok", f"Loaded {c.name}. Check every number: "
                                   + (" ".join(warns) if warns else "all main fields found."))
    except Exception as e:
        st.session_state["msg"] = ("err", f"Could not fetch {t}: {e}. You may be offline; load a sample or "
                                          "upload a file instead.")


def cb_upload():
    f = st.session_state.get("upload_file")
    if f is None:
        st.session_state["msg"] = ("err", "Choose a file first.")
        return
    try:
        put_company(company_from_upload(f))
        st.session_state["msg"] = ("ok", "File loaded. Review the numbers on the Company tab.")
    except Exception as e:
        st.session_state["msg"] = ("err", f"Could not read the file: {e}")


def cb_load_json():
    f = st.session_state.get("json_file")
    if f is None:
        return
    try:
        c, a, narr, extras = from_json(f.read())
        put_company(c)
        put_assumptions(a)
        for k, v in narr.items():
            st.session_state[k] = v
        if "peers" in extras:
            st.session_state["peers_df"] = pd.DataFrame(extras["peers"])
        st.session_state["msg"] = ("ok", "Saved valuation restored.")
    except Exception as e:
        st.session_state["msg"] = ("err", f"Could not read the saved file: {e}")


def cb_apply_story():
    key = st.session_state["story_key"]
    p = STORIES.get(key)
    if not p:
        return
    c = get_company()
    cur_m = c.ebit / c.revenue if c.revenue else 0.1
    s = st.session_state
    s["a_growth_y1"] = s["a_growth_stage1"] = p["growth"] * 100
    target = p["margin_target"] if "margin_target" in p else cur_m + p.get("margin_delta", 0.0)
    s["a_target_margin"] = round(target * 100, 2)
    s["a_margin_y1"] = round(cur_m * 100, 2)
    s["a_margin_converge_year"] = p["converge"]
    s["a_terminal_roic"] = p["roic"] * 100
    s["a_failure_prob"] = p["failure"] * 100
    base_s2c = defaults_from_company(c).sales_to_capital
    s["a_sales_to_capital"] = round(base_s2c * p["s2c_mult"], 2)
    s["a_reinvest_method"] = "sales_to_capital"
    for k in [k for k in s.keys() if str(k).startswith("yby_")]:
        del s[k]
    s["msg"] = ("ok", f"Story applied: {key}. The inputs on the Forecast tab now reflect it; adjust freely.")


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown(f"<h2 style='margin-bottom:0'>📈 Valuation Studio</h2>"
                f"<div class='credit'>LiFin Lab · end-to-end equity valuation</div>", unsafe_allow_html=True)
    st.radio("Mode", ["Learning", "Expert"], key="mode", horizontal=True,
             help="Learning shows an explanation beside every input. Expert hides them.")
    st.divider()
    st.subheader("1 · Get company data")
    src = st.radio("Source", ["Sample company", "Live ticker", "Upload file"], key="src")
    if src == "Sample company":
        st.selectbox("Sample", list(SAMPLES), key="sample_key")
        st.button("Load sample", on_click=cb_load_sample, width="stretch")
        st.caption("Fictional companies with realistic numbers. Work offline.")
    elif src == "Live ticker":
        st.text_input("Ticker (Yahoo Finance)", key="ticker_in", placeholder="INFY.NS, HINDUNILVR.NS, AAPL")
        st.button("Fetch financials", on_click=cb_fetch, width="stretch")
        st.caption("Needs internet. Indian tickers end in .NS (NSE) or .BO (BSE) and load in ₹ crore.")
    else:
        st.download_button("Download blank template (CSV)", template_csv(), "company_template.csv",
                           width="stretch")
        st.file_uploader("Upload filled template (CSV or Excel)", type=["csv", "xlsx"], key="upload_file")
        st.button("Load file", on_click=cb_upload, width="stretch")
    if st.session_state.get("msg"):
        kind, text = st.session_state["msg"]
        (st.success if kind == "ok" else st.error)(text)
    st.divider()
    st.subheader("Save / restore")
    st.file_uploader("Restore a saved valuation (.json)", type=["json"], key="json_file",
                     on_change=cb_load_json)
    st.divider()
    st.markdown("<div class='credit'>Curated by <b>Prof. Shalini Velappan</b>, IIM Tiruchirappalli · "
                "LiFin Lab.<br>For teaching. Not investment advice.</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------- compute
ss = st.session_state
ss["a_years"] = int(min(15, max(5, ss["a_years"])))
ss["a_high_growth_years"] = int(min(ss["a_years"], max(1, ss["a_high_growth_years"])))
ss["a_margin_converge_year"] = int(min(ss["a_years"], max(1, ss["a_margin_converge_year"])))
C = get_company()
A = get_assumptions()
missing = [lab for ok, lab in ((C.shares > 0, "shares outstanding"), (C.price > 0, "share price")) if not ok]
VALID = not missing
pershare_msg = ("Enter " + " and ".join(missing) + " on the ① Company tab to see value per share and run the "
                "other tabs. The forecast still works.") if missing else None
if C.revenue <= 0:
    R, dcf_error = None, "Enter revenue (and EBIT) on the ① Company tab to build the forecast."
else:
    try:
        R = run_dcf(C, A)
        dcf_error = None
    except Exception as e:
        R, dcf_error = None, str(e)

st.markdown(f"## {C.name} <span style='font-size:1rem;color:{GREY}'>{C.ticker} · {C.sector}</span>",
            unsafe_allow_html=True)
k1, k2, k3, k4, k5 = st.columns(5)
kpi(k1, "Market price", money(C.price, C, 2))
if R:
    kpi(k2, "DCF value per share", money(R["per_share"], C, 2) if VALID else "–")
    up = R["upside"]
    kpi(k3, "Upside vs price", f"<span style='color:{TEAL if up >= 0 else RED}'>{up:+.1%}</span>"
        if VALID and np.isfinite(up) else "–")
    kpi(k4, "WACC", pct(R["wacc_used"]))
    kpi(k5, "Terminal value share", pct(R["tv_share"], 0))
    if pershare_msg:
        st.warning(pershare_msg, icon="⚠️")
else:
    k2.error("Valuation paused")
    st.warning(dcf_error, icon="⚠️")

tabs = st.tabs(["① Company", "② Story & assumptions", "③ Cost of capital", "④ Forecast & DCF", "⑤ Other models",
                "⑥ Risk & judgement", "⑦ Health check", "⑧ Summary & export"])
U = f"{C.currency} {C.unit}"

# ---------------------------------------------------------------- 1 company
with tabs[0], guard("Company"):
    tip("company")
    a1, a2, a3 = st.columns(3)
    a1.text_input("Company name", key="c_name")
    a2.text_input("Ticker", key="c_ticker")
    a3.text_input("Sector", key="c_sector")
    b1, b2 = st.columns(2)
    b1.text_input("Currency symbol", key="c_currency")
    b2.selectbox("Money unit", ["crore", "million"], key="c_unit",
                 help="Shares must be in the same unit: crore shares with ₹ crore, million shares with $ million.")
    st.markdown(f"**Financials** (money in {U}; shares in {C.unit})")
    cols = st.columns(4)
    numeric = [f for f in FIELDS if f[2] == "num"]
    for i, (f, label, _) in enumerate(numeric):
        with cols[i % 4]:
            if f == "eff_tax":
                num("Effective tax rate (%)", f"c_{f}", 0.5, "%.1f")
            elif f in ("price", "beta"):
                num(label, f"c_{f}", 0.01, "%.2f")
            else:
                num(label, f"c_{f}", 1.0, "%.1f")
    st.markdown("**What the numbers say today**")
    mcap = C.price * C.shares
    ev = mcap + C.debt + C.minority - C.cash - C.non_operating
    d = st.columns(6)
    kpi(d[0], "Market cap", money(mcap, C))
    kpi(d[1], "Enterprise value", money(ev, C))
    kpi(d[2], "EBIT margin", pct(C.ebit / C.revenue if C.revenue else np.nan))
    ic = C.ic0()
    kpi(d[3], "ROIC (after tax)", pct(C.ebit * (1 - C.eff_tax) / ic if ic else np.nan))
    kpi(d[4], "Sales / capital", f"{C.revenue / ic:.2f}" if ic and C.revenue else "–")
    kpi(d[5], "P/E", f"{C.price / (C.net_income / C.shares):.1f}×"
        if C.net_income > 0 and C.shares > 0 and C.price > 0 else "–")
    h = C.history or {}
    if h.get("revenue"):
        fig = go.Figure()
        fig.add_bar(x=[str(y) for y in h["year"]], y=h["revenue"], name="Revenue", marker_color=NAVY)
        if h.get("ebit") and len(h["ebit"]) == len(h["revenue"]):
            mg = [e / r * 100 if r else None for e, r in zip(h["ebit"], h["revenue"])]
            fig.add_scatter(x=[str(y) for y in h["year"]], y=mg, name="EBIT margin %", yaxis="y2",
                            mode="lines+markers", line=dict(color=GOLD, width=3))
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), title="History",
                          yaxis=dict(title=U), yaxis2=dict(overlaying="y", side="right", title="%"),
                          plot_bgcolor="white", legend=dict(orientation="h"))
        st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- 2 story
with tabs[1], guard("Story"):
    tip("story")
    st.markdown("#### Pick a starting story, then make it yours")
    s1, s2 = st.columns([2, 1])
    s1.selectbox("Story archetype", list(STORIES), key="story_key")
    s2.write("")
    s2.write("")
    s2.button("Apply story to the numbers", on_click=cb_apply_story, width="stretch")
    p = STORIES.get(st.session_state["story_key"])
    if p:
        st.caption(p["blurb"])
    st.markdown("#### Your narrative")
    n1, n2 = st.columns(2)
    n1.text_area("The story in three sentences", key="n_story", height=110,
                 placeholder="What does the company do, why will customers keep choosing it, how big can it get?")
    n2.text_area("Size of the opportunity", key="n_market", height=110,
                 placeholder="Total market, its growth, and the share you think the company can win.")
    n3, n4, n5 = st.columns(3)
    n3.text_area("Competitive edge", key="n_edge", height=110, placeholder="Brand, cost, network, regulation…")
    n4.text_area("What could go wrong", key="n_risks", height=110, placeholder="Competition, regulation, execution…")
    n5.text_area("What would change your mind", key="n_catalyst", height=110,
                 placeholder="The data point that would make you revise the story.")
    st.markdown("#### Turn the story into numbers")
    st.caption("These are the value drivers: every number should trace back to a sentence in your story. "
               "To set growth or margin for each year separately, use the year-by-year table on ④ Forecast & DCF.")
    if st.session_state.get("yby_on"):
        st.info("Year-by-year growth and margins are switched on (④ Forecast & DCF). They override the growth and "
                "margin inputs below until you switch them off.", icon="📅")
    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown("##### Growth")
        tip("growth")
        st.slider("Forecast years", 5, 15, key="a_years")
        st.slider("High-growth years", 1, int(st.session_state["a_years"]), key="a_high_growth_years")
        num("Year-1 revenue growth (%)", "a_growth_y1", 0.5)
        num("Stage-1 growth, years 2 to H (%)", "a_growth_stage1", 0.5)
    with g2:
        st.markdown("##### Profitability")
        tip("margin")
        num("Year-1 EBIT margin (%)", "a_margin_y1", 0.5)
        num("Target EBIT margin (%)", "a_target_margin", 0.5)
        st.slider("Year target margin is reached", 1, int(st.session_state["a_years"]), key="a_margin_converge_year")
        st.checkbox("Tax rate moves from effective to marginal after high growth", key="a_tax_converge")
    with g3:
        st.markdown("##### Reinvestment")
        tip("s2c")
        st.radio("Method", ["sales_to_capital", "explicit"], key="a_reinvest_method", horizontal=True,
                 format_func={"sales_to_capital": "Sales-to-capital", "explicit": "Capex, D&A, NWC"}.get)
        num("Sales-to-capital ratio", "a_sales_to_capital", 0.05)
        e1, e2, e3 = st.columns(3)
        with e1:
            num("Capex % rev", "a_capex_pct", 0.1)
        with e2:
            num("D&A % rev", "a_da_pct", 0.1)
        with e3:
            num("NWC % rev", "a_nwc_pct", 0.5)
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("##### Terminal value")
        tip("terminal")
        st.radio("Method", ["gordon", "exit_multiple"], key="a_terminal_method", horizontal=True,
                 format_func={"gordon": "Perpetual growth", "exit_multiple": "Exit EV/EBITDA multiple"}.get)
        tt1, tt2, tt3 = st.columns(3)
        with tt1:
            num("Terminal growth (%)", "a_terminal_growth", 0.1)
        with tt2:
            num("Terminal ROIC (%)", "a_terminal_roic", 0.5)
        with tt3:
            num("Exit EV/EBITDA (×)", "a_exit_multiple", 0.5)
        st.checkbox("Use a separate mature WACC in the terminal years", key="a_twacc_on")
        num("Terminal WACC (%)", "a_twacc", 0.05, disabled=not st.session_state["a_twacc_on"])
    with t2:
        st.markdown("##### Risk of failure and timing")
        tip("mid_year")
        st.checkbox("Mid-year discounting", key="a_mid_year")
        tip("failure")
        f1, f2 = st.columns(2)
        with f1:
            num("Probability of failure (%)", "a_failure_prob", 1.0, "%.1f")
        with f2:
            num("Recovery if it fails (% of value)", "a_distress_recovery", 5.0, "%.1f")

    with st.expander("Story check: which number answers which question?"):
        if R:
            mapping = pd.DataFrame([
                ["How fast can it grow?", "Stage-1 revenue growth", pct(A.growth_stage1)],
                ["How long does the edge last?", "High-growth years", f"{A.high_growth_years} years"],
                ["How profitable when mature?", "Target EBIT margin", pct(A.target_margin)],
                ["How quickly does it get there?", "Years to target margin", f"{A.margin_converge_year}"],
                ["How much must it invest to grow?", "Sales-to-capital", f"{A.sales_to_capital:.2f}"],
                ["How risky is it?", "WACC", pct(R["wacc_used"])],
                ["Does the moat last for ever?", "Terminal ROIC", pct(A.terminal_roic)],
                ["Could it fail?", "Probability of failure", pct(A.failure_prob, 0)],
            ], columns=["Story question", "Model input", "Your number"])
            st.dataframe(mapping, hide_index=True, width="stretch")

# ---------------------------------------------------------------- 3 cost of capital
with tabs[2], guard("Cost of capital"):
    l, r = st.columns([1, 1])
    with l:
        st.markdown("#### Cost of equity (CAPM)")
        tip("rf")
        num("Risk-free rate (%)", "w_rf", 0.05)
        tip("erp")
        num("Mature-market equity risk premium (%)", "w_erp", 0.05)
        tip("crp")
        num("Country risk premium (%)", "w_crp", 0.05)
        st.checkbox("Scale country risk by beta", key="w_crp_scaled_by_beta")
        tip("beta")
        st.radio("Beta", ["given", "blume", "bottom_up"], key="w_beta_method", horizontal=True,
                 format_func={"given": "Regression beta", "blume": "Blume-adjusted",
                              "bottom_up": "Bottom-up (relevered)"}.get)
        b1, b2 = st.columns(2)
        with b1:
            num("Regression beta", "w_beta_given", 0.01)
        with b2:
            num("Industry unlevered beta", "w_unlevered_beta", 0.01)
        num("Target debt / equity (%)", "w_target_de", 1.0, "%.1f")
        c1, c2 = st.columns(2)
        with c1:
            num("Size premium (%)", "w_size_premium", 0.1)
        with c2:
            num("Company-specific premium (%)", "w_alpha", 0.1,
                help="Judgement. Use sparingly and say why in your story.")
    with r:
        st.markdown("#### Cost of debt and WACC")
        tip("kd")
        st.radio("Cost of debt", ["synthetic", "direct"], key="w_kd_method", horizontal=True,
                 format_func={"synthetic": "Synthetic rating (from coverage)", "direct": "Enter directly"}.get)
        num("Pre-tax cost of debt if entered directly (%)", "w_kd_direct", 0.05)
        st.checkbox("Add country risk to cost of debt", key="w_add_crp_to_kd")
        num("Marginal tax rate (%)", "w_marginal_tax", 0.5)
        tip("wacc")
        st.checkbox("Override WACC", key="w_ovr_on")
        num("WACC override (%)", "w_ovr", 0.05, disabled=not st.session_state["w_ovr_on"])
        if R:
            w = R["wacc"]
            steps = pd.DataFrame([
                ["Beta used", f"{w['beta']:.2f}"], ["Cost of equity", pct(w["ke"], 2)],
                ["Interest coverage", f"{w['coverage']:.1f}×" if np.isfinite(w["coverage"]) else "no interest"],
                ["Synthetic rating / spread", f"{w['rating']} / {pct(w['spread'], 2)}"],
                ["Pre-tax cost of debt", pct(w["kd_pre"], 2)], ["After-tax cost of debt", pct(w["kd_post"], 2)],
                ["Weights E / D", f"{w['e_v']:.0%} / {w['d_v']:.0%}"], ["WACC", pct(R["wacc_used"], 2)],
            ], columns=["Step", "Value"])
            st.dataframe(steps, hide_index=True, width="stretch")
            wi = A.wacc
            crp_part = w["beta"] * wi.crp if wi.crp_scaled_by_beta else wi.crp
            fig = go.Figure(go.Waterfall(
                x=["Risk-free", "β × ERP", "Country risk", "Size + specific", "Cost of equity"],
                y=[wi.rf * 100, w["beta"] * wi.erp * 100, crp_part * 100, (wi.size_premium + wi.alpha) * 100, 0],
                measure=["absolute", "relative", "relative", "relative", "total"],
                connector=dict(line=dict(color=GREY)), increasing=dict(marker_color=GOLD),
                totals=dict(marker_color=NAVY), texttemplate="%{y:.2f}%"))
            fig.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10), title="Cost of equity build-up (%)",
                              plot_bgcolor="white", showlegend=False)
            st.plotly_chart(fig, width="stretch")
    with st.expander("Rating table used for the synthetic cost of debt"):
        st.dataframe(pd.DataFrame([(f"> {f:.2f}" if np.isfinite(f) else "below", r_, f"{s:.2%}")
                                   for f, r_, s in RATING_TABLE],
                                  columns=["Interest coverage", "Rating", "Default spread"]),
                     hide_index=True, width="stretch")
        st.caption("Illustrative spreads in the style of Damodaran's annual table. Update them each year.")

# ---------------------------------------------------------------- 4 forecast & dcf
with tabs[3], guard("Forecast & DCF"):
    tip("forecast")
    st.markdown("#### Year-by-year assumptions")
    yon = st.toggle("Set revenue growth and EBIT margin for each year myself", key="yby_on",
                    help="Off: growth fades and margins converge using the inputs on the Story & assumptions tab. "
                         "On: type any path you like, year by year.")
    if yon:
        if not isinstance(st.session_state.get("yby_base"), pd.DataFrame) or \
                len(st.session_state["yby_base"]) != A.years:
            g0, m0 = base_paths(A)
            st.session_state["yby_base"] = yby_frame(g0, m0)
            st.session_state.pop("yby_editor", None)
        st.data_editor(st.session_state["yby_base"], key="yby_editor", width="stretch",
                       column_config={"Revenue growth %": st.column_config.NumberColumn(format="%.2f", step=0.25),
                                      "EBIT margin %": st.column_config.NumberColumn(format="%.2f", step=0.25)})
        c_a, c_b = st.columns([1, 3])
        if c_a.button("Reset to the fade path"):
            for k in ("yby_base", "yby_editor"):
                st.session_state.pop(k, None)
            st.rerun()
        c_b.caption("Edit any cell, then press Enter. The forecast below updates. Sensitivity, scenarios and "
                    "Monte Carlo shift your whole path up or down.")
    else:
        st.caption("Growth and margins follow the fade inputs on ② Story & assumptions. Switch on to type each "
                   "year yourself.")

    if dcf_error:
        st.error(dcf_error)
    elif R:
        st.divider()
        m = st.columns(5)
        kpi(m[0], f"PV of forecast FCFF ({U})", money(R["pv_fcff"], C))
        kpi(m[1], f"PV of terminal value ({U})", money(R["pv_tv"], C))
        kpi(m[2], f"Equity value ({U})", money(R["equity"], C))
        kpi(m[3], "Value per share", money(R["per_share"], C, 2) if VALID else "–")
        if A.terminal_method == "gordon":
            kpi(m[4], "Implied exit EV/EBITDA", f"{R['implied_exit_multiple']:.1f}×")
        else:
            kpi(m[4], "Implied perpetual growth", pct(R["implied_terminal_growth"], 2))
        T = R["table"]
        show = T.T.copy()
        pct_rows = ["Revenue growth", "EBIT margin", "Tax rate", "WACC", "ROIC"]
        fmt = show.copy().astype(object)
        for idx in show.index:
            for col in show.columns:
                v = show.loc[idx, col]
                fmt.loc[idx, col] = pct(v) if idx in pct_rows else (f"{v:.4f}" if idx == "Discount factor"
                                                                   else f"{v:,.0f}")
        st.markdown(f"#### Forecast ({U})")
        st.dataframe(fmt, width="stretch", height=600)
        ch1, ch2 = st.columns(2)
        fig = go.Figure()
        fig.add_bar(x=T.index, y=T["NOPAT"], name="NOPAT", marker_color=NAVY)
        fig.add_bar(x=T.index, y=-T["Reinvestment"], name="− Reinvestment", marker_color=GOLD)
        fig.add_scatter(x=T.index, y=T["FCFF"], name="FCFF", mode="lines+markers", line=dict(color=TEAL, width=3))
        fig.update_layout(barmode="relative", height=340, title="Where free cash flow comes from",
                          plot_bgcolor="white", margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h"))
        ch1.plotly_chart(fig, width="stretch")
        adj_fail = R["op_value_adj"] - R["op_value"]
        wf = go.Figure(go.Waterfall(
            x=["PV forecast FCFF", "PV terminal value", "Failure adj.", "+ Cash", "+ Non-operating", "− Debt",
               "− Minority", "− Options", "Equity value"],
            y=[R["pv_fcff"], R["pv_tv"], adj_fail, C.cash, C.non_operating, -C.debt, -C.minority, -C.options, 0],
            measure=["relative"] * 8 + ["total"], increasing=dict(marker_color=TEAL),
            decreasing=dict(marker_color=RED), totals=dict(marker_color=NAVY),
            connector=dict(line=dict(color=GREY))))
        wf.update_layout(height=340, title=f"From operating value to equity ({U})", plot_bgcolor="white",
                         margin=dict(l=10, r=10, t=40, b=10))
        ch2.plotly_chart(wf, width="stretch")

# ---------------------------------------------------------------- 5 other models
if not VALID and R is not None:
    R, dcf_error = None, pershare_msg
F = D = RI = REL = None
with tabs[4], guard("Other models"):
    sub = st.tabs(["FCFE", "Dividend discount", "Residual income", "Relative (peers)"])
    with sub[0]:
        tip("fcfe")
        if R:
            ic0 = C.ic0()
            dr = st.slider("Debt as % of invested capital (kept constant)", 0, 80,
                           int(round(min(80, max(0, C.debt / ic0 * 100 if ic0 else 0)))), key="fcfe_dr")
            try:
                F = run_fcfe(C, A, dr / 100)
                q = st.columns(4)
                kpi(q[0], "Cost of equity", pct(F["ke"], 2))
                kpi(q[1], f"PV terminal ({U})", money(F["pv_tv"], C))
                kpi(q[2], "FCFE value per share", money(F["per_share"], C, 2))
                kpi(q[3], "vs FCFF value", f"{F['per_share'] / R['per_share'] - 1:+.1%}")
                st.dataframe(F["table"].T.style.format("{:,.0f}"), width="stretch")
            except ValueError as e:
                st.error(str(e))
    with sub[1]:
        tip("ddm")
        if R:
            dps0 = C.dividends / C.shares if C.shares else 0.0
            d1, d2, d3, d4, d5 = st.columns(5)
            dps = d1.number_input("Dividend per share now", value=float(round(dps0, 2)), key="ddm_dps")
            g1d = d2.number_input("Growth, stage 1 (%)", value=float(round(A.growth_stage1 * 100, 1)), key="ddm_g1")
            yd = d3.number_input("Stage-1 years", value=5, min_value=1, max_value=20, key="ddm_y")
            gtd = d4.number_input("Terminal growth (%)", value=float(round(A.terminal_growth * 100, 1)), key="ddm_gt")
            ked = d5.number_input("Cost of equity (%)", value=float(round(R["wacc"]["ke"] * 100, 2)), key="ddm_ke")
            if dps <= 0:
                st.warning("This company pays no dividend: the DDM cannot value it. Use FCFE or FCFF.")
            else:
                try:
                    D = run_ddm(dps, g1d / 100, int(yd), gtd / 100, ked / 100)
                    q = st.columns(3)
                    kpi(q[0], "PV of stage-1 dividends", money(D["pv_stage1"], C, 2))
                    kpi(q[1], "PV of terminal value", money(D["pv_tv"], C, 2))
                    kpi(q[2], "DDM value per share", money(D["per_share"], C, 2))
                    st.dataframe(D["table"].T.style.format("{:,.2f}"), width="stretch")
                except ValueError as e:
                    st.error(str(e))
    with sub[2]:
        tip("ri")
        if R:
            roe0 = C.net_income / C.book_equity if C.book_equity else 0.15
            payout0 = C.dividends / C.net_income if C.net_income > 0 else 0.3
            r1, r2, r3, r4, r5 = st.columns(5)
            roe1 = r1.number_input("ROE, year 1 (%)", value=float(round(roe0 * 100, 1)), key="ri_roe1")
            roet = r2.number_input("ROE, terminal (%)", value=float(round(roe0 * 100, 1)), key="ri_roet",
                                   help="Default keeps today's ROE, consistent with a lasting moat. Lower it toward "
                                        "the cost of equity if competition will erode returns.")
            pay = r3.number_input("Payout ratio (%)", value=float(round(min(payout0, 1.0) * 100, 1)), key="ri_pay")
            yri = r4.number_input("Years", value=int(A.years), min_value=1, max_value=20, key="ri_y")
            keri = r5.number_input("Cost of equity (%)", value=float(round(R["wacc"]["ke"] * 100, 2)), key="ri_ke")
            try:
                RI = run_residual_income(C.book_equity, roe1 / 100, roet / 100, pay / 100, keri / 100, int(yri),
                                         A.terminal_growth, C.shares)
                q = st.columns(3)
                kpi(q[0], f"Book value today ({U})", money(C.book_equity, C))
                kpi(q[1], f"PV of residual income ({U})", money(RI["equity"] - C.book_equity, C))
                kpi(q[2], "RI value per share", money(RI["per_share"], C, 2))
                st.dataframe(RI["table"].T.style.format("{:,.2f}"), width="stretch")
            except ValueError as e:
                st.error(str(e))
    with sub[3]:
        tip("relative")
        st.caption("Edit the peer table: add rows, paste from Excel, or clear cells you do not trust.")
        PEERS = st.data_editor(st.session_state["peers_df"], num_rows="dynamic",
                               width="stretch", key="peer_editor")
        st.session_state["peers_edited"] = PEERS
        stat = st.radio("Peer statistic", ["median", "mean", "harmonic mean"], horizontal=True, key="rel_stat")
        REL = run_relative(C, PEERS, stat) if R else pd.DataFrame()
        if not R:
            st.warning(dcf_error or "Complete the company data first.")
        elif REL.empty:
            st.warning("Enter at least one peer multiple.")
        else:
            st.dataframe(REL.style.format({"Peer value used": "{:.1f}", "Peer low": "{:.1f}", "Peer high": "{:.1f}",
                                           "Company today": "{:.1f}", "Implied price": "{:,.2f}",
                                           "Implied low": "{:,.2f}", "Implied high": "{:,.2f}"}),
                         hide_index=True, width="stretch")

# ---------------------------------------------------------------- 6 risk
SCN = MC = None
with tabs[5], guard("Risk & judgement"):
    sub = st.tabs(["Sensitivity", "Tornado", "Scenarios", "Monte Carlo", "Reverse DCF"])
    if R:
        w0 = R["wacc_used"]
        with sub[0]:
            tip("sensitivity")
            step_w = st.select_slider("WACC step", [0.0025, 0.005, 0.01], value=0.005, format_func=pct, key="sw")
            wv = [w0 + step_w * k for k in range(-2, 3)]
            gv = [A.terminal_growth + 0.005 * k for k in range(-2, 3)]
            S1 = sensitivity_wacc_growth(C, A, wv, gv)
            st.markdown("**Value per share: terminal growth (rows) × WACC (columns)**")
            st.dataframe(S1.style.format("{:,.2f}", na_rep="g ≥ WACC").background_gradient(cmap="RdYlGn", axis=None),
                         width="stretch")
            gs = [A.growth_stage1 + d for d in (-0.06, -0.03, 0, 0.03, 0.06)]
            ms = [A.target_margin + d for d in (-0.04, -0.02, 0, 0.02, 0.04)]
            S2 = sensitivity_two(C, A, "growth_stage1", gs, "target_margin", ms)
            S2.index.name = "Stage-1 growth \\ target margin"
            st.markdown("**Value per share: stage-1 growth (rows) × target margin (columns)**")
            st.dataframe(S2.style.format("{:,.2f}").background_gradient(cmap="RdYlGn", axis=None),
                         width="stretch")
        with sub[1]:
            shocks = {
                "Stage-1 growth ±3 pts": ("growth_stage1", A.growth_stage1 - 0.03, A.growth_stage1 + 0.03),
                "Target margin ±2 pts": ("target_margin", A.target_margin - 0.02, A.target_margin + 0.02),
                "WACC ±1 pt": ("wacc_shift", 0.01, -0.01),
                "Terminal growth ±0.5 pt": ("terminal_growth", A.terminal_growth - 0.005, A.terminal_growth + 0.005),
                "Sales-to-capital ±25%": ("sales_to_capital", A.sales_to_capital * 0.75, A.sales_to_capital * 1.25),
                "Terminal ROIC ±5 pts": ("terminal_roic", max(0.02, A.terminal_roic - 0.05), A.terminal_roic + 0.05),
            }
            TD, base = tornado(C, A, shocks)
            fig = go.Figure()
            fig.add_bar(y=TD["Driver"], x=TD["Value at low"] - base, base=base, orientation="h", name="Low case",
                        marker_color=RED)
            fig.add_bar(y=TD["Driver"], x=TD["Value at high"] - base, base=base, orientation="h", name="High case",
                        marker_color=TEAL)
            fig.add_vline(x=base, line_color=NAVY)
            fig.update_layout(barmode="overlay", height=380, title="Which assumption moves value most?",
                              plot_bgcolor="white", margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h"))
            st.plotly_chart(fig, width="stretch")
            st.caption("Each bar changes one input and holds the rest at your base case.")
        with sub[2]:
            tip("scenarios")
            if "scn_df" not in st.session_state or st.session_state.get("scn_for") != C.name:
                st.session_state["scn_df"] = pd.DataFrame({
                    "Scenario": ["Bear", "Base", "Bull"], "Probability %": [25.0, 50.0, 25.0],
                    "Stage-1 growth %": [round((A.growth_stage1 - 0.04) * 100, 1), round(A.growth_stage1 * 100, 1),
                                         round((A.growth_stage1 + 0.04) * 100, 1)],
                    "Target margin %": [round((A.target_margin - 0.03) * 100, 1), round(A.target_margin * 100, 1),
                                        round((A.target_margin + 0.02) * 100, 1)],
                    "WACC shift (pts)": [1.0, 0.0, -0.5],
                    "Terminal growth %": [round((A.terminal_growth - 0.01) * 100, 1),
                                          round(A.terminal_growth * 100, 1), round(A.terminal_growth * 100, 1)]})
                st.session_state["scn_for"] = C.name
            sdf = st.data_editor(st.session_state["scn_df"], num_rows="dynamic", width="stretch",
                                 key="scn_editor", hide_index=True)
            spec = {r["Scenario"]: {"prob": r["Probability %"] / 100, "growth": r["Stage-1 growth %"] / 100,
                                    "margin": r["Target margin %"] / 100, "wacc_shift": r["WACC shift (pts)"] / 100,
                                    "terminal_growth": r["Terminal growth %"] / 100}
                    for _, r in sdf.dropna().iterrows()}
            SCN = scenarios(C, A, spec)
            tot = SCN["Probability"].sum()
            wv_ = (SCN["Probability"] * SCN["Value per share"]).sum() / tot if tot else np.nan
            q = st.columns(3)
            kpi(q[0], "Probability-weighted value", money(wv_, C, 2))
            kpi(q[1], "Range", f"{money(SCN['Value per share'].min(), C)} – {money(SCN['Value per share'].max(), C)}")
            kpi(q[2], "Probabilities sum to", f"{tot:.0%}")
            if abs(tot - 1) > 1e-6:
                st.warning("Probabilities do not add to 100%; the weighted value is rescaled.")
            st.dataframe(SCN.style.format({"Probability": "{:.0%}", "Stage-1 growth": "{:.1%}",
                                           "Target margin": "{:.1%}", "WACC": "{:.2%}", "Terminal growth": "{:.1%}",
                                           "Value per share": "{:,.2f}"}), hide_index=True, width="stretch")
        with sub[3]:
            tip("mc")
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                st.markdown("**Stage-1 growth (%)**")
                gl = st.number_input("Low", value=round((A.growth_stage1 - 0.05) * 100, 1), key="mc_gl")
                gm = st.number_input("Most likely", value=round(A.growth_stage1 * 100, 1), key="mc_gm")
                gh = st.number_input("High", value=round((A.growth_stage1 + 0.05) * 100, 1), key="mc_gh")
            with mc2:
                st.markdown("**Target margin (%)**")
                ml = st.number_input("Low ", value=round((A.target_margin - 0.04) * 100, 1), key="mc_ml")
                mm = st.number_input("Most likely ", value=round(A.target_margin * 100, 1), key="mc_mm")
                mh = st.number_input("High ", value=round((A.target_margin + 0.03) * 100, 1), key="mc_mh")
            with mc3:
                st.markdown("**Terminal growth (%) and WACC**")
                tl = st.number_input("Low  ", value=round((A.terminal_growth - 0.01) * 100, 1), key="mc_tl")
                tm = st.number_input("Most likely  ", value=round(A.terminal_growth * 100, 1), key="mc_tm")
                th = st.number_input("High  ", value=round((A.terminal_growth + 0.005) * 100, 1), key="mc_th")
                wsd = st.number_input("WACC standard deviation (pts)", value=0.75, key="mc_wsd")
            o1, o2, o3 = st.columns(3)
            nsim = o1.select_slider("Simulations", [500, 1000, 2000, 5000], value=1000, key="mc_n")
            rho = o2.slider("Correlation: growth and margin", -0.9, 0.9, 0.3, 0.1, key="mc_rho",
                            help="Positive: fast growers also get better margins (scale). Negative: growth is bought "
                                 "with price cuts.")
            seed = o3.number_input("Random seed", value=42, step=1, key="mc_seed")
            if st.button("Run simulation", type="primary"):
                with st.spinner("Simulating…"):
                    st.session_state["mc_res"] = monte_carlo(
                        C, A, int(nsim), tuple(sorted([gl / 100, gm / 100, gh / 100])),
                        tuple(sorted([ml / 100, mm / 100, mh / 100])), wsd / 100,
                        tuple(sorted([tl / 100, tm / 100, th / 100])), rho, int(seed))
            MC = st.session_state.get("mc_res")
            if MC:
                p_ = MC["percentiles"]
                q = st.columns(4)
                kpi(q[0], "Median value", money(p_["P50"], C, 2))
                kpi(q[1], "80% range (P10–P90)", f"{money(p_['P10'], C)} – {money(p_['P90'], C)}")
                kpi(q[2], "Mean", money(MC["mean"], C, 2))
                kpi(q[3], "Chance value > price", pct(MC["prob_above_price"], 0))
                fig = go.Figure(go.Histogram(x=MC["values"], nbinsx=60, marker_color=NAVY, opacity=0.85))
                fig.add_vline(x=C.price, line_color=RED, line_width=3, annotation_text="Price")
                fig.add_vline(x=p_["P50"], line_color=GOLD, line_width=3, annotation_text="Median")
                fig.update_layout(height=360, title="Distribution of value per share", plot_bgcolor="white",
                                  margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, width="stretch")
                st.caption("Inputs that are not simulated stay at your base case. Draws where terminal growth would "
                           "exceed WACC are capped 1 point below WACC.")
        with sub[4]:
            tip("reverse")
            choice = st.selectbox("Solve for", ["Stage-1 revenue growth", "Target EBIT margin", "Terminal growth"],
                                  key="rev_choice")
            fmap = {"Stage-1 revenue growth": ("growth_stage1", -0.10, 0.60),
                    "Target EBIT margin": ("target_margin", -0.10, 0.80),
                    "Terminal growth": ("terminal_growth", -0.03, R["wacc_terminal"] - 0.001)}
            fld, lo_, hi_ = fmap[choice]
            sol = reverse_dcf(C, A, fld, lo_, hi_)
            yours = getattr(A, fld)
            q = st.columns(3)
            kpi(q[0], "Market price", money(C.price, C, 2))
            kpi(q[1], f"{choice} the price implies", pct(sol) if sol is not None else "no solution in range")
            kpi(q[2], "Your assumption", pct(yours))
            if sol is not None:
                gap = sol - yours
                st.markdown(f"Holding everything else at your base case, today's price needs **{choice.lower()} of "
                            f"{pct(sol)}** against your **{pct(yours)}**. "
                            + ("The market is more optimistic than you." if gap > 0 else
                               "You are more optimistic than the market."))
            else:
                st.warning("No value in a reasonable range matches the price with your other assumptions. "
                           "The gap is too large for this one driver to explain.")

# ---------------------------------------------------------------- 7 health
with tabs[6], guard("Health check"):
    tip("checks")
    if R:
        peer_med = None
        pdf = st.session_state.get("peers_edited", st.session_state.get("peers_df"))
        if pdf is not None and "EV/EBITDA" in pdf:
            vals = pd.to_numeric(pdf["EV/EBITDA"], errors="coerce").dropna()
            peer_med = float(vals.median()) if not vals.empty else None
        checks = health_checks(C, A, R, peer_med)
        nred = sum(1 for c_ in checks if c_[0] == "red")
        namb = sum(1 for c_ in checks if c_[0] == "amber")
        q = st.columns(3)
        kpi(q[0], "Fix", f"<span style='color:{RED}'>{nred}</span>")
        kpi(q[1], "Explain in your story", f"<span style='color:#E0A100'>{namb}</span>")
        kpi(q[2], "Fine", f"<span style='color:{TEAL}'>{len(checks) - nred - namb}</span>")
        icon = {"red": "⛔", "amber": "⚠️", "green": "✅"}
        for lvl, title, msg in sorted(checks, key=lambda x: {"red": 0, "amber": 1, "green": 2}[x[0]]):
            st.markdown(f"<div class='chk {lvl}'><b>{icon[lvl]} {title}</b><br>{msg}</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------- 8 summary
with tabs[7], guard("Summary & export"):
    tip("summary")
    if R:
        ranges = []
        dcf_lo, dcf_hi = R["per_share"], R["per_share"]
        if MC:
            dcf_lo, dcf_hi = MC["percentiles"]["P10"], MC["percentiles"]["P90"]
            lab = "DCF (FCFF), Monte Carlo P10–P90"
        elif SCN is not None and not SCN.empty:
            dcf_lo, dcf_hi = SCN["Value per share"].min(), SCN["Value per share"].max()
            lab = "DCF (FCFF), bear to bull"
        else:
            lab = "DCF (FCFF)"
        ranges.append((lab, dcf_lo, dcf_hi, R["per_share"]))
        if F:
            ranges.append(("FCFE", F["per_share"], F["per_share"], F["per_share"]))
        if D:
            ranges.append(("Dividend discount", D["per_share"], D["per_share"], D["per_share"]))
        if RI:
            ranges.append(("Residual income", RI["per_share"], RI["per_share"], RI["per_share"]))
        if REL is not None and not REL.empty:
            for _, r_ in REL.iterrows():
                ranges.append((f"Peers {r_['Multiple']}", r_["Implied low"], r_["Implied high"], r_["Implied price"]))
        fig = go.Figure()
        for name, lo, hi, mid in ranges[::-1]:
            if hi - lo < 1e-9:
                fig.add_scatter(x=[mid], y=[name], mode="markers", marker=dict(size=16, color=GOLD, symbol="diamond"),
                                showlegend=False)
            else:
                fig.add_bar(y=[name], x=[hi - lo], base=[lo], orientation="h", marker_color=NAVY, opacity=0.8,
                            showlegend=False)
                fig.add_scatter(x=[mid], y=[name], mode="markers", marker=dict(size=12, color=GOLD),
                                showlegend=False)
        fig.add_vline(x=C.price, line_color=RED, line_width=3, annotation_text=f"Price {money(C.price, C, 0)}")
        fig.update_layout(height=80 + 46 * len(ranges), title="Football field: value per share by method",
                          plot_bgcolor="white", margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, width="stretch")

        st.markdown("#### Your final call")
        methods = {"DCF (FCFF)": R["per_share"]}
        if F:
            methods["FCFE"] = F["per_share"]
        if D:
            methods["Dividend discount"] = D["per_share"]
        if RI:
            methods["Residual income"] = RI["per_share"]
        if REL is not None and not REL.empty:
            methods["Relative (median of multiples)"] = float(REL["Implied price"].median())
        wcols = st.columns(len(methods))
        weights = {}
        for (mname, v), col in zip(methods.items(), wcols):
            default = 60 if mname == "DCF (FCFF)" else (40 if mname.startswith("Relative") else 0)
            weights[mname] = col.number_input(f"{mname} weight (%)", 0, 100, default, 5, key=f"wt_{mname}")
            col.caption(money(v, C, 2))
        tw = sum(weights.values())
        final = sum(methods[k] * w_ for k, w_ in weights.items()) / tw if tw else np.nan
        band = st.slider("Margin of safety for a Buy / Sell call (%)", 0, 40, 15, key="band")
        upside = final / C.price - 1 if C.price else np.nan
        call = "BUY" if upside > band / 100 else "SELL" if upside < -band / 100 else "HOLD"
        colr = TEAL if call == "BUY" else RED if call == "SELL" else GOLD
        q = st.columns(3)
        kpi(q[0], "Blended value per share", money(final, C, 2))
        kpi(q[1], "Upside vs price", f"{upside:+.1%}")
        kpi(q[2], "Call", f"<span style='color:{colr}'>{call}</span>")
        st.caption("Teaching tool, not investment advice. The call is only as good as the story and the numbers "
                   "behind it.")

        st.markdown("#### Export")
        narr = {k: st.session_state.get(k, "") for k in STORY_KEYS}
        inputs = pd.DataFrame({"Value": {**{f"Company · {k}": v for k, v in asdict(C).items() if k != "history"},
                                         **{f"Assumption · {k}": v for k, v in asdict(A).items() if k != "wacc"},
                                         **{f"WACC input · {k}": v for k, v in asdict(A.wacc).items()}}})
        inputs["Value"] = inputs["Value"].astype(str)
        summary = pd.DataFrame({"Value": {
            "Value per share (DCF)": R["per_share"], "Equity value": R["equity"], "Operating value": R["op_value_adj"],
            "PV forecast FCFF": R["pv_fcff"], "PV terminal value": R["pv_tv"], "WACC": R["wacc_used"],
            "Terminal value share %": R["tv_share"], "Blended value per share": final, "Price": C.price,
            "Upside": upside, "Call": call}})
        sheets = {"Summary": summary, "Inputs": inputs, "DCF forecast": R["table"].T,
                  "WACC": pd.DataFrame({"Value": R["wacc"]})}
        if F:
            sheets["FCFE"] = F["table"].T
        if D:
            sheets["DDM"] = D["table"].T
        if RI:
            sheets["Residual income"] = RI["table"].T
        if REL is not None and not REL.empty:
            sheets["Relative"] = REL.set_index("Multiple")
        if SCN is not None:
            sheets["Scenarios"] = SCN.set_index("Scenario")
        if MC:
            sheets["Monte Carlo"] = pd.DataFrame({"Value": {**MC["percentiles"], "Mean": MC["mean"],
                                                            "Prob > price": MC["prob_above_price"]}})
        sheets["Narrative"] = pd.DataFrame({"Text": narr})
        e1, e2, e3 = st.columns(3)
        safe = "".join(ch if ch.isalnum() else "_" for ch in C.name)[:40]
        e1.download_button("⬇ Excel workbook", to_excel(sheets), f"{safe}_valuation.xlsx",
                           width="stretch")
        e2.download_button("⬇ Save inputs (.json)",
                           to_json(C, A, narr, {"peers": st.session_state.get("peers_edited", st.session_state["peers_df"])
                                               .to_dict(orient="list")}),
                           f"{safe}_valuation.json", width="stretch")
        memo = [f"# {C.name}: valuation memo", "",
                f"**Value per share:** {money(final, C, 2)} vs price {money(C.price, C, 2)} ({upside:+.1%}) → **{call}**",
                "", "## Story", narr["n_story"] or "_(not written)_", "", "## Opportunity", narr["n_market"] or "–",
                "", "## Edge", narr["n_edge"] or "–", "", "## Risks", narr["n_risks"] or "–", "",
                "## What would change my mind", narr["n_catalyst"] or "–", "", "## Key numbers",
                f"- Stage-1 growth {pct(A.growth_stage1)}, target margin {pct(A.target_margin)}, "
                f"sales-to-capital {A.sales_to_capital:.2f}",
                f"- WACC {pct(R['wacc_used'], 2)}, terminal growth {pct(A.terminal_growth)}, "
                f"terminal ROIC {pct(A.terminal_roic)}",
                f"- Terminal value is {pct(R['tv_share'], 0)} of operating value",
                *[f"- {k}: {money(v, C, 2)} (weight {weights[k]}%)" for k, v in methods.items()],
                "", "_Prepared with LiFin Lab Valuation Studio. Teaching tool, not investment advice._"]
        e3.download_button("⬇ One-page memo (.md)", "\n".join(memo).encode(), f"{safe}_memo.md",
                           width="stretch")
