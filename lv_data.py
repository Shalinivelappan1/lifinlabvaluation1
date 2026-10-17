"""Company data: built-in samples (offline), live fetch by ticker (Yahoo Finance), file upload."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

from lv_engine import Company

# ---------------------------------------------------------------------------
# Sample companies. FICTIONAL, for teaching: realistic orders of magnitude, not real firms.
# ---------------------------------------------------------------------------
SAMPLES = {
    "Deccan Digital Services Ltd (sample IT services, ₹ crore)": dict(
        name="Deccan Digital Services Ltd", ticker="SAMPLE-IT", sector="IT services", currency="₹",
        unit="crore", revenue=24000, ebit=5040, da=720, capex=840, nwc=3600, cash=6500, debt=400,
        minority=50, non_operating=0, options=120, shares=100, price=1050, eff_tax=0.252, interest=40,
        net_income=3850, dividends=2000, book_equity=18000, beta=0.85,
        history={"year": [2022, 2023, 2024, 2025], "revenue": [18500, 20400, 22300, 24000],
                 "ebit": [3900, 4300, 4700, 5040]}),
    "Kaveri Consumer Products Ltd (sample FMCG, ₹ crore)": dict(
        name="Kaveri Consumer Products Ltd", ticker="SAMPLE-FMCG", sector="FMCG", currency="₹",
        unit="crore", revenue=15000, ebit=3000, da=300, capex=450, nwc=300, cash=2200, debt=300,
        minority=0, non_operating=500, options=0, shares=50, price=1900, eff_tax=0.255, interest=25,
        net_income=2300, dividends=1700, book_equity=8000, beta=0.65,
        history={"year": [2022, 2023, 2024, 2025], "revenue": [12100, 13000, 14100, 15000],
                 "ebit": [2300, 2550, 2800, 3000]}),
    "Nimbus Cloud Inc (sample US SaaS, $ million)": dict(
        name="Nimbus Cloud Inc", ticker="SAMPLE-SAAS", sector="Software (SaaS)", currency="$",
        unit="million", revenue=4200, ebit=380, da=260, capex=300, nwc=-400, cash=2800, debt=1500,
        minority=0, non_operating=0, options=600, shares=310, price=68, eff_tax=0.21, interest=45,
        net_income=330, dividends=0, book_equity=3200, beta=1.3,
        history={"year": [2022, 2023, 2024, 2025], "revenue": [2300, 2900, 3550, 4200],
                 "ebit": [-120, 60, 230, 380]}),
}

SAMPLE_PEERS = {
    "SAMPLE-IT": pd.DataFrame({
        "Peer": ["Peer IT A", "Peer IT B", "Peer IT C", "Peer IT D", "Peer IT E"],
        "P/E": [24.0, 27.5, 31.0, 22.0, 29.0], "EV/EBITDA": [16.5, 18.0, 21.5, 15.0, 19.5],
        "EV/Sales": [3.6, 4.1, 5.0, 3.2, 4.4], "P/B": [6.5, 7.8, 9.0, 5.6, 8.1]}),
    "SAMPLE-FMCG": pd.DataFrame({
        "Peer": ["Peer FMCG A", "Peer FMCG B", "Peer FMCG C", "Peer FMCG D"],
        "P/E": [52.0, 45.0, 38.0, 60.0], "EV/EBITDA": [36.0, 31.0, 27.0, 41.0],
        "EV/Sales": [7.5, 6.2, 5.1, 8.8], "P/B": [12.0, 10.5, 8.0, 15.0]}),
    "SAMPLE-SAAS": pd.DataFrame({
        "Peer": ["Peer SaaS A", "Peer SaaS B", "Peer SaaS C", "Peer SaaS D", "Peer SaaS E"],
        "P/E": [65.0, 48.0, 90.0, 55.0, 70.0], "EV/EBITDA": [32.0, 25.0, 45.0, 28.0, 36.0],
        "EV/Sales": [6.0, 4.8, 9.5, 5.4, 7.2], "P/B": [8.0, 6.5, 12.0, 7.0, 9.5]}),
}

EMPTY_PEERS = pd.DataFrame({"Peer": ["Peer 1", "Peer 2", "Peer 3"], "P/E": [np.nan] * 3,
                            "EV/EBITDA": [np.nan] * 3, "EV/Sales": [np.nan] * 3, "P/B": [np.nan] * 3})

FIELDS = [
    ("name", "Company name", "text"), ("ticker", "Ticker", "text"), ("sector", "Sector", "text"),
    ("currency", "Currency symbol (₹, $, €)", "text"), ("unit", "Money unit (crore or million)", "text"),
    ("revenue", "Revenue, last full year", "num"), ("ebit", "EBIT (operating profit)", "num"),
    ("da", "Depreciation & amortisation", "num"), ("capex", "Capital expenditure", "num"),
    ("nwc", "Net working capital (non-cash)", "num"), ("cash", "Cash & liquid investments", "num"),
    ("debt", "Total debt (incl. lease liabilities)", "num"), ("minority", "Minority interest", "num"),
    ("non_operating", "Non-operating assets / cross-holdings", "num"),
    ("options", "Value of employee options (ESOPs)", "num"), ("shares", "Shares outstanding", "num"),
    ("price", "Current share price", "num"), ("eff_tax", "Effective tax rate (decimal)", "num"),
    ("interest", "Interest expense", "num"), ("net_income", "Net income", "num"),
    ("dividends", "Total dividends paid", "num"), ("book_equity", "Book value of equity", "num"),
    ("beta", "Regression beta", "num"),
]


def sample_company(key: str) -> Company:
    return Company(**SAMPLES[key])


def template_csv() -> bytes:
    c = sample_company(list(SAMPLES)[0])
    rows = [{"field": f, "description": d, "value": getattr(c, f)} for f, d, _ in FIELDS]
    rows += [{"field": f"rev_{y}", "description": f"Revenue in {y} (history, optional)", "value": r}
             for y, r in zip(c.history["year"], c.history["revenue"])]
    return pd.DataFrame(rows).to_csv(index=False).encode()


def company_from_upload(file) -> Company:
    name = getattr(file, "name", "upload.csv").lower()
    raw = file.read() if hasattr(file, "read") else file
    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(raw))
    else:
        df = pd.read_csv(io.BytesIO(raw))
    df.columns = [str(x).strip().lower() for x in df.columns]
    kv = dict(zip(df["field"].astype(str).str.strip(), df["value"]))
    kwargs = {}
    for f, _, kind in FIELDS:
        if f in kv and pd.notna(kv[f]):
            kwargs[f] = str(kv[f]) if kind == "text" else float(kv[f])
    hist = sorted((int(k[4:]), float(v)) for k, v in kv.items() if k.startswith("rev_") and pd.notna(v))
    if hist:
        kwargs["history"] = {"year": [y for y, _ in hist], "revenue": [r for _, r in hist], "ebit": []}
    return Company(**kwargs)


# ---------------------------------------------------------------------------
# Live data (needs internet). Every number stays editable afterwards.
# ---------------------------------------------------------------------------
def _pick(df, names):
    if df is None or df.empty:
        return None
    for n in names:
        if n in df.index:
            s = pd.to_numeric(df.loc[n], errors="coerce").dropna()
            if not s.empty:
                return s
    return None


def fetch_ticker(ticker: str):
    """Return (Company, warnings). Indian tickers end in .NS or .BO and are converted to Rs crore."""
    import yfinance as yf

    warnings = []
    tk = yf.Ticker(ticker)
    try:
        info = tk.info or {}
    except Exception as e:  # network or API change
        info = {}
        warnings.append(f"Could not read company info ({e}).")
    inc = getattr(tk, "income_stmt", None)
    cf = getattr(tk, "cashflow", None)
    bs = getattr(tk, "balance_sheet", None)
    cur = info.get("currency") or ("INR" if ticker.upper().endswith((".NS", ".BO")) else "USD")
    if cur == "INR":
        scale, sym, unit = 1e7, "₹", "crore"
    else:
        scale, sym, unit = 1e6, {"USD": "$", "EUR": "€", "GBP": "£"}.get(cur, cur + " "), "million"

    def latest(df, names, label, default=0.0, sign=1.0):
        s = _pick(df, names)
        if s is None:
            warnings.append(f"{label} not found: set it by hand.")
            return default
        return sign * float(s.iloc[0]) / scale

    rev_s = _pick(inc, ["Total Revenue", "Operating Revenue"])
    ebit_s = _pick(inc, ["Operating Income", "EBIT"])
    revenue = float(rev_s.iloc[0]) / scale if rev_s is not None else 0.0
    if rev_s is None:
        warnings.append("Revenue not found: set it by hand.")
    ebit = float(ebit_s.iloc[0]) / scale if ebit_s is not None else 0.0
    if ebit_s is None:
        warnings.append("EBIT not found: set it by hand.")
    pretax = _pick(inc, ["Pretax Income"])
    taxp = _pick(inc, ["Tax Provision"])
    eff_tax = 0.25
    if pretax is not None and taxp is not None and float(pretax.iloc[0]) > 0:
        eff_tax = float(np.clip(float(taxp.iloc[0]) / float(pretax.iloc[0]), 0.0, 0.45))
    else:
        warnings.append("Effective tax rate estimated at 25%.")
    ca = _pick(bs, ["Current Assets"])
    cl = _pick(bs, ["Current Liabilities"])
    cash_s = _pick(bs, ["Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents"])
    nwc = 0.0
    if ca is not None and cl is not None:
        nwc = (float(ca.iloc[0]) - float(cl.iloc[0]) - (float(cash_s.iloc[0]) if cash_s is not None else 0)) / scale
    else:
        warnings.append("Working capital not found: set it by hand.")
    hist_years, hist_rev, hist_ebit = [], [], []
    if rev_s is not None:
        for d, v in list(rev_s.items())[::-1]:
            hist_years.append(getattr(d, "year", str(d)))
            hist_rev.append(float(v) / scale)
            if ebit_s is not None and d in ebit_s.index:
                hist_ebit.append(float(ebit_s[d]) / scale)
    # shares: company info -> fast_info -> balance sheet -> income statement average shares
    shares_raw = info.get("sharesOutstanding") or 0
    fi = None
    try:
        fi = tk.fast_info
    except Exception:
        fi = None
    if not shares_raw and fi is not None:
        try:
            shares_raw = float(fi["shares"] or 0)
        except Exception:
            shares_raw = 0
    if not shares_raw:
        s_ = _pick(bs, ["Ordinary Shares Number", "Share Issued"])
        if s_ is not None:
            shares_raw = float(s_.iloc[0])
    if not shares_raw:
        s_ = _pick(inc, ["Diluted Average Shares", "Basic Average Shares"])
        if s_ is not None:
            shares_raw = float(s_.iloc[0])
    shares = shares_raw / scale
    if not shares:
        warnings.append("Shares outstanding not found: enter it on the Company tab.")
    # price: company info -> fast_info -> last close
    price = float(info.get("currentPrice") or info.get("regularMarketPrice") or 0.0)
    if not price and fi is not None:
        try:
            price = float(fi["last_price"] or 0.0)
        except Exception:
            price = 0.0
    if not price:
        try:
            h_ = tk.history(period="5d")
            if not h_.empty:
                price = float(h_["Close"].dropna().iloc[-1])
        except Exception:
            pass
    if not price:
        warnings.append("Share price not found: enter it on the Company tab.")
    c = Company(
        name=info.get("longName") or info.get("shortName") or ticker, ticker=ticker,
        sector=info.get("sector") or info.get("industry") or "", currency=sym, unit=unit,
        revenue=revenue, ebit=ebit,
        da=latest(cf, ["Depreciation And Amortization", "Depreciation Amortization Depletion"], "D&A"),
        capex=latest(cf, ["Capital Expenditure"], "Capex", sign=-1.0),
        nwc=nwc,
        cash=(float(cash_s.iloc[0]) / scale) if cash_s is not None else (info.get("totalCash") or 0) / scale,
        debt=latest(bs, ["Total Debt"], "Total debt", default=(info.get("totalDebt") or 0) / scale),
        minority=latest(bs, ["Minority Interest"], "Minority interest"),
        non_operating=0.0, options=0.0, shares=shares,
        price=price,
        eff_tax=eff_tax,
        interest=latest(inc, ["Interest Expense", "Interest Expense Non Operating"], "Interest expense"),
        net_income=latest(inc, ["Net Income", "Net Income Common Stockholders"], "Net income"),
        dividends=latest(cf, ["Cash Dividends Paid", "Common Stock Dividend Paid"], "Dividends", sign=-1.0),
        book_equity=latest(bs, ["Stockholders Equity", "Common Stock Equity"], "Book equity"),
        beta=float(info.get("beta") or 1.0),
        history={"year": hist_years, "revenue": hist_rev, "ebit": hist_ebit},
    )
    if c.capex < 0:
        c.capex = -c.capex
    if c.dividends < 0:
        c.dividends = -c.dividends
    if not info.get("beta"):
        warnings.append("Beta not available: defaulted to 1.0.")
    return c, warnings
