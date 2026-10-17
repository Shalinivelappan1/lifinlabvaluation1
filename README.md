# LiFin Lab · Equity Valuation Studio

An end-to-end stock valuation app in Streamlit: from company data and a written story, through the cost of
capital and a full FCFF DCF, to cross-checks, risk analysis and a final call. Built for MBA teaching, with a
**Learning / Expert** toggle.

Curated by Prof. Shalini Velappan, IIM Tiruchirappalli · LiFin Lab. Teaching tool, not investment advice.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

The browser opens at http://localhost:8501. Python 3.10 or later.

**Share it with a class:** push this folder to a GitHub repository and deploy it free on
[Streamlit Community Cloud](https://streamlit.io/cloud) (main file: `app.py`). Students then need only a link.

## The eight tabs

| Tab | What it does |
|---|---|
| ① Company | Every input is editable: revenue, EBIT, D&A, capex, working capital, cash, debt, minority interest, options, shares, price, tax, beta. Shows market cap, EV, margin, ROIC and sales-to-capital today. |
| ② Story & assumptions | The "art" and the value drivers together: pick a story archetype (steady compounder, high-growth disruptor, mature cash cow, cyclical at the peak, turnaround) and apply it, write your narrative, then set every value driver beside it: forecast years, high-growth years, growth, year-1 and target margins, reinvestment (sales-to-capital or capex/D&A/NWC), terminal value method, terminal growth and ROIC, mature WACC, mid-year convention, probability of failure. |
| ③ Cost of capital | CAPM with country risk (scaled by beta or not), size and company-specific premiums; regression, Blume-adjusted or bottom-up relevered beta; synthetic rating from interest coverage or a direct cost of debt; target weights; optional override. |
| ④ Forecast & DCF | An optional year-by-year table: type revenue growth and EBIT margin for each year yourself (overrides the fade; reset any time). Then the full year-by-year forecast (revenue, margin, EBIT, tax, NOPAT, D&A, capex, reinvestment, FCFF, WACC, discount factor, PV, invested capital, ROIC), the FCFF chart and the bridge from operating value to equity. The forecast shows even if shares or price are missing. |
| ⑤ Other models | FCFE (constant debt ratio), two-stage dividend discount, residual income, and relative valuation on an editable peer table (median, mean or harmonic mean). |
| ⑥ Risk & judgement | Two-way sensitivity tables (WACC × terminal growth, growth × margin), tornado chart, editable bear/base/bull scenarios with probabilities, Monte Carlo with triangular distributions and growth–margin correlation, and reverse DCF (what growth, margin or terminal growth the price implies). |
| ⑦ Health check | Red/amber/green flags: terminal growth above the risk-free rate, terminal value share, ROIC vs WACC in perpetuity, extreme forecast ROIC, aggressive margin expansion, odd beta, cost of debt below risk-free, implied exit multiple vs peers, value far from price. |
| ⑧ Summary & export | Football field of all methods against the price, your weights for each method, a blended value and a Buy/Hold/Sell call with your margin of safety. Downloads: Excel workbook of every table, the full inputs as JSON (restore later from the sidebar), and a one-page memo. |

## Getting company data

- **Sample companies** (work offline): three *fictional* firms with realistic numbers: an Indian IT services
  company and an Indian FMCG company in ₹ crore, and a US SaaS company in $ million, each with a peer set.
- **Live ticker** (needs internet): any Yahoo Finance ticker. Indian stocks end in `.NS` (NSE) or `.BO` (BSE)
  and are converted to ₹ crore and crore shares. Fields the feed lacks are listed so you can fill them in.
  Always check the numbers; free data feeds have gaps and reclassifications.
- **Upload**: download the CSV template from the sidebar, fill it in, upload it (CSV or Excel).

Units: money in one unit (₹ crore or $ million) and shares in the matching unit (crore or million), so
equity value ÷ shares is the price per share.

## How the DCF works

For each year t: revenue grows at g_t; EBIT = revenue × margin_t; NOPAT = EBIT × (1 − tax_t);
reinvestment = Δrevenue ÷ sales-to-capital (or capex − D&A + ΔNWC); FCFF = NOPAT − reinvestment.
Cash flows are discounted at a WACC that can move from today's rate to a mature rate after the high-growth
years. Terminal value = NOPAT_{n+1} × (1 − g/ROIC) ÷ (WACC − g), or EBITDA_n × exit multiple.
Equity = operating value (adjusted for failure risk) + cash + non-operating assets − debt − minority
interest − employee options.

## Checks

`test_engine.py` recomputes the DCF, WACC, DDM, residual income, mid-year adjustment, exit-multiple
round trip, reverse DCF and failure adjustment independently, and runs every model on every sample:

```bash
python -m pytest -q test_engine.py
```

## Notes for instructors

- The synthetic-rating spreads are illustrative: update `RATING_TABLE` in `lv_wacc.py` each year
  (for example from Damodaran's data page). Default risk-free rates, ERP and country risk are starting points:
  update them on the day you teach.
- The sample IT company deliberately values below its price with default inputs. That is the teaching
  moment: run the reverse DCF to see what growth the price implies, then argue about the story.
