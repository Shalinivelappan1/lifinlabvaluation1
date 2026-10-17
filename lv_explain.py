"""Plain-language explanations shown in Learning mode."""

TIPS = {
    "company": "Start from the last full financial year. Money is in one unit (₹ crore or $ million) and shares "
               "in the matching unit (crore or million), so equity value ÷ shares gives the price per share. "
               "Every number here can be edited, even after a live fetch.",
    "story": "Valuation is a story told with numbers. First say in words what the company is, how big it can get "
             "and what could go wrong. Then every number should trace back to a sentence in that story. A number "
             "with no story behind it is a guess; a story with no numbers is a dream.",
    "rf": "Risk-free rate: the yield on a long-term government bond in the same currency as the cash flows "
          "(10-year G-sec for ₹, 10-year Treasury for $). It is the return for waiting, with no default risk.",
    "erp": "Equity risk premium: the extra return investors demand to hold a diversified stock portfolio instead "
           "of the risk-free asset. Use a mature-market figure here and add country risk separately.",
    "crp": "Country risk premium: extra premium for operating in a riskier economy. Scaling it by beta assumes a "
           "company's exposure to country risk moves with its market risk.",
    "beta": "Beta measures how much the stock moves with the market. A regression beta is noisy; a bottom-up "
            "beta takes the average unlevered beta of the industry and relevers it at the company's own debt.",
    "kd": "Cost of debt: what the firm pays to borrow today, not the coupon on old loans. The synthetic rating "
          "reads it off interest coverage (EBIT ÷ interest).",
    "wacc": "WACC weights the cost of equity and the after-tax cost of debt by target market-value weights. It is "
            "the return the firm's operations must earn to satisfy all capital providers.",
    "growth": "Revenue growth is the most story-driven number. High growth can last only as long as the market "
              "is large and the firm keeps an edge; after that it fades toward the economy's growth rate.",
    "margin": "Operating margin: what the business keeps from each rupee of sales before interest and tax. The "
              "target margin is where the story says the firm ends up when it matures.",
    "s2c": "Sales-to-capital ratio: rupees of extra revenue generated per rupee reinvested. Higher means growth "
           "is cheaper. Reinvestment = change in revenue ÷ sales-to-capital.",
    "terminal": "After the forecast, the firm grows at a constant rate for ever. Terminal growth cannot exceed "
                "the economy's growth (use the risk-free rate as a cap). Reinvestment in perpetuity = g ÷ ROIC, "
                "so growth is not free.",
    "roic": "ROIC in perpetuity: the return on new capital once the firm is mature. If it equals the WACC, growth "
            "adds no value; only ROIC above WACC creates value.",
    "exit": "An exit multiple values the firm at the end of the forecast like its peers today. It is quick but "
            "imports the market's mood into your intrinsic value.",
    "mid_year": "Mid-year convention: cash arrives through the year, not on 31 March, so each year's cash flow is "
                "discounted half a year less.",
    "failure": "Failure risk: the chance the firm does not survive as a going concern. DCF assumes survival, so "
               "blend in a distress value for young or troubled firms.",
    "forecast": "This is the engine room. Each column is one year: revenue grows, EBIT margin turns revenue into "
                "operating profit, tax leaves NOPAT, reinvestment is what growth costs, and what is left is free cash "
                "flow to the firm (FCFF). Discount each year's FCFF, add the terminal value, and you have the value "
                "of the operations. Switch on the year-by-year table to shape the path yourself.",
    "fcfe": "FCFE values equity directly: cash left for shareholders after debt payments and new borrowing, "
            "discounted at the cost of equity. It should land near the FCFF value; a big gap means the debt "
            "assumptions differ.",
    "ddm": "Dividend discount model: value = present value of expected dividends. Works for stable, high-payout "
           "firms; it undervalues firms that hoard cash or buy back shares.",
    "ri": "Residual income: value = book value today + present value of profits above the cost of equity. "
          "If ROE equals the cost of equity, the stock is worth exactly its book value.",
    "relative": "Relative valuation prices the company like its peers. It tells you what the market is paying, "
                "not what the company is worth. Use the median: averages get pulled by outliers.",
    "sensitivity": "Sensitivity tables change two inputs at a time. If small changes move the value a lot, the "
                   "conclusion is fragile; say so.",
    "scenarios": "Scenarios tell three coherent stories (bad, expected, good), not three random number sets. "
                 "Weight them by how likely you think each is.",
    "mc": "Monte Carlo draws thousands of input combinations from ranges you set and shows the spread of value. "
          "It does not remove uncertainty; it shows it honestly.",
    "reverse": "Reverse DCF asks: what must I believe to justify today's price? If the implied growth looks "
               "impossible, the stock is expensive whatever the multiples say.",
    "checks": "Health checks flag assumptions that break economic logic. Amber means 'explain this in your "
              "story'; red means 'fix it'.",
    "summary": "Different methods answer different questions. The football field shows where they agree. Weight "
               "them by how much you trust each for this company, then compare with the price.",
}
